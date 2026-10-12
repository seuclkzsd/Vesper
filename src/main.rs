//! GOSIM survey26 team agent ("rust-pro"), participant-agent-protocol-v4. Started from the official rust-pro port;
//! the planner, the fault detection and the src/llm/ stages are rewritten. One JSON object per line on stdin and
//! stdout, logs on stderr. One search per decision picks pointing, fibres, duration and program (planner/); fault
//! reports come from a change-point detector (fault.rs) and, on duty-log cards, from the duty log (duty.rs).
//!
//! LLM stages (src/llm/, table in README.md), on by default, asked at fixed decision points with a bounded wait,
//! answers validated and clamped by rules, rule fallback on any failure, a hard per-run call budget:
//!    - week plan & review (`llm::week_plan`): the per-night plan of a forecast week and its time price;
//!    - fault judge (`llm::fault_judge`): a fault report the rules decided on goes now or after one confirmation;
//!    - request triage (`llm::request_triage`): the priority of each new observation request;
//!    - duty log (`llm::duty_log`): the rule decoder (planner/dutylog/) reads every duty-log message; the lines it is not
//!      sure of go to the model, whose facts (fault times, stops, closures, terrain) join the decoder's view.
//!    Without an API key, or under OBSERVER_MODEL_DISABLED=1, the agent runs on its rules alone.

mod alloc;
mod band;
#[path = "planner/duty.rs"]
mod duty; // the planner's duty-log facts (kept with the planner)
#[path = "planner/dutylog/mod.rs"]
mod dutylog; // the duty-log rule decoder
mod fault;
mod guard;
mod llm;
mod planner;
mod requests;
mod seasonlp;
mod skymath;
mod terrain;
mod urgency;

use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{BufRead, Write};
use std::time::Instant;

use duty::Duty;
use fault::FaultMonitor;
use llm::client::{api_key, load_dotenv, model_disabled, LlmClient};
use llm::fault_judge::FaultJudge;
use llm::week_plan::{self, WeekPlanner};
use llm::duty_log::{self, DutyLog as DutyQuestions};
use llm::{request_triage, Quotas};
use planner::{env_f, env_i, Planner};

#[global_allocator]
static ALLOC: alloc::Alloc = alloc::Alloc;
use skymath::{format_date, format_hour_stamp, format_utc, parse_utc, psum, round_to};

const PROTOCOL: &str = "participant-agent-protocol-v4";

pub fn log(text: &str) {
    let mut err = std::io::stderr().lock();
    let _ = writeln!(err, "{text}");
    let _ = err.flush();
}

/// User + system CPU time of this process (all threads), like Python's `time.process_time()`.
#[cfg(unix)]
fn process_time() -> f64 {
    // SAFETY: getrusage only writes into the zeroed struct we hand it.
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) } != 0 {
        return 0.0;
    }
    let seconds = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 / 1e6;
    seconds(usage.ru_utime) + seconds(usage.ru_stime)
}

#[cfg(not(unix))]
fn process_time() -> f64 {
    0.0
}

/// Tunable constants of the agent (each one overridable with `PRO_<NAME>`).
struct Knobs {
    // --- pace ---
    pace_safety: f64,
    // --- LLM stages ---
    llm_wall_share: f64, // all LLM waits of a run together: at most this share of the wall-clock cap
    duty_timeout: f64,   // longest a background duty-log question may run (long answers; no decision waits)
    duty_chars: usize,   // decoded text of one duty-log question, at most (bytes; long messages go in chunks)
    fixed_level: i64,      // testing only: pin the search level (deterministic runs)
    wide: bool,            // level 0 starts with the wide search (0: the narrow search only, the single-breadth ladder)
    xtier: bool,           // the xwide breadth between wide and mid (0: wide, mid, narrow)
}

impl Knobs {
    fn load() -> Knobs {
        Knobs {
            pace_safety: env_f("PACE_SAFETY", 0.75),
            llm_wall_share: env_f("LLM_WALL_SHARE", 0.4),
            duty_timeout: env_f("DUTY_TIMEOUT", 180.0),
            duty_chars: env_i("DUTY_CHARS", 2000).max(300) as usize,
            fixed_level: env_i("FIXED_LEVEL", -1),
            wide: env_i("WIDE", 1) != 0,
            xtier: env_i("XTIER", 1) != 0,
        }
    }
}

/// Log throughput below which a reading is the instrument, not the sky: 1 % of normal (a dead fault is 0.5 %; the
/// strongest earthquake night is 5 % (max_degradation 0.95), times an overcast sky about 2 %).
const DEEP_Y: f64 = -4.6;

fn median_of(values: &[f64]) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[v.len() / 2]
}

fn action_wait_until(until: f64, reason: &str) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("action".into(), json!("wait"));
    m.insert("until_utc".into(), json!(format_utc(until)));
    m.insert("reason".into(), json!(reason));
    m
}

fn action_wait_for(seconds: i64, reason: &str) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("action".into(), json!("wait"));
    m.insert("duration_seconds".into(), json!(seconds));
    m.insert("reason".into(), json!(reason));
    m
}

fn action_finish(reason: &str) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("action".into(), json!("finish"));
    m.insert("reason".into(), json!(reason));
    m
}

/// Counters at the last week plan, for the review of the period since.
#[derive(Clone, Copy, Default)]
struct PeriodMark {
    night: usize,
    observes: i64,
    running_total: f64,
    required_done: usize,
    reports: i64,
    correct_reports: i64,
}

struct ObserverAgent {
    k: Knobs,
    planner: Planner,
    // LLM stages (src/llm/)
    client: LlmClient,
    quotas: Quotas,
    week: WeekPlanner,
    judge: FaultJudge,
    forecast_pending: Option<Value>, // a forecast not yet planned (planned at the next night start)
    plan_mark: PeriodMark,
    triaged: BTreeSet<String>, // observation requests already shown to the request triage
    dlog: dutylog::DutyLog,    // the duty-log rule decoder: every chat message decoded, the card's view resolved
    dl_llm: DutyQuestions,     // LLM duty-log stage: merged facts and the chunks the model has not read
    dl_asked: Option<duty_log::Asked>, // the duty-log question running in the background
    chat_quota: usize,         // calls added for the duty-log stage when a card has a chat
    chat_extended: bool,
    utc_off: i64,              // site UTC offset, minutes (chat times are local)
    last_repair: f64,          // time of the last correct report
    duty: Duty,                // duty-log facts (duty.rs); empty on cards without a duty log
    duty_y: Vec<(f64, f64, f64)>, // (start, end, log throughput) of readings outside dead time
    duty_probe: Option<usize>, // a duty-log stop whose probe exposure is in flight
    duty_alarm: Option<(f64, bool)>, // an unexplained drop in the readings: (start of the first low reading, held 35 min)
    duty_kinds: BTreeSet<String>, // ALL-sky notice kinds at the last reading
    duty_ref_from: f64,        // readings since the last report are the fallback's reference level
    duty_deep: Option<(f64, f64, u32)>, // a run of readings below 1 % of normal throughput: (start, end, readings)
    duty_rep: f64,             // time of the last correct or deep report (a deep run must start after it)
    deep_sent: bool,           // the report in flight is a deep-drop report
    deep_paid_false: i64,      // deep-drop reports that were paid and false
    held: Option<(f64, String, i64)>, // (hours, reason, observes): a rule report held for a confirmation exposure
    wall_at_start: Option<f64>,
    running_total: f64,
    model_wait: f64, // wall seconds spent waiting for the model (not planning cost)
    scale_hours: BTreeMap<i64, Vec<f64>>, // hour -> [planner.scale samples] (for the fault judge's table)
    start: f64,
    forecast_notices: Vec<Value>,
    night_seen: Option<usize>,
    observes: i64,
    // fault reporting state
    faults: FaultMonitor,
    reports: i64,
    correct_reports: i64,
    paid_false: i64,
    /// Report actions since the last other action, and the engine's limit on them (scoring.reporting).
    consecutive_reports: i64,
    max_consecutive_reports: i64,
    last_report_hours: f64,
    ref_from_hours: f64,
    quake_on: bool,
    quake_onset_hours: f64,
    // pace state
    // pace tiers: 0 = level 0 wide, 1 = xwide, 2 = mid, 3 = narrow, 4..6 = levels 1..3, 7 = level 4 (out of time)
    tier: usize,
    cost_ema: [f64; 7], // CPU seconds per observe decision at each tier
    wall_ema: [f64; 7], // real seconds per observe decision (own turn, model waits excluded)
    turn_end: Option<Instant>,
    decisions: i64,
    engine_ema: Option<f64>,
    sim_step_ema: Option<f64>,
    last_now: Option<f64>,
}

impl ObserverAgent {
    fn new(init: &Value) -> ObserverAgent {
        let started = Instant::now();
        let planner = Planner::new(init);
        let client = LlmClient::new();
        let quotas = Quotas::split(client.budget());
        // one forecast every 7 days: the week plans of the season share the stage's quota
        let span_days = match (planner.nights.first(), planner.nights.last()) {
            (Some(a), Some(b)) => (b.0 - a.0) / 86400.0 + 1.0,
            _ => 0.0,
        };
        let week = WeekPlanner::new((span_days / 7.0).ceil().max(1.0) as usize, quotas.week_plan);
        let faults = FaultMonitor::new(init, &planner.nights, &planner.night_slots, planner.slot_seconds);
        let k = Knobs::load();
        let tier = if k.wide { 0 } else { 3 };
        let agent = ObserverAgent {
            k,
            start: parse_utc(init["survey"]["start_utc"].as_str().unwrap_or("")),
            judge: FaultJudge::new(
                init["scoring"]["reporting"]["correct_reward"].as_f64().unwrap_or(100.0),
                init["scoring"]["reporting"]["false_penalty"].as_f64().unwrap_or(-150.0),
            ),
            quotas,
            week,
            forecast_pending: None,
            plan_mark: PeriodMark::default(),
            triaged: BTreeSet::new(),
            dlog: dutylog::DutyLog::new(),
            dl_llm: DutyQuestions::default(),
            dl_asked: None,
            chat_quota: env_i("CHAT_CALLS", 24).max(0) as usize,
            chat_extended: false,
            utc_off: (init["site"]["utc_offset_hours"].as_f64().unwrap_or(0.0) * 60.0).round() as i64,
            last_repair: -1e18,
            duty: Duty::default(),
            duty_y: Vec::new(),
            duty_probe: None,
            duty_alarm: None,
            duty_kinds: BTreeSet::new(),
            duty_ref_from: -1e18,
            duty_deep: None,
            duty_rep: -1e18,
            deep_sent: false,
            deep_paid_false: 0,
            held: None,
            wall_at_start: None,
            running_total: 0.0,
            model_wait: 0.0,
            scale_hours: BTreeMap::new(),
            forecast_notices: Vec::new(),
            night_seen: None,
            observes: 0,
            faults,
            reports: 0,
            correct_reports: 0,
            paid_false: 0,
            consecutive_reports: 0,
            max_consecutive_reports: {
                let from = |v: &Value| v.as_f64().filter(|m| *m >= 1.0).map(|m| m as i64);
                from(&init["scoring"]["reporting"]["max_consecutive_reports"])
                    .or_else(|| from(&init["limits"]["max_consecutive_reports"]))
                    .unwrap_or(32)
            },
            last_report_hours: -1e9,
            ref_from_hours: -1e9,
            quake_on: false,
            quake_onset_hours: -1e9,
            tier,
            cost_ema: [0.0; 7],
            wall_ema: [0.0; 7],
            turn_end: None,
            decisions: 0,
            engine_ema: None,
            sim_step_ema: None,
            last_now: None,
            planner,
            client,
        };
        log(&format!(
            "pro: {} targets, {} required, {} nights; init {:.2}s; model {} (call budget {}: week plan {}, fault judge {}, request triage {})",
            agent.planner.ids.len(),
            agent.planner.required.iter().filter(|&&r| r).count(),
            agent.planner.nights.len(),
            started.elapsed().as_secs_f64(),
            agent.client.model,
            agent.client.calls_left(),
            agent.quotas.week_plan,
            agent.quotas.fault_judge,
            agent.quotas.request_triage
        ));
        agent
    }

    // --- decision loop ------------------------------------------------------------------------------

    fn respond(&mut self, payload: &Value) -> Map<String, Value> {
        let started = Instant::now();
        let cpu_started = process_time();
        if let Some(end) = self.turn_end {
            // engine time between our turns (charged only by the old real-time clock)
            let gap = started.duration_since(end).as_secs_f64();
            if (0.0..5.0).contains(&gap) {
                self.engine_ema = Some(match self.engine_ema {
                    None => gap,
                    Some(e) => 0.95 * e + 0.05 * gap,
                });
            }
        }
        let model_before = self.model_wait;
        let tier = self.tier;
        let action = self.respond_inner(payload);
        // last check against the engine's action rules (guard.rs): a valid action passes unchanged
        let action = {
            let now = parse_utc(payload["now_utc"].as_str().unwrap_or(""));
            let fallback_wait = match self.planner.current_night(now) {
                Some((_, night_start, _)) => self.to_next_slot(now, night_start),
                None => self.clamp_wait(900.0),
            };
            let lim = guard::Limits {
                min_exposure: self.planner.min_exposure as i64,
                max_exposure: self.planner.max_exposure as i64,
                n_fibers: self.planner.n_fibers(),
                now,
                fallback_wait,
            };
            let planner = &self.planner;
            guard::check(action, &lim, |id| planner.is_target(id))
        };
        // the engine rejects a report past max_consecutive_reports in a row (any other action resets the count)
        if action.get("action").and_then(Value::as_str) == Some("report") {
            self.consecutive_reports += 1;
        } else {
            self.consecutive_reports = 0;
        }
        // the platform charges CPU time inside our turns; waiting for the model is free of CPU, so keep it
        // out of the real-time estimate as well
        let cpu = process_time() - cpu_started;
        let wall = started.elapsed().as_secs_f64() - (self.model_wait - model_before);
        // the charge is min(turn, CPU of the container): with two search threads that is the turn
        let cpu = if self.planner.threads() > 1 { cpu.min(wall) } else { cpu };
        if action.get("action").and_then(Value::as_str) == Some("observe") && tier < 7 {
            let xtier = self.k.xtier;
            for (ema, cost) in [(&mut self.cost_ema, cpu), (&mut self.wall_ema, wall)] {
                let c = ema[tier];
                ema[tier] = if c == 0.0 { cost } else { 0.9 * c + 0.1 * cost };
                for k in tier + 1..7 {
                    // cheaper tiers not measured yet (empirical ratios): mid about 0.6 of wide (xwide 0.8 of wide,
                    // mid 0.75 of xwide), narrow about 0.6 of mid, each lower level a third of the one above
                    if k == 1 && !xtier {
                        continue;
                    }
                    let (above, ratio) = match k {
                        1 => (0, 0.8),
                        2 if xtier => (1, 0.75),
                        2 => (0, 0.6),
                        3 => (2, 0.6),
                        _ => (k - 1, 1.0 / 3.0),
                    };
                    if ema[k] == 0.0 || ema[k] > ema[above] {
                        ema[k] = ema[above] * ratio;
                    }
                }
            }
        }
        self.turn_end = Some(Instant::now());
        action
    }

    fn respond_inner(&mut self, payload: &Value) -> Map<String, Value> {
        let now_utc = payload["now_utc"].as_str().unwrap_or("").to_string();
        let now = parse_utc(&now_utc);
        if let Some(last_now) = self.last_now {
            if self.planner.current_night(now).is_some() {
                let step = now - last_now;
                if 0.0 < step && step <= 3600.0 {
                    self.sim_step_ema = Some(match self.sim_step_ema {
                        None => step,
                        Some(e) => 0.95 * e + 0.05 * step,
                    });
                }
            }
        }
        self.last_now = Some(now);
        let hours = (now - self.start) / 3600.0;
        let empty = Vec::new();
        let messages = payload.get("new_messages").and_then(Value::as_array).unwrap_or(&empty);
        for message in messages {
            if message.get("record_type").and_then(Value::as_str) == Some("forecast") {
                self.forecast_notices = message.get("notices").and_then(Value::as_array).cloned().unwrap_or_default();
                self.forecast_pending = Some(message.clone());
            }
        }
        if self.wall_at_start.is_none() {
            self.wall_at_start = Some(Self::clock(payload).1);
        }
        self.running_total = payload.get("running_total").map(planner::num).unwrap_or(self.running_total);
        let last = payload.get("last_result").cloned().unwrap_or(Value::Null);
        if last.get("action").and_then(Value::as_str) == Some("report") {
            self.on_report_result(&last, hours, now);
        }
        let bulletin = payload.get("latest_bulletin").cloned().unwrap_or(Value::Null);
        self.planner.on_messages(messages, &bulletin);
        let quake = self.planner.notices.iter().any(|(kind, _)| kind == "earthquake");
        if quake && !self.quake_on {
            self.quake_onset_hours = hours;
            log(&format!("pro: earthquake notice at {now_utc}"));
        }
        self.quake_on = quake;
        self.planner.quake_active = quake;
        self.read_chat(messages, now, payload);
        let duty = self.duty.active();
        if duty {
            self.planner.duty_closed = self.duty.closed(now);
            self.planner.duty_closing = self.duty.closing(now, self.planner.max_exposure);
            if self.duty.has_terrain() {
                self.planner.terrain_top = self.duty.terrain(now);
            }
        }
        let requests = payload.get("active_requests").and_then(Value::as_array).unwrap_or(&empty);
        self.triage_new_requests(requests, now, payload);
        self.planner.on_requests(requests, now);
        // index the engine gives the next observe (for an exact state_resync); own count as a fallback
        self.planner.observe_index = Some(payload.get("observe_action_index").and_then(Value::as_i64).unwrap_or(self.observes));
        self.planner.on_result(&last, hours);
        if let Some(reading) = self.planner.reading.take() {
            let end = reading.start + reading.duration;
            // dead time (flat-field tests, stops) says nothing about the instrument
            if !(duty && self.duty.overlaps_dead(reading.start, end, now)) {
                if let (true, Some(y)) = (duty, reading.y) {
                    // a fallback for drops the log missed: half the level since the last report, same ALL-sky
                    // notices, held for 35 min (a flat test the log missed, or a closure, ends sooner)
                    // deep: no weather, shaking or moon takes the throughput below 1 % (flats and stops are dead
                    // time, a closed sky scores nothing): a drop the log missed, whatever the reference level
                    // (one such reading can be an exposure mostly in unannounced closed slots: two in a row)
                    self.duty_deep = (y < DEEP_Y).then(|| self.duty_deep.map_or((reading.start, end, 1), |d| (d.0, end, d.2 + 1)));
                    let kinds = self.planner.all_sky_kinds();
                    if kinds != self.duty_kinds {
                        self.duty_ref_from = self.duty_ref_from.max(reading.start); // new ALL-sky weather: a new level
                        self.duty_alarm = None;
                    }
                    let t0 = self.duty_alarm.map_or(f64::MAX, |a| a.0);
                    let r: Vec<f64> = self.duty_y.iter().rev().take_while(|r| r.0 >= self.duty_ref_from).filter(|r| r.0 < t0).take(8).map(|r| r.2).collect();
                    self.duty_alarm = if r.len() >= 3 && y < median_of(&r) - 0.7 {
                        let t0 = t0.min(reading.start);
                        Some((t0, end - t0 >= 2100.0))
                    } else {
                        None
                    };
                    self.duty_kinds = kinds;
                    if self.duty_y.len() >= 400 {
                        self.duty_y.drain(..200);
                    }
                    self.duty_y.push((reading.start, end, y));
                }
                if duty && reading.y.is_none() && reading.zero > 0 && reading.positive == 0 {
                    self.duty_alarm = None; // a closed sky, not the instrument
                }
                self.faults.quake_now = quake;
                self.faults.feed(&reading);
            }
        }
        self.pace(payload, now);

        let Some((night_index, night_start, night_end)) = self.planner.current_night(now) else {
            return match self.planner.next_night_start(now) {
                None => action_finish("no observing night left"),
                Some(nxt) => action_wait_until(nxt, "daytime: sleep until the next night"),
            };
        };
        if self.night_seen != Some(night_index) {
            self.night_seen = Some(night_index);
            self.night_advice(night_index, night_start, payload, hours);
        }
        let scale = self.planner.scale;
        self.scale_hours.entry(hours.trunc() as i64).or_default().push(scale);
        if night_end - now < self.planner.min_exposure {
            return match self.planner.next_night_start(now) {
                None => action_finish("survey over"),
                Some(nxt) => action_wait_until(nxt, "night ending"),
            };
        }
        if self.planner.site_closed() {
            // a due duty-log drop is reported while the dome is shut (a later drop would replace it)
            if duty && self.consecutive_reports < self.max_consecutive_reports {
                if let Some(report) = self.maybe_report(hours, payload, now, &now_utc) {
                    return report;
                }
            }
            return action_wait_for(self.to_next_slot(now, night_start), "bulletin: rain/storm over the whole sky");
        }
        let mut probe = false;
        if self.consecutive_reports < self.max_consecutive_reports {
            if let Some(report) = self.maybe_report(hours, payload, now, &now_utc) {
                return report;
            }
        }
        // duty log: dead time (flat-field tests; stops after a probe exposure scored nothing) is waited out
        if duty {
            if let Some(k) = self.duty_probe.take() {
                if last.get("action").and_then(Value::as_str) == Some("observe") {
                    let scored = last.get("hits").and_then(Value::as_array).map_or(false, |h| h.iter().any(|x| x.get("score").map_or(false, |s| planner::num(s) > 0.0)));
                    self.duty.probed(k, scored);
                    log(&format!("duty log: stop {} by the probe exposure", if scored { "refuted" } else { "confirmed" }));
                }
            }
            if let Some((k, end, stop, state)) = self.duty.dead_at(now) {
                if !stop || state == 1 {
                    return action_wait_until(end, if stop { "duty log: telescope stopped" } else { "duty log: flat-field test" });
                }
                probe = true;
                self.duty_probe = Some(k);
            }
        }
        // a confirmation exposure stays inside its slot (fault.rs; also while the fault judge holds a report)
        self.planner.max_t = if self.held.is_some() { self.faults.slot_cap(now) } else { self.faults.duration_cap(now) };
        if probe {
            self.planner.max_t = Some(self.planner.max_t.unwrap_or(f64::MAX).min(300.0));
        }
        if duty {
            // a due drop waits for its confirming reading: stay inside the slot
            if self.duty.awaiting(now) {
                if let Some(cap) = self.faults.slot_cap(now) {
                    self.planner.max_t = Some(self.planner.max_t.unwrap_or(f64::MAX).min(cap));
                }
            }
            // end the exposure at the next drop, dead window or urgent request the log announces
            if let Some(t) = self.duty.next_boundary(now) {
                if t - now < self.planner.min_exposure {
                    return action_wait_until(t, "duty log: an announced event is due");
                }
                if t - now < self.planner.max_exposure {
                    self.planner.max_t = Some(self.planner.max_t.unwrap_or(f64::MAX).min(t - now));
                }
            }
        }
        self.planner.idle_off = duty;
        let mut planned = self.planner.plan(now, night_end, night_index, hours);
        if planned.is_none()
            && !duty
            && !probe
            && self.held.is_none()
            && self.planner.max_t.is_some()
            && self.planner.idle_card()
            && self.planner.idle_ready(now)
        {
            // idle card: a fault confirmation keeps every exposure inside one slot, and on a time-rich
            // card with nothing left at that length the run would wait for months without a reading to clear it;
            // a full-length exposure is both the confirmation and the only useful work left
            let cap = self.planner.max_t.take();
            planned = self.planner.plan(now, night_end, night_index, hours);
            if planned.is_none() {
                self.planner.max_t = cap;
            }
        }
        match planned {
            None => action_wait_for(self.to_next_slot(now, night_start), "nothing useful is up"),
            Some(mut action) => {
                self.observes += 1;
                self.faults.note_observe(self.planner.all_sky_kinds());
                let fibres = action.get("assignments").and_then(Value::as_object).map_or(0, |a| a.len());
                let program = action.get("program").and_then(Value::as_str).unwrap_or("").to_string();
                action.insert("reason".into(), json!(format!("{fibres} fibres, program {program}")));
                action
            }
        }
    }

    fn to_next_slot(&self, now: f64, night_start: f64) -> i64 {
        let slot = if self.planner.slot_seconds > 0 { self.planner.slot_seconds as f64 } else { 900.0 };
        let into = skymath::pmod(now - night_start, slot);
        // a wait's duration_seconds must lie in the card's exposure range (60-3600 s on the known cards)
        self.clamp_wait((slot - into).min(3600.0).max(60.0))
    }

    /// A wait duration the engine accepts: inside [min, max] exposure seconds.
    fn clamp_wait(&self, seconds: f64) -> i64 {
        seconds.min(self.planner.max_exposure).max(self.planner.min_exposure) as i64
    }

    // --- pace ---------------------------------------------------------------------------------------

    /// (CPU seconds left, real seconds left, fair clock?) from the request's wallclock block.
    fn clock(payload: &Value) -> (f64, f64, bool) {
        let wall = &payload["wallclock"];
        if let Some(cpu) = wall.get("remaining_real_cpu_seconds").filter(|v| !v.is_null()) {
            let wall_left = wall.get("wall_remaining_seconds").map(planner::num).unwrap_or(1e9);
            return (planner::num(cpu), wall_left, true);
        }
        let remaining = wall.get("remaining_seconds").map(planner::num).unwrap_or(1e9);
        (remaining, remaining, false)
    }

    fn decisions_left(&self, now: f64) -> f64 {
        let night_seconds = psum(self.planner.nights.iter().filter(|&&(_, e)| e > now).map(|&(s, e)| (e - s.max(now)).max(0.0)));
        (night_seconds / self.sim_step_ema.unwrap_or(900.0)).max(1.0) // daytime waits cost nothing
    }

    /// Pick the search level from the measured cost per decision and the decisions still to come.
    fn pace(&mut self, payload: &Value, now: f64) {
        let (cpu_left, wall_left, fair) = Self::clock(payload);
        let decisions_left = self.decisions_left(now);
        let engine = self.engine_ema.unwrap_or(0.0);
        let cpu_budget = if fair { self.k.pace_safety * cpu_left / decisions_left } else { 1e9 };
        let wall_budget = self.k.pace_safety * wall_left / decisions_left - engine;
        // estimates of tiers not used for a while decay, so the agent climbs back up and re-measures them
        self.decisions += 1;
        if self.decisions % 50 == 0 {
            for k in 0..7 {
                if k != self.tier {
                    self.cost_ema[k] *= 0.85;
                    self.wall_ema[k] *= 0.85;
                }
            }
        }
        let top = if self.k.wide { 0 } else { 3 };
        let tier = if self.k.fixed_level >= 0 {
            if self.k.fixed_level == 0 { top } else { (self.k.fixed_level as usize + 3).min(7) }
        } else {
            let mut tier = top;
            while tier < 6
                && ((tier == 1 && !self.k.xtier) || self.cost_ema[tier] > cpu_budget || self.wall_ema[tier] > wall_budget)
            {
                tier += 1;
            }
            if cpu_left.min(wall_left) < 15.0 {
                tier = 7;
            }
            tier
        };
        let level = tier.saturating_sub(3);
        if self.k.fixed_level < 0 && level != self.planner.fast_level {
            // same line as the single-breadth ladder (costs of levels 0..3 = tiers 3..6; tools parse it)
            log(&format!(
                "pro: pace level {} (cpu budget {:.0} ms, wall budget {:.0} ms, cpu costs {:?} ms, {:.0} decisions left)",
                level,
                cpu_budget.min(99.0) * 1000.0,
                wall_budget * 1000.0,
                self.cost_ema[3..].iter().map(|c| (c * 1000.0).round() as i64).collect::<Vec<_>>(),
                decisions_left
            ));
        }
        if self.k.fixed_level < 0 && tier.min(4) != self.tier.min(4) && self.k.wide {
            log(&format!(
                "pro: search breadth {} (cpu budget {:.0} ms, wall budget {:.0} ms, wide / xwide / mid / narrow cost {:?} ms)",
                ["wide", "xwide", "mid", "narrow", "below level 0"][tier.min(4)],
                cpu_budget.min(99.0) * 1000.0,
                wall_budget * 1000.0,
                self.cost_ema[..4].iter().map(|c| (c * 1000.0).round() as i64).collect::<Vec<_>>()
            ));
        }
        self.tier = tier;
        self.planner.fast_level = level;
        self.planner.breadth = tier.min(3);
    }

    // --- LLM stages (src/llm/): asked at fixed decision points, answers clamped, rule fallback --------------

    /// Night start: a new forecast gets its week plan (every stride-th one asks the model), then tonight's
    /// rule plan, replaced by the week plan's advice for tonight when there is one.
    fn night_advice(&mut self, night_index: usize, night_start: f64, payload: &Value, hours: f64) {
        // the calendar's night_date (forecasts name nights by it)
        let night_date = match self.planner.night_dates.get(night_index) {
            Some(d) if d.len() == 10 => d.clone(),
            _ => format_date(night_start - 12.0 * 3600.0),
        };
        if let Some(forecast) = self.forecast_pending.take() {
            if self.week.on_forecast() && self.client.active() {
                let notices = forecast.get("notices").and_then(Value::as_array).cloned().unwrap_or_default();
                let dates = self.forecast_dates(&forecast, night_index);
                let context = self.week_context(night_index, hours);
                let wait = self.llm_wait(payload);
                let started = Instant::now();
                self.week.plan(&mut self.client, self.quotas.week_plan, &notices, &dates, context, wait);
                self.model_wait += started.elapsed().as_secs_f64();
                self.plan_mark = self.period_mark(night_index);
            }
            self.planner.lambda_scale = self.week.lambda_scale;
        }
        // rule plan for tonight: bad night = ALL-sky rain/storm/overcast/haze forecast; no extra sectors
        let rule_bad = week_plan::rule_bad_night(&self.forecast_notices, &night_date);
        self.planner.bad_forecast = rule_bad;
        self.planner.extra_avoid = BTreeSet::new();
        // nights of the current forecast that are bad over the whole sky (the required outlook's open chance)
        let bad_dates: BTreeSet<&str> = self
            .forecast_notices
            .iter()
            .filter(|n| {
                n.get("direction").and_then(Value::as_str) == Some("ALL")
                    && n.get("event_kind").and_then(Value::as_str).map_or(false, |k| week_plan::BAD_KINDS.contains(&k))
            })
            .flat_map(|n| n.get("nights").and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_str))
            .collect();
        self.planner.forecast_bad_nights =
            self.planner.night_dates.iter().enumerate().filter(|(_, d)| bad_dates.contains(d.as_str())).map(|(k, _)| k).collect();
        if let Some(advice) = self.week.tonight(&night_date) {
            self.planner.bad_forecast = advice.bad_night;
            self.planner.extra_avoid = advice.avoid.clone();
            if advice.bad_night != rule_bad || !advice.avoid.is_empty() {
                log(&format!("llm week plan, night {night_date}: bad_night={} avoid={:?}", advice.bad_night, advice.avoid));
            }
        }
    }

    /// The night dates a forecast covers (its coverage window; the next 7 nights if that is missing).
    fn forecast_dates(&self, forecast: &Value, night_index: usize) -> Vec<String> {
        let from = parse_utc(forecast.get("coverage_start_utc").and_then(Value::as_str).unwrap_or(""));
        let to = parse_utc(forecast.get("coverage_end_utc").and_then(Value::as_str).unwrap_or(""));
        let nights = &self.planner.nights;
        let mut indices: Vec<usize> = (0..nights.len()).filter(|&k| from <= nights[k].0 && nights[k].0 < to).collect();
        if indices.is_empty() {
            indices = (night_index..(night_index + 7).min(nights.len())).collect();
        }
        indices.into_iter().filter_map(|k| self.planner.night_dates.get(k).cloned()).filter(|d| !d.is_empty()).collect()
    }

    fn period_mark(&self, night_index: usize) -> PeriodMark {
        PeriodMark {
            night: night_index,
            observes: self.observes,
            running_total: self.running_total,
            required_done: self.planner.required_progress().1,
            reports: self.reports,
            correct_reports: self.correct_reports,
        }
    }

    /// The week plan's review input: what happened since the last plan and where the season stands (all
    /// computed by rules; the model gets summaries, not raw logs).
    fn week_context(&self, night_index: usize, hours: f64) -> Value {
        let (required_total, required_done) = self.planner.required_progress();
        let mark = self.plan_mark;
        let n_nights = self.planner.nights.len();
        let recent = |values: Vec<f64>| if values.is_empty() { Value::Null } else { json!(round_to(median_of(&values), 2)) };
        let e_rows: Vec<f64> = self.planner.e_hours.iter().rev().take(24).map(|(_, _, v)| median_of(v)).collect();
        let scale_rows: Vec<f64> = self.scale_hours.values().rev().take(24).filter(|v| !v.is_empty()).map(|v| median_of(v)).collect();
        json!({
            "season": {"nights_total": n_nights, "night_index": night_index, "nights_left": n_nights.saturating_sub(night_index),
                       "fraction_elapsed": round_to(night_index as f64 / n_nights.max(1) as f64, 3)},
            "since_last_plan": {
                "nights": night_index.saturating_sub(mark.night),
                "observations": self.observes - mark.observes,
                "score_gained": round_to(self.running_total - mark.running_total, 1),
                "required_finished": required_done.saturating_sub(mark.required_done),
                "fault_reports": self.reports - mark.reports,
                "correct_fault_reports": self.correct_reports - mark.correct_reports,
            },
            "score_so_far": round_to(self.running_total, 1),
            "required_targets": {"total": required_total, "done": required_done},
            "instrument": {
                "median_E_last_24_hours": recent(e_rows),
                "median_scale_last_24_hours": recent(scale_rows),
                "usual_scale": round_to(self.scale_ref(), 2),
                "earthquake_notice_now": self.quake_on,
                "hours_since_earthquake_notice_began":
                    if self.quake_onset_hours < -1e8 { Value::Null } else { json!(round_to(hours - self.quake_onset_hours, 1)) },
                "fault_detector": self.faults.summary(),
            },
            "observation_requests_seen": self.triaged.len(),
        })
    }

    /// How long one LLM question may wait (wall seconds; waiting costs no CPU). All waits of a run share
    /// LLM_WALL_SHARE of the wall-clock cap, spread over the calls still allowed, and never use the real time the
    /// planner and the engine are projected to need.
    fn llm_wait(&self, payload: &Value) -> f64 {
        self.llm_wait_cap(payload, self.client.timeout, 1.0)
    }

    /// The wait for one LLM question: `weight` shares of the run's LLM wall time left, at most `cap_seconds`.
    fn llm_wait_cap(&self, payload: &Value, cap_seconds: f64, weight: f64) -> f64 {
        let (_, wall_left, _) = Self::clock(payload);
        let cap = self.wall_at_start.unwrap_or(wall_left);
        let reserve = if self.observes >= 30 {
            let now = self.last_now.unwrap_or(0.0);
            let per_decision = self.wall_ema[self.tier.min(6)].max(0.01) + self.engine_ema.unwrap_or(0.02);
            1.5 * self.decisions_left(now) * per_decision + 60.0
        } else {
            0.5 * cap // nothing measured yet: keep half of the wall clock for the run
        };
        let spare = (self.k.llm_wall_share * cap - self.client.waited).min(wall_left - reserve);
        (weight * spare / self.client.calls_left().max(1) as f64).min(spare).min(cap_seconds).max(0.0)
    }

    /// New observation requests: the request triage sets their priorities before the planner values them.
    fn triage_new_requests(&mut self, requests: &[Value], now: f64, payload: &Value) {
        let fresh: Vec<&Value> = requests
            .iter()
            .filter(|r| {
                let id = r.get("request_id").and_then(Value::as_str).unwrap_or("");
                let open = r.get("remaining_count").map_or(true, |c| planner::num(c) > 0.0);
                let paid = r.get("completion_reward").map_or(true, |c| planner::num(c) > 0.0); // chat-only requests pay 0
                !id.is_empty() && open && paid && !self.triaged.contains(id)
            })
            .collect();
        if fresh.is_empty() {
            return;
        }
        for r in &fresh {
            self.triaged.insert(r["request_id"].as_str().unwrap_or("").to_string());
        }
        if !self.client.active() {
            return;
        }
        let facts: Vec<Value> = fresh.iter().map(|r| self.planner.request_facts(r, now)).collect();
        let wait = self.llm_wait(payload);
        let started = Instant::now();
        let priorities = request_triage::triage(&mut self.client, self.quotas.request_triage, facts, wait);
        self.model_wait += started.elapsed().as_secs_f64();
        self.planner.request_priority.extend(priorities);
    }

    /// New staff-chat text (request reasons with CJK text, or an English staff log; the one-line reasons of A-D cards
    /// never get here): the duty-log rule decoder reads every message (`dutylog`), then the duty-log stage (LLM): the lines the decoder is
    /// not sure of go to the model in chunks (spread over the season), one question at a time in the background:
    /// no decision waits for it, the decoder's facts count at once and the model's additions join when its answer
    /// is in (checked at every decision). The planner's duty-log facts (`duty`) are the decoder's view plus them.
    fn read_chat(&mut self, messages: &[Value], now: f64, _payload: &Value) {
        let mut new = false;
        for m in messages.iter().filter(|m| m.get("record_type").and_then(Value::as_str) == Some("observation_request")) {
            let text = m.get("reason").and_then(Value::as_str).unwrap_or("");
            if duty_log::is_chat(text) || dutylog::is_duty_log(text) {
                let t = parse_utc(m.get("issued_at_utc").and_then(Value::as_str).unwrap_or(""));
                let t = if t.is_finite() { t } else { now };
                let id = m.get("request_id").and_then(Value::as_str).unwrap_or("");
                let decoded = self.dlog.ingest(text, t as i64, self.utc_off as f64 / 60.0, id);
                let mut msg = duty_log::predecode(id, &self.dlog.lines, t);
                duty_log::attach(&mut msg, &decoded);
                self.dl_llm.add(msg, self.k.duty_chars);
                new = true;
            }
        }
        // the model's answer to the background question, once it is in (the decisions never wait for it)
        let mut answered = false;
        if let Some(asked) = &self.dl_asked {
            if let Some(merged) = duty_log::answered(&mut self.client, asked, self.utc_off) {
                if let Some(merged) = merged {
                    self.dl_llm.set(&asked.batch, merged);
                    answered = true;
                }
                self.dl_asked = None;
            }
        }
        if new && !self.chat_extended && self.client.active() {
            self.chat_extended = true;
            self.client.extend_budget(self.chat_quota);
        }
        let (a, b) = (self.planner.nights.first().map_or(now, |n| n.0), self.planner.nights.last().map_or(now, |n| n.1));
        let frac = ((now - a) / (b - a).max(1.0)).clamp(0.0, 1.0);
        let asked = self.client.stage_sent(duty_log::STAGE) as f64;
        if self.dl_asked.is_none() && self.client.active() && !self.dl_llm.pending.is_empty() && asked < 1.0 + self.chat_quota as f64 * frac {
            let batch = self.dl_llm.take_batch(self.k.duty_chars);
            self.dl_asked = duty_log::ask(&mut self.client, self.chat_quota, batch, &self.dl_llm.codebook, self.utc_off, self.k.duty_timeout);
        }
        if !new && !answered {
            return;
        }
        // the model's own facts (the decoder's view already holds the rule facts)
        let (more, gone) = self.dl_llm.resolve(self.utc_off);
        let v = &self.dlog.view;
        log(&format!("duty log: decoder {} faults, {} dead, {} closures; model {} faults, {} stops, {} closures, {} terrain; {} ruled out",
                     v.faults.len(), v.dead.len(), v.closures.len(), more.faults.len(), more.stops.len(), more.closures.len(), more.terrain.len(), gone.len()));
        self.duty.set_view(now, duty::view_in(&self.dlog, &more));
    }

    /// Usual clear-sky scale since the last repair: 75th percentile of the hourly medians.
    fn scale_ref(&self) -> f64 {
        let mut values: Vec<f64> = self
            .scale_hours
            .iter()
            .filter(|(h, v)| **h as f64 >= self.ref_from_hours && !v.is_empty())
            .map(|(_, v)| median_of(v))
            .collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if values.len() >= 4 {
            values[(3 * values.len()) / 4]
        } else {
            1.0
        }
    }

    /// The evidence the fault review reads: the last ~30 observed hours.
    fn fault_table(&self, hours: f64) -> Value {
        let e_by_hour: BTreeMap<i64, f64> = self
            .planner
            .e_hours
            .iter()
            .filter(|(h, _, _)| *h as f64 >= self.ref_from_hours)
            .map(|(h, _, v)| (*h, median_of(v)))
            .collect();
        let keys: Vec<i64> = self.scale_hours.keys().cloned().collect();
        let mut rows = Vec::new();
        for &hour in &keys[keys.len().saturating_sub(30)..] {
            if (hour as f64) < self.ref_from_hours {
                continue;
            }
            let stamp = format_hour_stamp(self.start + hour as f64 * 3600.0);
            let e = e_by_hour.get(&hour).map_or(Value::Null, |e| json!(round_to(*e, 2)));
            rows.push(json!([stamp, e, round_to(median_of(&self.scale_hours[&hour]), 2)]));
        }
        let notices: BTreeSet<String> = self.planner.notices.iter().map(|(k, d)| format!("{k} {d}")).collect();
        json!({
            "columns": ["utc_hour", "E", "scale"],
            "rows": rows,
            "ref": round_to(self.scale_ref(), 2),
            "notices_now": notices.into_iter().collect::<Vec<_>>(),
            "hours_since_earthquake_notice_began":
                if self.quake_onset_hours < -1e8 { Value::Null } else { json!(round_to(hours - self.quake_onset_hours, 1)) },
            "detector": self.faults.summary(),
            "paid_false_reports_so_far": self.paid_false,
            "correct_reports_so_far": self.correct_reports,
            "hours_since_last_report": if self.reports == 0 { Value::Null } else { json!(round_to(hours - self.last_report_hours, 1)) },
        })
    }

    // --- instrument faults ---------------------------------------------------------------------------

    /// Report when the fault monitor (fault.rs) says it pays. A report costs no time and its answer arrives at
    /// once. The fault judge (LLM) sees every rule report except the certain backstop first; it can only delay
    /// it by one confirmation exposure kept inside the current slot (or one slot of time when nothing is
    /// observed) -- the held report then always goes out at the next decision -- and it can never start or
    /// drop a report. A wrong hold costs at most one slot of faulty time.
    fn maybe_report(&mut self, hours: f64, payload: &Value, now: f64, now_utc: &str) -> Option<Map<String, Value>> {
        if let Some((t0, reason, observes)) = self.held.clone() {
            if self.observes == observes && hours - t0 < self.planner.slot_seconds as f64 / 3600.0 {
                return None; // held by the fault judge: the confirmation exposure comes first
            }
            self.held = None;
            return Some(self.send_report(hours, now_utc, &format!("{reason}; sent after the fault judge's confirmation exposure")));
        }
        if self.duty.active() {
            // the duty log's drops: reported at their time (duty.rs)
            let ys = &self.duty_y;
            let dropped = |r: f64, t: f64| {
                let level = |r: f64| ys.iter().rev().filter(|x| x.1 <= r).take(8).map(|x| x.2).collect::<Vec<f64>>();
                let before = Some(level(r)).filter(|b| b.len() >= 3).unwrap_or_else(|| level(t));
                before.len() >= 3 && ys.iter().rev().take_while(|x| x.0 >= t).any(|x| x.2 < median_of(&before) - 0.5)
            };
            if let Some((k, why, paid)) = self.duty.report_due(now, self.last_repair, self.faults.free_left(), dropped) {
                if let Some((reason, _)) = self.faults.chat_fire(now, &why) {
                    self.duty.sent(k);
                    return Some(self.send_report(hours, now_utc, &format!("{reason}{}", if paid { ", paid" } else { "" })));
                }
            }
            // a deep drop held 35 min over 2 readings (longer than any flat test) is the instrument even inside a window the log
            // explains (shaking or cloud never go that deep), whatever the log's own reports did (wrong times, paid false
            // ones): reported at once unless a logged drop is due within 15 min; paid after 3 readings and an hour, until a
            // deep report was a paid false
            let deep = self.duty_deep.filter(|d| d.2 >= 2 && d.1 - d.0 >= 2100.0 && d.0 >= self.duty_rep);
            if let (Some(d), None) = (deep, self.duty.dead_at(now)) {
                let paid_ok = self.deep_paid_false == 0 && d.2 >= 3 && d.1 - d.0 >= 3600.0;
                if (self.faults.free_left() >= 1 || paid_ok) && !self.duty.drop_due_within(now, 900.0) {
                    self.duty_deep = None;
                    self.duty_alarm = None;
                    if let Some((reason, paid)) = self.faults.chat_fire(now, "duty log: deep drop in the readings") {
                        log(&format!("duty log: deep drop since {}", format_utc(d.0)));
                        self.deep_sent = true;
                        return Some(self.send_report(hours, now_utc, &format!("{reason}{}", if paid { ", paid" } else { "" })));
                    }
                }
            }
            if self.duty.quiet(now) {
                self.faults.drop_alarm(); // a drop in dead time or one the log explains: no detector report
                self.duty_alarm = None;
                return None;
            }
            let near = self.duty.drop_near(now, 7200.0);
            if let Some((t, _)) = self.duty_alarm.filter(|a| a.1 && self.duty_ref_from < a.0 && self.faults.free_left() >= 1 && !near && self.duty.dense(now)) {
                self.duty_alarm = None;
                if let Some((reason, _)) = self.faults.chat_fire(now, "duty log: unexplained drop in the readings") {
                    log(&format!("duty log: drop seen at {}", format_utc(t)));
                    return Some(self.send_report(hours, now_utc, &reason));
                }
            }
        }
        let (reason, paid) = self.faults.decide(now)?;
        let reason = format!("{reason}{}", if paid { ", paid" } else { "" });
        if !reason.starts_with("backstop") && !self.judge_agrees(hours, payload, now_utc, &reason, paid) {
            self.held = Some((hours, reason, self.observes));
            return None;
        }
        Some(self.send_report(hours, now_utc, &reason))
    }

    fn send_report(&mut self, hours: f64, now_utc: &str, reason: &str) -> Map<String, Value> {
        self.last_report_hours = hours;
        self.reports += 1;
        log(&format!("pro: report at {now_utc} ({reason})"));
        let mut m = Map::new();
        m.insert("action".into(), json!("report"));
        m.insert("reason".into(), json!(format!("instrument fault check: {reason}")));
        m
    }

    /// The fault judge (LLM) sees the evidence before a report the rules decided on (see maybe_report for the
    /// shield). No client, no quota or no usable answer: the rule's report stands.
    fn judge_agrees(&mut self, hours: f64, payload: &Value, now_utc: &str, reason: &str, paid: bool) -> bool {
        if !self.client.active() {
            return true;
        }
        let rule_decision = json!({
            "kind": if paid { "paid report" } else { "free report" },
            "why": reason,
            "if_wait": "one more exposure inside the current slot first, then the report is sent",
        });
        let wait = self.llm_wait(payload);
        let started = Instant::now();
        let evidence = self.fault_table(hours);
        let verdict = self.judge.judge(&mut self.client, self.quotas.fault_judge, evidence, rule_decision, wait);
        self.model_wait += started.elapsed().as_secs_f64();
        if verdict == Some(false) {
            log(&format!("pro: fault judge holds the report at {now_utc} for a confirmation exposure"));
            return false;
        }
        true
    }

    fn on_report_result(&mut self, result: &Value, hours: f64, now: f64) {
        let delta = result.get("score_delta").cloned().unwrap_or(Value::Null);
        let correct = result.get("correct").and_then(Value::as_bool).unwrap_or(false);
        self.faults.on_report(correct, now);
        if self.duty.active() {
            self.duty.on_report(correct, result.get("score_delta").map_or(false, |d| planner::num(d) < 0.0), now);
            self.duty_ref_from = now;
            // a deep run restarts after a repair or its own report (a false logged report leaves the drop in place)
            let deep = std::mem::take(&mut self.deep_sent);
            if correct || deep {
                self.duty_rep = now;
                self.duty_deep = None;
            }
            if deep && !correct && result.get("score_delta").map_or(false, |d| planner::num(d) < 0.0) {
                self.deep_paid_false += 1;
            }
        }
        if correct {
            log(&format!("pro: report correct, fault repaired (delta {delta})"));
            self.correct_reports += 1;
            self.last_repair = now;
            self.planner.forget_quality_history();
            self.ref_from_hours = hours;
        } else {
            if result.get("score_delta").map_or(false, |d| planner::num(d) < 0.0) {
                self.paid_false += 1;
            }
            log(&format!("pro: report false (delta {delta})"));
        }
    }
}

fn main() {
    alloc::cache(true);
    load_dotenv(std::path::Path::new(".env"));
    let rules_only = model_disabled() || api_key().is_empty();
    if model_disabled() {
        log("pro: OBSERVER_MODEL_DISABLED=1, running rules only (no model calls)");
    } else if rules_only {
        // no key: never give up the card; every model stage keeps its rule default
        log("pro: no API key (OPENAI_API_KEY / KIMI_API_KEY); running rules only (no model calls)");
    }
    // a panic inside a decision is caught and answered with a wait; keep its message short
    std::panic::set_hook(Box::new(|info| log(&format!("pro: panic {info}"))));
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout().lock();
    let mut agent: Option<ObserverAgent> = None;
    // lines as bytes: a line that is not valid UTF-8 is read lossily (and then skipped as unreadable JSON if it is
    // broken) instead of ending the loop, which would leave every later request unanswered
    for line in stdin.lock().split(b'\n') {
        let Ok(line) = line else { break };
        let line = String::from_utf8_lossy(&line);
        if line.trim().is_empty() {
            continue;
        }
        let message: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                log(&format!("pro: unreadable line ({e})"));
                continue;
            }
        };
        let kind = message.get("message_type").and_then(Value::as_str).unwrap_or("");
        if message.get("protocol_version").and_then(Value::as_str) != Some(PROTOCOL) {
            log(&format!("pro: unexpected protocol {}", message.get("protocol_version").unwrap_or(&Value::Null)));
        }
        match kind {
            "initialize" => agent = Some(ObserverAgent::new(&message["payload"])),
            "decision_request" => {
                let payload = &message["payload"];
                let result = match agent.as_mut() {
                    Some(a) => std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| a.respond(payload))).ok(),
                    None => None,
                };
                let mut action = result.unwrap_or_else(|| {
                    // never crash the run: wait one slot instead (a duration the card accepts)
                    log("pro: error in a decision; waiting one slot");
                    action_wait_for(agent.as_ref().map_or(900, |a| a.clamp_wait(900.0)), "internal error")
                });
                action.entry("decision_source").or_insert(json!(if rules_only { "rules" } else { "llm-advised" }));
                let mut out = Map::new();
                out.insert("protocol_version".into(), json!(PROTOCOL));
                out.insert("message_type".into(), json!("decision_response"));
                out.insert("decision_sequence".into(), message.get("decision_sequence").cloned().unwrap_or(Value::Null));
                out.extend(action);
                let _ = writeln!(stdout, "{}", Value::Object(out));
                let _ = stdout.flush();
            }
            "finish" => {
                let reason = message["payload"].get("termination_reason").cloned().unwrap_or(Value::Null);
                let (observes, reports) = agent.as_ref().map_or((0, 0), |a| (a.observes, a.reports));
                log(&format!("pro finished: termination_reason={reason} observes={observes} reports={reports}"));
                if let Some(a) = agent.as_ref() {
                    log(&a.client.usage_line());
                }
            }
            _ => {}
        }
    }
}
