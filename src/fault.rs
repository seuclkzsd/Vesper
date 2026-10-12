//! Instrument faults: per-exposure change-point detector and expected-value report policy.
//!
//! Engine facts used (v4_weather_simulator.py, v4_scorer.py):
//! - A card has c faults (c unknown). The k-th starts at observing slot int(k N / (c + 1)) + U{-200..200}
//!   (N = observing slots of the calendar) and lasts until the next one starts. Faults are never announced.
//!   A report while an unrepaired fault is active repairs it (+reward); otherwise it is false: free for the
//!   first `false_report_free_allowance` after each correct report, then `false_penalty`.
//! - Weather is constant inside a slot; closures are whole slots. ALL-sky weather and earthquakes are
//!   announced at slot starts, but bulletins dedupe (kind, direction): an earthquake under a standing
//!   earthquake notice is silent and looks like a fault, except that it fades night by night (degradation
//!   D0 e^{-n/4} on night n). Earthquakes are not repaired by reports.
//!
//! Detector, fed one log throughput reading y per observe (planner/learn.rs `read_throughput`):
//! - Segments are (night, ALL-sky notice kinds). Inside a segment a bounded-LLR GLR-CUSUM tests y against the
//!   median of the last <= 6 accepted readings. Readings from one slot see one W, so the run of drops counts
//!   slots, not exposures. A strong alarm needs consistent drops in two slots read by exposures that stay
//!   inside their slot: after the first drop the next exposures are kept inside a slot (`duration_cap`).
//! - Across a boundary (new night, rain, cold snap, end of cloud/haze) the new level is compared with the
//!   level before (weak alarm). The start of a cloud, haze or earthquake notice is a blind boundary.
//! - An alarm becomes P(unrepaired fault now) from the window prior over c (with a hazard floor in case the
//!   generator differs) and the per-slot rate of the same shape without a fault.
//! - A drop that does not justify a report is held; on later nights its level is compared with what a fault
//!   (stays), an earthquake (fades) or weather (reverts, night correlation 0.82) predicts, which updates P.
//!
//! Policy (expected value): a free report when P >= PI_FREE (PI_FREE_LAST for the last free
//! one: it is worth more), a paid one when P >= PI_PAID for an in-night alarm or PI_PAID_PERSIST once later
//! nights confirmed it; at most MAX_PAID_FALSE paid false reports. Backstops: before the first correct report
//! a fault is certainly active at slot N/2 + 200 (every c >= 1); every fault window is inspected at its end;
//! the last night spends a free report. A report of either outcome restarts the search after it (`quiet`).

use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

use crate::log;
use crate::planner::{env_f, env_i, Reading};

const WINDOW_HALF: i64 = 200; // the engine's +-200 slot jitter of each fault start
const STEP_MIN: f64 = 0.357; // log(1 / 0.7): the smallest fault step
const STEP_MAX: f64 = 1.204; // log(1 / 0.3): the largest (Hard cards)
const RUN_DROP: f64 = 0.25; // a reading this far below the reference joins the alarm run
const CLEAR_DROP: f64 = 0.15; // ... and one closer than this clears it
const ONSET_TOLERANCE: i64 = 8; // slots: an onset this close to a window counts as inside it
const PRIOR_FAULTS_PER_NIGHT: f64 = 1.0 / 122.0; // reference config: 1 fault on a ~123-night card
const NIGHT_SD: f64 = 0.15; // night-to-night sd of the nightly log level (empirical, from simulated ground truth)
const NIGHT_CORRELATION: f64 = 0.82; // weather AR(1) between nights (reference weather config)
const QUAKE_DECAY_NIGHTS: f64 = 4.0; // earthquake degradation D0 e^{-n/4}
const MISS: f64 = 0.2; // P(the detector misses a fault until the next one starts)

#[derive(Clone, Copy, PartialEq, Debug)]
enum Alarm {
    /// consistent in-night drops in two slots, no earthquake notice
    Strong,
    /// the same under a standing earthquake notice (a silent new earthquake looks identical)
    Quake,
    /// a level drop across a segment boundary
    Weak,
    /// a drop in the last slots of a night (too late to confirm in-night) that the next night still shows
    Carry,
    /// one fault-sized drop read inside its slot, cut off by a new segment before a second slot could confirm it
    Single,
    /// the efficiency bound (band bits) below any healthy efficiency in two slots, no earthquake notice
    Calib,
    /// the same under an earthquake notice, below what the fading earthquake alone allows
    CalibQuake,
}

impl Alarm {
    fn name(self) -> &'static str {
        match self {
            Alarm::Strong => "strong",
            Alarm::Quake => "quake?",
            Alarm::Weak => "weak",
            Alarm::Carry => "night-end",
            Alarm::Single => "single-slot",
            Alarm::Calib => "efficiency",
            Alarm::CalibQuake => "efficiency-quake",
        }
    }

    /// Per-slot rate of this alarm without a fault (empirical, fitted to simulated ground truth): natural
    /// in-night runs of >= 2 slots happen ~1e-4/slot but stayed below 0.33 in log in simulated runs; under an
    /// earthquake notice silent earthquakes add large ones; boundary drops are common.
    fn false_rate(self, step: f64) -> f64 {
        match self {
            Alarm::Strong if step >= STEP_MIN - 0.03 => 2e-5,
            Alarm::Strong => 1.5e-4,
            Alarm::Quake => 3e-4,
            Alarm::Weak => 2e-3,
            // a single-slot drop (natural rate ~3e-4/slot) that the next night still shows: weather persists
            // across nights (correlation 0.82), so the second night adds little against weather
            Alarm::Carry => 3e-4,
            // natural single-slot drops >= 0.33 in log: ~3.5e-5/slot, plus multiplier mix-ups
            Alarm::Single => 1e-4,
            // per alarm, not per slot (alarm_p): two readings whose bound is below 0.9 / model error
            Alarm::Calib => 1e-5,
            // a second earthquake under a standing notice is silent and fades like the first: common enough that
            // only a fault window makes this a fault
            Alarm::CalibQuake => 1e-2,
        }
    }

    /// P(this alarm | a fault started in its onset interval).
    fn p_detect(self) -> f64 {
        match self {
            Alarm::Weak => 0.6,
            _ => 0.9,
        }
    }
}

/// A drop that did not justify a report when it was seen: re-tested on the following nights.
#[derive(Clone, Copy, Debug)]
struct Suspect {
    kind: Alarm,
    onset: Onset,
    level: f64, // reference level before the drop
    step: f64,  // size of the drop (log)
    night: usize,
    checked: usize, // last night compared
    llr: f64,       // log LR of the later nights: fault vs the alternative of its kind
}

/// Slots [first, last] in which the fault behind an alarm may have started.
type Onset = (i64, i64);

#[derive(Clone, PartialEq, Debug)]
struct SegKey {
    night: usize,
    kinds: BTreeSet<String>,
}

/// Tunable constants (each one overridable with `PRO_<NAME>`).
struct Knobs {
    pi_free: f64,
    pi_free_last: f64,
    pi_paid: f64,
    pi_paid_persist: f64,
    max_paid_false: i64,
    hazard_floor: f64,
    sigma: f64,
    s_alarm: f64,
    calib_eff: f64,
    calib_n: usize,
    health_eff: f64,
}

impl Knobs {
    fn load() -> Knobs {
        Knobs {
            pi_free: env_f("FAULT_PI_FREE", 0.15),
            pi_free_last: env_f("FAULT_PI_FREE_LAST", 0.35),
            pi_paid: env_f("FAULT_PI_PAID", 0.96),
            pi_paid_persist: env_f("FAULT_PI_PERSIST", 0.75),
            max_paid_false: env_i("FAULT_MAX_PAID_FALSE", 2),
            hazard_floor: env_f("FAULT_HAZARD_FLOOR", 0.07),
            sigma: env_f("FAULT_SIGMA", 0.06),
            s_alarm: env_f("FAULT_S_ALARM", 12.0),
            calib_eff: env_f("FAULT_CALIB_EFF", 0.85),
            calib_n: env_i("FAULT_CALIB_N", 2).max(1) as usize,
            health_eff: env_f("FAULT_HEALTH_EFF", 0.75),
        }
    }
}

pub struct FaultMonitor {
    k: Knobs,
    /// (observing start, end, first global slot, slot count) per night
    nights: Vec<(f64, f64, usize, usize)>,
    n_slots: usize,
    slot_s: f64,
    free_allowance: i64,
    /// posterior over the number of faults c (index = c, 0 unused)
    pc: Vec<f64>,
    prior: Vec<f64>,
    correct: usize, // correct reports so far
    /// slot of the last report: an unrepaired fault started later, in a window with mass `left` (default 1)
    quiet: i64,
    left: BTreeMap<(usize, usize), f64>,
    onset: Option<Onset>, // of the report in flight
    // --- detector
    key: Option<SegKey>,
    hist: Vec<f64>,
    run: Vec<(f64, usize, bool)>, // (y, slot, slot-contained), one entry per slot
    run_ref: f64,
    run_onset: Onset,
    last_slot: usize, // slot in which the last reading ended
    run_alarmed: bool,
    cusum: f64,
    last_contained: Option<usize>,
    prev_level: Option<(f64, SegKey, usize)>, // (level, its segment, slot of its last reading)
    boundary: Option<(f64, f64, usize)>, // (reference level, sigma, slot it was read) for a new segment's first readings
    carry: Option<(f64, Onset, f64, bool)>, // unconfirmed drop at the end of the last night: (level, onset, step, quake)
    newseg: Vec<f64>,
    pre_event: BTreeMap<String, (f64, usize, usize)>, // cloud kind -> (level before it, night, slot)
    confirm: bool,
    night_vals: BTreeMap<usize, Vec<f64>>,
    pending_kinds: BTreeSet<String>,
    // --- policy
    false_since_correct: i64,
    paid_false: i64,
    suspend_until: usize,
    suspect: Option<Suspect>,
    last_run: Vec<f64>,
    want: Option<(Alarm, Onset, f64)>, // an alarm to settle at the next decision (kind, onset, step)
    checked_night: Option<usize>,
    last_night_probe: bool,
    backstop_done: bool,
    /// Slot of the certain backstop: N/2 + 200, or the last night's first slot on a short card (N < ~400 slots).
    backstop_slot: usize,
    windows_checked: BTreeSet<(usize, usize)>, // fault windows (c, k) inspected at their end
    last_pi: f64,
    /// slots since the last report whose efficiency bound fell below calib_eff (no earthquake notice)
    calib: Vec<(usize, usize)>, // (slot, night)
    calib_night: Option<usize>, // night of the last efficiency alarm (one per night)
    calib_false: i32,           // false reports it caused: each asks one more slot, 10 % lower
    calib_fired: bool,          // the report in flight came from it
    quake_since: Option<usize>, // night the standing earthquake notice began
    /// an earthquake notice is out now (a quake that began during the last exposure lowered it unannounced)
    pub quake_now: bool,
}

/// Median of a non-empty slice.
fn median(values: &[f64]) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

fn odds_to_p(odds: f64) -> f64 {
    odds / (1.0 + odds)
}

impl FaultMonitor {
    pub fn new(init: &Value, nights: &[(f64, f64)], night_slots: &[usize], slot_seconds: i64) -> FaultMonitor {
        let mut table = Vec::with_capacity(nights.len());
        let mut first = 0;
        for (k, &(s, e)) in nights.iter().enumerate() {
            let count = night_slots.get(k).cloned().unwrap_or(1).max(1);
            table.push((s, e, first, count));
            first += count;
        }
        let n_slots = first.max(1);
        // prior over c: half uniform over 1..=cmax, half centred on the reference rate (1 per ~122 nights)
        let expected = (nights.len() as f64 * PRIOR_FAULTS_PER_NIGHT).max(1.0);
        let cmax = ((expected.round() as usize) + 2).max(4).min(8).min(n_slots);
        let gauss: Vec<f64> = (1..=cmax).map(|c| (-(c as f64 - expected).powi(2) / (2.0 * 0.7 * 0.7)).exp()).collect();
        let gsum: f64 = gauss.iter().sum();
        let mut prior = vec![0.0; cmax + 1];
        for c in 1..=cmax {
            prior[c] = 0.5 / cmax as f64 + 0.5 * gauss[c - 1] / gsum;
        }
        let free_allowance = init["scoring"]["reporting"]["false_report_free_allowance"].as_f64().unwrap_or(0.0) as i64;
        let backstop_slot = (n_slots / 2 + WINDOW_HALF as usize).min(table.last().map_or(usize::MAX, |n| n.2));
        log(&format!(
            "fault: {} observing slots, certain backstop at slot {}, prior P(c) {:?}, free reports {}",
            n_slots,
            backstop_slot,
            prior.iter().skip(1).map(|p| (p * 100.0).round() / 100.0).collect::<Vec<_>>(),
            free_allowance
        ));
        FaultMonitor {
            k: Knobs::load(),
            nights: table,
            n_slots,
            slot_s: if slot_seconds > 0 { slot_seconds as f64 } else { 900.0 },
            free_allowance,
            pc: prior.clone(),
            prior,
            correct: 0,
            quiet: -1,
            left: BTreeMap::new(),
            onset: None,
            key: None,
            hist: Vec::new(),
            run: Vec::new(),
            run_ref: 0.0,
            run_onset: (0, 0),
            last_slot: 0,
            run_alarmed: false,
            cusum: 0.0,
            last_contained: None,
            prev_level: None,
            boundary: None,
            carry: None,
            newseg: Vec::new(),
            pre_event: BTreeMap::new(),
            confirm: false,
            night_vals: BTreeMap::new(),
            pending_kinds: BTreeSet::new(),
            false_since_correct: 0,
            paid_false: 0,
            suspend_until: 0,
            suspect: None,
            last_run: Vec::new(),
            want: None,
            checked_night: None,
            last_night_probe: false,
            backstop_done: false,
            backstop_slot,
            windows_checked: BTreeSet::new(),
            last_pi: 0.0,
            calib: Vec::new(),
            calib_night: None,
            calib_false: 0,
            calib_fired: false,
            quake_since: None,
            quake_now: false,
        }
    }

    // --- calendar ------------------------------------------------------------------------------

    /// (night, global slot index, slot end) of a moment inside a night.
    fn locate(&self, t: f64) -> Option<(usize, usize, f64)> {
        let k = self.nights.iter().position(|&(s, e, _, _)| s <= t && t < e)?;
        let (s, _, first, count) = self.nights[k];
        let j = (((t - s) / self.slot_s + 1e-9).floor() as usize).min(count - 1);
        Some((k, first + j, s + (j + 1) as f64 * self.slot_s))
    }

    /// Global slot index of a moment (the next night's first slot during the day).
    fn slot_of(&self, t: f64) -> usize {
        if let Some((_, j, _)) = self.locate(t) {
            return j;
        }
        self.nights.iter().find(|n| n.0 > t).map_or(self.n_slots, |n| n.2)
    }

    // --- the window prior ----------------------------------------------------------------------

    pub fn free_left(&self) -> i64 {
        self.free_allowance - self.false_since_correct
    }

    /// Start window of fault k (1-based) when the card has c faults.
    fn window(&self, c: usize, k: usize) -> (i64, i64) {
        let nominal = (k as f64 * self.n_slots as f64 / (c as f64 + 1.0)) as i64;
        (nominal - WINDOW_HALF, nominal + WINDOW_HALF)
    }

    /// Fault k's window after the last report: (first slot, last slot, slots / mass left); None if over or
    /// repaired. Every such window counts: an undetected fault is replaced by the next one.
    fn rest(&self, c: usize, k: usize) -> Option<(i64, i64, f64)> {
        let (a, b) = self.window(c, k);
        let (from, w) = ((a - 1).max(self.quiet), *self.left.get(&(c, k)).unwrap_or(&1.0));
        (b > from && w > 0.0).then(|| (from + 1, b, (b - from) as f64 / w))
    }

    /// Prior per-slot hazard that an unrepaired fault starts at slot j (mixture over c, with a floor).
    fn hazard(&self, j: i64) -> f64 {
        let mut h = 0.0;
        for c in 1..self.pc.len() {
            for (a, b, n) in (1..=c).filter_map(|k| self.rest(c, k)) {
                if a <= j && j <= b {
                    h += self.pc[c] / n;
                }
            }
        }
        (1.0 - self.k.hazard_floor) * h + self.k.hazard_floor / self.n_slots as f64
    }

    /// P(a fault started after the last report, by slot j | c faults).
    fn started_c(&self, c: usize, j: i64) -> f64 {
        1.0 - (1..=c).filter_map(|k| self.rest(c, k)).map(|(a, b, n)| 1.0 - ((j.min(b) - a + 1) as f64 / n).max(0.0)).product::<f64>()
    }

    fn p_started(&self, j: i64) -> f64 {
        (1..self.pc.len()).map(|c| self.pc[c] * self.started_c(c, j)).sum()
    }

    /// Prior odds that an alarm of this kind and step, with its onset in the given slots, is a fault.
    fn odds(&self, kind: Alarm, onset: Onset, step: f64) -> f64 {
        let (lo, hi) = (onset.0.min(onset.1), onset.0.max(onset.1));
        let h = (lo..=hi).map(|j| self.hazard(j)).sum::<f64>() / (hi - lo + 1) as f64;
        h * kind.p_detect() / kind.false_rate(step)
    }

    /// Bayes update of P(c) from a report at slot j. A correct one takes the window of the fault it repaired: the
    /// first meeting the alarm's onset, else the first left if it holds j; other windows begun by j keep the mass
    /// after j. A dated one implies c's earlier faults were missed (MISS).
    fn update_c(&mut self, j: i64, correct: bool) {
        let (lo, hi) = self.onset.map_or((j, j), |(lo, hi)| (lo - ONSET_TOLERANCE, hi + ONSET_TOLERANCE));
        let mut new = self.pc.clone();
        let mut took = Vec::new();
        for c in 1..self.pc.len() {
            let (mut met, mut missed, mut first) = (0.0, 0, true);
            for k in 1..=c {
                let Some((a, b, n)) = self.rest(c, k) else { continue };
                if b < lo {
                    missed += 1;
                } else if a <= hi {
                    met += ((hi.min(b) - lo.max(a) + 1) as f64 / n).min(1.0);
                    let rep = first && (missed == 0 || self.onset.is_some());
                    took.push(((c, k), if rep { 0.0 } else { (b - j.max(a - 1)).max(0) as f64 / n }));
                    first = false;
                }
            }
            let started = self.started_c(c, j);
            new[c] *= match (correct, self.onset) {
                (true, Some(_)) => (met + 1e-4 / (2 * WINDOW_HALF + 1) as f64) * MISS.powi(missed), // dated
                (true, None) => started,
                (false, _) => 1.0 - started,
            };
        }
        let z: f64 = new.iter().sum();
        // no c fits a false report before any correct one: the certain backstop failed, no fault at all
        self.pc = if z > 0.0 { new.iter().map(|v| v / z).collect() } else if correct || self.correct > 0 { self.prior.clone() } else { new };
        if correct {
            self.left.extend(took);
        }
        self.quiet = j;
    }

    /// No unrepaired fault at slot j (efficiency proven healthy): P(c) and the windows as after a false report.
    fn healthy(&mut self, j: i64) {
        let new: Vec<f64> = (0..self.pc.len()).map(|c| if c == 0 { 0.0 } else { self.pc[c] * (1.0 - self.started_c(c, j)) }).collect();
        let z: f64 = new.iter().sum();
        if z > 0.0 {
            self.pc = new.iter().map(|v| v / z).collect();
        }
        self.quiet = j;
        self.suspect = None;
    }

    // --- detector ------------------------------------------------------------------------------

    fn reset_detector(&mut self) {
        self.hist.clear();
        self.run.clear();
        self.cusum = 0.0;
        self.boundary = None;
        self.newseg.clear();
        self.confirm = false;
    }

    fn ref_level(&self) -> f64 {
        median(&self.hist[self.hist.len().saturating_sub(6)..])
    }

    /// Bounded GLR log-likelihood ratio of a drop x for a step of unknown size in [STEP_MIN, STEP_MAX].
    fn llr(x: f64, sigma: f64) -> f64 {
        let d = x.clamp(STEP_MIN, STEP_MAX);
        ((d * x - d * d / 2.0) / (sigma * sigma)).clamp(-4.0, 12.0)
    }

    /// Only a cold snap or a closure (rain, storm) differs between two keys: the level carries over.
    fn continues(old: &SegKey, new: &SegKey) -> bool {
        old.kinds.symmetric_difference(&new.kinds).all(|k| matches!(k.as_str(), "cold_snap" | "rain" | "storm"))
    }

    /// Log change of W expected from a cold snap starting (~0.92) or ending.
    fn cold_snap_offset(old: &SegKey, new: &SegKey) -> f64 {
        match (old.kinds.contains("cold_snap"), new.kinds.contains("cold_snap")) {
            (false, true) => 0.92f64.ln(),
            (true, false) => -(0.92f64.ln()),
            _ => 0.0,
        }
    }

    /// A new segment starts: keep the old level and decide what the first readings are compared with.
    /// Returns an alarm when the old segment ended on an unconfirmed fault-sized drop read inside its slot.
    fn new_segment(&mut self, key: &SegKey) -> Option<(Alarm, Onset, f64, f64)> {
        let mut cut = None;
        self.carry = None;
        if let (Some(old), false) = (self.key.clone(), self.run.is_empty() || self.run_alarmed) {
            // the segment ends on a drop before a second slot could confirm it
            let steps: Vec<f64> = self.run.iter().map(|q| self.run_ref - q.0).collect();
            let step = median(&steps);
            let quake = old.kinds.contains("earthquake");
            if self.run.iter().any(|q| q.2) && step >= STEP_MIN - 0.03 {
                // read inside its slot (no closure dilution) and fault-sized: decide on it now
                let kind = if quake { Alarm::Quake } else { Alarm::Single };
                cut = Some((kind, self.run_onset, step, self.run_ref));
            } else if old.night != key.night {
                // the night ended on it: let the next night's level decide
                self.carry = Some((self.run_ref, self.run_onset, step, quake));
            }
        }
        self.start_segment(key);
        if self.boundary.is_none() {
            self.carry = None; // no comparison across this boundary: a carried drop cannot be tested
        }
        cut
    }

    fn start_segment(&mut self, key: &SegKey) {
        if !self.hist.is_empty() {
            if let Some(old) = self.key.clone() {
                self.prev_level = Some((self.ref_level(), old, self.last_slot));
            }
        }
        self.reset_detector();
        let Some((level, before, since)) = self.prev_level.clone() else { return };
        let added: BTreeSet<&String> = key.kinds.difference(&before.kinds).collect();
        let removed: BTreeSet<&String> = before.kinds.difference(&key.kinds).collect();
        let quake = key.kinds.contains("earthquake");
        let cloud = |k: &&String| k.as_str() == "overcast" || k.as_str() == "haze";
        for kind in added.iter().filter(|k| cloud(k)) {
            self.pre_event.insert((*kind).clone(), (level, before.night, since));
        }
        if added.iter().any(|k| k.as_str() == "earthquake") || added.iter().any(|k| cloud(k)) {
            self.carry = None;
            return; // an announced drop of unknown size: blind boundary
        }
        if removed.iter().any(|k| cloud(k)) {
            // end of a cloud/haze event: the level must come back to the one before it (a fault would not)
            let refs: Vec<(f64, usize, usize)> = removed.iter().filter_map(|k| self.pre_event.remove(k.as_str())).collect();
            let still_cloudy = key.kinds.iter().any(|k| k == "overcast" || k == "haze");
            if let (Some(&(level0, night0, since0)), false) = (refs.first(), still_cloudy) {
                let nights = key.night.saturating_sub(night0);
                let mut sigma = if nights == 0 { 0.10 } else { NIGHT_SD + 0.03 * nights.min(5) as f64 };
                if quake {
                    sigma += 0.10;
                }
                self.boundary = Some((level0, sigma, since0));
            }
            return;
        }
        let offset = Self::cold_snap_offset(&before, key);
        if let Some(c) = self.carry.as_mut() {
            c.0 += offset;
        }
        let sigma = match (key.night != before.night, quake) {
            (true, true) => 0.25,
            (true, false) => NIGHT_SD,
            _ => 0.08,
        };
        self.boundary = Some((level + offset, sigma, since));
    }

    /// Feed the reading of the last observe (its ALL-sky notice kinds were given to `note_observe`).
    pub fn feed(&mut self, r: &Reading) {
        let Some(y) = r.y else { return };
        let Some((night, j, slot_end)) = self.locate(r.start) else { return };
        let contained = r.start + r.duration <= slot_end + 1.0;
        let end_slot = self.locate(r.start + r.duration - 1.0).map_or(j, |l| l.1);
        let key = SegKey { night, kinds: std::mem::take(&mut self.pending_kinds) };
        let mut cut = None;
        if self.key.as_ref() != Some(&key) {
            match self.key.clone() {
                Some(old) if old.night == key.night && Self::continues(&old, &key) => {
                    // same night, only a cold snap or a closure came or went: keep the run, rescale the level
                    let offset = Self::cold_snap_offset(&old, &key);
                    self.hist.iter_mut().for_each(|h| *h += offset);
                    self.run.iter_mut().for_each(|q| q.0 += offset);
                    self.run_ref += offset;
                }
                _ => cut = self.new_segment(&key),
            }
            self.key = Some(key.clone());
            self.last_contained = None;
        }
        if contained && self.last_contained == Some(j) {
            return; // a second exposure inside the same slot sees the same W: no new evidence
        }
        if contained {
            self.last_contained = Some(j);
        }
        if !key.kinds.iter().any(|k| k == "overcast" || k == "haze") {
            self.night_vals.entry(night).or_default().push(y); // nightly levels for the suspect checks
        }
        let mut alarm: Option<(Alarm, Onset, f64, f64)> = cut; // (kind, onset, step, level before)
        if !self.hist.is_empty() {
            let mu = self.ref_level();
            let sigma = if self.hist.len() >= 2 { self.k.sigma } else { 1.4 * self.k.sigma };
            let x = mu - y;
            self.cusum = (self.cusum + Self::llr(x, sigma)).max(0.0);
            if x > RUN_DROP {
                if self.run.is_empty() {
                    self.run_ref = mu;
                    self.run_alarmed = false;
                    // the drop began after the previous reading and before this one ended
                    self.run_onset = (self.last_slot as i64, end_slot as i64);
                }
                match self.run.last_mut() {
                    Some(last) if last.1 == j => {
                        if contained && !last.2 {
                            *last = (y, j, true);
                        }
                    }
                    _ => self.run.push((y, j, contained)),
                }
                self.confirm = !self.run_alarmed;
            } else {
                if x < CLEAR_DROP {
                    self.run.clear();
                    self.cusum = 0.0;
                }
                self.hist.push(y);
                self.confirm = false;
            }
            // drops in two slots, at least one read inside its slot (not a closure diluting a long exposure)
            let enough = self.run.len() >= 2 && self.run.iter().any(|q| q.2);
            if !self.run_alarmed && self.cusum >= self.k.s_alarm && enough {
                let n = self.run.len();
                let (a, b) = (mu - self.run[n - 2].0, mu - self.run[n - 1].0);
                // the same step twice (one fault multiplier), not two different closure dilutions
                if (a - b).abs() < 0.25 || n >= 3 {
                    let steps: Vec<f64> = self.run.iter().map(|q| self.run_ref - q.0).collect();
                    let kind = if key.kinds.contains("earthquake") { Alarm::Quake } else { Alarm::Strong };
                    alarm = Some((kind, self.run_onset, median(&steps), self.run_ref));
                    self.run_alarmed = true;
                    self.confirm = false;
                }
            }
        } else {
            self.hist.push(y); // the first reading of a segment starts its reference
        }
        if let Some((level, sigma, since)) = self.boundary {
            // the level across the boundary: closures and bad slots only lower a reading, so compare the
            // better of the first two readings
            self.newseg.push(y);
            if self.newseg.len() >= 2 {
                let top = self.newseg.iter().cloned().fold(f64::MIN, f64::max);
                let x = level - top;
                if let Some((level0, onset, step0, quake)) = self.carry.take() {
                    // last night ended on an unconfirmed drop: is tonight's level still down by about that step?
                    let x0 = level0 - top;
                    if x0 > RUN_DROP && alarm.is_none() {
                        let kind = if quake { Alarm::Quake } else { Alarm::Carry };
                        alarm = Some((kind, onset, 0.5 * (step0 + x0), level0));
                    }
                }
                if x > STEP_MIN + 2.0 * sigma && alarm.is_none() {
                    alarm = Some((Alarm::Weak, (since as i64, end_slot as i64), x, level));
                }
                self.boundary = None;
            }
        }
        // Calibration: the saturated hits bound the band level tau*K/s, which carries no efficiency, so
        // W / band_lo bounds the efficiency from above whatever the reference (w_hi: uncertain multipliers
        // taken as mismatched). Healthy efficiency is >= 0.9 outside earthquakes: two slots well below it
        // mean an unrepaired fault, however the path placed its readings.
        let ub = if r.band_lo > 0.0 && r.w_hi > 0.0 { r.w_hi / r.band_lo } else { f64::NAN };
        let (mut thr, need) = (self.k.calib_eff * 0.9f64.powi(self.calib_false), self.k.calib_n + self.calib_false as usize);
        // under an earthquake notice the efficiency may be down by D0 e^{-n/4} on its n-th night (D0 <= 0.95):
        // allow for it with a slower fade, and stop trusting the notice's start after 15 nights (a second
        // earthquake under a standing notice is silent)
        let quake = key.kinds.contains("earthquake") || self.quake_now;
        if quake {
            let since = *self.quake_since.get_or_insert(night);
            let n = if key.kinds.contains("earthquake") { night - since } else { 0 }; // began during the exposure
            thr = if n <= 15 { thr * (1.0 - 0.95 * (-(n as f64) / 6.0).exp()) } else { 0.0 };
        } else {
            self.quake_since = None;
        }
        if ub < thr {
            if self.calib.last().map(|c| c.0) != Some(j) {
                self.calib.push((j, night));
            }
            self.calib.retain(|c| c.1 + 1 >= night); // tonight's and last night's
            if self.calib.len() >= need && alarm.is_none() && self.calib_night != Some(night) {
                self.calib_night = Some(night);
                let kind = if quake { Alarm::CalibQuake } else { Alarm::Calib };
                alarm = Some((kind, (self.quiet + 1, j as i64), (0.95 / ub).ln(), 0.0));
            }
        }
        // Health: W over the band level's upper bound (mismatched or BRIGHT-matched saturated hits) bounds the
        // efficiency from below. Above any faulty efficiency (<= 0.7) no unrepaired fault was active when the
        // exposure began, so none started since the last report: the news of a false report, for free.
        let lb = if r.band_hi.is_finite() && r.w_lo > 0.0 { r.w_lo / r.band_hi } else { f64::NAN };
        if self.k.health_eff > 0.0 && lb > self.k.health_eff && (j as i64) > self.quiet && alarm.is_none() && self.want.is_none() {
            self.healthy(j as i64);
        }
        self.last_slot = end_slot;
        if let Some((kind, onset, step, level)) = alarm {
            self.on_alarm(kind, onset, step, level, j, night);
        }
    }

    /// Report P for a fresh alarm.
    fn alarm_p(&self, kind: Alarm, onset: Onset, step: f64) -> f64 {
        if matches!(kind, Alarm::Calib | Alarm::CalibQuake) {
            // not dated: P(a fault started since the last report) against the alarm's own false rate
            let j = onset.1;
            let floor = self.k.hazard_floor * (j - self.quiet).max(1) as f64 / self.n_slots as f64;
            let ps = self.p_started(j) + (1.0 - self.p_started(j)) * floor.min(1.0);
            return ps * kind.p_detect() / (ps * kind.p_detect() + (1.0 - ps) * kind.false_rate(step));
        }
        odds_to_p(self.odds(kind, onset, step))
    }

    /// Does a report at probability p pay now? Free reports readily (the last free one is worth more),
    /// paid ones only above `paid_at` and under the paid-false cap.
    fn pays(&self, p: f64, paid_at: f64) -> bool {
        let free = self.free_left();
        (free >= 2 && p >= self.k.pi_free)
            || (free == 1 && p >= self.k.pi_free_last)
            || (free <= 0 && self.paid_false < self.k.max_paid_false && p >= paid_at)
    }

    /// An alarm: report at the next decision if it pays, else hold it for the checks on later nights.
    fn on_alarm(&mut self, kind: Alarm, onset: Onset, step: f64, level: f64, j: usize, night: usize) {
        if j < self.suspend_until {
            return;
        }
        let p = self.alarm_p(kind, onset, step);
        self.last_pi = p;
        let go = self.pays(p, self.k.pi_paid);
        log(&format!(
            "fault: {} alarm at slot {} (onset in {:?}, step {:.2}), P(fault) {:.3}, free left {}, P(c) {:?} -> {}",
            kind.name(),
            j,
            onset,
            step,
            p,
            self.free_left(),
            self.pc.iter().skip(1).map(|p| (p * 100.0).round() / 100.0).collect::<Vec<_>>(),
            if go { "report" } else { "hold" }
        ));
        if go {
            self.want = Some((kind, onset, step));
        } else if matches!(kind, Alarm::Calib | Alarm::CalibQuake) {
            // not a level drop: nothing to re-test on later nights; it alarms again on a later night
        } else if self.suspect.is_none() || p > self.suspect.map_or(0.0, |s| self.alarm_p(s.kind, s.onset, s.step)) {
            self.suspect = Some(Suspect { kind, onset, level, step, night, checked: night, llr: 0.0 });
        }
    }

    /// The next observe should stay inside one slot (a confirmation exposure): its longest duration.
    pub fn duration_cap(&self, now: f64) -> Option<f64> {
        if !self.confirm {
            return None;
        }
        self.slot_cap(now)
    }

    /// The longest observe that stays inside the current slot.
    pub fn slot_cap(&self, now: f64) -> Option<f64> {
        let (_, _, slot_end) = self.locate(now)?;
        let left = slot_end - now;
        // under a minute left: take the rest of this slot plus the next one (at most 1/16 diluted)
        Some(if left >= 60.0 { left } else { left + self.slot_s })
    }

    /// Remember the ALL-sky notice kinds of the observe about to start (its reading arrives next decision).
    pub fn note_observe(&mut self, kinds: BTreeSet<String>) {
        self.pending_kinds = kinds;
    }

    // --- policy --------------------------------------------------------------------------------

    /// Report now? Returns (reason, paid) when a report is due. Called at every night decision.
    pub fn decide(&mut self, now: f64) -> Option<(String, bool)> {
        let (night, j, _) = self.locate(now)?;
        if j < self.suspend_until {
            return None;
        }
        // 1. the certain backstop: no correct report yet, and every c >= 1 has a fault active by N/2 + 200
        if self.correct == 0 && !self.backstop_done && j >= self.backstop_slot {
            self.backstop_done = true; // false only if the card has no fault at all: never repeat it
            return self.fire("backstop N/2+200", None, j);
        }
        // 2. an alarm from the last reading
        if let Some((kind, onset, step)) = self.want.take() {
            let p = self.alarm_p(kind, onset, step);
            if self.pays(p, self.k.pi_paid) {
                let calib = matches!(kind, Alarm::Calib | Alarm::CalibQuake);
                let dated = if calib { None } else { Some(onset) };
                self.calib_fired = calib;
                return self.fire(&format!("{} alarm P={:.2}", kind.name(), p), dated, j);
            }
        }
        // 3. a held drop, once a later night has enough readings
        if let Some(r) = self.check_suspect(j) {
            return Some(r);
        }
        // 4. once per night: on the last night a free report costs nothing
        if self.checked_night != Some(night) {
            self.checked_night = Some(night);
            if night + 1 == self.nights.len() && self.free_left() > 0 && !self.last_night_probe && self.p_started(j as i64) >= 0.02 {
                self.last_night_probe = true;
                return self.fire("last night, free report", None, j);
            }
        }
        // 5. inspection at the end of each fault window (Barlow-Hunter-Proschan): report if a fault has started
        //    since the last report with enough P (summed over c), whatever free reports are left
        for c in 1..self.pc.len() {
            for k in 1..=c {
                let Some((_, b, _)) = self.rest(c, k) else { continue };
                if (j as i64) < b || !self.windows_checked.insert((c, k)) {
                    continue;
                }
                let p = self.p_started(j as i64);
                let free = self.free_left();
                // keep free reports for the detector while the certain backstop is still to come
                let (two, one) = if self.correct == 0 && !self.backstop_done { (0.5, 0.85) } else { (0.3, 0.6) };
                if (free >= 2 && p >= two)
                    || (free == 1 && p >= one)
                    || (free <= 0 && ((self.paid_false < self.k.max_paid_false && p >= one) || p >= 0.9))
                {
                    return self.fire(&format!("end of fault {k}'s window for c={c} (P started {p:.2})"), None, j);
                }
            }
        }
        None
    }

    /// A held drop on a later night: is it still there? A fault stays at its step, an earthquake fades
    /// (D0 e^{-n/4}), weather reverts (night correlation 0.82). The nightly level is the median of the
    /// night's readings outside cloud notices (night-to-night sd ~0.15).
    fn check_suspect(&mut self, j: usize) -> Option<(String, bool)> {
        let mut s = self.suspect?;
        let (&n, vals) = self.night_vals.range(s.checked + 1..).find(|(_, v)| v.len() >= 4)?;
        let x = s.level - median(vals);
        s.checked = n;
        if x <= RUN_DROP {
            self.suspect = None; // the level came back: weather or a fading earthquake
            log(&format!("fault: held {} drop gone on night {} (drop now {:.2})", s.kind.name(), n, x));
            return None;
        }
        let nights = (n - s.night) as f64;
        let other = match s.kind {
            Alarm::Quake => {
                let d0 = (1.0 - (-s.step).exp()).clamp(0.0, 0.95);
                -(1.0 - d0 * (-nights / QUAKE_DECAY_NIGHTS).exp()).ln()
            }
            _ => s.step * NIGHT_CORRELATION.powf(nights),
        };
        s.llr += ((x - other).powi(2) - (x - s.step).powi(2)) / (2.0 * NIGHT_SD * NIGHT_SD);
        self.suspect = Some(s);
        let p = odds_to_p(self.odds(s.kind, s.onset, s.step) * s.llr.clamp(-20.0, 20.0).exp());
        let go = self.pays(p, self.k.pi_paid_persist);
        log(&format!(
            "fault: held {} drop on night {}: drop {:.2} (fault {:.2}, other {:.2}), P(fault) {:.3} -> {}",
            s.kind.name(),
            n,
            x,
            s.step,
            other,
            p,
            if go { "report" } else { "hold" }
        ));
        if go {
            self.suspect = None;
            return self.fire(&format!("{} drop stayed {} night(s), P={:.2}", s.kind.name(), nights, p), None, j);
        }
        None
    }

    fn fire(&mut self, reason: &str, onset: Option<Onset>, j: usize) -> Option<(String, bool)> {
        let paid = self.free_left() <= 0;
        self.onset = onset;
        self.last_run = self.run.iter().map(|r| r.0).collect();
        self.suspend_until = j + 1;
        self.want = None;
        log(&format!("fault: report at slot {} ({}), {}", j, reason, if paid { "paid" } else { "free" }));
        Some((reason.to_string(), paid))
    }

    /// Forget the pending alarm and held drop (duty log: dead time, or a drop it explains).
    pub fn drop_alarm(&mut self) {
        self.want = None;
        self.suspect = None;
    }

    /// A report the duty log calls for (duty.rs).
    pub fn chat_fire(&mut self, now: f64, reason: &str) -> Option<(String, bool)> {
        let (_, j, _) = self.locate(now)?;
        self.fire(reason, None, j)
    }

    /// The engine's answer to our report.
    pub fn on_report(&mut self, correct: bool, now: f64) {
        let j = self.slot_of(now) as i64;
        self.update_c(j, correct);
        // an alarm raised while the fault judge held this report is answered by it
        self.want = None;
        self.calib.clear();
        if !correct && std::mem::take(&mut self.calib_fired) {
            self.calib_false += 1;
        }
        self.calib_fired = false;
        if correct {
            self.correct += 1;
            self.false_since_correct = 0;
            // the instrument is repaired: the level jumps up, start afresh
            self.reset_detector();
            self.prev_level = None;
            self.pre_event.clear();
            self.suspect = None;
        } else {
            self.false_since_correct += 1;
            if self.false_since_correct > self.free_allowance {
                self.paid_false += 1;
            }
            // the engine says no fault is active: the low level is healthy, keep it as the reference
            let seed = std::mem::take(&mut self.last_run);
            self.reset_detector();
            self.hist = seed;
            self.suspect = None;
        }
        self.onset = None;
        log(&format!(
            "fault: report {} at slot {}; P(c) {:?}; free left {}, paid false {}",
            if correct { "correct" } else { "false" },
            j,
            self.pc.iter().skip(1).map(|p| (p * 100.0).round() / 100.0).collect::<Vec<_>>(),
            self.free_left(),
            self.paid_false
        ));
    }

    /// Detector state for the model's fault review.
    pub fn summary(&self) -> Value {
        json!({
            "fault_count_posterior": self.pc.iter().skip(1).map(|p| (p * 100.0).round() / 100.0).collect::<Vec<_>>(),
            "correct_reports": self.correct,
            "last_alarm_probability": (self.last_pi * 100.0).round() / 100.0,
            "held_drop": self.suspect.map(|s| json!({"kind": s.kind.name(), "step": (s.step * 100.0).round() / 100.0})),
        })
    }
}
