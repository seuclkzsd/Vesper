//! Duty log ("值班日志"): the staff chat in zero-reward observation_request reasons (A1-D1 style cards) decoded into
//! DutyFacts (schema: dutylog/schema.md, notes/28) and resolved into a card view: guide-camera fault schedule, dead
//! time (flat tests + stops), closures, terrain. Deterministic; only messages with CJK text should be fed in.
#![allow(dead_code)]
macro_rules! re {
    ($p:expr) => {{
        static R: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
        R.get_or_init(|| regex::Regex::new($p).unwrap())
    }};
}
mod decode;
mod extract;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wx { Clear, ThinCloud, Haze, ThinOrHaze, ThickCloudDiscount, ThickCloud, Rain, Wind, Launch, BrightLowAlt }
impl Wx {
    /// true = observing still possible (maybe degraded); false = closed / avoid.
    pub fn observe(self) -> bool { matches!(self, Wx::Clear | Wx::ThinCloud | Wx::Haze | Wx::ThinOrHaze | Wx::ThickCloudDiscount) }
}
/// Sky region: whole sky, or clockwise 8-point sectors (from, to), 0 = N, 1 = NE … 7 = NW, both inclusive.
#[derive(Clone, Debug, PartialEq)]
pub enum Dirs { All, Sectors(Vec<(u8, u8)>) }
impl Dirs {
    pub fn covers(&self, d: u8) -> bool {
        match self { Dirs::All => true, Dirs::Sectors(v) => v.iter().any(|&(a, b)| (d + 8 - a) % 8 <= (b + 8 - a) % 8) }
    }
}
/// Part of a night; minutes are local clock minutes.
#[derive(Clone, Debug, PartialEq)]
pub enum NightPart { All, FirstHalf, SecondHalf, From(i64), Window(i64, i64) }
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Component { GuideCamera, Unannounced, MaskedByQuake, Positioner, GuideCameraCooling, CcdReadout }
/// Total: throughput ~0, persists until reported. Partial ("效率会降"): degraded, gone by the next night.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Expect { Total, Partial }
#[derive(Clone, Debug, PartialEq)]
pub enum Anchor {
    /// "D 那晚，第 k 段平场灯测试结束以后再等 N"
    FlatTestEnd { night: i64, segment: u8, delay_min: i64 },
    /// "D2 的导星相机，按钟点比前面 D1 那次早/晚 N": local clock of the most recently mentioned D1 job, on D2, + delta
    PreviousJobClock { job_date: i64, date: i64, delta_min: i64 },
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Grade { Good, Normal, Poor }
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason { Coating, HydraulicOil, DomeMotor, GuideCameraExchange, Transformer, Maintenance }
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IgnoreReason { Rumour, Neighbor, LiteralCodeword, Chatter }
#[derive(Clone, Debug, PartialEq)]
pub struct CodeEntry { pub phrase: String, pub wx: Wx, pub directional: bool, pub all_day: bool }

/// Times are UTC unix seconds; `night` / dates are local evening dates as days since 1970-01-01.
#[derive(Clone, Debug, PartialEq)]
pub enum DutyFact {
    Codebook { entries: Vec<CodeEntry> },
    CaesarRule,
    SolfegeRule,
    Terrain { dirs: Option<Dirs>, ridge_alt_deg: Option<f32>, clear_above_deg: Option<f32> },
    Forecast { start: i64, end: i64, dirs: Option<Dirs>, wx: Wx, end_date_implicit: bool },
    Closure { start: i64, end: i64, dirs: Dirs, wx: Wx, below_alt_deg: Option<f32> },
    /// window whose weather/sector comes in the next ("接上…") line
    Pending { start: i64, end: i64, dirs: Option<Dirs> },
    Continuation { wx: Option<Wx>, dirs: Option<Dirs> },
    NightNote { night: i64, wx: Wx, dirs: Option<Dirs>, part: NightPart, below_alt_deg: Option<f32> },
    NightQuality { night: i64, grade: Grade },
    Stop { start: i64, end: i64, reason: StopReason },
    FlatTest { night: i64, start: i64, end: i64 },
    InstrumentDrop { at: Option<i64>, component: Component, expect: Expect, observed: bool, anchor: Option<Anchor>,
                     rain_conditional_night: Option<i64>, postponed_from: Option<i64> },
    InstrumentDropCancel { at: i64 },
    InstrumentDropReject { at: i64 },
    RainDeferralTriggered { night: Option<i64> },
    InstrumentWarning { component: Component },
    Earthquake { at: Option<i64> },
    EarthquakeVerdict,
    DropQuestion { at: Option<i64> },
    DropExplainedCloud,
    ReportAdvice,
    UrgentRequest { expected_issue: Option<i64>, n_targets: Option<u32>, deadline: Option<i64> },
    UrgentReason { complete_by_local_min: Option<i64>, n_needed: Option<u32> },
    OfficialOverride { night: i64, dirs: Dirs, actual: Wx },
    Temperature { night: Option<i64>, min_c: f32 },
    Ignore { reason: IgnoreReason },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status { Assert, Correction, Retract, Rumour, Neighbor, Lamp, LampQuestion, Question, Info }

#[derive(Clone, Debug)]
pub struct Fact {
    pub id: u32,
    pub fact: DutyFact,
    pub request_id: String,
    /// non-empty line index inside the message
    pub line: u16,
    pub speaker: Option<String>,
    pub issued_at: i64,
    pub status: Status,
    pub confidence: f32,
    /// clock of the source times, seconds east of UTC: None = site local, Some(0) = UTC, Some(9h) = Tokyo, UTC-N
    pub clock_utc_offset: Option<i64>,
    pub superseded_by: Option<u32>,
    pub category: &'static str,
}

impl Fact {
    /// This fact in the schema form of dutylog/schema.md (UTC times; the fields the LLM duty-log stage uses: anchors
    /// for its relative times, times ruled out, lines it reviews), `"certain": false` = a guess; None = chatter.
    pub fn schema(&self) -> Option<serde_json::Value> {
        use serde_json::json;
        use DutyFact::*;
        let t = |x: i64| { let m = x.div_euclid(60); let (y, mo, d) = crate::skymath::civil_from_days(m.div_euclid(1440)); json!(format!("{y:04}-{mo:02}-{d:02}T{:02}:{:02}:00Z", m.rem_euclid(1440) / 60, m.rem_euclid(60))) };
        let day = |n: i64| { let (y, m, d) = crate::skymath::civil_from_days(n); json!(format!("{y:04}-{m:02}-{d:02}")) };
        let dirs = |d: &Dirs| match d { Dirs::All => json!("ALL"), Dirs::Sectors(s) => json!(s.iter().map(|&(a, b)| [crate::llm::duty_log::DIRS[a as usize], crate::llm::duty_log::DIRS[b as usize]]).collect::<Vec<_>>()) };
        let mut v = match &self.fact {
            Ignore { reason: IgnoreReason::Chatter } => return None,
            Ignore { reason } => json!({"kind": "Ignore", "reason": match reason { IgnoreReason::Rumour => "rumour", IgnoreReason::Neighbor => "neighbor", _ => "literal_codeword" }}),
            InstrumentDrop { at, expect, observed, anchor, .. } => json!({"kind": "InstrumentDrop", "at": at.map(t), "observed": observed, "certain": at.is_some() || anchor.is_some(),
                "expect": if *expect == Expect::Partial { "partial" } else { "total" }}),
            InstrumentDropCancel { at } => json!({"kind": "InstrumentDropCancel", "at": t(*at)}),
            InstrumentDropReject { at } => json!({"kind": "InstrumentDropReject", "at": t(*at)}),
            RainDeferralTriggered { night } => json!({"kind": "RainDeferralTriggered", "night": night.map(day)}),
            FlatTest { night, start, end } => json!({"kind": "FlatTest", "night": day(*night), "start": t(*start), "end": t(*end)}),
            Terrain { dirs: d, ridge_alt_deg, clear_above_deg } => json!({"kind": "Terrain", "dirs": d.as_ref().map(dirs), "ridge_alt_deg": ridge_alt_deg, "clear_above_deg": clear_above_deg}),
            Pending { .. } => json!({"kind": "Closure", "certain": false}),
            other => json!({"kind": format!("{other:?}").split(|c: char| !c.is_alphanumeric()).next().unwrap_or("")}),
        };
        match self.status {
            Status::Rumour => v["status"] = json!("rumour"),
            Status::Neighbor => v["status"] = json!("neighbor"),
            Status::LampQuestion | Status::Question => v["status"] = json!("lamp_question"),
            _ => {}
        }
        if self.confidence <= 0.3 && self.confidence > 0.1 { v["certain"] = json!(false); }
        Some(v)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Fault { pub at: i64, pub expect: Expect, pub observed: bool, pub first_heard: i64 }

/// Card-level resolved view, recomputed after every ingest.
#[derive(Clone, Debug, Default)]
pub struct View {
    /// guide-camera onsets (cancellations, postponements, rain deferrals, relative times applied) + observed drops
    pub faults: Vec<Fault>,
    pub cancelled: Vec<i64>,
    /// flat tests and stops: whole-sky Q = 0, never report
    pub dead: Vec<(i64, i64)>,
    pub flats: BTreeMap<i64, Vec<(i64, i64)>>,
    /// (start, end, sectors, weather, closed below this altitude)
    pub closures: Vec<(i64, i64, Dirs, Wx, Option<f32>)>,
    /// per direction (0 = N … 7 = NW): altitude above which the sky clears the ridge (latest non-灯 statement)
    pub terrain_clear: [Option<f32>; 8],
    pub warnings: Vec<(i64, Component)>,
    pub urgent: Vec<(Option<i64>, Option<i64>, Option<u32>)>,
    /// fact id -> resolved onset of relative / rain-conditional guide-camera facts
    pub resolved: BTreeMap<u32, i64>,
}

#[derive(Default)]
pub struct DutyLog {
    pub facts: Vec<Fact>,
    pub view: View,
    off: i64,
    codes: Vec<(CodeEntry, regex::Regex)>,
    sol: Option<decode::Sol>,
    /// the last message's non-empty lines decoded: (speaker, text without carriers, Caesar undone)
    pub lines: Vec<(String, String)>,
    /// speaker names seen in line tags; the unreliable ones (whom others correct: "X记错了", "No, X - not …")
    speakers: BTreeSet<String>,
    unreliable: BTreeSet<String>,
}

/// Duty-log text: CJK chat, or an English multi-line staff log (A-D style one-line reasons such as "time-critical
/// follow-up" must never reach the decoder).
pub fn is_duty_log(text: &str) -> bool {
    text.chars().any(|c| ('\u{3040}'..='\u{9fff}').contains(&c))
        || re!(r"^\s*(?:\[[^\]\d]{1,20}\]|<[^>\d]{1,20}>)").is_match(text)
        || re!(r"(?i-u)guide camera|guider|flat-field|flat test|flat-lamp|mirror-cover|dome (?:shut|closed)|telescope down|shutdown|code phrase|\bROT-?\d").is_match(text)
        || (text.lines().filter(|l| !l.trim().is_empty()).count() >= 2
            && re!(r"(?i-u)guide|flat|dome|engineering|throughput|telescope|ridge|cloud|wind|rain|report|shut|launch").is_match(text))
}

/// "Name：" / "Name:" line tags: used when most lines of the message carry one (a name seen twice here, or before).
fn prefix_tags(lines: &[&str], known: &BTreeSet<String>) -> Vec<Option<(String, usize)>> {
    let cand: Vec<Option<(String, usize)>> = lines.iter().map(|l| re!(r"^\s*([^\s\d:：，。,.!?？！/【\[<（(「『“\x22]{1,10}(?: [^\s\d:：，。,.!?？！/【\[<（(「『“\x22]{1,10})?)\s?[：:]\s*")
        .captures(l).map(|c| (c[1].to_string(), c.get(0).unwrap().end()))
        .filter(|(n, _)| !re!(r"^(?:(?i-u:correction|update|note|notice|heads-up|repeat|repeating|delay|shutdown|tonight|tomorrow night|final word|house rules|long-range|scratch that)|更正|订正|注意|通知|补充|预告|改期|规矩|规则|ルール|追記|訂正|結論|お知らせ|予告|今夜|今晚|明晚|备忘|小道消息|长期预报|長期予報|工程组.*)$").is_match(n))).collect();
    let n = cand.iter().filter(|c| c.is_some()).count();
    cand.iter().map(|c| c.clone().filter(|(name, _)| 2 * n > lines.len() && (known.contains(name) || cand.iter().flatten().filter(|x| &x.0 == name).count() >= 2))).collect()
}

impl DutyLog {
    pub fn new() -> Self { Self::default() }

    /// Decode one observation_request reason (all its lines); `utc_off_h` = initialize.site.utc_offset_hours.
    /// Returns the new facts (the view is updated).
    pub fn ingest(&mut self, text: &str, issued_at: i64, utc_off_h: f64, request_id: &str) -> Vec<Fact> {
        self.off = (utc_off_h * 3600.0).round() as i64;
        let ctx = extract::Ctx::new(issued_at, self.off);
        let first = self.facts.len();
        let mut group: Vec<usize> = vec![];
        self.lines.clear();
        let raws: Vec<&str> = text.split('\n').filter(|l| !l.trim().is_empty()).collect();
        let tags = prefix_tags(&raws, &self.speakers);
        for (i, raw) in raws.iter().enumerate() {
            let sol = self.sol.get_or_insert_with(decode::default_sol);
            let (spk, s) = match &tags[i] { Some((n, e)) => (Some(n.clone()), decode::clean(&raw[*e..], sol).1), None => decode::clean(raw, sol) };
            let (s, _) = decode::caesar(&s, ctx.dom());
            if let Some(n) = &spk { self.speakers.insert(n.clone()); }
            // someone corrects another speaker by name -> that speaker is the unreliable one (like 灯 on A1-D1)
            if re!(r"记错|记混|记岔|搞错|弄错|不对|不是|旧图|旧的|ではなく|違います|古い|(?i-u)\bnot\b|wrong|correction|old chart").is_match(&s) {
                // the one addressed ("灯，是…", "X记错了", "No, X -", "Correction for X", "Xさん、"), not "以X的通知为准"
                let named = |n: &str| s.match_indices(n).any(|(i, _)| {
                    let (pre, post) = (&s[..i], &s[i + n.len()..]);
                    pre.ends_with("No, ") || pre.ends_with("for ") || pre.ends_with("to ")
                        || (re!(r"^(?:记|搞|弄|说的|看的|又|さん|，|,|、|。|\s-|:)").is_match(post) && !post.starts_with("的") && !post.starts_with("'s"))
                });
                let hit: Vec<String> = self.speakers.iter().filter(|n| Some(*n) != spk.as_ref() && named(n)).cloned().collect();
                self.unreliable.extend(hit);
            }
            self.lines.push((spk.clone().unwrap_or_default(), s.clone()));
            let lamp = spk.as_deref() == Some("灯") || spk.as_ref().map_or(false, |n| self.unreliable.contains(n));
            let l = extract::line(&ctx, if lamp { Some("灯") } else { spk.as_deref() }, &s, &self.codes);
            if l.cat == "codebook" {
                for f in &l.facts {
                    if let DutyFact::Codebook { entries } = f { self.codes = entries.iter().map(|e| (e.clone(), extract::code_re(&e.phrase))).collect(); }
                }
                if let Some(m) = decode::learn_sol(&s) { self.sol = Some(m); }
            }
            let mut facts = l.facts;
            match l.st {
                Status::Rumour => facts = vec![DutyFact::Ignore { reason: IgnoreReason::Rumour }],
                Status::Neighbor => facts = vec![DutyFact::Ignore { reason: IgnoreReason::Neighbor }],
                _ if facts.is_empty() && l.cat == "other" => facts.push(DutyFact::Ignore { reason: IgnoreReason::Chatter }),
                _ => {}
            }
            let conf = l.conf.unwrap_or(match l.st {
                Status::Rumour | Status::Neighbor => 0.0, Status::LampQuestion => 0.05, _ if lamp => 0.5, _ => 0.9 });
            let base = self.facts.len();
            for fact in facts {
                let id = self.facts.len() as u32;
                self.facts.push(Fact { id, fact, request_id: request_id.into(), line: i as u16, speaker: spk.clone(), issued_at, status: l.st,
                    confidence: conf, clock_utc_offset: l.clock, superseded_by: None, category: l.cat });
            }
            self.window_group(base, l.cat, l.st, &mut group);
        }
        self.resolve();
        self.facts[first..].to_vec()
    }

    /// Message-level forecast resolution: "接上…" completes the previous window, a Morse line echoes it, a correction
    /// line supersedes the previous window group (whoever said it).
    fn window_group(&mut self, base: usize, cat: &str, st: Status, group: &mut Vec<usize>) {
        use DutyFact::*;
        if matches!(st, Status::Rumour | Status::Neighbor) { return; }
        let win = |f: &DutyFact| match f { Forecast { start, .. } | Closure { start, .. } | Pending { start, .. } => Some(*start), _ => None };
        let Some(k) = (base..self.facts.len()).find(|&k| win(&self.facts[k].fact).is_some() || matches!(self.facts[k].fact, Continuation { .. })) else { return };
        let last = group.first().copied();
        if let (Continuation { wx, dirs }, Some(j)) = (self.facts[k].fact.clone(), last) {
            let (start, end, d0) = match &self.facts[j].fact {
                Forecast { start, end, dirs, .. } | Pending { start, end, dirs } => (*start, *end, dirs.clone()),
                Closure { start, end, dirs, .. } => (*start, *end, Some(dirs.clone())),
                _ => return,
            };
            let old = match &self.facts[j].fact { Forecast { wx, .. } | Closure { wx, .. } => Some(*wx), _ => None };
            let dirs = dirs.or(d0);
            self.facts[j].fact = match wx.or(old) {
                Some(w) if w.observe() => Forecast { start, end, dirs, wx: w, end_date_implicit: false },
                Some(w) => Closure { start, end, dirs: dirs.unwrap_or(Dirs::All), wx: w, below_alt_deg: None },
                None => Pending { start, end, dirs },
            };
            return;
        }
        if matches!(self.facts[k].fact, Continuation { .. }) { return; }
        if cat == "morse" {
            if let Some(j) = last.filter(|&j| win(&self.facts[j].fact) == win(&self.facts[k].fact)) {
                self.facts[k].superseded_by = Some(j as u32);
                group.push(k);
                return;
            }
        } else if st == Status::Correction {
            for &j in group.iter() { self.facts[j].superseded_by.get_or_insert(k as u32); }
        }
        *group = vec![k];
    }

    fn resolve(&mut self) {
        use DutyFact::*;
        let off = self.off;
        let mut v = View::default();
        let mut flats: BTreeMap<i64, BTreeSet<(i64, i64)>> = BTreeMap::new();
        let mut drops: BTreeMap<i64, (Expect, Vec<u32>, i64)> = BTreeMap::new();
        let (mut cancels, mut triggers, mut rel, mut cond, mut obs) = (BTreeSet::new(), BTreeSet::new(), vec![], vec![], vec![]);
        let mut asked: Option<(i64, &str)> = None; // "又是机器吗？" awaiting "是硬件掉的，现在提单" in the same message
        let add = |d: &mut BTreeMap<i64, (Expect, Vec<u32>, i64)>, at: i64, e: Expect, id: u32, t: i64| {
            let x = d.entry(at).or_insert((e, vec![], t));
            if e == Expect::Partial { x.0 = e; }
            x.1.push(id);
        };
        for f in &self.facts {
            let ok = !matches!(f.status, Status::Rumour | Status::Neighbor | Status::LampQuestion);
            match &f.fact {
                FlatTest { night, start, end } if f.status == Status::Assert => { flats.entry(*night).or_default().insert((*start, *end)); }
                InstrumentDrop { at, component, expect, anchor, rain_conditional_night, .. } if ok => {
                    if let Some(a) = anchor { rel.push((f.id, a.clone(), *expect, f.issued_at)); }
                    else if let (Some(at), Some(n)) = (at, rain_conditional_night) { cond.push((f.id, *at, *n, f.issued_at)); }
                    else if let Some(at) = at {
                        if *component == Component::GuideCamera { add(&mut drops, *at, *expect, f.id, f.issued_at); } else { obs.push((*at, f.issued_at)); }
                    }
                }
                InstrumentDropCancel { at } => { cancels.insert(*at); }
                DropQuestion { at: Some(at) } => asked = Some((*at, &f.request_id)),
                ReportAdvice => if let Some((at, _)) = asked.take().filter(|a| a.1 == f.request_id) { obs.push((at, f.issued_at)); },
                RainDeferralTriggered { night: Some(n) } => { triggers.insert(*n); }
                InstrumentWarning { component } => v.warnings.push((f.issued_at, *component)),
                Stop { start, end, .. } if matches!(f.status, Status::Assert | Status::Correction) && f.confidence >= 0.5 => v.dead.push((*start, *end)),
                UrgentRequest { expected_issue, n_targets, deadline } if ok => v.urgent.push((*expected_issue, *deadline, *n_targets)),
                Terrain { dirs: Some(d), ridge_alt_deg, clear_above_deg } if ok && f.confidence > 0.5 => {
                    if let Some(h) = clear_above_deg.or(ridge_alt_deg.map(|r| r + 2.0)) {
                        for k in 0..8 { if d.covers(k) { v.terrain_clear[k as usize] = Some(h); } }
                    }
                }
                _ => {}
            }
        }
        for (id, at, n, t) in cond {
            let at = if triggers.contains(&n) { at + 86400 } else { at };
            add(&mut drops, at, Expect::Total, id, t);
            v.resolved.insert(id, at);
        }
        for (id, a, e, t) in &rel {
            if let Anchor::FlatTestEnd { night, segment, delay_min } = a {
                if let Some(&(_, end)) = flats.get(night).and_then(|s| s.iter().nth((*segment as usize).max(1) - 1)) {
                    add(&mut drops, end + delay_min * 60, *e, *id, *t);
                    v.resolved.insert(*id, end + delay_min * 60);
                }
            }
        }
        for (id, a, _, t) in &rel {
            if let Anchor::PreviousJobClock { job_date, date, delta_min } = a {
                let key = |x: &(Expect, Vec<u32>, i64)| x.1.iter().filter(|&&s| s < *id).max().copied();
                let base = drops.iter().filter(|(at, _)| (*at + off).div_euclid(86400) == *job_date && !cancels.contains(*at))
                    .fold(None::<(i64, Option<u32>)>, |b, (at, x)| if b.map_or(true, |b| key(x) > b.1) { Some((*at, key(x))) } else { b });
                if let Some((at, _)) = base {
                    let at = date * 86400 + (at + off).rem_euclid(86400) + delta_min * 60 - off;
                    add(&mut drops, at, Expect::Total, *id, *t);
                    v.resolved.insert(*id, at);
                }
            }
        }
        v.faults = drops.into_iter().filter(|(at, _)| !cancels.contains(at)).map(|(at, x)| Fault { at, expect: x.0, observed: false, first_heard: x.2 }).collect();
        // an observed-drop dialogue retold in a later message lands a day later: same clock within 3 days = same drop
        let mut kept: Vec<(i64, i64)> = vec![];
        for (at, t) in obs { if !kept.iter().any(|k| (k.0 - at).rem_euclid(86400) == 0 && (k.0 - at).abs() <= 3 * 86400) { kept.push((at, t)); } }
        v.faults.extend(kept.into_iter().map(|(at, t)| Fault { at, expect: Expect::Partial, observed: true, first_heard: t }));
        v.faults.sort_by_key(|f| f.at);
        v.cancelled = cancels.into_iter().collect();
        v.flats = flats.into_iter().map(|(n, s)| (n, s.into_iter().collect())).collect();
        v.dead.extend(v.flats.values().flatten().copied());
        v.dead.sort();
        for f in self.facts.iter().filter(|f| f.superseded_by.is_none() && !matches!(f.status, Status::Rumour | Status::Neighbor)) {
            match &f.fact {
                DutyFact::Closure { dirs, wx, below_alt_deg, .. } => v.closures.push((self.span(f).unwrap().0, self.span(f).unwrap().1, dirs.clone(), *wx, *below_alt_deg)),
                DutyFact::NightNote { wx, dirs, below_alt_deg, .. } if !wx.observe() => {
                    let (a, b) = self.span(f).unwrap();
                    v.closures.push((a, b, dirs.clone().unwrap_or(Dirs::All), *wx, *below_alt_deg));
                }
                _ => {}
            }
        }
        v.closures.sort_by_key(|c| c.0);
        self.view = v;
    }

    /// UTC interval a windowed fact covers (night notes: local noon-to-noon night, halves split at midnight).
    pub fn span(&self, f: &Fact) -> Option<(i64, i64)> {
        use DutyFact::*;
        let loc = |n: i64, m: i64| n * 86400 + m * 60 - self.off;
        match &f.fact {
            Forecast { start, end, .. } | Closure { start, end, .. } | Pending { start, end, .. } | Stop { start, end, .. } | FlatTest { start, end, .. } => Some((*start, *end)),
            NightNote { night: n, part, .. } => Some(match part {
                NightPart::All => (loc(*n, 720), loc(n + 1, 720)),
                NightPart::FirstHalf => (loc(*n, 720), loc(n + 1, 0)),
                NightPart::SecondHalf => (loc(n + 1, 0), loc(n + 1, 720)),
                NightPart::From(a) => (loc(*n + (*a < 720) as i64, *a), loc(n + 1, 720)),
                NightPart::Window(a, b) => { let s = loc(*n + (*a < 720) as i64, *a); let e = loc(*n + (*b < 720) as i64, *b); (s, if e <= s { e + 86400 } else { e }) }
            }),
            _ => None,
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/dutylog.rs"]
mod tests;
