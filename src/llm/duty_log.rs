//! LLM stage D: duty-log facts (natural-language understanding and data parsing of the staff duty log, "值班日志":
//! staff chat in zero-reward request reasons with engineering notices, stops, flat-lamp tests, closures and terrain
//! behind codewords, ciphers, time zones, rumours and corrections). The rule decoder (`crate::dutylog`) decodes every
//! line (`predecode`) and its facts ride on the lines (`attach`); the lines it is not sure of that are worth reading
//! (`KEYS`) go to the model in chunks, spread over the season and asked in the background (no decision waits).
//! The `submit_duty_facts` answer lists facts in the rule decoder's fact schema (`dutylog::Fact::schema`; times as
//! written plus their clock, the agent converts to UTC) and a review of the rule facts. `merge`: a number the rules
//! decoded with certainty wins; the model adds the facts the rules missed, fills open fields and sets the meaning
//! (rumour, 灯's question, another site, a literal codeword, wrong).
//! `DutyLog::resolve` turns the model's facts into facts in force (cancellations, corrections, rain deferrals,
//! flat-test or previous-job anchors applied), which join the decoder's view (`duty::view_in`): fault times reported
//! (free false reports only), stops waited out after a probe exposure, closures and terrain applied.

use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};

use super::client::{Background, LlmClient, Tool};
use super::{boolean, number, reason_of};
use crate::skymath::{civil_from_days as civil, days_from_civil as days};

pub const STAGE: &str = "duty_log";
pub const DIRS: [&str; 8] = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];

/// Facts in force from the model (UTC seconds), added to the decoder's view (`duty::view_in`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Facts {
    pub terrain: Vec<(usize, f64)>,            // (direction, top deg)
    pub faults: Vec<(f64, bool)>,              // (onset, announced ahead)
    pub stops: Vec<(f64, f64)>,                // [start, end)
    pub closures: Vec<(f64, f64, Vec<usize>)>, // [start, end), directions
}

/// Text that can be a duty log (CJK); a plain request reason such as "time-critical follow-up" is not.
pub fn is_chat(text: &str) -> bool {
    text.chars().any(|c| c as u32 >= 0x3000)
}

pub fn local_str(t: f64, off: i64) -> String {
    let m = (t / 60.0).round() as i64 + off;
    let (y, mo, d) = civil(m.div_euclid(1440));
    format!("{y:04}-{mo:02}-{d:02} {:02}:{:02}", m.rem_euclid(1440) / 60, m.rem_euclid(60))
}
const KINDS: [&str; 11] = ["InstrumentDrop", "InstrumentDropCancel", "InstrumentDropReject", "RainDeferralTriggered", "Stop", "FlatTest",
                           "Closure", "Terrain", "InstrumentWarning", "UrgentRequest", "Ignore"];
const WX: [&str; 5] = ["rain", "wind", "launch", "thick_cloud", "bright_low_alt"];
const VERDICTS: [&str; 5] = ["rumour", "neighbor", "lamp_question", "literal_codeword", "wrong"];

const SYSTEM: &str = concat!(
    "You read the duty log (值班日志) of a survey telescope: staff chat (e.g. 灯, 立希, 祥子) in Chinese, Japanese or ",
    "English, already pre-decoded (emoticons gone; text after a cipher note like （凯撒…） is already plain; solfege is ",
    "digits; \"(Morse)\" lines only echo the line before). Call submit_duty_facts once with every fact below found in ",
    "the messages (m = message, l = line).\n",
    "Trust: later corrections win (更正, 以…为准, 旧图, 最新, 改了, 记错, \"X不对…是Y\"). Rumours (听说, 有人说, 据说, 没确认, ",
    "うわさ, 未確認), other sites (隔壁, 隣の観測所, 另一座山) and a speaker's unsure question about a guide-camera time ",
    "(…那次…是X吧/对吗/对吧/是不是; the junior who misremembers, e.g. 灯) are never evidence: report them as Ignore. ",
    "A codeword from the codebook used with a time window gives that window's weather (a Closure when it closes the ",
    "sky: rain, wind, thick cloud); one without a window (打印机卡纸了，弄好了) is literal: Ignore.\n",
    "Times: at/start/end = \"YYYY-MM-DD HH:MM\" as written (the text's date and clock; never convert time zones) and ",
    "clock_utc_offset = the clock the text uses: null = site local, 0 = UTC, 9 = Tokyo, -11 for \"UTC-11\", etc. A window ",
    "inside one night (\"M/D 当晚/那晚/夜里/の夜 a到b\", flat tests, stops) is night = \"YYYY-MM-DD\" (its evening) with ",
    "start/end = \"HH:MM\". A window end without a date may be \"HH:MM\". 今晚/明天/后天/大后天/下周X count from the ",
    "message's written_local.\nKinds:\n",
    "- InstrumentDrop: guide-camera work (导星相机, ガイダーカメラ, guider camera) or another engineering notice that ",
    "throughput will drop; also a drop someone saw that is not weather or quake shaking (吞吐量掉了…不是天的事, 提单, ",
    "叫工程组, 硬件掉的). at, clock_utc_offset; expect total (吞吐量几乎归零, 数据全废, 不会自行恢复) or partial (效率会降); ",
    "observed=true if it already happened (at = when it started; \"HH:MM\" alone is fine; after shaking (楼晃) the ",
    "later separate jump is the drop). relative_to {\"anchor\":\"flat_test_end\",\"night\",\"segment\":k,\"delay_min\"} ",
    "for \"第k段平场测试结束后再等N\"; {\"anchor\":\"previous_job_clock\",\"job_date\",\"date\",\"delta_min\"} for \"D2 ",
    "比前面 D1 那次早/晚 N (按钟点)\" (the D1 job mentioned last; give at too when you can). conditional ",
    "{\"if_rain_night\"} for \"若那晚下雨，顺延一天\".\n",
    "- InstrumentDropCancel: at of a job that will not happen (取消; the original time of a postponed job, whose new ",
    "time is an InstrumentDrop). InstrumentDropReject: at that a correction rules out (\"X不对\"). Several jobs can share ",
    "a date or a clock time: a later notice cancels or corrects nothing unless it says so.\n",
    "- RainDeferralTriggered: night that rained, so its job slips a day.\n",
    "- Stop: whole telescope stopped (maintenance, coating, hydraulic oil, dome motor, transformer; 换导星相机…停机 / ",
    "ガイドカメラ交換…停止 is a Stop, not a drop): night + start/end, or dated start/end for long stops; reason.\n",
    "- FlatTest: flat-lamp / lens-cap test segments (平场灯, フラット試験): data lost but not a fault; one per segment: ",
    "night, start, end.\n",
    "- Closure: directions that must not be observed: wx rain|wind|launch|thick_cloud|bright_low_alt; start/end, or ",
    "night (+ start/end); dirs \"ALL\" or [[\"FROM\",\"TO\"]] clockwise (N NE E SE S SW W NW); below_alt_deg for 低空发亮.\n",
    "- Terrain: dirs + ridge_alt_deg, or zenith_deg for \"天顶距 N° 以内可以拍\".\n",
    "- InstrumentWarning: component (positioner, guide_camera_cooling, ccd_readout, ...) of a warning with no time.\n",
    "- UrgentRequest: a pre-announced urgent follow-up request: at (expected), n_targets, deadline.\n",
    "- Ignore: reason rumour|neighbor|lamp_question|literal_codeword, with at when the line names a guide-camera time.\n",
    "Skip everything else (observable thin cloud or haze, night grades, temperatures, chatter) and restatements of a ",
    "fact already given (another language, clock or cipher, \"再念一遍\"). rule_facts = what the rule decoder found ",
    "(site-local times): do not repeat them; review lists only the wrong ones as {\"id\", \"verdict\": ",
    "rumour|neighbor|lamp_question|literal_codeword|wrong}. One pass over the lines; reason = one short sentence."
);

fn tool() -> Tool {
    let s = json!({"type": "string"});
    let n = json!({"type": "number"});
    Tool {
        name: "submit_duty_facts",
        description: "Report the duty-log facts of the given messages and the wrong rule facts.",
        parameters: json!({
            "type": "object",
            "properties": {
                "facts": {"type": "array", "items": {"type": "object", "properties": {
                    "m": {"type": "integer"}, "l": {"type": "integer"}, "kind": {"type": "string", "enum": KINDS},
                    "at": s, "night": s, "start": s, "end": s, "clock_utc_offset": {"type": ["number", "null"]},
                    "expect": {"type": "string", "enum": ["total", "partial"]}, "observed": {"type": "boolean"},
                    "relative_to": {"type": "object"}, "conditional": {"type": "object"},
                    "wx": {"type": "string", "enum": WX}, "dirs": {"description": "\"ALL\" or [[from, to], ...]"},
                    "below_alt_deg": n, "ridge_alt_deg": n, "zenith_deg": n, "component": s, "reason": s,
                    "n_targets": {"type": "integer"}, "deadline": s},
                    "required": ["m", "l", "kind"]}},
                "review": {"type": "array", "items": {"type": "object", "properties": {
                    "id": s, "verdict": {"type": "string", "enum": VERDICTS}}}},
                "reason": s
            },
            "required": ["facts", "review", "reason"]
        }),
    }
}

// --- pre-decode ---

/// One pre-decoded line: speaker, decoded text, and the facts the rule decoder found on it (schema JSON with UTC
/// times; `"certain": false` marks a guess).
#[derive(Clone, Debug, Default)]
pub struct Line {
    pub who: String,
    pub text: String,
    pub rule: Vec<Value>,
}

#[derive(Clone, Debug)]
pub struct Msg {
    pub id: String,
    pub issued: f64,
    pub lines: Vec<Line>,
    pub show: Vec<usize>, // the lines a question shows (a chunk of the message)
}

/// One message pre-decoded (`issued`: UTC seconds) from the rule decoder's decoded lines (`dutylog::DutyLog::lines`:
/// emoticons, zero-width and full-width characters, filler asides, traditional characters, solfege and Chinese
/// clock times, Caesar-shifted English); Morse lines become "(Morse)" (they echo the line before).
pub fn predecode(id: &str, lines: &[(String, String)], issued: f64) -> Msg {
    let morse = |s: &str| {
        let glyphs = s.chars().filter(|c| "○●·—-.tiTA短长♪♫口曰嘀嗒".contains(*c)).count();
        glyphs >= 20 && 10 * glyphs >= 6 * s.chars().filter(|c| !c.is_whitespace() && *c != '/').count()
    };
    let lines: Vec<Line> = lines.iter().map(|(who, text)| Line { who: who.clone(), text: if morse(text) { "(Morse)".into() } else { text.clone() }, rule: Vec::new() }).collect();
    let show = (0..lines.len()).collect();
    Msg { id: id.to_string(), issued, lines, show }
}

/// The pre-decode hook: the rule decoder's facts on the lines of `msg` in the schema form (`dutylog::Fact::schema`,
/// UTC times). A line it is sure of is not asked about; a hedged 灯 line, a drop with no time and a window whose
/// weather is missing are shown to the model with `"certain": false`; a line with nothing decoded has no rule fact.
pub fn attach(msg: &mut Msg, facts: &[crate::dutylog::Fact]) {
    for f in facts {
        if let (Some(v), Some(line)) = (f.schema(), msg.lines.get_mut(f.line as usize)) {
            line.rule.push(v);
        }
    }
}

/// Words of lines worth the model's time (operations, instruments, closures, terrain, corrections), in Chinese,
/// Japanese and English; plus the learned codewords that close the sky.
const KEYS: [&str; 64] = ["导星", "導星", "ガイド", "guider", "camera", "工程", "吞吐", "スループット", "throughput", "计数", "計数", "提单",
    "电话", "電話", "叫人", "硬件", "停", "stop", "平场", "平場", "フラット", "flat", "镜盖", "风", "風", "wind", "关", "閉", "clos", "封",
    "发射", "発射", "launch", "ロケット", "厚", "雨", "rain", "低空", "发亮", "山", "稜線", "ridge", "天顶距", "天頂", "zenith", "紧急",
    "緊急", "urgent", "预告", "更正", "旧图", "为准", "记错", "不对", "最新", "取消", "推迟", "顺延", "cancel", "postpone", "晃", "定位",
    "制冷", "CCD"];

// --- times ---

fn digits(s: &str) -> Vec<i64> {
    s.split(|c: char| !c.is_ascii_digit()).filter(|p| !p.is_empty()).filter_map(|p| p.parse().ok()).collect()
}

/// UTC seconds of local day `day` (days since 1970) and minute `min` on a clock `off` minutes ahead of UTC.
fn utc(day: i64, min: i64, off: i64) -> f64 {
    ((day * 1440 + min - off) * 60) as f64
}

fn day_of(s: &str) -> Option<i64> {
    match digits(s)[..] {
        [y, mo, d, ..] if (1..=12).contains(&mo) && (1..=31).contains(&d) => Some(days(y, mo, d)),
        _ => None,
    }
}

/// "YYYY-MM-DD HH:MM" on clock `off`; "HH:MM" alone inside the night of evening `night` (before noon = next day).
fn when(s: &str, night: Option<i64>, off: i64) -> Option<f64> {
    match digits(s)[..] {
        [y, mo, d, h, mi, ..] if (1..=12).contains(&mo) && (1..=31).contains(&d) && h < 24 && mi < 60 => Some(utc(days(y, mo, d), h * 60 + mi, off)),
        [h, mi] if h < 24 && mi < 60 => Some(utc(night? + i64::from(h < 12), h * 60 + mi, off)),
        _ => None,
    }
}

pub fn iso(t: f64) -> String {
    let m = (t / 60.0).round() as i64;
    let (y, mo, d) = civil(m.div_euclid(1440));
    format!("{y:04}-{mo:02}-{d:02}T{:02}:{:02}:00Z", m.rem_euclid(1440) / 60, m.rem_euclid(60))
}

pub fn parse_iso(s: &str) -> Option<f64> {
    s.ends_with('Z').then(|| when(s, None, 0))?
}

fn local_day(t: f64, off: i64) -> i64 {
    ((t / 60.0).round() as i64 + off).div_euclid(1440)
}

/// "2026-10-01 20:00 Thu": the message's local writing time.
fn written(t: f64, off: i64) -> String {
    let wd = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"][(local_day(t, off) + 4).rem_euclid(7) as usize];
    format!("{} {wd}", local_str(t, off))
}

/// A rule fact as the model sees it: UTC times shown site-local.
fn shown(f: &Value, off: i64) -> Value {
    match f {
        Value::String(s) => parse_iso(s).map_or_else(|| f.clone(), |t| json!(local_str(t, off))),
        Value::Array(a) => Value::Array(a.iter().map(|x| shown(x, off)).collect()),
        Value::Object(o) => Value::Object(o.iter().filter(|(k, _)| *k != "certain").map(|(k, v)| (k.clone(), shown(v, off))).collect()),
        _ => f.clone(),
    }
}

// --- the model's answer ---

/// 8-point directions of `dirs` ("ALL" = all eight; [[from, to]] clockwise; or plain names).
pub fn dirs_of(v: &Value) -> Vec<usize> {
    let ix = |v: &Value| v.as_str().and_then(|s| DIRS.iter().position(|d| d.eq_ignore_ascii_case(s.trim())));
    let mut out = Vec::new();
    match v {
        Value::String(s) if s.eq_ignore_ascii_case("all") => out.extend(0..8),
        Value::Array(a) => {
            for e in a {
                match e {
                    Value::Array(p) if p.len() == 2 => {
                        if let (Some(a), Some(b)) = (ix(&p[0]), ix(&p[1])) {
                            out.extend((0..=(b + 8 - a) % 8).map(|k| (a + k) % 8));
                        }
                    }
                    _ => out.extend(ix(e)),
                }
            }
        }
        _ => out.extend(ix(v)),
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Words a line must carry before the model may rule a time out on it: a cancellation or postponement, a correction,
/// a rumour or an unsure question (a wrong veto would silence a real fault; a missed one costs a free report).
const VETO: [&[&str]; 3] = [
    &["取消", "推迟", "延期", "延后", "顺延", "改到", "改为", "不动", "中止", "取り消", "延期", "cancel", "postpone", "delay", "moved"],
    &["不对", "记错", "記錯", "记岔", "有误", "错", "更正", "订正", "纠正", "間違", "wrong", "not "],
    &["?", "？", "吗", "吧", "听说", "聽說", "有人说", "據說", "据说", "确认", "確認", "传言", "うわさ", "rumo", "heard", "隔壁", "隣"],
];

/// Words the line of a model fault, stop or flat test must carry (the model's facts sit on lines that say so).
const TOPIC: [(&str, &[&str]); 3] = [
    ("InstrumentDrop", &["导星", "導星", "ガイド", "ガイダー", "guider", "camera", "工程", "吞吐", "スループット", "throughput", "计数",
                         "計数", "提单", "电话", "電話", "硬件", "机器", "機器", "定位", "制冷", "ccd", "掉", "跳", "drop"]),
    ("Stop", &["停", "stop", "中止", "維持", "维护", "維護", "保守", "点検", "检查", "檢查", "镀膜", "液压", "变压", "交換", "换"]),
    ("FlatTest", &["平场", "平場", "フラット", "flat", "镜盖", "鏡蓋"]),
];

/// One fact of the model's answer about message `msg`, checked and put in schema form (UTC times), or None.
fn norm(f: &Map<String, Value>, msg: &Msg, l: usize, off: i64) -> Option<Value> {
    let kind = KINDS.iter().find(|k| f.get("kind").and_then(Value::as_str) == Some(**k))?;
    let line = msg.lines[l].text.to_lowercase();
    if TOPIC.iter().any(|(k, words)| k == kind && !words.iter().any(|w| line.contains(w))) {
        return None;
    }
    let clock = match f.get("clock_utc_offset") {
        Some(v) if !v.is_null() => (number(Some(v))? * 60.0).round() as i64,
        _ => off,
    };
    if clock.abs() > 14 * 60 {
        return None;
    }
    let text = |k: &str| f.get(k).and_then(Value::as_str);
    let night = text("night").and_then(day_of);
    let tm = |k: &str| text(k).and_then(|s| when(s, night, clock));
    let near = |t: f64| t > msg.issued - 10.0 * 86400.0 && t < msg.issued + 60.0 * 86400.0;
    let mut out = json!({"kind": kind});
    let mut put = |k: &str, v: Value| {
        out[k] = v;
    };
    match *kind {
        "InstrumentDrop" => {
            let observed = boolean(f.get("observed")).unwrap_or(false);
            let mut at = tm("at");
            if observed {
                // a drop already seen: the latest such clock time before the entry was written
                let w = msg.issued + 600.0;
                if let (None, [h, mi]) = (at, &digits(text("at").unwrap_or(""))[..]) {
                    at = Some(utc(local_day(w, clock), h * 60 + mi, clock));
                }
                at = at.map(|t| if t > w { t - 86400.0 } else { t }).filter(|&t| t > w - 3.0 * 86400.0);
            }
            let rel = f.get("relative_to").and_then(Value::as_object).and_then(|r| {
                let int = |k: &str| number(r.get(k)).map(|x| x.round() as i64);
                let date = |k: &str| r.get(k).and_then(Value::as_str).filter(|s| day_of(s).is_some());
                match r.get("anchor").and_then(Value::as_str)? {
                    "flat_test_end" => {
                        let (n, k, d) = (date("night")?, int("segment").filter(|k| (1..=6).contains(k))?, int("delay_min").filter(|d| (0..=900).contains(d))?);
                        Some(json!({"anchor": "flat_test_end", "night": n, "segment": k, "delay_min": d}))
                    }
                    "previous_job_clock" => {
                        let (j, d, x) = (date("job_date")?, date("date")?, int("delta_min").filter(|x| x.abs() <= 900)?);
                        Some(json!({"anchor": "previous_job_clock", "job_date": j, "date": d, "delta_min": x}))
                    }
                    _ => None,
                }
            });
            if at.map_or(rel.is_none(), |t| !near(t)) {
                return None;
            }
            put("at", at.map_or(Value::Null, |t| json!(iso(t))));
            put("expect", json!(if text("expect") == Some("partial") { "partial" } else { "total" }));
            put("observed", json!(observed));
            put("component", json!(text("component").unwrap_or(if observed { "unannounced" } else { "guide_camera" })));
            if let Some(r) = rel {
                put("relative_to", r);
            }
            let rain = f.get("conditional").and_then(|c| c.get("if_rain_night")).and_then(Value::as_str).and_then(day_of);
            if let Some(d) = rain {
                let (y, m, dd) = civil(d);
                put("conditional", json!({"if_rain_night": format!("{y:04}-{m:02}-{dd:02}"), "then_shift_days": 1}));
            }
        }
        "InstrumentDropCancel" | "InstrumentDropReject" | "Ignore" => {
            let words = VETO[match *kind {
                "InstrumentDropCancel" => 0,
                "InstrumentDropReject" => 1,
                _ => 2,
            }];
            let at = tm("at").filter(|&t| near(t) && words.iter().any(|w| line.contains(w)));
            if *kind == "Ignore" {
                put("reason", json!(text("reason").filter(|r| VERDICTS[..4].contains(r))?));
                if let Some(t) = at {
                    put("at", json!(iso(t)));
                }
            } else {
                put("at", json!(iso(at?)));
            }
        }
        "RainDeferralTriggered" => put("night", json!(text("night").filter(|_| night.is_some())?)),
        "Stop" | "FlatTest" | "Closure" => {
            let whole = *kind == "Closure" && text("start").is_none();
            let start = if whole { night.map(|n| utc(n, 720, clock)) } else { tm("start") }?;
            let mut end = if whole { start + 86400.0 } else { tm("end").or_else(|| when(text("end")?, Some(local_day(start, clock)), clock))? };
            if end <= start && digits(text("end").unwrap_or("")).len() == 2 {
                end += 86400.0; // "HH:MM" end: the next such time after the start
            }
            let long = match *kind {
                "Stop" => 16.0,
                "FlatTest" => 0.25,
                _ => 4.0,
            };
            if end <= start || end - start > long * 86400.0 || !near(start) {
                return None;
            }
            put("start", json!(iso(start)));
            put("end", json!(iso(end)));
            if let Some(n) = text("night").filter(|_| night.is_some()) {
                put("night", json!(n));
            }
            match *kind {
                "Stop" => put("reason", json!(text("reason").unwrap_or("maintenance"))),
                "Closure" => {
                    let wx = text("wx").filter(|w| WX.contains(w))?;
                    let d = dirs_of(f.get("dirs").unwrap_or(&json!("ALL")));
                    put("wx", json!(wx));
                    put("dirs", if d.len() == 8 || d.is_empty() { json!("ALL") } else { json!(d.iter().map(|&k| [DIRS[k], DIRS[k]]).collect::<Vec<_>>()) });
                    if let Some(b) = number(f.get("below_alt_deg")).filter(|b| (5.0..=80.0).contains(b)) {
                        put("below_alt_deg", json!(b));
                    }
                }
                _ => {}
            }
        }
        "Terrain" => {
            let d = dirs_of(f.get("dirs")?);
            let ridge = number(f.get("ridge_alt_deg")).filter(|h| (10.0..=60.0).contains(h));
            let clear = number(f.get("zenith_deg")).filter(|z| (30.0..=80.0).contains(z)).map(|z| 90.0 - z);
            if d.is_empty() || d.len() > 4 || (ridge.is_none() && clear.is_none()) {
                return None;
            }
            put("dirs", json!(d.iter().map(|&k| [DIRS[k], DIRS[k]]).collect::<Vec<_>>()));
            put("ridge_alt_deg", json!(ridge));
            put("clear_above_deg", json!(clear));
        }
        "InstrumentWarning" => put("component", json!(text("component").unwrap_or("unknown"))),
        "UrgentRequest" => {
            let (a, b) = (tm("at")?, tm("deadline")?);
            if !near(a) || b <= a {
                return None;
            }
            put("expected_issue", json!(iso(a)));
            put("deadline", json!(iso(b)));
            put("n_targets", json!(number(f.get("n_targets")).unwrap_or(0.0).round()));
        }
        _ => return None,
    }
    if f.get("clock_utc_offset").map_or(false, |v| !v.is_null()) {
        out["clock_utc_offset"] = json!(clock as f64 / 60.0);
    }
    Some(out)
}

/// Rule and model facts of a batch put together (see the module docs). `model`: (message, line, fact);
/// `wrong`: rule facts (by id, in batch order) the model's review rejects.
pub fn merge(batch: &[Msg], model: &[(usize, usize, Value)], wrong: &BTreeSet<usize>) -> Vec<(usize, usize, Value)> {
    let mut out = Vec::new();
    let mut id = 0;
    for (m, msg) in batch.iter().enumerate() {
        for &l in &msg.show {
            let line = &msg.lines[l];
            let rule: Vec<&Value> = line.rule.iter().filter(|_| {
                id += 1;
                !wrong.contains(&(id - 1))
            }).collect();
            let mine: Vec<&Value> = model.iter().filter(|x| x.0 == m && x.1 == l).map(|x| &x.2).collect();
            let kinds: BTreeSet<&str> = rule.iter().chain(mine.iter()).filter_map(|f| f["kind"].as_str()).collect();
            for kind in kinds {
                let r: Vec<&Value> = rule.iter().copied().filter(|f| f["kind"] == kind).collect();
                let g: Vec<&Value> = mine.iter().copied().filter(|f| f["kind"] == kind).collect();
                if r.is_empty() || (!g.is_empty() && r.iter().all(|f| f.get("certain") == Some(&json!(false)))) {
                    out.extend(g.iter().map(|f| (m, l, (*f).clone()))); // nothing certain from the rules: the model's
                    if !g.is_empty() {
                        continue;
                    }
                }
                for (k, f) in r.iter().enumerate() {
                    let mut f = (*f).clone();
                    if let (Some(o), Some(Value::Object(x))) = (f.as_object_mut(), g.get(k).map(|v| (*v).clone())) {
                        for (key, v) in x {
                            if o.get(&key).map_or(true, Value::is_null) {
                                o.insert(key, v); // a field the rules left open
                            }
                        }
                    }
                    out.push((m, l, f));
                }
                out.extend(g.iter().skip(r.len()).map(|f| (m, l, (*f).clone())));
            }
        }
    }
    out
}

/// The question about `batch`: the pre-decoded lines, the rule facts (site-local times) and the codebook.
pub fn question(batch: &[Msg], codebook: &[String], off: i64) -> Value {
    let mut rule_facts = Vec::new();
    for (m, msg) in batch.iter().enumerate() {
        for &l in &msg.show {
            for f in &msg.lines[l].rule {
                let mut v = shown(f, off);
                v["id"] = json!(format!("r{}", rule_facts.len()));
                v["m"] = json!(m);
                v["l"] = json!(l);
                rule_facts.push(v);
            }
        }
    }
    json!({
        "site_utc_offset_hours": off as f64 / 60.0,
        "codebook": codebook,
        "messages": batch.iter().enumerate().map(|(m, msg)| json!({"m": m, "written_local": written(msg.issued, off),
            "lines": msg.show.iter().map(|&l| format!("{l} {}: {}", msg.lines[l].who, msg.lines[l].text)).collect::<Vec<_>>()})).collect::<Vec<_>>(),
        "rule_facts": rule_facts,
    })
}

/// A duty-log question in flight: the chunks asked about and the background request.
pub struct Asked {
    pub batch: Vec<Msg>,
    q: Background,
}

/// Ask the model about `batch` in the background (no decision waits for it; the rule facts stand meanwhile);
/// `codebook`: the codeword definitions seen so far. None: not asked (no model, budget, breaker).
pub fn ask(client: &mut LlmClient, quota: usize, batch: Vec<Msg>, codebook: &[String], off: i64, timeout: f64) -> Option<Asked> {
    let q = client.ask_background(STAGE, quota, SYSTEM, question(&batch, codebook, off), Some(tool()), timeout)?;
    Some(Asked { batch, q })
}

/// The merged facts of the asked batch once its answer is in: None while the question runs; Some(None) when it
/// ended without a usable answer (the batch keeps its rule facts).
pub fn answered(client: &mut LlmClient, asked: &Asked, off: i64) -> Option<Option<Vec<(usize, usize, Value)>>> {
    let answer = client.background_done(&asked.q)?;
    Some(answer.and_then(|a| {
        let (model, wrong) = read_answer(&a, &asked.batch, off)?;
        crate::log(&format!("llm duty log: {} facts, {} rule facts rejected ({})", model.len(), wrong.len(), reason_of(&a)));
        Some(merge(&asked.batch, &model, &wrong))
    }))
}

/// The model's facts (checked) and the rule facts its review rejects.
pub fn read_answer(a: &Map<String, Value>, batch: &[Msg], off: i64) -> Option<(Vec<(usize, usize, Value)>, BTreeSet<usize>)> {
    let facts = a.get("facts")?.as_array()?;
    let index = |v: Option<&Value>| number(v).filter(|x| *x >= 0.0).map(|x| x as usize);
    let mut model = Vec::new();
    for f in facts.iter().filter_map(Value::as_object) {
        let (Some(m), Some(l)) = (index(f.get("m")), index(f.get("l"))) else { continue };
        if let Some(mut v) = batch.get(m).filter(|msg| msg.show.contains(&l)).and_then(|msg| norm(f, msg, l, off)) {
            v["src"] = json!("model");
            model.push((m, l, v));
        }
    }
    let wrong = a.get("review").and_then(Value::as_array).into_iter().flatten().filter_map(|r| {
        let ok = r.get("verdict").and_then(Value::as_str).map_or(false, |v| VERDICTS.contains(&v));
        r.get("id").and_then(Value::as_str).and_then(|s| s.trim_start_matches('r').parse().ok()).filter(|_| ok)
    });
    Some((model, wrong.collect()))
}

// --- facts in force ---

/// The duty log of a card: every message's merged facts, and the messages not yet read by the model.
#[derive(Default)]
pub struct DutyLog {
    pub pending: Vec<Msg>,
    pub codebook: Vec<String>,
    closing: Vec<String>, // codewords (first characters) that close the sky
    facts: Vec<(f64, String, usize, Value)>, // (issued, request id, line, fact)
}

impl DutyLog {
    /// A new message: its rule facts count at once; its lines worth reading (`KEYS`, the closing codewords, and
    /// any line the rule decoder is unsure of) wait in chunks of at most `chars` for the model.
    pub fn add(&mut self, msg: Msg, chars: usize) {
        self.add_kept(msg, chars, 8 * chars)
    }

    /// `add`, keeping only the newest pending chunks that fit in `cap` (at least one; `cap` >= `chars`).
    fn add_kept(&mut self, msg: Msg, chars: usize, cap: usize) {
        for l in &msg.lines {
            if l.text.contains('「') && (l.text.contains('=') || l.text.contains('＝')) && self.codebook.len() < 3 {
                self.codebook.push(l.text.clone());
                // 「X」＝ meaning: X closes the sky when its meaning says so (learned from the text)
                for part in l.text.split('「').skip(1) {
                    if let Some((word, meaning)) = part.split_once('」') {
                        let meaning: String = meaning.chars().take_while(|c| *c != '「').collect();
                        if ["关", "不了", "停", "别"].iter().any(|w| meaning.contains(w)) {
                            self.closing.push(word.chars().take(3).collect());
                        }
                    }
                }
            }
        }
        let rules: Vec<(usize, usize, Value)> = msg.lines.iter().enumerate().flat_map(|(l, x)| x.rule.iter().map(move |f| (0, l, f.clone()))).collect();
        self.set(std::slice::from_ref(&msg), rules);
        let worth = |x: &Line| {
            let low = x.text.to_lowercase();
            x.text != "(Morse)" && (KEYS.iter().any(|k| low.contains(&k.to_lowercase())) || self.closing.iter().any(|c| x.text.contains(c.as_str())))
                && (x.rule.is_empty() || x.rule.iter().any(|f| f.get("certain") == Some(&json!(false))))
        };
        let mut chunks: Vec<Vec<usize>> = vec![Vec::new()];
        let mut size = 0;
        for (l, x) in msg.lines.iter().enumerate().filter(|(_, x)| worth(x)) {
            if size + x.text.len() > chars && chunks.last().map_or(false, |c| !c.is_empty()) {
                chunks.push(Vec::new());
                size = 0;
            }
            chunks.last_mut().map(|c| c.push(l));
            size += x.text.len();
        }
        // Only the newest chunks can ever be asked: `take_batch` asks the newest that fit in `chars` and drops the rest.
        // Each chunk carries a copy of its whole message, so a huge message (MBs, hundreds of chunks) would need GBs:
        // keep just the newest chunks that fit in `cap` (8 x `chars`, at least one), which always hold what `take_batch` takes.
        let chunks: Vec<Vec<usize>> = chunks.into_iter().filter(|c| !c.is_empty()).collect();
        let size = |m: &Msg, c: &[usize]| c.iter().map(|&l| m.lines[l].text.len()).sum::<usize>();
        let mut total = 0;
        let fresh = chunks.iter().rev().take_while(|c| { total += size(&msg, c); total <= cap }).count().max(1).min(chunks.len());
        self.pending.extend(chunks[chunks.len() - fresh..].iter().map(|show| Msg { show: show.clone(), ..msg.clone() }));
        let mut total = 0;
        let keep = self.pending.iter().rev().take_while(|m| { total += size(m, &m.show); total <= cap }).count().max(1).min(self.pending.len());
        self.pending.drain(..self.pending.len() - keep);
    }

    /// The newest pending chunks that fit in `chars` (at least one); older ones stay with their rule facts.
    pub fn take_batch(&mut self, chars: usize) -> Vec<Msg> {
        let mut size = 0;
        let keep = self.pending.iter().rev().take_while(|m| {
            size += m.show.iter().map(|&l| m.lines[l].text.len()).sum::<usize>();
            size <= chars
        }).count().max(1);
        let all = std::mem::take(&mut self.pending);
        all[all.len().saturating_sub(keep)..].to_vec()
    }

    /// Replace the facts of the batch's lines by `facts` (message index in the batch, line, fact).
    pub fn set(&mut self, batch: &[Msg], facts: Vec<(usize, usize, Value)>) {
        self.facts.retain(|f| !batch.iter().any(|m| m.id == f.1 && m.show.contains(&f.2)));
        self.facts.extend(facts.into_iter().map(|(m, l, f)| (batch[m].issued, batch[m].id.clone(), l, f)));
        self.facts.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.2.cmp(&b.2)));
    }

    /// The model's facts in force (faults, stops, closures, terrain; the decoder's own view carries the rule facts,
    /// which here anchor the model's relative times and rule times out), and the times the log rules out
    /// (cancelled, corrected, a rumour or 灯's guess).
    pub fn resolve(&self, off: i64) -> (Facts, Vec<f64>) {
        let t = |f: &Value, k: &str| f.get(k).and_then(Value::as_str).and_then(parse_iso);
        let s = |f: &Value, k: &str| f.get(k).and_then(Value::as_str).unwrap_or("").to_string();
        let mut flats: BTreeMap<String, Vec<(f64, f64)>> = BTreeMap::new();
        let mut rained = BTreeSet::new();
        let mut gone = Vec::new();
        let mut out = Facts::default();
        for (_, _, _, f) in &self.facts {
            let span = (t(f, "start"), t(f, "end"));
            let mine = f["src"] == "model";
            match (f["kind"].as_str().unwrap_or(""), span) {
                ("FlatTest", (Some(a), Some(b))) => flats.entry(s(f, "night")).or_default().push((a, b)),
                ("RainDeferralTriggered", _) => {
                    rained.insert(s(f, "night"));
                }
                ("InstrumentDropCancel" | "InstrumentDropReject" | "Ignore", _) => gone.extend(t(f, "at")),
                ("Stop", (Some(a), Some(b))) if mine => out.stops.push((a, b)),
                ("Closure", (Some(a), Some(b))) if mine && ["wind", "launch", "thick_cloud"].contains(&s(f, "wx").as_str()) => {
                    let d = dirs_of(&f["dirs"]);
                    if (1..=3).contains(&d.len()) && b - a <= 16.0 * 3600.0 {
                        out.closures.push((a, b, d));
                    }
                }
                ("Terrain", _) if mine => {
                    let top = number(f.get("ridge_alt_deg")).or_else(|| number(f.get("clear_above_deg")).map(|c| c - 1.5));
                    out.terrain.extend(top.into_iter().flat_map(|h| dirs_of(&f["dirs"]).into_iter().map(move |d| (d, h))));
                }
                _ => {}
            }
        }
        for v in flats.values_mut() {
            v.sort_by(|a, b| a.0.total_cmp(&b.0));
            v.dedup();
        }
        for (k, (_, id, _, f)) in self.facts.iter().enumerate() {
            let unreliable = f["unreliable"] == json!(true) || ["rumour", "neighbor", "lamp_question"].contains(&s(f, "status").as_str());
            if f["kind"] != "InstrumentDrop" || unreliable || f["src"] != "model" {
                continue;
            }
            let mut at = t(f, "at");
            let r = &f["relative_to"];
            let int = |k: &str| r[k].as_i64().unwrap_or(0);
            match r["anchor"].as_str() {
                Some("flat_test_end") => {
                    if let Some(seg) = flats.get(r["night"].as_str().unwrap_or("")).and_then(|v| v.get(int("segment").max(1) as usize - 1)) {
                        at = Some(seg.1 + 60.0 * int("delay_min") as f64);
                    }
                }
                Some("previous_job_clock") if at.is_none() => {
                    // the latest job before it in the same message on job_date, same local clock + delta on date
                    let job_day = r["job_date"].as_str().and_then(day_of);
                    let job = self.facts[..k].iter().rev().filter(|x| x.1 == *id && x.3["kind"] == "InstrumentDrop").filter_map(|x| t(&x.3, "at")).find(|&u| Some(local_day(u, off)) == job_day);
                    if let (Some(u), Some(d)) = (job, r["date"].as_str().and_then(day_of)) {
                        at = Some(utc(d, ((u / 60.0).round() as i64 + off).rem_euclid(1440) + int("delta_min"), off));
                    }
                }
                _ => {}
            }
            if f["conditional"]["if_rain_night"].as_str().map_or(false, |n| rained.contains(n)) {
                at = at.map(|u| u + 86400.0);
            }
            if let Some(u) = at.filter(|u| !gone.iter().any(|g| (g - u).abs() < 61.0)) {
                out.faults.push((u, f["observed"] != json!(true)));
            }
        }
        (out, gone)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/duty_log.rs"]
mod tests;
