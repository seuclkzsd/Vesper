//! Line classifier + fact extractor (one cleaned, decoded duty-log line -> DutyFacts). Dates: single times are
//! calendar dates, "M/D 那晚 a到b" is night-owned (< 12:00 = next day), observed events are the most recent
//! occurrence of that clock time before the entry was written.
use super::decode::{cn_num, duration, morse};
use super::*;
use crate::skymath::{civil_from_days as civil, days_from_civil as days};

macro_rules! D { () => { r"(\d{1,2})/(\d{1,2})" } }
macro_rules! W { () => { r"\s*(?:（(?:周|週|星期)?([一二三四五六日天月火水木金土])）)?" } }
macro_rules! T { () => { r"(\d{1,2}:\d{2})" } }
macro_rules! REL { () => { "大后天|明々后日|明后日|后天|后儿|明天|明晚|明晩|明儿|明日の夜|今晚|今晩|今天晚上|今夜|今儿夜里|今儿晚上|(?:下|这|本)?(?:周|星期|礼拜|週)[一二三四五六日天]|(?:今週|来週)?[月火水木金土日]曜|(?i-u:the night after tomorrow|tomorrow night|tonight|(?:this |next )?(?:mon|tues|wednes|thurs|fri|satur|sun)day night)" } }
macro_rules! DIR { () => { "(东北|东南|西北|西南|北东|南东|南西|北西|北|东|南|西)" } }
macro_rules! TO { () => { r"(?:到|至|–|-|〜|~|から|\bto\b|\buntil\b|\btill\b)" } }

pub struct Ctx { pub off: i64, pub local: i64, pub day: i64, pub tonight: i64 }
impl Ctx {
    pub fn new(issued: i64, off: i64) -> Ctx {
        let local = issued + off;
        let day = local.div_euclid(86400);
        Ctx { off, local, day, tonight: if local.rem_euclid(86400) < 21600 { day - 1 } else { day } }
    }
    pub fn dom(&self) -> u32 { civil(self.day).2 as u32 }
    /// "M/D" -> the nearest such date within [-120, +300] days of the entry.
    fn ymd(&self, mo: &str, d: &str) -> Option<i64> {
        let (mo, d): (i64, i64) = (mo.parse().ok()?, d.parse().ok()?);
        let y = civil(self.day).0;
        (y - 1..=y + 1).filter_map(|y| { let z = days(y, mo, d); (civil(z) == (y, mo, d)).then_some(z) })
            .filter(|z| (-120..=300).contains(&(z - self.day))).min_by_key(|z| (z - self.day).abs())
    }
    fn utc(&self, day: i64, hm: i64, clock: Option<i64>) -> i64 { day * 86400 + hm * 60 - clock.unwrap_or(self.off) }
    fn night(&self, n: i64, hm: i64, clock: Option<i64>) -> i64 { self.utc(if hm < 720 { n + 1 } else { n }, hm, clock) }
    /// most recent occurrence of local clock `hm` at or before the entry
    fn past(&self, hm: i64) -> i64 {
        let t = self.day * 86400 + (hm % 1440) * 60;
        (if t > self.local { t - 86400 } else { t }) - self.off
    }
    fn span(&self, n: i64, a: &str, b: &str, clock: Option<i64>) -> (i64, i64) {
        let (t0, t1) = (self.night(n, hm(a), clock), self.night(n, hm(b), clock));
        (t0, if t1 <= t0 { t1 + 86400 } else { t1 })
    }
}

pub struct Line { pub cat: &'static str, pub st: Status, pub facts: Vec<DutyFact>, pub clock: Option<i64>, pub conf: Option<f32> }
fn ln(cat: &'static str, st: Status, facts: Vec<DutyFact>) -> Line { Line { cat, st, facts, clock: None, conf: None } }

pub fn hm(s: &str) -> i64 { let (h, m) = s.split_once(':').unwrap(); h.parse::<i64>().unwrap_or(0) * 60 + m.parse::<i64>().unwrap_or(0) }
fn has(s: &str, r: &regex::Regex) -> bool { r.is_match(s) }
fn wd(z: i64) -> i64 { (z + 3).rem_euclid(7) }

fn dir(s: &str) -> u8 {
    match s { "北" | "N" => 0, "东北" | "北东" | "NE" => 1, "东" | "E" => 2, "东南" | "南东" | "SE" => 3, "南" | "S" => 4,
        "西南" | "南西" | "SW" => 5, "西" | "W" => 6, _ => 7 }
}

fn dir_en(s: &str) -> u8 {
    ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"].iter().position(|d| s.eq_ignore_ascii_case(d)).unwrap_or(0) as u8
}

fn rel_day(c: &Ctx, s: &str, base: Option<i64>) -> Option<i64> {
    let b = base.unwrap_or(c.tonight);
    if has(s, re!("大后天|明々后日")) { return Some(b + 3); }
    if has(s, re!("后天|明后日|后儿|(?i-u)after tomorrow")) { return Some(b + 2); }
    if has(s, re!("明天|明晚|明晩|明日|明儿|(?i-u)tomorrow")) { return Some(b + 1); }
    // weekday: 下周X / 来週X曜 = next calendar week (Monday start), 这周 / 今週 / this = this week, bare = next occurrence
    let wk = re!("(下|这|本|这个|今週|来週)?(?:周|星期|礼拜|週)?([一二三四五六日天月火水木金土])曜?|(?i-u)(this |next )?(mon|tues|wednes|thurs|fri|satur|sun)day")
        .captures_iter(s).find(|m| m.get(4).is_some() || m.get(0).unwrap().as_str().chars().count() >= 2);
    if let Some(m) = wk {
        let w = match m.get(2) {
            Some(x) => "一二三四五六日".chars().position(|y| x.as_str().starts_with(y))
                .or("月火水木金土日".chars().position(|y| x.as_str().starts_with(y))).unwrap_or(6) as i64,
            None => ["mon", "tues", "wednes", "thurs", "fri", "satur", "sun"].iter().position(|d| m[4].eq_ignore_ascii_case(d)).unwrap_or(6) as i64,
        };
        let mon = b - wd(b);
        let p = m.get(1).or(m.get(3)).map(|x| x.as_str().trim().to_lowercase());
        // "this Tuesday" said on a Saturday is the coming one: a plan never names a past evening
        return Some(match p.as_deref() { Some("下" | "来週" | "next") => mon + 7 + w, Some(_) => if mon + w < b { mon + 7 + w } else { mon + w }, None => b + (w - wd(b)).rem_euclid(7) });
    }
    if let Some(m) = re!(r"(\d{1,2})号").captures(s) {
        let d: i64 = m[1].parse().ok()?;
        return (0..40).map(|k| b + k).find(|&z| civil(z).2 == d);
    }
    if has(s, re!("今晚|今晩|今夜|今天|今儿|今日|(?i-u)tonight")) { return Some(b); }
    if has(s, re!("昨晚|昨夜|昨天")) { return Some(b - 1); }
    None
}

pub fn parse_dirs(s: &str) -> Option<Dirs> {
    if has(s, re!("满天|整片天|全天|空全体|整个天|全方位|满空|天空全|(?i-u)whole sky|entire sky|all over the sky")) { return Some(Dirs::All); }
    let sec = |v: Vec<(u8, u8)>| (!v.is_empty()).then_some(Dirs::Sectors(v));
    let v: Vec<_> = re!(concat!("(?:从)?", DIR!(), "(?:边|面|方)?(?:到|至|~|–|-|〜|から)", DIR!())).captures_iter(s).map(|c| (dir(&c[1]), dir(&c[2]))).collect();
    if !v.is_empty() { return sec(v); }
    if let Some(c) = re!(r"\b(NE|NW|SE|SW|N|E|S|W)\s?[-–]\s?(NE|NW|SE|SW|N|E|S|W)\b").captures(s) { return sec(vec![(dir(&c[1]), dir(&c[2]))]); }
    const EN: &str = "(?i-u:(northeast|northwest|southeast|southwest|north|south|east|west))";
    if let Some(c) = re!(&format!(r"\b{EN}\s*(?:-|–|to)\s*{EN}\b")).captures(s) { return sec(vec![(dir_en(&c[1]), dir_en(&c[2]))]); }
    if let Some(c) = re!(&format!(r"\b(?:the|toward|towards|from|in the|to the)\s+{EN}\b|\b{EN}\s+(?:sector|side|ridge)\b")).captures(s) {
        let d = dir_en(c.get(1).or(c.get(2)).unwrap().as_str());
        return sec(vec![(d, d)]);
    }
    let one = |r: &regex::Regex| sec(r.captures_iter(s).map(|c| (dir(&c[1]), dir(&c[1]))).collect());
    one(re!(concat!(DIR!(), "(?:・|は|が|の|で|、)"))).or_else(|| one(re!(concat!(DIR!(), "(?:边|面|方向|天|角|一带|那边|那一片|那一侧|侧)"))))
}

/// First "D (wd) T 到 [D] T" (or relative-day / English) window -> (start, end, start day).
fn window(c: &Ctx, s: &str, clock: Option<i64>) -> Option<(i64, i64, i64)> {
    let fix = |t0: i64, t1: i64| if t1 <= t0 { t1 + 86400 } else { t1 };
    if let Some(m) = re!(concat!(D!(), W!(), r"\s*(?:（UTC）)?\s*(?:[,，]|の|\bat\b)?\s*(?:从)?\s*", T!(), r"\s*(?:起)?\s*", TO!(), r"\s*(?:(?:the\s+)?", D!(), W!(), r"\s*(?:（UTC）)?\s*(?:[,，]|の)?\s*)?", T!())).captures(s) {
        let d0 = c.ymd(&m[1], &m[2])?;
        let t0 = c.utc(d0, hm(&m[4]), clock);
        let t1 = match m.get(5) { Some(_) => c.utc(c.ymd(&m[5], &m[6])?, hm(&m[8]), clock), None => fix(t0, c.utc(d0, hm(&m[8]), clock)) };
        return Some((t0, t1, d0));
    }
    if let Some(m) = re!(concat!(r"(", REL!(), r")\s*(?:晚上|夜里|夜间|那晚|の夜)?\s*[,，]?\s*", T!(), r"\s*", TO!(), r"\s*(?:", D!(), W!(), r"\s*[,，]?\s*)?", T!())).captures(s) {
        let base = has(s, re!("昨天写|昨天的|忘了发|忘发|忘记发|没发")).then(|| c.tonight - 1);
        let d0 = rel_day(c, &m[1], base)?;
        let t0 = c.night(d0, hm(&m[2]), clock);
        let t1 = match m.get(3) { Some(_) => c.utc(c.ymd(&m[3], &m[4])?, hm(&m[6]), clock), None => fix(t0, c.night(d0, hm(&m[6]), clock)) };
        return Some((t0, t1, d0));
    }
    let m = re!(concat!(D!(), r"\s+", T!(), r"\s*(?:-|until|to)\s*(?:", D!(), r"\s+)?", T!())).captures(s)?;
    let d0 = c.ymd(&m[1], &m[2])?;
    let t0 = c.utc(d0, hm(&m[3]), clock);
    let t1 = match m.get(4) { Some(_) => c.utc(c.ymd(&m[4], &m[5])?, hm(&m[6]), clock), None => fix(t0, c.utc(d0, hm(&m[6]), clock)) };
    Some((t0, t1, d0))
}

pub fn wx_kind(s: &str) -> Option<Wx> {
    let low = s.to_lowercase();
    let s = if s.chars().any(|c| c.is_ascii_uppercase()) { &low } else { s };
    use Wx::*;
    [(re!(r"(?i-u:\brain|\bshower)|下雨|有雨|雨一直|一直在下雨|雨が|雨で|整段有雨|全程有雨|一直有雨|关顶|関顶|dome closed|dome shut|ドーム閉|[^伞]雨[^伞]|雨$"), Rain),
     (re!("发射|launch|ロケット|打ち上げ"), Launch), (re!(r"风大|风很大|大风|(?i-u:\bwind\b|\bwindy\b|\bgusts?\b|\bgale)|风速超限|风限|风が強|強风|强风|強い风"), Wind), (re!("效果打折|打折"), ThickCloudDiscount),
     (re!("厚云|浓云|密云|云厚|云很厚|云偏厚|云比较厚|厚い雲|厚い云|thick|overcast|heavy cloud|云层较厚|拍不了|别去|请回避|回避|観測不可|cannot observe|not observable"), ThickCloud),
     (re!("低空发亮|发亮|明るく|明るい|bright"), BrightLowAlt), (re!("薄霾|うすいもや|薄いもや|haze|霾|もや"), Haze),
     (re!(r"薄云|淡云|うす雲|うす云|薄い雲|薄い云|薄曇|(?i-u:thin cloud|light cloud|\bthin\b)"), ThinCloud)].into_iter().find(|(r, _)| r.is_match(s)).map(|x| x.1)
}

fn wx_name(w: Wx) -> &'static str {
    use Wx::*;
    match w { Rain => "rain", Launch => "launch", Wind => "wind", ThickCloudDiscount => "打折", ThickCloud => "thick", BrightLowAlt => "bright",
        Haze => "haze", ThinCloud | ThinOrHaze => "thin", Clear => "clear" }
}

/// Learn the codebook from a definition line: two or more quoted phrases each defined as a weather meaning
/// (「X」＝Y, “X”=Y, 『X』：Y, "X" -> Y, "X" means Y, 「X」はY).
pub fn codebook(s: &str) -> Vec<CodeEntry> {
    let v: Vec<CodeEntry> = re!(r#"[「『“"‘〈《]([^」』”"’〉》]{1,24})[」』”"’〉》]\s*(?:[=＝:：]|->|→|=>|代表|表示|意思是|意为|指的是|就是|即|\bmeans?\b|\bstands for\b|\bdenotes\b|\bis code for\b|は)\s*([^;；。「『“"‘〈《]+)"#).captures_iter(s).filter_map(|m| {
        let wx = wx_kind(&m[2])?;
        Some(CodeEntry { phrase: m[1].trim().to_string(), wx, directional: has(&m[2], re!("方向|(?i-u)that side|direction")), all_day: has(&m[2], re!("全天|那段时间|(?i-u)whole sky")) })
    }).collect();
    if v.len() >= 2 { v } else { vec![] }
}

/// Codeword usage pattern. CJK: the phrase's characters (without the particles 又/了/要/在/会/还) in order with up to 3
/// filler characters between neighbours ("咖啡机没水" ~ "咖啡机又要没水了", "洗濯機が止まった" ~ "洗濯機がまた止まった").
/// Latin: the content words in order with up to 3 words between ("the kettle is leaking" ~ "the kettle will be leaking again").
pub fn code_re(p: &str) -> regex::Regex {
    let words: Vec<String> = re!(r"[A-Za-z]+").find_iter(p).map(|m| m.as_str().to_lowercase())
        .filter(|w| !matches!(w.as_str(), "the" | "a" | "an" | "is" | "are" | "was" | "will" | "be" | "been" | "again" | "it" | "its" | "of" | "to" | "in" | "on" | "at" | "has" | "have")).collect();
    if words.len() >= 2 && !p.chars().any(|c| ('\u{3040}'..='\u{9fff}').contains(&c)) {
        let ci = |w: &String| { let mut c = w.chars(); let f = c.next().unwrap(); format!("[{}{}]{}", f, f.to_ascii_uppercase(), regex::escape(c.as_str())) };
        return regex::Regex::new(&format!(r"\b{}\b", words.iter().map(ci).collect::<Vec<_>>().join(r"\b(?:\W+\w+){0,3}?\W+\b"))).unwrap();
    }
    let cs: Vec<String> = p.chars().filter(|c| !matches!(c, '又' | '了' | '要' | '在' | '会' | '还' | ' ')).map(|c| regex::escape(&c.to_string())).collect();
    regex::Regex::new(&format!("{}(?:了)?", cs.join("[^，。;；、,.]{0,3}?"))).unwrap()
}

pub fn line(c: &Ctx, spk: Option<&str>, s: &str, codes: &[(CodeEntry, regex::Regex)]) -> Line {
    use DutyFact::*;
    let st = if has(s, re!("隔壁|隣の|另一座山|他们那边|山谷里|别的山|別の山|别の山|邻站|鄰站|山下|(?i-u)other observatory|down the valley|neighbou?r")) { Status::Neighbor }
        else if has(s, re!("听说|听人说|听人讲|有人说|有人传|有传言|传言|传闻|据说|小道消息|坊间|未经证实|未經證實|不作数|当没听见|没确认|未確認|未确认|うわさ|噂|没看到正式通知|没正式通知|没见正式通知|正式な通知はない|还没确认|确かめて|確かめて|別当真|别当真|先别信|别急着信|别信|無視|(?i-u)rumou?r|hearsay|word is|unconfirmed|not verified|nothing official|no official|not official|don't act on|ignore it|some(?:one|body) (?:said|mentioned)|heard (?:a|that)")) { Status::Rumour }
        else { Status::Assert };
    let cb = codebook(s);
    if !cb.is_empty() || (s.contains("暗号") && s.contains('=')) {
        let mut f = vec![];
        if !cb.is_empty() { f.push(Codebook { entries: cb }); }
        if has(s, re!("凯撒|シーザー|(?i-u)caesar|shifted")) { f.push(CaesarRule); }
        if s.contains("唱名") { f.push(SolfegeRule); }
        return ln("codebook", Status::Assert, f);
    }
    if has(s, re!("暗号照旧|合言葉はいつも通り|(?i-u)code phrases unchanged")) { return ln("codebook_reuse", Status::Assert, vec![]); }
    if has(s, re!("规矩再讲|交接时请注意|每条通知报一次|每条导星相机通知")) { return ln("rules", Status::Info, vec![]); }
    let mut sub = s.to_string();
    // a codeword in the past tense with no window ("打印机卡纸了，弄好了") is the real office item, not weather
    let mut literal = has(s, re!("コーヒーメーカー|プリンター"));
    for (e, r) in codes {
        if r.find_iter(&sub).any(|m| has(&sub[m.end()..], re!("^，?(?:弄好|修好|好了|换好)"))) {
            literal = true;
            sub = r.replace_all(&sub, "").into_owned();
        } else {
            sub = r.replace_all(&sub, format!("<{}>", wx_name(e.wx)).as_str()).into_owned();
        }
    }
    if has(s, re!(DEFER)) && has(s, re!(RAINED)) && !has(s, re!(COND)) {
        let n = re!(D!()).captures(s).and_then(|m| c.ymd(&m[1], &m[2]));
        return ln("guide_camera_defer_trigger", Status::Assert, vec![RainDeferralTriggered { night: n }]);
    }
    let gc = has(s, re!(r"导星(?:相机|相機|镜|鏡|器|仪|儀|头|頭|装置|裝置|模块|单元|單元|系统|系統|摄像头|攝像頭| ?CCD|作业|的作业|工作)|ガイダー|ガイド(?:カメラ|鏡|装置|系)|(?i-u)guider|guid(?:e|ing) (?:camera|cam|unit|system|ccd)|autoguider"))
        // "D T 的作业取消" / "D Tの作業は取りやめ" / "the D T job is called off": a cancelled job needs no camera word
        || (has(s, re!(CANCEL)) && has(s, re!("作業|作业|(?i-u)\\bjob\\b|\\bwork\\b")) && !find_dt(s).is_empty())
        // any device: a timed notice whose damage lasts until it is reported is the guide-camera fault of these cards
        || (has(s, re!(r"直到(?:有人|我们)?报|报修前|报修之前|不报修?就|废到有人报|报告するまで|報告するまで|通報するまで|(?i-u)until (?:someone |we |it is |it's )?(?:report|file|fix)|until (?:the fault report|reported)"))
            && !has(s, re!(r"不是故障|不算故障|故障ではない|(?i-u)not a fault")) && !find_dt(s).is_empty());
    if (has(s, re!(STOPW)) && (!gc || has(s, re!("更换|换导星|交[換换]|(?i-u)swap|replace|exchange"))))
        || has(s, re!("ガイドカメラ.{0,3}交[換换].*停止|停止.*ガイドカメラ|换导星相机[^。]*停机|导星相机的交换")) {
        return with_notes(c, &sub, literal, stop(c, spk, s, st));
    }
    if gc { return with_notes(c, &sub, literal, guide(c, spk, s, st)); }
    // no time given: a later unexplained throughput drop is the real fault (report at once)
    if has(s, re!("定位器|电机声音|モーター|制冷|冷却が|读出电路|読み出し|(?i-u)positioner|motor sounds|cooling|readout")) && !has(s, re!("点検|检修圆顶电机|(?i-u)dome motor")) {
        let comp = if has(s, re!("定位器|电机|ポジショナー|(?i-u)positioner|motor")) { Component::Positioner } else if has(s, re!("读出|読み出し|(?i-u)readout")) { Component::CcdReadout } else { Component::GuideCameraCooling };
        let mut f = vec![InstrumentWarning { component: comp }];
        f.extend(terrain(s));
        return with_notes(c, &sub, literal, ln("instrument_warning", Status::Assert, f));
    }
    if has(s, re!("平场|フラット|镜盖|ミラーカバー|(?i-u)flat-lamp|flat lamp|flat-field|flat field|flat test|mirror-cover|mirror cover|dome flat|calibration flat|sky flat|twilight flat")) { return flat(c, s, st); }
    if let Some(m) = re!(concat!("官方[^，。]*?", D!(), r"[^，。]*?(?:那晚)?", DIR!(), "[^，。]*?(多云|雨)")).captures(s) {
        if let Some(n) = c.ymd(&m[1], &m[2]) {
            let actual = if has(s, re!("只是薄云|薄云，照拍")) { Wx::ThinCloud } else if &m[4] == "雨" { Wx::Rain } else { Wx::ThickCloud };
            return ln("official_override", st, vec![OfficialOverride { night: n, dirs: Dirs::Sectors(vec![(dir(&m[3]), dir(&m[3]))]), actual }]);
        }
    }
    if has(s, re!("晃|揺れ|地震|吞吐量|通量|计数|效率掉|提单|叫工程组|该打那个电话|又是机器|机器吗|是硬件|不是云|那就是云|有云飘|云图|スループット|カウント|機械|ハード|雲画像|全天カメラ|(?i-u)allsky|all-sky|throughput|count rate|counts|quake|shook|machine|hardware|cloud map")) { return with_notes(c, &sub, literal, drop_talk(c, s, st)); }
    if has(s, re!("紧急|緊急|(?i-u)urgent")) { return urgent(c, s, st); }
    if has(s, re!("GRB|LVK|IceCube|引力波|中微子|Swift")) {
        let m = re!(concat!(T!(), r"\s*(?:前|之前)拍(?:到)?\s*(\d+)\s*个")).captures(s);
        return ln("urgent_reason", Status::Info, vec![UrgentReason { complete_by_local_min: m.as_ref().map(|m| hm(&m[1])), n_needed: m.and_then(|m| m[2].parse().ok()) }]);
    }
    if (has(s, re!("山|稜線|天顶距|天頂距|挡到|(?i-u)ridge|zenith")) || (has(s, re!(concat!(DIR!(), r"(?:边|面|侧|那边)?\s*(?:是|应该是|只有|就到|实测)?\s*\d{2}\s*度"))) && !has(s, re!(T!()))))
        && !has(s, re!("山谷|山头|山上要断电|另一座山")) {
        let f = terrain(s);
        if !f.is_empty() {
            let mut l = ln("terrain", if has(s, re!("记岔|记错|又错了|不对|又记错|旧图纸|那是旧")) && st == Status::Assert { Status::Correction } else { st }, f);
            if has(s, re!(HEDGE)) { l.conf = Some(0.3); }
            return with_notes(c, &sub, literal, l);
        }
    }
    if has(s, re!("°F|°C|気温|冷え込み|最低")) && has(s, re!(r"\d")) {
        let mut f = vec![];
        if let Some(m) = re!(r"(-?\d+)\s*°([FC])").captures(s) {
            let v: f32 = m[1].parse().unwrap_or(0.0);
            let n = re!(D!()).captures(s).and_then(|d| c.ymd(&d[1], &d[2]));
            f.push(Temperature { night: n, min_c: if &m[2] == "F" { (v - 32.0) * 5.0 / 9.0 } else { v } });
        }
        return ln("temperature", st, f);
    }
    if has(s, re!("中规中矩|普普通通|普通|一般|中等|平平|不好不坏|不咸不淡|平常|偏差|较差|整体差|通透|好夜|好晚上|没啥特别|没什么特别|没有特别|中不溜秋|没好没坏|平稳|完美|正合适|拖后腿|大晴天"))
        && !has(s, re!(T!())) && !has(s, re!("官方|多云")) {
        if let Some(n) = re!(D!()).captures(s).and_then(|d| c.ymd(&d[1], &d[2])) {
            let sarcasm = has(s, re!("呵呵|'完美'|‘完美’|“完美”|'大晴天'|‘大晴天’|“大晴天”|好极了"));
            let grade = if sarcasm || (has(s, re!("偏差|不好|差|别押|别放|避开|不宜")) && !has(s, re!("不好不坏|不出挑也不拖后腿"))) { Grade::Poor }
                else if has(s, re!("通透|好夜|好晚上|视宁度也好|视宁度同样不错|正合适")) { Grade::Good } else { Grade::Normal };
            return ln("night_quality", st, vec![NightQuality { night: n, grade }]);
        }
    }
    with_notes(c, &sub, literal, wx(c, spk, s, &sub, st))
}

/// Lines about something else that also carry "今晚/明天晚上…" summaries keep those NightNotes.
fn with_notes(c: &Ctx, sub: &str, literal: bool, mut l: Line) -> Line {
    if matches!(l.cat, "earthquake" | "terrain" | "drop_is_cloud" | "instrument_warning" | "stop") {
        l.facts.extend(tonight(c, sub).into_iter().filter(|f| matches!(f, DutyFact::NightNote { .. })));
    }
    if literal { l.facts.push(DutyFact::Ignore { reason: IgnoreReason::LiteralCodeword }); }
    l
}

fn terrain(s: &str) -> Vec<DutyFact> {
    let mut out: Vec<DutyFact> = re!(r"天[顶頂]距離?\s*(?:要)?(?:小于|在|不超过|不到|以内)?\s*(\d+)\s*(?:°|度)|(?i-u:within)\s*(\d+)\s*(?:°|deg|degrees)\s*of the zenith").captures_iter(s)
        .map(|m| DutyFact::Terrain { dirs: parse_dirs(s), ridge_alt_deg: None, clear_above_deg: Some(90.0 - m.get(1).or(m.get(2)).unwrap().as_str().parse::<f32>().unwrap_or(0.0)) }).collect();
    if !out.is_empty() { return out; }
    if let Some(m) = re!(r"\b(?i-u:(northeast|northwest|southeast|southwest|north|south|east|west))\b[^;\d]{0,30}?(\d{2})\s*(?:°|deg|degrees)").captures(s) {
        let d = dir_en(&m[1]);
        return vec![DutyFact::Terrain { dirs: Some(Dirs::Sectors(vec![(d, d)])), ridge_alt_deg: m[2].parse().ok(), clear_above_deg: None }];
    }
    for m in re!(concat!(DIR!(), r"[^;\d]{0,30}?(\d{2})\s*度")).captures_iter(s) {
        let d = dir(&m[1]);
        out.push(DutyFact::Terrain { dirs: Some(Dirs::Sectors(vec![(d, d)])), ridge_alt_deg: m[2].parse().ok(), clear_above_deg: None });
    }
    if let (Some(m), Some(DutyFact::Terrain { clear_above_deg, .. })) = (re!(r"(\d{2})\s*度?より上|(\d{2})度を超え").captures(s), out.last_mut()) {
        *clear_above_deg = m.get(1).or(m.get(2)).and_then(|x| x.as_str().parse().ok());
    }
    out
}

fn stop_reason(s: &str) -> StopReason {
    use StopReason::*;
    [(re!("ガイドカメラ.{0,3}交[換换]|导星相机交[换換]|换导星相机"), GuideCameraExchange), (re!("コーティング|镀膜"), Coating), (re!("油圧|液压"), HydraulicOil),
     (re!("ドームモーター|圆顶电机"), DomeMotor), (re!("变压器|変圧器|断电"), Transformer)].into_iter().find(|x| x.0.is_match(s)).map_or(Maintenance, |x| x.1)
}

fn stop(c: &Ctx, spk: Option<&str>, s: &str, st: Status) -> Line {
    let corr = has(s, re!(CORR));
    let reason = stop_reason(s);
    let mut f = vec![];
    let noon = |n: i64| c.utc(n, 720, None);
    if let Some(m) = re!(concat!(D!(), W!(), r"\s*(?:の夜|那晚|夜|晚上|的夜里|夜里|当晚)?[^0-9]{0,30}?", T!(), r"\s*", TO!(), r"\s*", T!())).captures(s) {
        if let Some(n) = c.ymd(&m[1], &m[2]) { let (a, b) = c.span(n, &m[4], &m[5], None); f.push(DutyFact::Stop { start: a, end: b, reason }); }
    } else if let Some(m) = re!(concat!(r"(", REL!(), r")[^0-9]{0,30}?", T!(), r"\s*", TO!(), r"\s*", T!())).captures(s) {
        if let Some(n) = rel_day(c, &m[1], None) { let (a, b) = c.span(n, &m[2], &m[3], None); f.push(DutyFact::Stop { start: a, end: b, reason }); }
    } else {
        let ds: Vec<_> = re!(D!()).captures_iter(s).collect();
        if ds.len() >= 2 && has(s, re!("恢复|まで|到|(?i-u)until|till|through")) {
            if let (Some(a), Some(b)) = (c.ymd(&ds[0][1], &ds[0][2]), c.ymd(&ds[1][1], &ds[1][2])) { f.push(DutyFact::Stop { start: noon(a), end: noon(b), reason }); }
            if spk == Some("灯") && !corr { let mut l = ln("stop", st, f); l.conf = Some(0.3); return l; }
        } else {
            let mut ns = vec![];
            if has(s, re!("今晚|今夜|今晩|(?i-u)tonight")) { ns.push(c.tonight); }
            if has(s, re!("明天|明日|明晚|(?i-u)tomorrow")) { ns.push(c.tonight + 1); }
            if let Some(n) = re!(r"(\d{1,2})号晚上").find(s).and_then(|m| rel_day(c, m.as_str(), None)) { ns.push(n); }
            f.extend(ns.into_iter().map(|n| DutyFact::Stop { start: noon(n), end: noon(n + 1), reason }));
        }
    }
    ln("stop", if corr && st == Status::Assert { Status::Correction } else { st }, f)
}

const CORR: &str = r"更正|订正|纠正|訂正|以这个为准|以此为准|以这条为准|旧图|古い図|改了|又改了|改过了|最新|不准|有误|改一下|Update \(latest\)|记错|记岔|记混|弄错|搞错|又错了|不对|不是这个|说的不对|ではなく|(?i-u)correction|scratch that|latest|old chart";
const CANCEL: &str = r"取消|作罢|撤了|撤销|撤掉|撤回|不搞了|不做了|不弄了|中止|取り消し|取りやめ|キャンセル|(?i-u)cancel|called off|scrapped|scrubbed|been dropped|is dropped|off the schedule";
const HEDGE: &str = r"好像|大概|说不准|不太确定|たぶん|かな|でしたっけ|吧|\?|？|(?i-u)\bmaybe\b|not sure|\bI think\b|might be|I guess|probably";
const STOPW: &str = r"停止|停机|停機|全停|暂停观测|停观|整台都停|整台停机|整机停|全面停止|停着|停電工事|今晚停机|停止中|(?i-u)shut ?down|telescope (?:is )?(?:fully )?(?:down|shut|offline)|still shut";
const DEFER: &str = r"顺延|順延|(?i-u)slides?\b|slip(?:s|ped)? (?:by )?(?:one|a|1) day|postponed? (?:by )?(?:one|a|1) day|moved? (?:by )?(?:one|a|1) day";
const RAINED: &str = r"下雨了|下了雨|雨だった|雨でした|雨が降った|(?i-u)\brained\b";
const COND: &str = r"若|如果|要是|倘若|假如|万一|なら|場合|(?i-u)\bif\b";

/// (mo, d, hh:mm) date-time mentions in CN/JP/EN word orders, in text order.
fn find_dt(s: &str) -> Vec<(String, String, String)> { find_dt_pos(s).into_iter().map(|x| (x.1, x.2, x.3)).collect() }

/// The same with the byte span of each mention.
fn find_dt_pos(s: &str) -> Vec<((usize, usize), String, String, String)> {
    let mut v: Vec<((usize, usize), String, String, String)> = vec![];
    let sp = |m: &regex::Captures| (m.get(0).unwrap().start(), m.get(0).unwrap().end());
    for m in re!(concat!(D!(), W!(), r"\s*(?:（UTC）|\(UTC\))?\s*(?:的|の|at|,|，|大约|约|[…\s]{1,8})?\s*", T!())).captures_iter(s) {
        v.push((sp(&m), m[1].into(), m[2].into(), m[4].into()));
    }
    for m in re!(concat!(T!(), r"[^/:]{0,40}?\bon\s+(?:the\s+)?(?:[A-Z][a-z]{2,8},?\s+)?", D!())).captures_iter(s) { v.push((sp(&m), m[2].into(), m[3].into(), m[1].into())); }
    for m in re!(concat!(r"\b[Oo]n\s+", D!(), r"[^/:]{0,40}?\bat\s+", T!())).captures_iter(s) { v.push((sp(&m), m[1].into(), m[2].into(), m[3].into())); }
    v.sort();
    let mut last = -10i64;
    v.into_iter().filter(|x| { let keep = x.0 .0 as i64 - last > 3; last = x.0 .0 as i64; keep }).collect()
}

/// Clock of a line's times: Tokyo / JST, Beijing, UTC±N or GMT±N engineering clock, UTC, else site local (None).
fn line_clock(s: &str) -> Option<i64> {
    if has(s, re!(r"北京时间|北京時間|(?i-u)beijing time")) { return Some(8 * 3600); }
    if has(s, re!(r"东京时间|东京|東京|日本時間|日本时间|(?:^|[^A-Za-z])JST(?:[^A-Za-z]|$)|(?i-u:tokyo)")) { return Some(9 * 3600); }
    if let Some(m) = re!(r"(?:UTC|GMT)\s*([+\-−])\s*(\d{1,2})(?::([0-5]\d)(?:[^\d/]|$))?").captures(s) {
        let v = m[2].parse::<i64>().unwrap_or(0) * 3600 + m.get(3).map_or(0, |x| x.as_str().parse::<i64>().unwrap_or(0) * 60);
        return Some(if &m[1] == "+" { v } else { -v });
    }
    has(s, re!(r"（UTC）|\(UTC\)|(?:^|[^A-Za-z])(?:UTC|GMT)(?:[^A-Za-z]|$)|格林尼治|格林威治")).then_some(0)
}

fn drop_f(at: Option<i64>, component: Component, expect: Expect) -> DutyFact {
    DutyFact::InstrumentDrop { at, component, expect, observed: false, anchor: None, rain_conditional_night: None, postponed_from: None }
}

/// A duration in the line once its dates and clock times are taken out.
fn dur_in(s: &str) -> Option<i64> { duration(&re!(r"\d{1,2}/\d{1,2}|\d{1,2}:\d{2}").replace_all(s, " ")) }

fn guide(c: &Ctx, spk: Option<&str>, s: &str, st: Status) -> Line {
    use DutyFact::*;
    let clock = line_clock(s);
    let times = find_dt_pos(s);
    let at = |t: &((usize, usize), String, String, String)| c.ymd(&t.1, &t.2).map(|d| c.utc(d, hm(&t.3), clock));
    let partial = has(s, re!(r"效率会|效率下降|效率会降|效率会往下|效率会下|效率.*下滑|会跟着往下掉|往下走|打折|往下掉|少し落ち|落ちます|下がります|(?i-u)throughput will (?:drop|fall|sag)|efficiency (?:will )?(?:be )?(?:reduced|drop|fall|go down)|drop somewhat|will be reduced"))
        && !has(s, re!(r"几乎|归零|为零|全废|废片|作废|不能用|全滅|使えない|(?i-u)\bvoid\b|\blost\b|junk|zero|useless"));
    let expect = if partial { Expect::Partial } else { Expect::Total };
    let ymd = |m: &regex::Captures, i: usize| c.ymd(&m[i], &m[i + 1]);
    let rel = |anchor: Option<Anchor>| { let mut f = drop_f(None, Component::GuideCamera, expect); if let InstrumentDrop { anchor: a, .. } = &mut f { *a = anchor; } f };
    // (1) N after the end of flat-test segment k ("第k段平场…以后再等 N", "k回目のフラット…終わってから N 後", "N after test window #k ends")
    if times.is_empty() && has(s, re!(r"平场|平場|フラット|(?i-u)\bflat|test window")) && has(s, re!(r"以后|以後|之后|结束|测完|跑完|終わ|(?i-u)\bafter\b|\bends?\b")) {
        let k = re!(r"第\s*([一二三四五1-5])\s*段|(\d)\s*(?:回目|段目|番目)|(?i-u)(?:#\s*|no\.\s*|window\s+|segment\s+)(\d)|\b(first|second|third)\b").captures(s)
            .and_then(|m| m.get(1).and_then(|x| cn_num(x.as_str())).or(m.get(2).or(m.get(3)).and_then(|x| x.as_str().parse().ok()))
                .or(m.get(4).map(|x| match x.as_str().to_lowercase().as_str() { "first" => 1, "second" => 2, _ => 3 })));
        if let (Some(k), Some(d)) = (k, dur_in(s)) {
            let a = night_of(c, s).map(|x| Anchor::FlatTestEnd { night: x.0, segment: k as u8, delay_min: d });
            return ln("guide_camera_relative_flat", st, vec![rel(a)]);
        }
    }
    // (2) same clock as / N earlier or later than the job on an earlier date (D2 is named first, D1 second)
    if let Some(m) = re!(concat!(D!(), "[^/]{0,20}?(?:时间与|与)(?:上面|前面|上述)?\\s*", D!(), r"\s*那次相同|on (?:again for )?", D!(), ", at the same local time as the ", D!())).captures(s) {
        let g: Vec<&str> = m.iter().skip(1).flatten().map(|x| x.as_str()).collect();
        let a = c.ymd(g[2], g[3]).zip(c.ymd(g[0], g[1])).map(|(j, d)| Anchor::PreviousJobClock { job_date: j, date: d, delta_min: 0 });
        return ln("guide_camera_relative_job", st, vec![rel(a)]);
    }
    if times.is_empty() {
        let cmp = re!(concat!(D!(), r"\s*(?:那次|的)?[^/]{0,24}?比(?:上面|前面|上述)?\s*", D!(), r"\s*(?:那次|的那次|的)?\s*(早|晚)"))
            .captures(s).map(|m| (ymd(&m, 1), ymd(&m, 3), &m[5] == "早"))
            .or_else(|| re!(concat!(D!(), r"[^/]{0,40}?", D!(), r"[^/]{0,16}?より[^/]{0,12}?(早|遅)")).captures(s).map(|m| (ymd(&m, 1), ymd(&m, 3), &m[5] == "早")))
            .or_else(|| re!(concat!(D!(), r"[^/]{0,60}?\b([Ee]arlier|[Ll]ater)\b[^/]{0,40}?than (?:the )?", D!())).captures(s)
                .map(|m| (ymd(&m, 1), ymd(&m, 4), m[3].eq_ignore_ascii_case("earlier"))));
        if let Some((d2, d1, early)) = cmp {
            let a = d1.zip(d2).map(|(j, d)| Anchor::PreviousJobClock { job_date: j, date: d, delta_min: if early { -1 } else { 1 } * dur_in(s).unwrap_or(0) });
            return ln("guide_camera_relative_job", st, vec![rel(a)]);
        }
    }
    let t0 = times.first().and_then(at);
    let l = |cat, st, facts| Line { cat, st, facts, clock, conf: None };
    if has(s, re!(CANCEL)) {
        if let Some(t) = t0 { return l("guide_camera_cancel", Status::Retract, vec![InstrumentDropCancel { at: t }]); }
    }
    // rain-conditional ("若 D 那晚下雨就顺延一天") and its trigger ("D 那晚下雨了，顺延")
    if has(s, re!(DEFER)) {
        if has(s, re!(RAINED)) && !has(s, re!(COND)) {
            let n = re!(D!()).captures(s).and_then(|m| ymd(&m, 1));
            return l("guide_camera_defer_trigger", Status::Assert, vec![RainDeferralTriggered { night: n }]);
        }
        if let (Some(t), true) = (t0, has(s, re!(COND))) {
            let ds: Vec<_> = re!(D!()).captures_iter(s).collect();
            let mut f = drop_f(Some(t), Component::GuideCamera, Expect::Total);
            if let InstrumentDrop { rain_conditional_night, .. } = &mut f { *rain_conditional_night = ds.get(1).and_then(|m| ymd(m, 1)); }
            return l("guide_camera_conditional", st, vec![f]);
        }
    }
    if has(s, re!(r"推迟|推后|延后|押后|挪后|往后挪|往后推|往后延|延期|遅れ|遅らせ|繰り下げ|(?i-u)postponed|pushed back|delayed|\bslip(?:s|ped)?\b|moved (?:back|later)")) {
        if let (Some(t), Some(d)) = (t0, dur_in(s)) {
            let mut f = drop_f(Some(t + d * 60), Component::GuideCamera, Expect::Total);
            if let InstrumentDrop { postponed_from, .. } = &mut f { *postponed_from = Some(t); }
            return l("guide_camera_postpone", st, vec![InstrumentDropCancel { at: t }, f]);
        }
    }
    // a time said to be wrong ("X 不对", "不是 X", "XではなくY", "not X", "X is wrong") is rejected; the other one stands
    let wrong = |x: &((usize, usize), String, String, String)| {
        let (a, b) = x.0;
        let after: String = s[b..].chars().take(14).collect();
        let before: String = s[..a].chars().rev().take(14).collect::<Vec<_>>().into_iter().rev().collect();
        has(&after, re!(r"^\s*(?:（[^）]*）)?\s*(?:不对|是记错|记错了|有误|是错的|是错|是记岔|记岔了|是我记错|是我们记岔|记混|是记错的|ではなく|(?i-u)is wrong|was wrong)"))
            || has(&before, re!(r"(?:(?:^|[^是])不是|(?i-u)\bnot)\s*(?:the\s*)?(?:[A-Z][a-z]{2}\s*)?$"))
    };
    if times.iter().any(|x| wrong(x)) {
        let mut f: Vec<DutyFact> = times.iter().filter(|x| wrong(x)).filter_map(at).map(|t| InstrumentDropReject { at: t }).collect();
        f.extend(times.iter().filter(|x| !wrong(x)).filter_map(at).next().map(|t| drop_f(Some(t), Component::GuideCamera, Expect::Total)));
        return l("guide_camera_correction", Status::Correction, f);
    }
    let Some(t) = t0 else {
        if has(s, re!("制冷|冷却|(?i-u)cooling")) {
            let f = [InstrumentWarning { component: Component::GuideCameraCooling }].into_iter().chain(terrain(s)).collect();
            return l("instrument_warning", Status::Assert, f);
        }
        return l("guide_camera_other", Status::Info, vec![]);
    };
    // 灯's "导星相机那次是 X 吧？" (always a wrong time); anyone's question or hedge about a time is not a notice
    let q = has(s, re!(r"吧|吗|是不是|难道|莫非|不就是|该不会|来着|会不会|没记错|对不对|でしたっけ|だったかな|かな|[?？]|(?i-u)\bI think\b|not sure|mixing it up|\bmaybe\b|I guess|wasn't the|isn't it"));
    if st == Status::Assert && !has(s, re!("到点报修|その時刻に|(?i-u)report (?:at|right|then)")) && ((spk == Some("灯") && (q || has(s, re!("那次|那回|的话")))) || q) {
        return l("guide_camera_lamp_question", Status::LampQuestion, vec![drop_f(Some(t), Component::GuideCamera, Expect::Total)]);
    }
    l(if st == Status::Assert { "guide_camera" } else { "guide_camera_rumour" }, st, vec![drop_f(Some(t), Component::GuideCamera, expect)])
}

/// The night a "M/D 那晚 / 后天晚上 / 周X晚上" statement is about, and where its date words end.
fn night_of(c: &Ctx, s: &str) -> Option<(i64, usize)> {
    let d = re!(D!()).captures(s).and_then(|m| Some((m.get(0)?.start(), c.ymd(&m[1], &m[2])?, m.get(0)?.end())));
    let r = re!(REL!()).find(s).and_then(|m| Some((m.start(), rel_day(c, m.as_str(), None)?, m.end())));
    [d, r].into_iter().flatten().min().map(|x| (x.1, x.2))
}

fn flat(c: &Ctx, s: &str, st: Status) -> Line {
    let Some((n, e)) = night_of(c, s) else { return ln("flat_test", Status::Info, vec![]) };
    let rest = &s[e..];
    let f = re!(concat!(T!(), r"\s*", TO!(), r"\s*", T!())).captures_iter(rest)
        .map(|x| { let (a, b) = c.span(n, &x[1], &x[2], None); DutyFact::FlatTest { night: n, start: a, end: b } }).collect();
    ln("flat_test", st, f)
}

fn drop_talk(c: &Ctx, s: &str, st: Status) -> Line {
    use DutyFact::*;
    let quake = has(s, re!("晃|揺れ|地震|(?i-u)quake|shook|shaking"));
    let second = has(s, re!("(?:又|另外又|再)(?:单独|自己)?(?:往下|跳|掉)|单独往下|单独又|又单独|又自己往下|もう一段|単独|(?i-u)on its own|dropped again|another step"));
    let s = re!(r"(\d{1,2})点(?:出头一点|刚过一点点)").replace_all(s, "$1:05");
    let tm: Vec<i64> = re!(T!()).find_iter(&s).map(|m| hm(m.as_str())).collect();
    let calm = re!("不用管|不必|不用担心|不用。|白班说不用上来|别乱叫人|别瞎叫人|放っておいて|放置|問題ない|(?i-u)leave it|recovers? (?:by itself|slowly)|itself is fine");
    if quake && !second {
        let mut f = vec![Earthquake { at: tm.first().map(|&t| c.past(t)) }];
        if calm.is_match(&s) {
            if !(has(&s, re!("楼|揺れ|(?i-u)building|shook")) && !tm.is_empty()) { f.clear(); }
            f.push(EarthquakeVerdict);
            f.extend(terrain(&s));
        }
        return ln("earthquake", st, f);
    }
    if second && !tm.is_empty() {
        let f = InstrumentDrop { at: Some(c.past(*tm.last().unwrap())), component: Component::MaskedByQuake, expect: Expect::Partial, observed: true,
            anchor: None, rain_conditional_night: None, postponed_from: None };
        return ln("masked_drop", Status::Assert, vec![f]);
    }
    if s.contains("后面那次突降请按流程提单") { return ln("masked_drop_confirm", Status::Assert, vec![ReportAdvice]); }
    if has(&s, re!("那就是云|有云飘|不是的|是云|雲が流れ|(?i-u)cloud drifted|cloud then")) && !has(&s, re!("不是云|不是天|(?i-u)not (?:the )?weather")) { return ln("drop_is_cloud", Status::Assert, vec![DropExplainedCloud]); }
    if has(&s, re!("不是云|是硬件|不是天|跟天没关系|不关天|机器的事|提单|该打那个电话|不怪天|工程组上来|叫一下工程组|打那个电话|报修|报障|天気じゃない|天気ではない|ハードの故障|今すぐ報告|通報|報告して|(?i-u)not (?:the )?weather|call it in|file it|report it|it's hardware")) {
        let f = match tm.first() {
            Some(&t) => InstrumentDrop { at: Some(c.past(t)), component: Component::Unannounced, expect: Expect::Partial, observed: true,
                anchor: None, rain_conditional_night: None, postponed_from: None },
            None => ReportAdvice,
        };
        return ln("observed_drop", Status::Assert, vec![f]);
    }
    if has(&s, re!(r"又是机器|机器吗|机器\?|机器又|又机器|机器的问题|判断不了|说不准|我也说不准|機械|判断できない|(?i-u)machine again|the machine\?|can't tell")) {
        return ln("drop_question", Status::Question, tm.first().map(|&t| DropQuestion { at: Some(c.past(t)) }).into_iter().collect());
    }
    ln("drop_other", Status::Info, vec![])
}

fn urgent(c: &Ctx, s: &str, st: Status) -> Line {
    let n = re!(r"(\d+)\s*个(?:我们的)?源|我们的源有\s*(\d+)\s*个|源\s*(\d+)\s*个|対象\s*(\d+)\s*個|(?i-u)(\d+) of our (?:targets|sources)").captures(s)
        .and_then(|m| m.iter().skip(1).flatten().next().and_then(|x| x.as_str().parse().ok()));
    let at = |t: &((usize, usize), String, String, String)| c.ymd(&t.1, &t.2).map(|d| c.utc(d, hm(&t.3), None));
    let dts = find_dt_pos(s);
    let k = re!("截止|締切|(?i-u)deadline").find(s).map(|m| m.end());
    let is_dl = |x: &((usize, usize), String, String, String)| k.map_or(false, |k| x.0 .0 >= k && x.0 .0 <= k + 4);
    let dl = dts.iter().find(|x| is_dl(x)).and_then(at);
    let iss = dts.iter().filter(|x| !is_dl(x)).find(|x| has(&s[x.0 .1..], re!(r"^\s*(?:左右|前后|ごろ|頃)")) || has(&s[..x.0 .0], re!(r"(?i-u)around\s*$")))
        .or(dts.iter().find(|x| !is_dl(x))).and_then(at);
    ln("urgent_preannounce", st, vec![DutyFact::UrgentRequest { expected_issue: iss, n_targets: n, deadline: dl }])
}

fn wx(c: &Ctx, spk: Option<&str>, s: &str, sub: &str, st: Status) -> Line {
    let corr = has(s, re!(CORR)) && spk != Some("灯");
    let cont = has(s, re!("^(?:……)?(?:接上|刚才那段的意思是|就是刚才那段|补一句|补充一下|……就是刚才那段)"));
    let utc = has(s, re!(r"（UTC）|UTC 算|时间按 UTC|均为 UTC|以 UTC 计|都是 UTC|UTC 口径|UTC 时间|\(UTC\)"));
    let clock = line_clock(s).or(utc.then_some(0));
    let mut f = vec![];
    for t in morse(s) {
        let m = re!(r"^(\d{1,2})/(\d{1,2}) (\d{2})(\d{2})-(\d{2})(\d{2}) (.*)$").captures(&t).unwrap();
        let Some(d0) = c.ymd(&m[1], &m[2]) else { continue };
        let (h0, h1) = (m[3].parse::<i64>().unwrap() * 60 + m[4].parse::<i64>().unwrap(), m[5].parse::<i64>().unwrap() * 60 + m[6].parse::<i64>().unwrap());
        let (t0, t1) = (c.utc(d0, h0, None), c.utc(d0, h1, None));
        let (start, end) = (t0, if t1 <= t0 { t1 + 86400 } else { t1 });
        f.push(match &m[7] {
            "OK THIN" => DutyFact::Forecast { start, end, dirs: None, wx: Wx::ThinOrHaze, end_date_implicit: true },
            w => DutyFact::Closure { start, end, dirs: Dirs::All, wx: wx_kind(w).unwrap_or(Wx::ThickCloud), below_alt_deg: None },
        });
    }
    if !f.is_empty() { return ln("morse", st, f); }
    let k = wx_kind(sub);
    if re!(MARKS).find_iter(sub).count() >= 2 {
        let f = tonight(c, sub);
        if f.iter().any(|f| !matches!(f, DutyFact::NightNote { wx: Wx::Clear, .. })) { return ln("tonight_summary", st, f); }
    }
    let w = window(c, s, clock);
    if w.map_or(false, |w| w.1 <= w.0) { return ln("forecast_invalid", Status::Info, vec![]); }
    let mut dirs = parse_dirs(sub);
    if k.is_none() && w.is_none() {
        return ln(if has(s, re!("晴|clear|月亮|月が|moon")) { "tonight_clear_or_moon" } else { "other" }, Status::Info, vec![]);
    }
    let Some((start, end, _)) = w else {
        if cont { return ln("continuation", st, vec![DutyFact::Continuation { wx: k, dirs }]); }
        let f = tonight(c, sub);
        return ln(if f.is_empty() { "other" } else { "tonight_summary" }, st, f);
    };
    if dirs.is_none() {
        dirs = re!(concat!(DIR!(), "(?:薄|厚|有|风|一带|方向|那边|<)")).captures(sub).map(|m| Dirs::Sectors(vec![(dir(&m[1]), dir(&m[1]))]));
    }
    let fact = match k {
        None => DutyFact::Pending { start, end, dirs },
        Some(w) if w.observe() => DutyFact::Forecast { start, end, dirs, wx: w, end_date_implicit: false },
        Some(w) => DutyFact::Closure { start, end, dirs: dirs.unwrap_or(Dirs::All), wx: w, below_alt_deg: below(sub) },
    };
    let st = if corr && st == Status::Assert { Status::Correction } else if spk == Some("灯") && st == Status::Assert { Status::Lamp } else { st };
    Line { cat: if cont { "continuation" } else { "forecast" }, st, facts: vec![fact], clock, conf: None }
}

fn below(s: &str) -> Option<f32> { re!(r"(\d{2})\s*度以下").captures(s).and_then(|m| m[1].parse().ok()) }

const MARKS: &str = r"(今天晚上|今晚|今儿夜里|今夜|今晩|今日の夜|明天夜里|明天晚上|明晚|明日の夜|后天晚上|后天夜里|(?:下|这|本)?周[一二三四五六日天]晚上|(?i-u:tonight|tomorrow night)|(\d{1,2})号晚上|(\d{1,2})/(\d{1,2})(?:（[^）]*）)?\s*(?:夜里|晚上|那晚|の夜))";

/// "今晚/明天夜里/后天晚上/周X晚上/N号晚上/M/D 夜里" free-text summaries -> NightNotes.
fn tonight(c: &Ctx, sub: &str) -> Vec<DutyFact> {
    let marks: Vec<_> = re!(MARKS).captures_iter(sub).collect();
    let mut out = vec![];
    for (i, m) in marks.iter().enumerate() {
        let (p, e) = (m.get(0).unwrap().start(), marks.get(i + 1).map_or(sub.len(), |x| x.get(0).unwrap().start()));
        let mk = &m[1];
        let night = if m.get(3).is_some() { c.ymd(&m[3], &m[4]) } else if m.get(2).is_some() || mk.contains('周') { rel_day(c, mk, None) }
            else { rel_day(c, mk, None).or(Some(c.tonight)) };
        let Some(night) = night else { continue };
        for piece in re!(r"[;；。]|\.\s").split(&sub[p..e]) {
            // split at 、 when the following clause names weather
            let mut segs = vec![];
            let mut from = 0;
            for (j, _) in piece.match_indices('、') {
                let nxt = piece[j + 3..].split('、').next().unwrap_or("");
                if has(nxt, re!("薄|雨|风|发射|厚|晴")) { segs.push(&piece[from..j]); from = j + 3; }
            }
            segs.push(&piece[from..]);
            for seg in segs {
                let Some(k) = wx_kind(seg) else {
                    if seg.contains('晴') { out.push(DutyFact::NightNote { night, wx: Wx::Clear, dirs: Some(Dirs::All), part: NightPart::All, below_alt_deg: None }); }
                    if has(seg, re!("停着|停机|停止中")) { out.push(DutyFact::Stop { start: c.utc(night, 720, None), end: c.utc(night + 1, 720, None), reason: StopReason::Maintenance }); }
                    continue;
                };
                if (seg.contains("零星") && !has(seg, re!(r"\d"))) || seg.chars().count() < 6 { continue; }
                let dirs = parse_dirs(seg).or_else(|| re!(concat!(DIR!(), "(?:边|面|天|角)?")).captures(seg).map(|m| Dirs::Sectors(vec![(dir(&m[1]), dir(&m[1]))])))
                    .or((k == Wx::Rain).then_some(Dirs::All));
                let part = if let Some(m) = re!(concat!(T!(), r"\s*", TO!(), r"\s*", T!())).captures(seg) { NightPart::Window(hm(&m[1]), hm(&m[2])) }
                    else if let Some(m) = re!(concat!(r"(?:从\s*|(?i-u:from)\s+)?", T!(), r"\s*(?:以后|以後|起|开始|から|之后)?")).captures(seg).filter(|m| has(m.get(0).unwrap().as_str(), re!("从|以后|以後|起|开始|から|之后|(?i-u)from"))) { NightPart::From(hm(&m[1])) }
                    else if has(seg, re!("前半夜|上半夜|前半|天黑后")) { NightPart::FirstHalf }
                    else if has(seg, re!("后半夜|下半夜|後半|后半")) { NightPart::SecondHalf } else { NightPart::All };
                out.push(DutyFact::NightNote { night, wx: k, dirs, part, below_alt_deg: below(seg) });
            }
        }
    }
    out
}
