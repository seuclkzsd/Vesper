//! Text layer: normalisation (zero-width, kaomoji, full-width, distractors, traditional chars), solfege and
//! Chinese-numeral clock times, Caesar (dictionary-checked), Morse (7 glyph sets), durations.
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

const KAOMOJI: [&str; 22] = ["(￣▽￣)", "(｀Д´)", "(｀・ω・´)", "( ´∀｀)", "(¬_¬)", "(^_^)", "(*^^*)", "ヽ(`Д´)ﾉ", "(╬ Ò﹏Ó)",
    "m(_ _)m", "( ´ー`)", "(-_-#)", "┐(´д`)┌", "(ﾟДﾟ;)", "(；´Д｀)", "( ;∀;)", "(・・;)", "(；・∀・)", "(´-ω-`)", "( ˘ω˘ )",
    "(´・ω・`)", "(..)"];
const TRAD: &str = "東帶雲預報說這沒裡關發邊頂個為準滿風時間後現測試點號開動確認聽讓機門氣壞從會視寧對錯記舊圖樣長層較區鐘換屬題嗎陣過傳據誤單鏡蓋場燈訊實際話還別當見來劃稱將處緊們種啟顯應該細離麼颳颶濃狀態稍內兩導請週";
const SIMP: &str = "东带云预报说这没里关发边顶个为准满风时间后现测试点号开动确认听让机门气坏从会视宁对错记旧图样长层较区钟换属题吗阵过传据误单镜盖场灯讯实际话还别当见来划称将处紧们种启显应该细离么刮飓浓状态稍内两导请周";

/// Solfege syllable -> digit (learned from the codebook line when it defines one; this is the fallback).
pub type Sol = Vec<(String, u32)>;
pub fn default_sol() -> Sol {
    [("高音do", 8), ("高音re", 9), ("休止", 0), ("do", 1), ("re", 2), ("mi", 3), ("fa", 4), ("sol", 5), ("la", 6), ("si", 7)]
        .iter().map(|&(a, b)| (a.to_string(), b)).collect()
}

/// "do=1 re=2 … 高音do=8 … 休止=0" -> map (None when the line defines nothing).
pub fn learn_sol(s: &str) -> Option<Sol> {
    let v: Sol = re!(r"(高音do|高音re|休止|do|re|mi|fa|sol|la|si)\s*=\s*(\d)").captures_iter(s)
        .map(|c| (c[1].to_string(), c[2].parse().unwrap())).collect();
    (v.len() >= 7).then(|| { let mut v = v; v.sort_by_key(|x| std::cmp::Reverse(x.0.len())); v })
}

fn t2s(c: char) -> char {
    static M: OnceLock<HashMap<char, char>> = OnceLock::new();
    *M.get_or_init(|| TRAD.chars().zip(SIMP.chars()).collect()).get(&c).unwrap_or(&c)
}

pub fn cn_num(s: &str) -> Option<i64> {
    if let Ok(v) = s.parse() { return Some(v); }
    let d = |c: char| "零一二三四五六七八九".chars().position(|x| x == c).map(|v| v as i64)
        .or(match c { '〇' => Some(0), '两' => Some(2), _ => None });
    if let Some((a, b)) = s.split_once('十') {
        let a = if a.is_empty() { 1 } else { d(a.chars().next()?)? };
        let b = if b.is_empty() { 0 } else { d(b.chars().next()?)? };
        return Some(a * 10 + b);
    }
    s.chars().try_fold(0, |v, c| Some(v * 10 + d(c)?))
}

/// -> (speaker, cleaned text, had distractor)
pub fn clean(raw: &str, sol: &Sol) -> (Option<String>, String) {
    let mut s: String = raw.chars().filter(|c| !matches!(c, '\u{200b}' | '\u{200c}' | '\u{200d}' | '\u{2060}' | '\u{feff}')).collect();
    for k in KAOMOJI { if s.contains(k) { s = s.replace(k, ""); } }
    let s: String = s.chars().map(|c| match c {
        '（' | '）' | '，' => c, '：' => ':', '\u{ff01}'..='\u{ff5e}' => char::from_u32(c as u32 - 0xfee0).unwrap(), _ => c }).collect();
    let s = re!(r"（(?:车停在[^（）]*号|有事打分机 ?\d+|第 ?\d+ ?杯咖啡了|顺便:工号[^（）]*)）").replace_all(&s, "");
    let (spk, s) = match re!(r"^\s*(?:【(.+?)】|\[([^\]\d]{1,20})\]|<([^>\d]{1,20})>)").captures(&s) {
        Some(c) => (c.iter().skip(1).flatten().next().map(|x| x.as_str().trim().to_string()), s[c.get(0).unwrap().end()..].trim().to_string()),
        None => (None, s.trim().to_string()),
    };
    let s: String = s.chars().map(t2s).collect();
    let tok = |t: &str| sol.iter().find(|x| x.0 == t).map_or(0, |x| x.1);
    let num = |g: &str| g.split_whitespace().fold(0, |v, t| v * 10 + tok(t));
    let sol_rep = |c: &regex::Captures| { let (h, m) = c[1].split_once(':').unwrap(); format!(" {}:{:02} ", num(h), num(m)) };
    let s = re!(r"（(?:哼一句|按谱子|唱名)[:：]\s*((?:高音do|高音re|do|re|mi|fa|sol|la|si|休止)(?:\s+(?:高音do|高音re|do|re|mi|fa|sol|la|si|休止))*\s*:\s*(?:高音do|高音re|do|re|mi|fa|sol|la|si|休止)(?:\s+(?:高音do|高音re|do|re|mi|fa|sol|la|si|休止))*)\s*）").replace_all(&s, sol_rep);
    let s = re!(r"♪\s*((?:高音do|高音re|do|re|mi|fa|sol|la|si|休止)(?:\s+(?:高音do|高音re|do|re|mi|fa|sol|la|si|休止))*\s*:\s*(?:高音do|高音re|do|re|mi|fa|sol|la|si|休止)(?:\s+(?:高音do|高音re|do|re|mi|fa|sol|la|si|休止))*)\s*♪").replace_all(&s, sol_rep);
    let s = cn_times(&s);
    (spk, re!(r"\s+").replace_all(&s, " ").into_owned())
}

/// Date and clock phrases -> "M/D " and "H:MM": Chinese numerals (零点一刻), 12-hour forms with a period word
/// (晚上十一点一刻, 凌晨一点半, 午後11時, 11:15 pm), 2315h, 0515Z (UTC), 23時半, ISO dates, English month names, 10月5日/号.
pub fn cn_times(s: &str) -> String {
    let n = |x: &str| cn_num(x).unwrap_or(0);
    let mon = |x: &str| ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"].iter()
        .position(|m| x.to_lowercase().starts_with(m)).map_or(0, |i| i + 1);
    let s = re!(r"(^|[^\d])20\d\d-(\d{1,2})-(\d{1,2})([^\d]|$)").replace_all(s, |c: &regex::Captures| format!("{}{}/{} {}", &c[1], n(&c[2]), n(&c[3]), &c[4]));
    const MON: &str = r"(?i-u:(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sept?(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?)";
    let s = re!(&format!(r"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+{MON}\b")).replace_all(&s, |c: &regex::Captures|
        if &c[2] == "may" { c[0].to_string() } else { format!("{}/{}", mon(&c[2]), n(&c[1])) });
    let s = re!(&format!(r"\b{MON}\s+(\d{{1,2}})(?:st|nd|rd|th)?\b")).replace_all(&s, |c: &regex::Captures|
        if &c[1] == "may" { c[0].to_string() } else { format!("{}/{}", mon(&c[1]), n(&c[2])) });
    let s = re!(r"(\d{1,2})(?::(\d{2}))?\s?([AaPp])\.?\s?[Mm]\.?([^A-Za-z]|$)").replace_all(&s, |c: &regex::Captures| {
        let h = n(&c[1]) % 12 + if c[3].eq_ignore_ascii_case("p") { 12 } else { 0 };
        format!("{}:{}{}", h, c.get(2).map_or("00", |m| m.as_str()), &c[4])
    });
    let s = re!(r"\b([0-2]\d)([0-5]\d)\s?([hZ])\b").replace_all(&s, |c: &regex::Captures|
        format!("{}:{}{}", n(&c[1]), &c[2], if &c[3] == "Z" { " UTC" } else { "" }));
    // a period word makes it 12-hour: 傍晚/下午/午後 +12; night words (晚上, 夜里, 今晚, 夜) 6-11 +12 and 12 = 0, before 6 kept
    // (night-owned: "今晚5点以后" = 05:00); 凌晨/半夜/深夜 12 = 0. A day mark in front (明天晚上, 今晚, 周六晚上) is kept.
    let s = re!(r"(\d?)((?:今天|明天|后天|大后天|今儿|明儿|今|明|那|(?:下|这|本)?(?:周|星期|礼拜)[一二三四五六日天])?)(凌晨|半夜|午夜|深夜|夜里|夜间|晚上|晚间|傍晚|下午|中午|午后|午後|午前|早上|早晨|清晨|上午|晚|夜)\s*(\d{1,2}|[零〇一二两三四五六七八九十]{1,3})\s*[点時时](?:(半)|(一刻)|(三刻)|(整)|\s*(\d{1,2}|[零〇一二三四五六七八九十]{1,3})\s*分?|(出头一点|刚过一点点|出头))?").replace_all(&s, |c: &regex::Captures| {
        let h = n(&c[4]);
        let m = c.get(9).map_or(if c.get(5).is_some() { 30 } else if c.get(6).is_some() { 15 } else if c.get(7).is_some() { 45 } else if c.get(10).is_some() { 5 } else { 0 }, |m| n(m.as_str()));
        let h = match &c[3] {
            "傍晚" | "下午" | "午后" | "午後" if h < 12 => h + 12,
            "晚上" | "晚间" | "夜里" | "夜间" | "晚" | "夜" if (6..12).contains(&h) => h + 12,
            "中午" if h < 6 => h + 12,
            "中午" | "傍晚" | "下午" | "午后" | "午後" => h,
            _ if h == 12 => 0,
            _ => h,
        };
        let keep = if c[2].is_empty() { String::new() } else { format!("{}{}", &c[2], &c[3]) };
        format!("{}{} {}:{:02}", if c[1].is_empty() { "" } else { &c[1] }, keep, h, m)
    });
    let s = re!("(\\d?)([零〇一二两三四五六七八九十]{1,3})点(?:([零〇一二两三四五六七八九十]{1,3})分|(半)|(一刻)|(三刻)|(整))").replace_all(&s, |c: &regex::Captures| {
        let m = c.get(3).map_or(if c.get(4).is_some() { 30 } else if c.get(5).is_some() { 15 } else if c.get(6).is_some() { 45 } else { 0 }, |m| n(m.as_str()));
        format!("{}{}:{:02}", if c[1].is_empty() { String::new() } else { format!("{} ", &c[1]) }, n(&c[2]), m)
    });
    let s = re!(r"(\d{1,2})[点時时](\d{1,2})分?").replace_all(&s, |c: &regex::Captures| format!("{}:{:02}", n(&c[1]), n(&c[2])));
    let s = re!(r"(\d{1,2})[点時时](半|整)").replace_all(&s, |c: &regex::Captures| format!("{}:{}", n(&c[1]), if &c[2] == "半" { "30" } else { "00" }));
    let s = re!(r"(\d{1,2})月(\d{1,2})[日号號]").replace_all(&s, "$1/$2 ");
    let s = re!(r"(\d{1,2})[時时]([^\d間间]|$)").replace_all(&s, "$1:00$2");
    re!(r"(\d{1,2})点(\s*(?:到|至|~|-|–|〜|左右|前后|上下|那会|那阵|以后|以後|起|开始|这样|的时候|开工|动|，|。|、|$))").replace_all(&s, "$1:00$2").into_owned()
}

fn shift(s: &str, k: u32) -> String {
    s.chars().map(|c| match c {
        'a'..='z' => (((c as u32 - 97 + 26 - k) % 26) + 97) as u8 as char,
        'A'..='Z' => (((c as u32 - 65 + 26 - k) % 26) + 65) as u8 as char,
        _ => c }).collect()
}

fn en_score(s: &str) -> f64 {
    static W: OnceLock<HashSet<&str>> = OnceLock::new();
    let w = W.get_or_init(|| "the a an of to and is at on in for from be by it that this not do no so if then until after than with will has have are was were been can we our you all as one by day night local time hour hours min minutes report fault frames data job work guide guider camera engineering flat lamp test tests window windows sky cloud thin thick haze wind rain closed launch dome sector update latest observe observable keep still over whole side strong cannot low bright avoid below above fine deg ridge zenith postponed pushed back delay later earlier cancelled called off scratch rumor rumour unconfirmed ignore official throughput drop quake machine hardware urgent targets deadline code phrases shifted shift telescope down shutdown notice correction wrong lost void junk useless zero report reported moment same again scheduled maintenance file clock convert rains slides night's am pm repeating unit".split_whitespace().collect());
    let low = s.to_lowercase();
    let words: Vec<&str> = re!("[a-z']+").find_iter(&low).map(|m| m.as_str()).collect();
    words.iter().filter(|x| w.contains(*x)).count() as f64 / words.len().max(1) as f64
}

/// Caesar layer: a cipher marker ("（凯撒，偏移 N）", "(ROT-7)", "[cipher 7]", "（シーザー…）", "（密文…）") is dropped and the
/// key is the stated number, else the writing day (local day of month), else whichever of the 26 keys reads as English
/// best; unmarked Latin text is kept when it already reads as English, else decoded only when some key clearly reads.
/// -> (text, Some(key)) when the line was English.
pub fn caesar(s: &str, day: u32) -> (String, Option<u32>) {
    let mark = re!(r"[（(\[【]([^）)\]】]*(?:凯撒|シーザー|加密|钥匙|挪|密文|位移|移位|ずらし|暗号|ROT|[Rr]ot|[Ss]hift|[Cc]ipher|[Cc]aesar)[^）)\]】]*)[）)\]】]").find(s);
    if mark.is_none() && re!("[A-Za-z]{2,}").find_iter(s).count() < 5 { return (s.to_string(), None); }
    let body = match mark { Some(m) => format!("{}{}", &s[..m.start()], &s[m.end()..]), None => s.to_string() };
    let stated = mark.and_then(|m| re!("[0-9]+|[一二三四五六七八九十]+").find(m.as_str())).and_then(|x| cn_num(x.as_str()));
    let plain = en_score(&body);
    if mark.is_none() && plain >= 0.3 { return (cn_times(&body), Some(0)); }
    let first = stated.map_or(day, |v| v as u32) % 26;
    for k in [first, day % 26] {
        if en_score(&shift(&body, k)) >= 0.3 { return (cn_times(&shift(&body, k)), Some(k)); }
    }
    // any other key: the best one, and it must read clearly (unmarked lines are often Morse ti/TA or romaji)
    let (k, sc) = (1..26).map(|k| (k, en_score(&shift(&body, k)))).fold((0, 0.0), |b, x| if x.1 > b.1 { x } else { b });
    if sc >= if mark.is_some() { 0.3 } else { 0.5 } && sc > plain + 0.2 { return (cn_times(&shift(&body, k)), Some(k)); }
    if mark.is_some() { (cn_times(&shift(&body, first)), Some(first)) } else { (s.to_string(), None) }
}

const MORSE: [(&str, char); 41] = [(".-", 'A'), ("-...", 'B'), ("-.-.", 'C'), ("-..", 'D'), (".", 'E'), ("..-.", 'F'), ("--.", 'G'),
    ("....", 'H'), ("..", 'I'), (".---", 'J'), ("-.-", 'K'), (".-..", 'L'), ("--", 'M'), ("-.", 'N'), ("---", 'O'), (".--.", 'P'),
    ("--.-", 'Q'), (".-.", 'R'), ("...", 'S'), ("-", 'T'), ("..-", 'U'), ("...-", 'V'), (".--", 'W'), ("-..-", 'X'), ("-.--", 'Y'),
    ("--..", 'Z'), ("-----", '0'), (".----", '1'), ("..---", '2'), ("...--", '3'), ("....-", '4'), (".....", '5'), ("-....", '6'),
    ("--...", '7'), ("---..", '8'), ("----.", '9'), ("-..-.", '/'), ("-....-", '-'), ("---...", ':'), (".-.-.-", '.'), ("..--..", '?')];

/// Morse in ANY two-symbol alphabet (●○, ·—, ti/TA, dit-dah, トン/ツー, ▪▬, . -, *_ …; letters split by single spaces,
/// words by "/", "|", "　" or 2+ spaces): the line without bracketed asides must be all Morse; the alphabet is the only
/// pair of equal-length units (1-3 chars) that tiles every letter; both polarities are tried and only decodes that read
/// "M/D HHMM-HHMM WORDS" are kept.
pub fn morse(s: &str) -> Vec<String> {
    let full = re!(r"[（(][^）)]*[）)]").replace_all(s, " ");
    // the Morse run is the line, or what follows its last "：" / "…" / "，" ("……换个写法：·———— …")
    let mut cands = vec![full.trim().to_string()];
    if let Some(i) = full.rfind(['：', ':', '…', '，', ',']) { cands.push(full[i..].trim_start_matches(['：', ':', '…', '，', ',']).trim().to_string()); }
    cands.into_iter().find_map(|b| morse_run(&b)).into_iter().collect()
}

fn morse_run(body: &str) -> Option<String> {
    let words: Vec<Vec<String>> = re!(r"\s*[/|｜]+\s*|\s{2,}|　+").split(body).map(|w| w.split(' ').filter(|t| !t.is_empty())
        .map(|t| if t.chars().any(|c| c.is_alphabetic() && c.is_ascii()) { t.replace('-', "") } else { t.to_string() }).collect::<Vec<String>>())
        .filter(|w: &Vec<String>| !w.is_empty()).collect();
    let toks: Vec<&String> = words.iter().flatten().collect();
    if toks.len() < 8 { return None; }
    for l in 1..=3usize {
        let mut units: Vec<String> = vec![];
        let ok = toks.iter().all(|t| {
            let c: Vec<char> = t.chars().collect();
            c.len() % l == 0 && c.chunks(l).all(|u| { let u: String = u.iter().collect(); if !units.contains(&u) { units.push(u); } units.len() <= 2 })
        });
        if !ok || units.len() != 2 { continue; }
        for (x, y) in [(&units[0], &units[1]), (&units[1], &units[0])] {
            let text = words.iter().map(|w| w.iter().map(|t| {
                let c: Vec<char> = t.chars().collect();
                let code: String = c.chunks(l).map(|u| if u.iter().collect::<String>() == **x { '.' } else { '-' }).collect();
                MORSE.iter().find(|p| p.0 == code).map_or('?', |p| p.1)
            }).collect::<String>()).collect::<Vec<_>>().join(" ");
            let _ = y;
            if re!(r"^\d{1,2}/\d{1,2} \d{4}-\d{4}( [A-Z]+)+$").is_match(&text) { return Some(text); }
        }
    }
    None
}

/// "两小时四十五分" / "一个半小时" / "half an hour" / "two hours and 45 minutes" -> minutes.
pub fn duration(s: &str) -> Option<i64> {
    let en = |w: &str| match w { "a" | "an" | "one" => 1, "two" => 2, "three" => 3, "four" => 4, _ => 5 };
    let half_only = s.contains("half an hour") && !re!(r"\b(?:one|two|three|four|five)\s+hours?|\ban hour and").is_match(s);
    if half_only { return Some(30); }
    if let Some(c) = re!(r"\b(an?|one|two|three|four|five)\s+and a half hours?").captures(s) { return Some(60 * en(&c[1]) + 30); }
    let mut t = 0;
    if let Some(c) = re!(r"\b(an?|one|two|three|four|five)\s+hours?").captures(s) {
        t += 60 * en(&c[1]) + if s.contains("and a half") { 30 } else { 0 };
    }
    if s.contains("half an hour") { t += 30; }
    if let Some(c) = re!(r"(\d+)\s*minutes").captures(s) { t += c[1].parse::<i64>().ok()?; }
    if t > 0 { return Some(t); }
    if let Some(c) = re!(r"\b(\d+)\s*h(?:rs?)?\b(?:\s*(\d+)\s*min)?|\b(\d+)\s*min(?:utes?|s)?\b|(\d+)\s*[時时][間间](半)?(?:\s*(\d+)\s*分)?|(\d+)\s*分(?:間)?(?:後|遅|早|$|[^钟鐘])").captures(s) {
        let g = |i: usize| c.get(i).and_then(|x| x.as_str().parse::<i64>().ok()).unwrap_or(0);
        let t = 60 * (g(1) + g(4)) + g(2) + g(3) + g(6) + g(7) + if c.get(5).is_some() { 30 } else { 0 };
        if t > 0 && !re!("小[时時]").is_match(s) { return Some(t); }
    }
    if let Some(c) = re!(r"([一二两三四五六\d]+)\s*个?(半)?\s*小[时時]").captures(s) {
        t += 60 * cn_num(&c[1])? + if c.get(2).is_some() { 30 } else { 0 };
    } else if re!(r"半小[时時]").is_match(s) { t += 30; }
    let m = re!(r"小[时時]\s*(?:零\s*)?([一二三四五六七八九十\d]+)\s*分").captures(s).map(|c| c[1].to_string()).or_else(|| {
        re!(r"(\d+|[一二三四五六七八九十]+)\s*分").captures_iter(s)
            .find(|c| !s[..c.get(0).unwrap().start()].ends_with(['小', '时', '時'])).map(|c| c[1].to_string())
    });
    if let Some(m) = m { t += cn_num(&m)?; }
    (t > 0).then_some(t)
}
