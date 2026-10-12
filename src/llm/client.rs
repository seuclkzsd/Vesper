//! OpenAI-compatible chat client of the LLM stages, built to protect a shared quota.
//!
//! Configuration (environment, or a local `.env` in the working directory that is never packed):
//!   OPENAI_API_KEY    required (KIMI_API_KEY is accepted as an alternate name)
//!   OPENAI_BASE_URL   default https://api.kimi.com/coding/v1 (Kimi Coding Plan; outside mainland China
//!                     use https://api.kimi.ai/coding/v1). On the platform this is injected automatically.
//!   OPENAI_MODEL      default k3
//!   LLM_MAX_CALLS     hard cap on HTTP requests per run, retries included (default 16; 0 = no calls)
//!   LLM_TIMEOUT       longest wait for one stage's answer, seconds (default 40)
//!   LLM_MAX_TOKENS    default 8000 (reasoning models spend part of it before the answer)
//!
//! Quota protection (the final runs share one key across 12 concurrent card runs):
//! - every HTTP request, first try or retry, counts against LLM_MAX_CALLS and against the asking stage's quota;
//! - a question is retried at most once, and only after a transient failure (5xx, network error, empty or
//!   unreadable reply) or when the endpoint rejects the `tools` field (then asked again without tools);
//! - HTTP 429 (rate limit, as the organizers ask): wait (Retry-After, else 2, 4, 8 s) and retry, up to 3 times
//!   within the question's deadline; 3 questions in a row that end rate-limited open the breaker;
//! - a circuit breaker stops every LLM call for the rest of the run after HTTP 401/403 or any reply that
//!   mentions a usage limit or quota, and after 3 questions in a row that failed; a stage whose own last 3
//!   questions failed (e.g. long duty-log questions timing out) stops alone.
//!
//! Questions are synchronous with a deadline: the agent asks at a fixed decision point and waits at most the
//! given seconds (no CPU is spent waiting); a late or failed answer leaves the rule decision in place. The
//! duty-log stage asks in the background instead (`ask_background`): no decision waits for its long answers,
//! the rule decoder's facts are used meanwhile and the model's additions join when they arrive.
//! Answers: the model may reply through the offered function tool (`tool_calls`) or as text; reasoning blocks
//! (`<think>...</think>`) are stripped and the last complete outermost JSON object is used.

use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

pub const DEFAULT_BASE_URL: &str = "https://api.kimi.com/coding/v1";
pub const DEFAULT_MODEL: &str = "k3";
const DEFAULT_MAX_CALLS: usize = 16;
const DEFAULT_TIMEOUT: f64 = 40.0;
const MAX_ATTEMPTS: usize = 2; // one try + at most one retry
const BREAKER_FAILURES: u32 = 3; // questions in a row that failed (one stage, or rate-limited)
const GLOBAL_FAILURES: u32 = 5; // questions in a row that failed, any stage
const MIN_FIRST_SECONDS: f64 = 3.0; // do not start a question with less time than this
const MIN_RETRY_SECONDS: f64 = 2.0;

fn env_trim(name: &str) -> String {
    std::env::var(name).map(|v| v.trim().to_string()).unwrap_or_default()
}

/// OBSERVER_MODEL_DISABLED=1: the platform runs this evaluation without a model (本次不提供模型 /
/// survey26 eval start --no-model). No key is needed and no call is made: every rule default stands.
pub fn model_disabled() -> bool {
    std::env::var("OBSERVER_MODEL_DISABLED").map(|v| v == "1").unwrap_or(false)
}

pub fn api_key() -> String {
    let key = env_trim("OPENAI_API_KEY");
    if key.is_empty() {
        env_trim("KIMI_API_KEY")
    } else {
        key
    }
}

/// Fill missing environment variables from a local .env (for local runs only).
pub fn load_dotenv(path: &std::path::Path) {
    let Ok(text) = std::fs::read_to_string(path) else { return };
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || !line.contains('=') {
            continue;
        }
        let (name, value) = line.split_once('=').unwrap();
        let name = name.trim();
        let value = value.trim().trim_matches('"').trim_matches('\'');
        if !name.is_empty() && !value.is_empty() && env_trim(name).is_empty() {
            std::env::set_var(name, value);
        }
    }
}

// --- reply parsing ---------------------------------------------------------------------------------

/// Remove `<think>...</think>` reasoning blocks (MiniMax / DeepSeek style). An unclosed `<think>` drops the
/// rest of the text; a stray `</think>` drops everything before it.
pub fn strip_think(text: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    loop {
        match rest.find("<think>") {
            None => {
                out.push_str(rest);
                break;
            }
            Some(a) => {
                out.push_str(&rest[..a]);
                match rest[a..].find("</think>") {
                    Some(b) => rest = &rest[a + b + "</think>".len()..],
                    None => break,
                }
            }
        }
    }
    match out.rfind("</think>") {
        Some(b) => out[b + "</think>".len()..].to_string(),
        None => out,
    }
}

/// Spans of the outermost balanced `{...}` objects in `text` (braces inside JSON strings ignored).
fn outer_objects(text: &str) -> Vec<(usize, usize)> {
    let bytes = text.as_bytes();
    let mut spans = Vec::new();
    let (mut depth, mut start, mut in_str, mut escaped) = (0usize, 0usize, false, false);
    for (k, &c) in bytes.iter().enumerate() {
        if depth > 0 && in_str {
            if escaped {
                escaped = false;
            } else if c == b'\\' {
                escaped = true;
            } else if c == b'"' {
                in_str = false;
            }
            continue;
        }
        match c {
            b'"' if depth > 0 => in_str = true,
            b'{' => {
                if depth == 0 {
                    start = k;
                }
                depth += 1;
            }
            b'}' if depth > 0 => {
                depth -= 1;
                if depth == 0 {
                    spans.push((start, k));
                }
            }
            _ => {}
        }
    }
    spans
}

/// The answer object of a reply: reasoning blocks stripped, then the LAST complete outermost JSON object that
/// parses (a reply may echo the input or an example first); falls back to the first '{' .. last '}'.
/// Objects inside the reasoning are never used (they are drafts).
pub fn parse_reply(text: &str) -> Option<Map<String, Value>> {
    let clean = strip_think(text);
    for &(a, b) in outer_objects(&clean).iter().rev() {
        if let Ok(Value::Object(map)) = serde_json::from_str::<Value>(&clean[a..=b]) {
            return Some(map);
        }
    }
    if let (Some(a), Some(b)) = (clean.find('{'), clean.rfind('}')) {
        if a < b {
            if let Ok(Value::Object(map)) = serde_json::from_str::<Value>(&clean[a..=b]) {
                return Some(map);
            }
        }
    }
    None
}

/// The answer in a chat-completions message: the arguments of a call of `tool` if the model called it,
/// otherwise the JSON object in the text content.
pub fn answer_of(message: &Value, tool: Option<&str>) -> Option<Map<String, Value>> {
    if let (Some(name), Some(calls)) = (tool, message.get("tool_calls").and_then(Value::as_array)) {
        for call in calls {
            let function = &call["function"];
            if function.get("name").and_then(Value::as_str) != Some(name) {
                continue;
            }
            match function.get("arguments") {
                Some(Value::Object(map)) => return Some(map.clone()),
                Some(Value::String(text)) => {
                    if let Some(map) = parse_reply(text) {
                        return Some(map);
                    }
                }
                _ => {}
            }
        }
    }
    parse_reply(message.get("content").and_then(Value::as_str).unwrap_or(""))
}

// --- one HTTP request --------------------------------------------------------------------------------

/// A function the model is offered for its answer (OpenAI `tools`). The agent executes the call: it validates
/// and clamps the arguments and applies them.
pub struct Tool {
    pub name: &'static str,
    pub description: &'static str,
    pub parameters: Value,
}

enum Failure {
    /// quota / rate limit / key rejected: no request may be sent for the rest of the run
    Fatal(String),
    /// the endpoint refused the `tools` field: ask again without it
    ToolsRejected(String),
    /// worth one retry (5xx, network, empty or unreadable reply)
    Transient(String),
    /// HTTP 429 rate limit: wait (seconds from Retry-After, if any) and ask again
    RateLimited(String, Option<f64>),
    /// a retry would not help (other 4xx)
    Final(String),
}

/// Seconds in a Retry-After header; a date or a non-finite number ("NaN", "inf") counts as absent.
fn retry_after(v: &str) -> Option<f64> {
    v.trim().parse::<f64>().ok().filter(|s| s.is_finite())
}

fn quota_words(text: &str) -> bool {
    let t = text.to_lowercase();
    ["usage limit", "quota", "insufficient_quota"].iter().any(|w| t.contains(w))
}

fn rate_words(text: &str) -> bool {
    let t = text.to_lowercase();
    ["rate limit", "rate_limit", "too many requests"].iter().any(|w| t.contains(w))
}

#[derive(Clone)]
struct Endpoint {
    base_url: String,
    key: String,
    model: String,
    max_tokens: u64,
}

impl Endpoint {
    fn request(&self, system: &str, user: &Value, tool: Option<&Tool>, timeout: f64) -> Result<Map<String, Value>, Failure> {
        let mut body = json!({
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": serde_json::to_string(user).unwrap_or_default()}],
            "max_tokens": self.max_tokens,
        });
        if let Some(t) = tool {
            body["tools"] = json!([{"type": "function",
                                    "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}]);
            body["tool_choice"] = json!("auto");
        }
        // the question's time bounds the connect too (ureq's own connect limit, 30 s, would take precedence)
        let limit = Duration::from_secs_f64(timeout.max(0.5));
        let agent = ureq::AgentBuilder::new().timeout(limit).timeout_connect(limit).build();
        let response = agent
            .post(&format!("{}/chat/completions", self.base_url))
            .set("Content-Type", "application/json")
            .set("Authorization", &format!("Bearer {}", self.key))
            .send_json(body);
        let data: Value = match response {
            Ok(r) => r.into_json().map_err(|e| Failure::Transient(format!("unreadable body: {}", e.kind())))?,
            Err(ureq::Error::Status(code, r)) => {
                let after = r.header("retry-after").and_then(retry_after);
                let text: String = r.into_string().unwrap_or_default().chars().filter(|c| !c.is_control()).take(160).collect();
                let what = format!("HTTP {code}: {text}");
                return Err(if code == 401 || code == 403 || quota_words(&text) {
                    Failure::Fatal(what)
                } else if code == 429 || rate_words(&text) {
                    Failure::RateLimited(what, after)
                } else if code >= 500 {
                    Failure::Transient(what)
                } else if tool.is_some() && (code == 400 || code == 422) {
                    Failure::ToolsRejected(what)
                } else {
                    Failure::Final(what)
                });
            }
            Err(ureq::Error::Transport(t)) => return Err(Failure::Transient(format!("transport: {}", t.kind()))),
        };
        if let Some(error) = data.get("error").filter(|e| !e.is_null()) {
            let text = error.to_string();
            return Err(if quota_words(&text) {
                Failure::Fatal(text)
            } else if rate_words(&text) {
                Failure::RateLimited(text, None)
            } else {
                Failure::Transient(text)
            });
        }
        let message = &data["choices"][0]["message"];
        match answer_of(message, tool.map(|t| t.name)) {
            Some(map) => Ok(map),
            None if message.get("content").and_then(Value::as_str).map_or(true, |c| c.trim().is_empty()) => {
                Err(Failure::Transient("empty reply (reasoning used up max_tokens?)".into()))
            }
            None => Err(Failure::Transient("no JSON object in the reply".into())),
        }
    }
}

// --- budgeted client ---------------------------------------------------------------------------------

#[derive(Default, Clone, Copy)]
struct StageUsage {
    sent: usize,    // HTTP requests
    ok: usize,      // questions answered
    failed: usize,  // questions that got no usable answer
    skipped: usize, // questions not sent (budget, quota, breaker, no time)
    in_a_row: u32,  // failed questions in a row
}

/// A question running on its own thread (`ask_background`).
pub struct Background {
    stage: &'static str,
    reserved: usize,
    rx: std::sync::mpsc::Receiver<Finished>,
}

#[derive(Default)]
struct Finished {
    answer: Option<Map<String, Value>>,
    error: String,
    requests: usize,
    limited: bool,
    fatal: bool,
    tools_rejected: bool,
    took: f64,
}

pub struct LlmClient {
    endpoint: Option<Endpoint>,
    pub model: String,
    budget: usize,
    pub timeout: f64,
    sent: usize,
    consecutive_failures: u32,
    rate_limited: u32, // questions in a row that ended rate-limited
    breaker: Option<String>,
    tools_ok: bool,
    stages: BTreeMap<&'static str, StageUsage>,
    skip_logged: BTreeSet<String>,
    pub waited: f64,
}

impl LlmClient {
    pub fn new() -> LlmClient {
        let base = env_trim("OPENAI_BASE_URL");
        let base_url = if base.is_empty() { DEFAULT_BASE_URL.to_string() } else { base.trim_end_matches('/').to_string() };
        let model = env_trim("OPENAI_MODEL");
        let model = if model.is_empty() { DEFAULT_MODEL.to_string() } else { model };
        let max_tokens = env_trim("LLM_MAX_TOKENS").parse::<u64>().ok().filter(|&t| t > 0).unwrap_or(8000);
        let budget = env_trim("LLM_MAX_CALLS").parse::<usize>().unwrap_or(DEFAULT_MAX_CALLS);
        let timeout = env_trim("LLM_TIMEOUT").parse::<f64>().ok().filter(|t| *t > 0.0).unwrap_or(DEFAULT_TIMEOUT);
        let key = api_key();
        let endpoint = if model_disabled() || key.is_empty() || budget == 0 {
            None
        } else {
            Some(Endpoint { base_url, key, model: model.clone(), max_tokens })
        };
        LlmClient {
            model: if endpoint.is_some() { model } else { "none".to_string() },
            endpoint,
            budget,
            timeout,
            sent: 0,
            consecutive_failures: 0,
            rate_limited: 0,
            breaker: None,
            tools_ok: true,
            stages: BTreeMap::new(),
            skip_logged: BTreeSet::new(),
            waited: 0.0,
        }
    }

    /// True while questions may still be sent (a model is configured, budget left, breaker closed).
    pub fn active(&self) -> bool {
        self.endpoint.is_some() && self.breaker.is_none() && self.sent < self.budget
    }

    pub fn budget(&self) -> usize {
        self.budget
    }

    /// More calls for a stage that only some cards need (the duty log); the other quotas are unchanged.
    pub fn extend_budget(&mut self, calls: usize) {
        self.budget += calls;
    }

    pub fn calls_left(&self) -> usize {
        if self.active() {
            self.budget - self.sent
        } else {
            0
        }
    }

    /// Requests a stage has sent so far.
    pub fn stage_sent(&self, stage: &str) -> usize {
        self.stages.get(stage).map_or(0, |u| u.sent)
    }

    fn skip(&mut self, stage: &'static str, why: &str) {
        if self.endpoint.is_none() {
            return; // rules-only run: nothing to account
        }
        self.stages.entry(stage).or_default().skipped += 1;
        if self.skip_logged.insert(format!("{stage}/{why}")) {
            crate::log(&format!("llm {stage}: not asked ({why}); rules decide"));
        }
    }

    /// Ask one question and wait at most `wait` seconds. `quota` caps the requests of this stage.
    /// Returns the answer object, or None (no model, budget or quota used up, breaker open, no time, failure):
    /// the caller then keeps its rule decision.
    pub fn ask(&mut self, stage: &'static str, quota: usize, system: &str, user: &Value, tool: Option<&Tool>, wait: f64) -> Option<Map<String, Value>> {
        let Some(endpoint) = self.endpoint.clone() else { return None };
        if self.breaker.is_some() {
            self.skip(stage, "circuit breaker open");
            return None;
        }
        if self.sent >= self.budget || self.stage_sent(stage) >= quota {
            self.skip(stage, "call budget used up");
            return None;
        }
        if self.stages.get(stage).map_or(false, |u| u.in_a_row >= BREAKER_FAILURES) {
            self.skip(stage, "stage breaker open");
            return None;
        }
        if wait < MIN_FIRST_SECONDS {
            self.skip(stage, "no wall time to spare");
            return None;
        }
        let started = Instant::now();
        let mut use_tools = tool.is_some() && self.tools_ok;
        let mut error = String::new();
        let mut requests = 0;
        let (mut waits, mut limited) = (0, false);
        let mut answer = None;
        while requests < MAX_ATTEMPTS + waits {
            let left = wait - started.elapsed().as_secs_f64();
            let need = if requests == 0 { MIN_FIRST_SECONDS } else { MIN_RETRY_SECONDS };
            if left < need || self.sent >= self.budget || self.stage_sent(stage) >= quota {
                break;
            }
            self.sent += 1;
            self.stages.entry(stage).or_default().sent += 1;
            requests += 1;
            match endpoint.request(system, user, if use_tools { tool } else { None }, left) {
                Ok(map) => {
                    answer = Some(map);
                    break;
                }
                Err(Failure::Fatal(e)) => {
                    crate::log(&format!("llm: circuit breaker open after {e}; every stage runs on rules for the rest of the run"));
                    self.breaker = Some(e.chars().take(60).collect());
                    error = "quota / rate limit".into();
                    break;
                }
                Err(Failure::ToolsRejected(e)) => {
                    self.tools_ok = false; // this endpoint takes no `tools`: the text answer is used from now on
                    use_tools = false;
                    error = e;
                }
                Err(Failure::RateLimited(e, after)) => {
                    // back off and ask again (does not use up the single transient retry)
                    let pause = after.unwrap_or(2.0 * 2f64.powi(waits as i32)).clamp(1.0, 20.0);
                    limited = true;
                    error = e;
                    if waits >= 3 || wait - started.elapsed().as_secs_f64() < pause + MIN_RETRY_SECONDS {
                        break;
                    }
                    waits += 1;
                    crate::log(&format!("llm {stage}: rate limited, retrying in {pause:.0}s"));
                    std::thread::sleep(Duration::from_secs_f64(pause));
                }
                Err(Failure::Transient(e)) => {
                    limited = false;
                    error = e;
                    std::thread::sleep(Duration::from_secs_f64((0.25 * (wait - started.elapsed().as_secs_f64())).clamp(0.0, 0.5)));
                }
                Err(Failure::Final(e)) => {
                    error = e;
                    break;
                }
            }
        }
        let took = started.elapsed().as_secs_f64();
        self.waited += took;
        self.settle(stage, answer, error, limited, took, requests)
    }

    /// The bookkeeping after a question (answered or not): per-stage counts, the stage and global breakers.
    fn settle(&mut self, stage: &'static str, answer: Option<Map<String, Value>>, mut error: String, limited: bool, took: f64, requests: usize) -> Option<Map<String, Value>> {
        let usage = self.stages.entry(stage).or_default();
        match answer {
            Some(map) => {
                usage.ok += 1;
                usage.in_a_row = 0;
                self.consecutive_failures = 0;
                self.rate_limited = 0;
                crate::log(&format!("llm {stage}: answer in {took:.1}s ({requests} request(s), {} of {} used)", self.sent, self.budget));
                Some(map)
            }
            None => {
                usage.failed += 1;
                usage.in_a_row += 1;
                if usage.in_a_row == BREAKER_FAILURES {
                    crate::log(&format!("llm {stage}: stage breaker open after {BREAKER_FAILURES} failed questions in a row"));
                }
                self.consecutive_failures += 1;
                self.rate_limited = if limited { self.rate_limited + 1 } else { 0 };
                if error.is_empty() {
                    error = "no time left for the request".into();
                }
                crate::log(&format!("llm {stage}: failed after {took:.1}s ({error}); rules decide"));
                if self.breaker.is_none() && (self.consecutive_failures >= GLOBAL_FAILURES || self.rate_limited >= BREAKER_FAILURES) {
                    crate::log(&format!("llm: circuit breaker open after {} failed questions in a row; rules decide from now on", self.consecutive_failures));
                    self.breaker = Some(format!("{} failures in a row", self.consecutive_failures));
                }
                None
            }
        }
    }

    /// Ask one question in the background (the duty log: long answers that no decision waits for). The same
    /// gates as `ask` (budget, stage quota, breakers); the requests it may need (one try, one retry, up to three
    /// 429 back-offs) are reserved from the budget now and the unused ones given back when it ends, so the
    /// budget is never exceeded. The decisions go on with the rule facts; `background_done` collects the answer.
    pub fn ask_background(&mut self, stage: &'static str, quota: usize, system: &'static str, user: Value, tool: Option<Tool>, timeout: f64) -> Option<Background> {
        let Some(endpoint) = self.endpoint.clone() else { return None };
        if self.breaker.is_some() {
            self.skip(stage, "circuit breaker open");
            return None;
        }
        let room = (self.budget.saturating_sub(self.sent)).min(quota.saturating_sub(self.stage_sent(stage)));
        if room == 0 {
            self.skip(stage, "call budget used up");
            return None;
        }
        if self.stages.get(stage).map_or(false, |u| u.in_a_row >= BREAKER_FAILURES) {
            self.skip(stage, "stage breaker open");
            return None;
        }
        let reserved = room.min(MAX_ATTEMPTS + 3);
        self.sent += reserved;
        self.stages.entry(stage).or_default().sent += reserved;
        let tools_ok = self.tools_ok;
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let started = Instant::now();
            let mut use_tools = tool.is_some() && tools_ok;
            let (mut requests, mut waits, mut limited, mut fatal) = (0, 0, false, false);
            let (mut error, mut answer) = (String::new(), None);
            while requests < reserved.min(MAX_ATTEMPTS + waits) {
                let left = timeout - started.elapsed().as_secs_f64();
                if left < if requests == 0 { MIN_FIRST_SECONDS } else { MIN_RETRY_SECONDS } {
                    break;
                }
                requests += 1;
                match endpoint.request(system, &user, if use_tools { tool.as_ref() } else { None }, left) {
                    Ok(map) => {
                        answer = Some(map);
                        break;
                    }
                    Err(Failure::Fatal(e)) => {
                        (fatal, error) = (true, e);
                        break;
                    }
                    Err(Failure::ToolsRejected(e)) => {
                        use_tools = false;
                        error = e;
                    }
                    Err(Failure::RateLimited(e, after)) => {
                        let pause = after.unwrap_or(2.0 * 2f64.powi(waits as i32)).clamp(1.0, 20.0);
                        (limited, error) = (true, e);
                        if waits >= 3 || timeout - started.elapsed().as_secs_f64() < pause + MIN_RETRY_SECONDS {
                            break;
                        }
                        waits += 1;
                        std::thread::sleep(Duration::from_secs_f64(pause));
                    }
                    Err(Failure::Transient(e)) => {
                        (limited, error) = (false, e);
                        std::thread::sleep(Duration::from_secs_f64(0.5));
                    }
                    Err(Failure::Final(e)) => {
                        error = e;
                        break;
                    }
                }
            }
            let tools_rejected = tool.is_some() && !use_tools && tools_ok;
            let _ = tx.send(Finished { answer, error, requests, limited, fatal, tools_rejected, took: started.elapsed().as_secs_f64() });
        });
        crate::log(&format!("llm {stage}: asked in the background ({reserved} request(s) reserved, {} of {} used)", self.sent, self.budget));
        Some(Background { stage, reserved, rx })
    }

    /// The background question's outcome once it has ended: None while it still runs, Some(answer or None).
    pub fn background_done(&mut self, q: &Background) -> Option<Option<Map<String, Value>>> {
        let f = match q.rx.try_recv() {
            Ok(f) => f,
            Err(std::sync::mpsc::TryRecvError::Empty) => return None,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => Finished { error: "worker gone".into(), ..Finished::default() },
        };
        let unused = q.reserved.saturating_sub(f.requests);
        self.sent -= unused;
        if let Some(u) = self.stages.get_mut(q.stage) {
            u.sent -= unused.min(u.sent);
        }
        if f.tools_rejected {
            self.tools_ok = false;
        }
        if f.fatal && self.breaker.is_none() {
            crate::log(&format!("llm: circuit breaker open after {}; every stage runs on rules for the rest of the run", f.error));
            self.breaker = Some(f.error.chars().take(60).collect());
        }
        Some(self.settle(q.stage, f.answer, f.error, f.limited, f.took, f.requests))
    }

    /// One stderr line for the end of the run.
    pub fn usage_line(&self) -> String {
        let (mut ok, mut failed, mut skipped) = (0, 0, 0);
        let per_stage: Vec<String> = self
            .stages
            .iter()
            .map(|(name, u)| {
                ok += u.ok;
                failed += u.failed;
                skipped += u.skipped;
                format!("{name}:{}/{}/{}/{}", u.sent, u.ok, u.failed, u.skipped)
            })
            .collect();
        format!(
            "llm_usage: model={} calls={} budget={} ok={} failed={} skipped={} breaker={} waited={:.1}s per_stage(sent/ok/failed/skipped)={}",
            self.model,
            self.sent,
            self.budget,
            ok,
            failed,
            skipped,
            self.breaker.as_deref().unwrap_or("closed"),
            self.waited,
            if per_stage.is_empty() { "-".to_string() } else { per_stage.join(",") }
        )
    }
}

#[cfg(test)]
#[path = "../../tests/unit/llm_client.rs"]
mod tests;
