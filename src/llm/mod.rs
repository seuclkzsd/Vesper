//! The LLM-driven stages of the agent. Each one is on by default (only the platform's
//! `OBSERVER_MODEL_DISABLED=1`, a missing API key or `LLM_MAX_CALLS=0` turns it off), asks at a fixed decision
//! point, validates and clamps the answer to a trust region the rules define, and falls back to the rule
//! decision on any failure (no answer in time, unusable answer, budget used up, circuit breaker open).
//!
//! | stage | file | asked when | the answer decides |
//! |---|---|---|---|
//! | week plan & review | `week_plan.rs` | a new 7-day forecast arrives (spread over the season to fit the quota) | per-night plan for the forecast week (bad night for faint required targets, sectors to avoid) and the week's time price (lambda scale) |
//! | fault judge | `fault_judge.rs` | the fault rule is about to report an unannounced instrument fault | report now, or after one confirmation exposure inside the slot |
//! | request triage | `request_triage.rs` | new observation requests appear | each request's priority (scales its planning value) |
//! | duty log | `duty_log.rs` (rule decoder: `crate::dutylog`) | new duty-log messages in request reasons (cards with a staff chat only): the lines the rule decoder is not sure of, in chunks | duty-log facts the rules missed (in the rule decoder's fact schema): fault times to report (free false reports only), times ruled out, stops (wait), closures, terrain tops |
//!
//! `client.rs` holds the HTTP client with the per-run call budget, the single retry, the rate-limit backoff and the
//! circuit breakers.

pub mod client;
pub mod duty_log;
pub mod fault_judge;
pub mod request_triage;
pub mod week_plan;

use serde_json::{Map, Value};

/// How the run's call budget is shared: 3/8 each for the week plans and the request triage (cards carry up to 6
/// requests), the rest (a quarter) for the fault judge. Every stage stops at its quota, so no stage can starve
/// another.
pub struct Quotas {
    pub week_plan: usize,
    pub fault_judge: usize,
    pub request_triage: usize,
}

impl Quotas {
    pub fn split(budget: usize) -> Quotas {
        let week_plan = 3 * budget / 8;
        let request_triage = 3 * budget / 8;
        Quotas { week_plan, fault_judge: budget - week_plan - request_triage, request_triage }
    }
}

/// The answer's free-text reason, shortened for the log.
pub fn reason_of(answer: &Map<String, Value>) -> String {
    let text = match answer.get("reason") {
        Some(Value::String(s)) => s.clone(),
        Some(v) => v.to_string(),
        None => String::new(),
    };
    text.chars().filter(|c| !c.is_control()).take(100).collect()
}

/// A finite number from a JSON number or numeric string.
pub fn number(v: Option<&Value>) -> Option<f64> {
    let x = match v? {
        Value::Number(n) => n.as_f64()?,
        Value::String(s) => s.trim().parse::<f64>().ok()?,
        _ => return None,
    };
    x.is_finite().then_some(x)
}

/// A boolean from a JSON bool or "true"/"false".
pub fn boolean(v: Option<&Value>) -> Option<bool> {
    match v? {
        Value::Bool(b) => Some(*b),
        Value::String(s) => match s.trim().to_lowercase().as_str() {
            "true" | "yes" => Some(true),
            "false" | "no" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

#[cfg(test)]
#[path = "../../tests/unit/llm_mod.rs"]
mod tests;
