//! LLM stage C: request triage (natural-language understanding, action decision-making).
//!
//! Observation requests arrive during the season with a free-text `reason`, a target list, a deadline and an
//! all-or-nothing reward. When new requests appear, the model reads each request's text together with the
//! rule-computed feasibility (nights and hours to the deadline, exposure each open target needs at typical sky,
//! how many targets are inside their observing window, whether the rules think it can be completed) and calls
//! `submit_request_triage` with a `priority` per request.
//!
//! Trust region: the priority multiplies the request's planning value in the planner (`request_priority` in
//! `Planner::on_requests`). A request the rules judge feasible (enough open targets inside their observing
//! window that can reach the threshold within the longest exposure) is worth 100 points the rules expect to
//! collect, so its priority is clamped to [1, PRIORITY_MAX]: the model can make it more urgent, never give it
//! up. A request the rules judge infeasible can also be made cheaper, down to PRIORITY_MIN (`pursue: false`
//! means PRIORITY_MIN); it is never dropped from the plan. Requests the model does not answer keep the rule
//! priority 1. PRIORITY_MAX is 1.5: the rules' value is already the request's marginal completion probability
//! times its reward, so a priority acts mostly where completion is uncertain, and a higher one would pull
//! fields away from the survey for requests the rules already expect to finish.

use serde_json::{json, Value};
use std::collections::BTreeMap;

use super::client::{LlmClient, Tool};
use super::{boolean, number, reason_of};

pub const STAGE: &str = "request_triage";
pub const PRIORITY_MIN: f64 = 0.5;
pub const PRIORITY_MAX: f64 = 1.5;

const SYSTEM: &str = concat!(
    "You triage observation requests for a robotic fibre-spectroscopy survey telescope. A request asks for a set ",
    "of targets to be observed to a quality threshold before its deadline; it pays its completion_reward only if at ",
    "least minimum_completed targets reach the threshold in time. Each request comes with its text (reason) and ",
    "with facts the scheduler computed: hours and nights left, the exposure each open target needs at typical sky ",
    "(an exposure can be at most max_exposure_s), how many open targets are inside their observing window before ",
    "the deadline, and rule_feasible.\n",
    "Set a priority per request: 1 = as the rules plan it; up to 1.5 for urgent, valuable and feasible requests (for ",
    "example a short deadline with few nights left); down to 0.5, or pursue=false, for requests that are unlikely ",
    "to complete or would take much telescope time for their reward (only requests with rule_feasible=false can ",
    "go below 1). The scheduler keeps working on every request; the priority only scales how much it is worth.\n",
    "Answer by calling submit_request_triage once (or reply with one JSON object): ",
    "{\"requests\": [{\"request_id\": \"...\", \"pursue\": true, \"priority\": 1.0}], \"reason\": \"<20 words\"}"
);

fn tool() -> Tool {
    Tool {
        name: "submit_request_triage",
        description: "Set the scheduling priority of each new observation request.",
        parameters: json!({
            "type": "object",
            "properties": {
                "requests": {"type": "array", "items": {"type": "object", "properties": {
                    "request_id": {"type": "string"},
                    "pursue": {"type": "boolean"},
                    "priority": {"type": "number", "minimum": PRIORITY_MIN, "maximum": PRIORITY_MAX}},
                    "required": ["request_id", "pursue", "priority"]}},
                "reason": {"type": "string"}
            },
            "required": ["requests", "reason"]
        }),
    }
}

/// Ask for the priorities of the new requests (`facts`: one object per request, with `request_id`).
/// Returns the validated priorities by request id; requests missing from it keep priority 1.
pub fn triage(client: &mut LlmClient, quota: usize, facts: Vec<Value>, wait: f64) -> BTreeMap<String, f64> {
    let feasible: BTreeMap<String, bool> = facts
        .iter()
        .filter_map(|f| {
            let id = f.get("request_id").and_then(Value::as_str)?;
            Some((id.to_string(), f.get("rule_feasible").and_then(Value::as_bool).unwrap_or(true)))
        })
        .collect();
    let ids: Vec<&String> = feasible.keys().collect();
    let user = json!({
        "requests": facts,
        "rule_proposal": {"requests": ids.iter().map(|id| json!({"request_id": id, "pursue": true, "priority": 1.0})).collect::<Vec<_>>(),
                          "reason": "rules: every request at its planning value"},
    });
    let tool = tool();
    let Some(answer) = client.ask(STAGE, quota, SYSTEM, &user, Some(&tool), wait) else { return BTreeMap::new() };
    let priorities = validate(&answer, &feasible);
    crate::log(&format!("llm request triage: {:?} ({})", priorities, reason_of(&answer)));
    priorities
}

/// Clamp the answer: `feasible` maps each asked request to the rules' feasibility verdict.
fn validate(answer: &serde_json::Map<String, Value>, feasible: &BTreeMap<String, bool>) -> BTreeMap<String, f64> {
    let mut out = BTreeMap::new();
    for item in answer.get("requests").and_then(Value::as_array).into_iter().flatten() {
        let Some(id) = item.get("request_id").and_then(Value::as_str) else { continue };
        let Some(&rule_feasible) = feasible.get(id) else { continue };
        let floor = if rule_feasible { 1.0 } else { PRIORITY_MIN };
        let priority = if boolean(item.get("pursue")) == Some(false) {
            Some(floor)
        } else {
            number(item.get("priority")).map(|p| p.clamp(floor, PRIORITY_MAX))
        };
        if let Some(p) = priority {
            out.insert(id.to_string(), p);
        }
    }
    out
}

#[cfg(test)]
#[path = "../../tests/unit/llm_request_triage.rs"]
mod tests;
