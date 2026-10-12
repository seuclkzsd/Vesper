//! LLM stage B: fault judge (action decision-making, data parsing).
//!
//! An instrument fault is never announced; the rules detect it (fault.rs) with a change-point detector on one
//! exact site-throughput reading per exposure and the engine's fault-window prior, and report when a report pays
//! in expectation. Right before the rules send such a report (except the certain N/2+200 backstop), the model
//! reads the evidence -- the detector's state (fault-count posterior, last alarm probability), the agent's
//! hour-by-hour E and sky-scale table, earthquake timing, the current bulletin, the free false reports left, the
//! rules' reason -- and calls `submit_fault_verdict` with `report_now` or `wait`.
//!
//! Trust region (the shield, in main.rs `maybe_report` / `judge_agrees`): the model can only hold a rule
//! report, never start a report the rules did not decide on, so it can never cause a false report. A `wait`
//! holds the report for one confirmation exposure kept inside the current slot; at the next decision the report
//! goes out without asking again (it is never dropped). So a wrong `wait` costs at most one slot of faulty time.
//! No answer: the rules report.

use serde_json::{json, Value};

use super::client::{LlmClient, Tool};
use super::reason_of;

pub const STAGE: &str = "fault_judge";

const SYSTEM: &str = concat!(
    "You check the evidence for an unannounced instrument fault on a robotic survey telescope before a fault ",
    "report is sent. A fault lowers the instrument efficiency, and so the quality of every exposure, until it is ",
    "reported; a correct report repairs it at once and earns {REWARD} points. A false report is free while ",
    "free_false_reports_left > 0, otherwise it costs {PENALTY} points. An earthquake (announced in the bulletin) ",
    "also lowers efficiency; that loss fades night by night and a report does not repair it. Weather lowers both ",
    "the measured quality and the program band, which the instrument does not affect.\n",
    "evidence.detector: the rules' change-point detector on an exact per-exposure reading of the site throughput ",
    "(fault_count_posterior, last_alarm_probability). Table columns per hour: E = measured quality / quality the ",
    "program bands allow (about 1 when healthy; it often stays near 1 under a fault, so it is weak evidence), ",
    "scale = measured sky quality relative to the clear-sky model; ref = the usual scale since the last repair. ",
    "rule_decision explains why the rules want to report now.\n",
    "Signs of a fault: a throughput drop that stays across slots and nights, not explained by announced weather or ",
    "a recent earthquake. Choose report_now to send the report now, or wait to take one more exposure (inside ",
    "the current slot) first: the report is then sent.\n",
    "Answer by calling submit_fault_verdict once (or reply with one JSON object): ",
    "{\"verdict\": \"report_now\" | \"wait\", \"confidence\": <0..1>, \"reason\": \"<20 words\"}"
);

fn tool() -> Tool {
    Tool {
        name: "submit_fault_verdict",
        description: "Send the instrument-fault report now, or after one more exposure inside the current slot.",
        parameters: json!({
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": ["report_now", "wait"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "reason": {"type": "string"}
            },
            "required": ["verdict", "reason"]
        }),
    }
}

/// The card's report reward and false-report cost, quoted in the prompt.
pub struct FaultJudge {
    system: String,
}

impl FaultJudge {
    pub fn new(correct_reward: f64, false_penalty: f64) -> FaultJudge {
        let fmt = |v: f64| if v.fract() == 0.0 { format!("{}", v.abs() as i64) } else { format!("{}", v.abs()) };
        FaultJudge { system: SYSTEM.replace("{REWARD}", &fmt(correct_reward)).replace("{PENALTY}", &fmt(false_penalty)) }
    }

    /// Some(true) = report now, Some(false) = wait; None = no usable answer (the rule's report stands).
    /// `evidence`: the fault table; `rule_decision`: why the rules report (free or paid probe).
    pub fn judge(&self, client: &mut LlmClient, quota: usize, evidence: Value, rule_decision: Value, wait: f64) -> Option<bool> {
        let user = json!({
            "evidence": evidence,
            "rule_decision": rule_decision,
            "rule_proposal": {"verdict": "report_now", "confidence": 0.5, "reason": "rules: the fault detector says a report pays"},
        });
        let tool = tool();
        let answer = client.ask(STAGE, quota, &self.system, &user, Some(&tool), wait)?;
        let verdict = answer.get("verdict").and_then(Value::as_str).map(|v| v.trim().to_lowercase());
        let report = match verdict.as_deref() {
            Some("report_now") | Some("report") => true,
            Some("wait") | Some("hold") => false,
            _ => {
                crate::log("llm fault_judge: answer not usable; the rule's report stands");
                return None;
            }
        };
        crate::log(&format!("llm fault judge: {} ({})", if report { "report now" } else { "wait" }, reason_of(&answer)));
        Some(report)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/llm_fault_judge.rs"]
mod tests;
