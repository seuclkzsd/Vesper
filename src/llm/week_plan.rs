//! LLM stage A: week plan and review (task planning, plan adaptation, data parsing, tool calling).
//!
//! When a new 7-day forecast arrives (at the start of a night), the model reads the forecast notices, the
//! rule-computed review of the period since its last plan (observations, score gained, required targets
//! finished, instrument quality E, reports) and the season's progress, and calls the function `set_week_plan`:
//!
//! - for each night of the forecast week: `bad_night` (faint required targets should wait for a better night)
//!   and `avoid` (compass sectors whose low sky the planner should lean away from that night: a soft value
//!   factor AVOID_FACTOR below 60 deg altitude within 45 deg of the sector, `direction_factor` in planner/mod.rs);
//! - `lambda_scale`: the week's multiplier on the planner's time price (higher = shorter, higher-rate fields;
//!   lower = longer exposures that finish more targets).
//!
//! Trust region (validated and clamped by rules), sized so that a wrong answer stays cheap (the rule
//! plan is tuned; a forecast is exact about the night but not the hour): `bad_night` can only be true on a
//! night the rule marks bad, so the model confirms or clears the rule's bad nights; `avoid` may only name
//! sectors the forecast names for that night; `lambda_scale` moves at most LAMBDA_STEP per plan inside
//! [LAMBDA_MIN, LAMBDA_MAX] (a 2 % price change; the cost of a mis-set price grows with its square) and relaxes
//! halfway back to 1 at every forecast that is not planned with the model. The plan is applied at fixed points: lambda from the night the plan is
//! made, the per-night advice at each night start of the forecast week. Missing or invalid fields, and any
//! failure, leave the rule plan (bad night = ALL-sky rain/storm/overcast/haze forecast, no extra sectors,
//! lambda unchanged).
//!
//! Quota: forecasts come every 7 days (52 on a one-year card); the model plans every `stride`-th forecast so
//! that the week plans of a whole season fit the stage's share of the call budget.

use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};

use super::client::{LlmClient, Tool};
use super::{boolean, number, reason_of};

pub const STAGE: &str = "week_plan";
pub const DIRECTIONS: [&str; 8] = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];
/// Weather kinds a forecast can announce (rocket launches are handled by the bulletin rules).
pub const WEATHER_KINDS: [&str; 5] = ["rain", "storm", "overcast", "haze", "cold_snap"];
/// ALL-sky kinds that make a night bad for faint required targets (the rule).
pub const BAD_KINDS: [&str; 4] = ["rain", "storm", "overcast", "haze"];
pub const LAMBDA_MIN: f64 = 0.98;
pub const LAMBDA_MAX: f64 = 1.02;
pub const LAMBDA_STEP: f64 = 0.02;

const SYSTEM: &str = concat!(
    "You are the weekly planner of a robotic fibre-spectroscopy survey telescope. A deterministic scheduler picks ",
    "every exposure; you set its plan for the coming forecast week, once per week, and review how the last period ",
    "went. Each forecast notice is a weather event kind and a compass sector (N, NE, E, SE, S, SW, W, NW) or ALL ",
    "(the whole sky), with the nights (local evening dates) it touches.\n",
    "For each night of the week decide:\n",
    "- bad_night: the rules call a night bad when the forecast gives it rain, storm, overcast or haze over ALL of ",
    "the sky. Faint must-observe (required) targets need about an hour in a clear sky, so on a bad night they wait ",
    "for a better night unless their season is ending. A forecast night is often only partly affected (rain ",
    "closes 30-50 % of a listed night); set false to let them try anyway. Only nights in allowed.may_be_bad ",
    "can be bad.\n",
    "- avoid: sectors with forecast weather that night whose low sky the scheduler should lean away from (a ",
    "mild nudge: the live bulletin already avoids a sector while its event is on). Only sectors in ",
    "allowed.sectors count; an empty list is fine.\n",
    "Also set lambda_scale, the multiplier on the scheduler's price of telescope time for the week (1 = unchanged, ",
    "allowed range in knobs). Raise it slightly when the review shows time is scarce (much science left, many ",
    "closed or poor nights ahead); lower it slightly when the survey is ahead and longer exposures can finish ",
    "faint targets. rule_proposal is what the rules would do; keep it when unsure.\n",
    "Answer by calling set_week_plan once (or reply with one JSON object with the same fields): ",
    "{\"nights\": [{\"date\": \"YYYY-MM-DD\", \"bad_night\": false, \"avoid\": []}], \"lambda_scale\": 1.0, ",
    "\"review\": \"<20 words on the last period\", \"reason\": \"<20 words\"}"
);

fn tool() -> Tool {
    Tool {
        name: "set_week_plan",
        description: "Set the survey plan for the forecast week: per-night bad-night flag and sectors to avoid, and the week's time-price multiplier.",
        parameters: json!({
            "type": "object",
            "properties": {
                "nights": {"type": "array", "items": {"type": "object", "properties": {
                    "date": {"type": "string", "description": "night date YYYY-MM-DD from the input"},
                    "bad_night": {"type": "boolean"},
                    "avoid": {"type": "array", "items": {"type": "string", "enum": DIRECTIONS}}},
                    "required": ["date", "bad_night", "avoid"]}},
                "lambda_scale": {"type": "number", "minimum": LAMBDA_MIN, "maximum": LAMBDA_MAX},
                "review": {"type": "string"},
                "reason": {"type": "string"}
            },
            "required": ["nights", "lambda_scale", "reason"]
        }),
    }
}

/// The plan for one night.
#[derive(Clone, Debug, PartialEq)]
pub struct NightAdvice {
    pub bad_night: bool,
    pub avoid: BTreeSet<String>,
}

/// What the forecast allows the plan to say about one night, and the rule's own answer.
struct NightRules {
    date: String,
    rule_bad: bool,
    may_be_bad: bool,
    sectors: BTreeSet<String>,
}

fn names_night(notice: &Value, date: &str) -> bool {
    notice.get("nights").and_then(Value::as_array).map_or(false, |a| a.iter().any(|d| d.as_str() == Some(date)))
}

fn night_rules(notices: &[Value], dates: &[String]) -> Vec<NightRules> {
    dates
        .iter()
        .map(|date| {
            let tonight: Vec<(&str, &str)> = notices
                .iter()
                .filter(|n| names_night(n, date))
                .map(|n| (n.get("event_kind").and_then(Value::as_str).unwrap_or(""), n.get("direction").and_then(Value::as_str).unwrap_or("")))
                .collect();
            NightRules {
                date: date.clone(),
                rule_bad: tonight.iter().any(|(k, d)| *d == "ALL" && BAD_KINDS.contains(k)),
                // the model confirms or clears the rule's bad nights; it cannot add one (a cold snap is harmless)
                may_be_bad: tonight.iter().any(|(k, d)| *d == "ALL" && BAD_KINDS.contains(k)),
                sectors: tonight
                    .iter()
                    .filter(|(k, d)| WEATHER_KINDS.contains(k) && DIRECTIONS.contains(d))
                    .map(|(_, d)| d.to_string())
                    .collect(),
            }
        })
        .collect()
}

/// The rule's answer for tonight: bad when the forecast gives ALL-sky rain, storm, overcast or haze.
pub fn rule_bad_night(notices: &[Value], date: &str) -> bool {
    notices.iter().any(|n| {
        names_night(n, date)
            && n.get("direction").and_then(Value::as_str) == Some("ALL")
            && n.get("event_kind").and_then(Value::as_str).map_or(false, |k| BAD_KINDS.contains(&k))
    })
}

/// Week plans of the run: which forecasts are planned with the model, the current plan and lambda scale.
pub struct WeekPlanner {
    stride: usize,
    forecasts: usize,
    /// Model advice by night date for the current forecast week (empty: the rules plan every night).
    nights: BTreeMap<String, NightAdvice>,
    pub lambda_scale: f64,
}

impl WeekPlanner {
    /// `forecasts_expected`: forecasts in the season (one per 7 days); `quota`: requests this stage may send.
    pub fn new(forecasts_expected: usize, quota: usize) -> WeekPlanner {
        let stride = if quota == 0 { usize::MAX } else { forecasts_expected.max(1).div_ceil(quota).max(1) };
        WeekPlanner { stride, forecasts: 0, nights: BTreeMap::new(), lambda_scale: 1.0 }
    }

    /// A new forecast arrived: the previous week's advice expires. True when this forecast is planned with the
    /// model; otherwise lambda relaxes halfway back to 1.
    pub fn on_forecast(&mut self) -> bool {
        let planned = self.stride != usize::MAX && self.forecasts % self.stride == 0;
        self.forecasts += 1;
        self.nights.clear();
        if !planned {
            self.lambda_scale = 1.0 + 0.5 * (self.lambda_scale - 1.0);
        }
        planned
    }

    /// The model's advice for one night, if the current plan has one.
    pub fn tonight(&self, date: &str) -> Option<&NightAdvice> {
        self.nights.get(date)
    }

    /// Ask the model for the week's plan. `notices`: the forecast's notices; `dates`: the night dates the
    /// forecast covers; `context`: the rule-computed review and season facts. The validated plan replaces the
    /// current one; on any failure the rules plan the week. Returns true when the model's plan was applied.
    pub fn plan(&mut self, client: &mut LlmClient, quota: usize, notices: &[Value], dates: &[String], context: Value, wait: f64) -> bool {
        let rules = night_rules(notices, dates);
        let rule_proposal = json!({
            "nights": rules.iter().map(|r| json!({"date": r.date, "bad_night": r.rule_bad, "avoid": []})).collect::<Vec<_>>(),
            "lambda_scale": self.lambda_scale,
            "reason": "rules: bad night = ALL-sky rain/storm/overcast/haze forecast; no extra sectors; time price unchanged",
        });
        let (lo, hi) = self.lambda_range();
        let user = json!({
            "week": dates,
            "forecast": notices.iter().map(|n| json!({"event_kind": n.get("event_kind"), "direction": n.get("direction"), "nights": n.get("nights")})).collect::<Vec<_>>(),
            "allowed": rules.iter().map(|r| json!({"date": r.date, "may_be_bad": r.may_be_bad, "sectors": r.sectors})).collect::<Vec<_>>(),
            "context": context,
            "knobs": {"lambda_scale": {"current": self.lambda_scale, "allowed": [lo, hi]}},
            "rule_proposal": rule_proposal,
        });
        let tool = tool();
        let Some(answer) = client.ask(STAGE, quota, SYSTEM, &user, Some(&tool), wait) else { return false };
        let Some((nights, lambda)) = self.validate(&answer, &rules) else {
            crate::log("llm week_plan: answer not usable; rules plan the week");
            return false;
        };
        crate::log(&format!(
            "llm week plan {}..{}: lambda_scale {:.3} -> {:.3}; bad nights {:?}; avoid {:?} ({}; review: {})",
            dates.first().map_or("", String::as_str),
            dates.last().map_or("", String::as_str),
            self.lambda_scale,
            lambda,
            nights.iter().filter(|(_, a)| a.bad_night).map(|(d, _)| d.as_str()).collect::<Vec<_>>(),
            nights.iter().filter(|(_, a)| !a.avoid.is_empty()).map(|(d, a)| format!("{d}:{}", a.avoid.iter().cloned().collect::<Vec<_>>().join("/"))).collect::<Vec<_>>(),
            reason_of(&answer),
            answer.get("review").and_then(Value::as_str).unwrap_or("").chars().take(100).collect::<String>(),
        ));
        self.nights = nights;
        self.lambda_scale = lambda;
        true
    }

    fn lambda_range(&self) -> (f64, f64) {
        ((self.lambda_scale - LAMBDA_STEP).max(LAMBDA_MIN), (self.lambda_scale + LAMBDA_STEP).min(LAMBDA_MAX))
    }

    /// Clamp the answer to the trust region. None when it has no usable field at all.
    fn validate(&self, answer: &Map<String, Value>, rules: &[NightRules]) -> Option<(BTreeMap<String, NightAdvice>, f64)> {
        let mut nights = BTreeMap::new();
        for item in answer.get("nights").and_then(Value::as_array).into_iter().flatten() {
            let Some(date) = item.get("date").and_then(Value::as_str) else { continue };
            let Some(rule) = rules.iter().find(|r| r.date == date) else { continue };
            // bad only where the rule says bad; a missing flag keeps the rule
            let bad_night = boolean(item.get("bad_night")).map_or(rule.rule_bad, |b| b && rule.may_be_bad);
            let avoid: BTreeSet<String> = item
                .get("avoid")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .map(|d| d.trim().to_uppercase())
                .filter(|d| rule.sectors.contains(d))
                .collect();
            nights.insert(date.to_string(), NightAdvice { bad_night, avoid });
        }
        let (lo, hi) = self.lambda_range();
        let lambda = number(answer.get("lambda_scale")).map(|x| x.clamp(lo, hi));
        if nights.is_empty() && lambda.is_none() {
            return None;
        }
        Some((nights, lambda.unwrap_or(self.lambda_scale)))
    }
}

#[cfg(test)]
#[path = "../../tests/unit/llm_week_plan.rs"]
mod tests;
