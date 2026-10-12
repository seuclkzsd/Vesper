//! Output guard: the last check of every action before it is sent, mirroring the engine's validation
//! (`normalize_action` in challenge/v4_runner.py). An action the engine would reject ends the run with
//! agent_error, so a defect anywhere in the planner (a NaN pointing, a duration outside the card's exposure
//! range, an unknown or repeated target) is repaired here, or the action is replaced by a wait. A valid action
//! passes through untouched (same map, same bytes), so the guard never changes a normal decision.

use serde_json::{json, Map, Value};
use std::collections::HashSet;

use crate::log;
use crate::skymath::{format_utc, parse_utc};

/// Envelope keys the engine allows next to the action fields.
const ENVELOPE: [&str; 2] = ["reason", "decision_source"];
const OBSERVE_KEYS: [&str; 5] = ["action", "pointing", "assignments", "duration_seconds", "program"];

pub struct Limits {
    pub min_exposure: i64,
    pub max_exposure: i64,
    pub n_fibers: usize,
    pub now: f64,
    /// A wait the engine accepts that ends at the next slot boundary (the fallback action).
    pub fallback_wait: i64,
}

fn wait_for(seconds: i64, reason: &str) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("action".into(), json!("wait"));
    m.insert("duration_seconds".into(), json!(seconds));
    m.insert("reason".into(), json!(reason));
    m
}

/// An integer-valued, finite JSON number (the engine accepts 900 and 900.0).
fn integer(v: Option<&Value>) -> Option<i64> {
    let x = v?.as_f64()?;
    if x.is_finite() && x.fract() == 0.0 && !v?.is_boolean() {
        Some(x as i64)
    } else {
        None
    }
}

fn finite(v: Option<&Value>) -> Option<f64> {
    v.and_then(Value::as_f64).filter(|x| x.is_finite())
}

/// Return the action unchanged when the engine accepts it, else a repaired action (logged).
pub fn check(mut action: Map<String, Value>, lim: &Limits, is_target: impl Fn(&str) -> bool) -> Map<String, Value> {
    let kind = action.get("action").and_then(Value::as_str).unwrap_or("").to_string();
    let lo = lim.min_exposure.max(1);
    let hi = lim.max_exposure.max(lo);
    let fallback = lim.fallback_wait.clamp(lo, hi);
    let mut fixes: Vec<String> = Vec::new();
    for key in ENVELOPE {
        if action.get(key).map_or(false, |v| !v.is_string()) {
            action.remove(key);
            fixes.push(format!("{key} not a string"));
        }
    }
    let extra = |action: &Map<String, Value>, allowed: &[&str]| -> Vec<String> {
        action.keys().filter(|k| !allowed.contains(&k.as_str()) && !ENVELOPE.contains(&k.as_str())).cloned().collect()
    };
    match kind.as_str() {
        "report" | "finish" => {
            for k in extra(&action, &["action"]) {
                action.remove(&k);
                fixes.push(format!("{kind} with field {k}"));
            }
        }
        "wait" => {
            let has_d = action.contains_key("duration_seconds");
            let has_u = action.contains_key("until_utc");
            let until_ok = action
                .get("until_utc")
                .and_then(Value::as_str)
                .filter(|s| s.ends_with('Z') && s.len() >= 19)
                .map_or(false, |s| parse_utc(s) > lim.now);
            let dur = integer(action.get("duration_seconds"));
            let ok = match (has_d, has_u) {
                (true, false) => dur.map_or(false, |d| lo <= d && d <= hi),
                (false, true) => until_ok,
                _ => false,
            };
            if !ok {
                let seconds = match (has_u, dur) {
                    (false, Some(d)) => d.clamp(lo, hi),
                    _ => fallback,
                };
                let reason = action.get("reason").cloned().unwrap_or(json!("wait"));
                action = wait_for(seconds, "");
                action.insert("reason".into(), reason);
                fixes.push("wait duration / until_utc".into());
            }
            for k in extra(&action, &["action", "duration_seconds", "until_utc"]) {
                action.remove(&k);
                fixes.push(format!("wait with field {k}"));
            }
        }
        "observe" => {
            let alt = action.get("pointing").and_then(|p| finite(p.get("alt_deg")));
            let az = action.get("pointing").and_then(|p| finite(p.get("az_deg")));
            let pointing_keys = action.get("pointing").and_then(Value::as_object).map_or(0, |p| p.len());
            match (alt, az) {
                (Some(a), Some(z)) => {
                    let az_ok = (0.0..360.0).contains(&z);
                    if !(0.0..=90.0).contains(&a) || !az_ok || pointing_keys != 2 {
                        let a2 = a.clamp(0.0, 90.0);
                        let mut z2 = z.rem_euclid(360.0);
                        if !(z2 < 360.0) {
                            z2 = 0.0;
                        }
                        action.insert("pointing".into(), json!({"alt_deg": a2, "az_deg": z2}));
                        fixes.push(format!("pointing ({a}, {z})"));
                    }
                }
                _ => {
                    log(&format!("pro: guard: observe without a usable pointing, waiting instead: {:?}", action.get("pointing")));
                    return wait_for(fallback, "guard: unusable pointing");
                }
            }
            match integer(action.get("duration_seconds")) {
                Some(d) if lo <= d && d <= hi => {}
                other => {
                    let d = finite(action.get("duration_seconds")).map_or(lo, |x| (x.round() as i64).clamp(lo, hi));
                    action.insert("duration_seconds".into(), json!(d));
                    fixes.push(format!("duration {other:?} -> {d}"));
                }
            }
            if let Some(p) = action.get("program") {
                if !p.as_str().map_or(false, |s| crate::planner::PROGRAMS.contains(&s)) {
                    action.insert("program".into(), json!("BACKUP"));
                    fixes.push("program".into());
                }
            }
            let assignments = action.get("assignments").and_then(Value::as_object).cloned();
            match assignments {
                None => {
                    action.insert("assignments".into(), json!({}));
                    fixes.push("assignments not an object".into());
                }
                Some(map) => {
                    let mut fibres: HashSet<usize> = HashSet::new();
                    let mut targets: HashSet<String> = HashSet::new();
                    let mut bad: Vec<String> = Vec::new();
                    for (key, target) in &map {
                        let fibre = key.trim().parse::<usize>().ok().filter(|_| key.trim().chars().all(|c| c.is_ascii_digit()));
                        let ok = match (fibre, target.as_str()) {
                            (Some(f), Some(t)) => f < lim.n_fibers && fibres.insert(f) && is_target(t) && targets.insert(t.to_string()),
                            _ => false,
                        };
                        if !ok {
                            bad.push(key.clone());
                        }
                    }
                    if !bad.is_empty() {
                        let mut kept = map.clone();
                        for k in &bad {
                            kept.remove(k);
                        }
                        action.insert("assignments".into(), Value::Object(kept));
                        fixes.push(format!("{} bad assignments dropped", bad.len()));
                    }
                }
            }
            for k in extra(&action, &OBSERVE_KEYS) {
                action.remove(&k);
                fixes.push(format!("observe with field {k}"));
            }
        }
        other => {
            log(&format!("pro: guard: unknown action {other:?}, waiting instead"));
            return wait_for(fallback, "guard: unknown action");
        }
    }
    if !fixes.is_empty() {
        log(&format!("pro: guard repaired a {kind} at {}: {}", format_utc(lim.now), fixes.join("; ")));
    }
    action
}

#[cfg(test)]
#[path = "../tests/unit/guard.rs"]
mod tests;
