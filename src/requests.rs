//! Observation requests: hard time windows and the Poisson-binomial value of each request target.
//!
//! Engine rule (v4_scorer.py `request_factors` / `observation_request_status`): a request pays its reward when at
//! least `minimum_completed` distinct targets reach factor >= `completion_factor_threshold` in valid exposures
//! lying wholly inside [issued_at, deadline] (start >= issued_at and end <= deadline). Data loss can withdraw a
//! completion; the engine's `active_requests` snapshot (completed ids, remaining_count) is recomputed from the valid
//! ledger, so it is the state to plan from at every decision.
//!
//! Value of one more completion. With r completions still needed and q_k the probability that remaining target k
//! can still be completed inside the window (see `Planner::request_feasibility`), let S_-j be the number of the
//! other remaining targets that complete (Poisson-binomial over their q_k). Completing j now is worth
//!
//!     R * max( P(S_-j = r-1),  P(S_-j >= r-1) / r )
//!
//! * P(S_-j = r-1) is the exact marginal value of j against never completing it (j is pivotal). It dominates when
//!   the request is tight: no spare targets, every needed completion carries the whole reward.
//! * P(S_-j >= r-1) / r is j's equal share of the r completions the request still needs, paid only if the others
//!   can supply the rest. It dominates when there are spare targets: then no single target is pivotal (its exact
//!   marginal is ~0) although r of them must still be done, and a planner that decides one exposure at a time
//!   would otherwise never start on any of them.
//! * A hopeless request (fewer feasible targets than needed) is worth ~0 everywhere: no time is spent on it.
//!
//! In the usual case (all targets feasible) the value is ~R / r, the stock planner's per-target split.

/// One request a target belongs to: its planning value and the window an exposure must lie in to count.
#[derive(Clone, Debug, PartialEq)]
pub struct RequestTerm {
    pub value: f64,
    pub threshold: f64,
    pub issued: f64,
    pub deadline: f64,
}

impl RequestTerm {
    /// Does an exposure [start, start + seconds] count for this request?
    pub fn window_holds(&self, start: f64, seconds: f64) -> bool {
        start >= self.issued && start + seconds <= self.deadline
    }
}

/// Distribution of S = sum of independent Bernoulli(q_k): P(S = k) for k = 0..=n (O(n^2) recursion).
pub fn poisson_binomial(q: &[f64]) -> Vec<f64> {
    let mut p = vec![0.0; q.len() + 1];
    p[0] = 1.0;
    for (n, &qk) in q.iter().enumerate() {
        let qk = qk.clamp(0.0, 1.0);
        for k in (0..=n + 1).rev() {
            let stay = p[k] * (1.0 - qk);
            let step = if k > 0 { p[k - 1] * qk } else { 0.0 };
            p[k] = stay + step;
        }
    }
    p
}

/// Value (in reward units) of completing each remaining target now; `q` are the completion probabilities of the
/// remaining targets, `needed` the completions still missing (>= 1).
pub fn target_values(q: &[f64], needed: usize, reward: f64) -> Vec<f64> {
    let needed = needed.max(1);
    (0..q.len())
        .map(|j| {
            let others: Vec<f64> = q.iter().enumerate().filter(|&(k, _)| k != j).map(|(_, &v)| v).collect();
            let dist = poisson_binomial(&others);
            let pivotal = dist.get(needed - 1).cloned().unwrap_or(0.0);
            let rest_feasible: f64 = dist.iter().skip(needed - 1).sum();
            reward * pivotal.max(rest_feasible / needed as f64)
        })
        .collect()
}

/// Standard normal CDF (Abramowitz & Stegun 7.1.26 erf, |error| < 1.5e-7).
pub fn normal_cdf(x: f64) -> f64 {
    let z = x.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.327_591_1 * z);
    let poly = t * (0.254_829_592 + t * (-0.284_496_736 + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let erf = 1.0 - poly * (-z * z).exp();
    if x >= 0.0 {
        0.5 * (1.0 + erf)
    } else {
        0.5 * (1.0 - erf)
    }
}

/// A usable start of a request target: start time, latest end (night end or deadline), need at the typical sky.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CapSample {
    pub t: f64,
    pub end: f64,
    pub need: f64,
}

/// Sky scenarios (need multiplier, weight): the window's quality scatters about its median by about +-40 %.
pub const CAP_SCENARIOS: [(f64, f64); 3] = [(1.0 / 1.4, 0.25), (1.0, 0.5), (1.0 / 0.7, 0.25)];

/// How many of the targets (time-ordered usable starts, `step` s apart) can still be completed one after the other
/// from `t0` (counting up to `cap`), needs x `scale`, at least `min_exp`, at most `max_exp`: greedy by earliest
/// completion.
pub fn schedulable(targets: &[&[CapSample]], t0: f64, step: f64, scale: f64, min_exp: f64, max_exp: f64, cap: usize) -> usize {
    let mut done = vec![false; targets.len()];
    let mut t = t0;
    let mut count = 0;
    while count < cap {
        let mut best: Option<(f64, usize)> = None;
        for (k, samples) in targets.iter().enumerate() {
            if done[k] {
                continue;
            }
            for s in samples.iter() {
                if s.t + step <= t {
                    continue; // this start has passed
                }
                let tau = (s.need * scale).max(min_exp);
                let start = s.t.max(t);
                if tau <= max_exp && start + tau <= s.end {
                    if best.map_or(true, |b| start + tau < b.0) {
                        best = Some((start + tau, k));
                    }
                    break;
                }
            }
        }
        match best {
            Some((end, k)) => {
                done[k] = true;
                t = end;
                count += 1;
            }
            None => break,
        }
    }
    count
}

/// P(capacity): the weight of the sky scenarios in which `served` completions plus the targets' own reach `needed`.
pub fn p_capacity(targets: &[&[CapSample]], t0: f64, step: f64, min_exp: f64, max_exp: f64, needed: usize, served: usize) -> f64 {
    if served >= needed {
        return 1.0;
    }
    let rest = needed - served;
    CAP_SCENARIOS
        .iter()
        .map(|&(scale, w)| if schedulable(targets, t0, step, scale, min_exp, max_exp, rest) >= rest { w } else { 0.0 })
        .sum()
}

#[cfg(test)]
#[path = "../tests/unit/requests.rs"]
mod tests;
