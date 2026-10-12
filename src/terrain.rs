//! Terrain obstruction sectors learnt from the own hits.
//!
//! The initial bulletin names each terrain sector only by the 8-point compass direction of its centre. The engine's
//! weather generator draws a sector as: start uniform on [0, 360), width uniform on [30, 70] deg, altitude from 0
//! up to a maximum uniform on [35, 50] deg (`v4_weather_simulator.py` terrain block, reference weather config),
//! and the notice direction is the compass point nearest to the sector centre. So given the label the centre lies
//! within +-22.5 deg of it. A target inside a sector scores 0 for that part of the exposure.
//!
//! A hard cut that avoids the whole bound (label +-60 deg, altitude below 50 deg) for the entire season can
//! hide a large part of the sky (on a site where the footprint culminates low in that direction, every target
//! there). Here every sector keeps a uniform prior over a grid of (centre, half width, maximum altitude); each
//! assigned target near the sector is evidence (clearly scored: its mid-exposure position is outside the sector;
//! scored 0 while others scored: inside it). A hypothesis the evidence contradicts keeps EPS of its weight (soft,
//! so a mislabelled outcome does not wipe out the belief); outcomes within 1.5 deg / 0.6 deg of a hypothesis' edge
//! do not count against it (grid step, a track crossing the edge). The planner uses P(clear) = 1 - P(blocked) as
//! an expected-value factor instead of the hard cut. If even the best hypothesis is contradicted often (two sectors
//! behind one label, or a config outside the reference ranges) the label falls back to that hard cut.

use crate::skymath::wrap180;

const HALF_SPAN: f64 = 22.5; // label = nearest compass point of the centre
const STEP: f64 = 1.25; // grid of centre offsets and half widths, degrees
const EPS_LN: f64 = -6.908; // ln(0.001): weight kept by a contradicted hypothesis
const EDGE_AZ: f64 = 1.5; // an outcome this close to a hypothesis' edge says nothing about it (grid step, track)
const EDGE_ALT: f64 = 0.6;
const HALF_WIDTH: (f64, f64) = (15.0, 35.0);
const MAX_ALT: (f64, f64) = (35.0, 50.0);
const COLS: usize = 121; // offsets -60..=60 deg, 1 deg
const ROWS: usize = 101; // altitudes 0..=50 deg, 0.5 deg
pub const STOCK_HALF_ANGLE: f64 = 60.0; // fallback hard cut: label +-60 deg ...
pub const STOCK_MAX_ALT: f64 = 50.0; // ... below 50 deg

pub struct TerrainBelief {
    pub label_az: f64,
    /// (centre offset from the label, half width, maximum altitude)
    hyps: Vec<(f64, f64, f64)>,
    /// ln weight and number of contradicting observations of each hypothesis
    ln_w: Vec<f64>,
    contradicted: Vec<u16>,
    /// P(blocked) per (altitude row, offset column)
    table: Vec<f32>,
    pub collapsed: bool,
    pub evidence: usize,
    /// highest altitude seen blocked (a chat top below it is wrong)
    pub top_blocked: f64,
}

impl TerrainBelief {
    pub fn new(label_az: f64) -> TerrainBelief {
        let mut hyps = Vec::new();
        let n_c = (2.0 * HALF_SPAN / STEP).round() as usize; // cell centres of the centre range
        let n_w = ((HALF_WIDTH.1 - HALF_WIDTH.0) / STEP).round() as usize + 1;
        for ci in 0..n_c {
            let c = -HALF_SPAN + STEP * (ci as f64 + 0.5);
            for wi in 0..n_w {
                let hw = HALF_WIDTH.0 + STEP * wi as f64;
                for hi in 0..15 {
                    hyps.push((c, hw, MAX_ALT.0 + 0.5 + hi as f64));
                }
            }
        }
        let n = hyps.len();
        let mut belief = TerrainBelief {
            label_az,
            hyps,
            ln_w: vec![0.0; n],
            contradicted: vec![0; n],
            table: vec![0.0; ROWS * COLS],
            collapsed: false,
            evidence: 0,
            top_blocked: -1.0,
        };
        belief.rebuild();
        belief
    }

    /// Does the outcome contradict the hypothesis (clearly, not within the edge tolerance)?
    fn contradicts(h: &(f64, f64, f64), offset: f64, alt: f64, blocked: bool) -> bool {
        let d = (offset - h.0).abs();
        if blocked {
            d > h.1 + EDGE_AZ || alt > h.2 + EDGE_ALT // the hypothesis clearly leaves the point open
        } else {
            d < h.1 - EDGE_AZ && alt < h.2 - EDGE_ALT // the hypothesis clearly covers the point
        }
    }

    /// Is the point inside the widest possible sector of this label?
    pub fn in_cone(&self, alt: f64, az: f64) -> bool {
        let offset = wrap180(az - self.label_az);
        offset.abs() <= HALF_SPAN + HALF_WIDTH.1 + 1.0 && alt <= MAX_ALT.1 + 0.5
    }

    /// P(the point is inside this label's sector).
    pub fn p_blocked(&self, alt: f64, az: f64) -> f64 {
        let offset = wrap180(az - self.label_az);
        if self.collapsed {
            return if offset.abs() <= STOCK_HALF_ANGLE && alt < STOCK_MAX_ALT { 1.0 } else { 0.0 };
        }
        if offset.abs() > 60.0 || alt >= 50.0 || alt < 0.0 {
            return 0.0;
        }
        let col = ((offset + 60.0).round() as usize).min(COLS - 1);
        let row = ((alt / 0.5).floor() as usize).min(ROWS - 1);
        self.table[row * COLS + col] as f64
    }

    /// One observation at a mid-exposure position: blocked (scored 0) or clear (scored normally).
    /// Returns true when the belief changed.
    pub fn observe(&mut self, alt: f64, az: f64, blocked: bool) -> bool {
        if blocked {
            self.top_blocked = self.top_blocked.max(alt);
        }
        if self.collapsed {
            return false;
        }
        let offset = wrap180(az - self.label_az);
        let mut changed = false;
        for (k, h) in self.hyps.iter().enumerate() {
            if Self::contradicts(h, offset, alt, blocked) {
                self.ln_w[k] += EPS_LN;
                self.contradicted[k] = self.contradicted[k].saturating_add(1);
                changed = true;
            }
        }
        self.evidence += 1;
        let best = self.contradicted.iter().copied().min().unwrap_or(0) as usize;
        if best >= 4 && best * 10 >= self.evidence {
            self.collapsed = true; // no single sector explains the outcomes: back to the hard cut
        } else if changed {
            self.rebuild();
        }
        changed
    }

    /// P(blocked) table from the hypothesis weights (2-D difference array, then prefix sums).
    fn rebuild(&mut self) {
        let top = self.ln_w.iter().cloned().fold(f64::MIN, f64::max);
        let mut diff = vec![0f64; (ROWS + 1) * (COLS + 1)];
        let mut total = 0.0;
        for (k, h) in self.hyps.iter().enumerate() {
            let w = (self.ln_w[k] - top).exp();
            if w < 1e-12 {
                continue;
            }
            total += w;
            let c0 = ((h.0 - h.1 + 60.0).ceil().max(0.0)) as usize;
            let c1 = ((h.0 + h.1 + 60.0).floor().min((COLS - 1) as f64)) as usize;
            let r1 = (((h.2 / 0.5).floor()) as usize).min(ROWS - 1); // rows r with r * 0.5 <= h
            if c0 > c1 {
                continue;
            }
            diff[c0] += w;
            diff[c1 + 1] -= w;
            diff[(r1 + 1) * (COLS + 1) + c0] -= w;
            diff[(r1 + 1) * (COLS + 1) + c1 + 1] += w;
        }
        let total = total.max(1e-300);
        let mut run = vec![0f64; COLS + 1];
        for r in 0..ROWS {
            let mut acc = 0f64;
            for c in 0..COLS {
                acc += diff[r * (COLS + 1) + c];
                run[c] += acc;
                self.table[r * COLS + c] = (run[c] / total).clamp(0.0, 1.0) as f32;
            }
        }
    }

    /// (hypotheses never contradicted, fewest contradictions, MAP centre offset, half width, maximum altitude)
    pub fn summary(&self) -> (usize, u16, f64, f64, f64) {
        let mut best = 0;
        for k in 0..self.hyps.len() {
            if self.ln_w[k] > self.ln_w[best] {
                best = k;
            }
        }
        let clean = self.contradicted.iter().filter(|&&c| c == 0).count();
        let h = self.hyps[best];
        (clean, self.contradicted[best], h.0, h.1, h.2)
    }
}

#[cfg(test)]
#[path = "../tests/unit/terrain.rs"]
mod tests;
