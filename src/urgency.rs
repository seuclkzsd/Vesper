//! Season supply/demand urgency (research note 07 item #8, after 07c U2).
//!
//! The stock urgency `1 + 1.5 / nights_left` only counts nights on which a target is above the altitude limit
//! for 20 minutes, so a field whose *good* hours run out early in the season (it sets in the evening) looks as
//! relaxed as one that stays up all season. This module multiplies it by a supply/demand factor per sky cell
//! (5 x 5 deg): demand D_c = fibre-seconds the cell's unfinished targets still need; every remaining time node
//! (one engine slot) is shared among the cells at a good altitude then, in proportion to their demand (DESI-style
//! LST matching). Cell pressure u_c = D_c / supply_c = 1 / sum_n(sec_n / Dtot_n); factor = clip((u_c / u_mean)^alpha,
//! lo, hi), u_mean = total demand / usable seconds. Cells with no good node left get `hi`. A boost (factor > 1)
//! applies only while the target is at a good altitude: it ranks a crowded cell first inside its own good hours
//! instead of chasing it at low altitude (notes/11); discounts apply at any altitude.
//!
//! The factors are normalised per decision to mean 1 over the candidate pool (planner/search.rs), so they reorder
//! targets that compete for the same time without changing the units of the learned time price. The planner
//! scales alpha with the card's time scarcity and its candidates per fibre, and builds no urgency at all on
//! sparse catalogues (planner/values.rs).
//!
//! Recomputed at each night start from the planner's own factor estimates. Everything comes from `initialize`
//! (site, night calendar, slot length, catalogue, score config): no card-specific constants.

use crate::skymath::{local_sidereal_deg, normalized_airmass};

pub struct UrgencyConfig {
    pub cell_deg: f64,
    pub rel_good: f64,
    pub alpha: f64,
    pub lo: f64,
    pub hi: f64,
    pub cap: f64,
    /// Boosts only at a good altitude (else the stock term alone).
    pub in_window: bool,
    /// A cell without (admitted) demand gets `lo`, the lowest pressure, instead of the neutral 1.
    pub idle_lo: bool,
}

pub struct Urgency {
    cfg: UrgencyConfig,
    /// Target -> cell (u32::MAX: never above the limit).
    cell_of: Vec<u32>,
    n_cells: usize,
    /// Time nodes (engine slots): length in seconds; CSR list of the cells at a good altitude in each node.
    node_sec: Vec<f64>,
    /// Per cell, the lowest sin(alt) that counts as good (2.0: never).
    sin_good: Vec<f64>,
    node_ptr: Vec<usize>,
    node_cells: Vec<u32>,
    /// First node of each night.
    night_first_node: Vec<usize>,
    /// Current supply/demand factor per cell; sum over its good nodes of seconds / demand.
    cell_band: Vec<f64>,
    inv: Vec<f64>,
    pub night: i64,
}

pub struct SiteGeometry<'a> {
    pub lat: f64,
    pub lon: f64,
    pub min_alt: f64,
    pub airmass_exponent: f64,
    pub nights: &'a [(f64, f64)],
    pub slot_seconds: f64,
}

impl Urgency {
    /// `visible[i]`: the target ever rises above the limit (the planner's hmax > 0).
    pub fn new(cfg: UrgencyConfig, geo: &SiteGeometry, ra: &[f64], dec: &[f64], visible: &[bool]) -> Urgency {
        let n = ra.len();
        let cell_deg = if cfg.cell_deg > 0.0 { cfg.cell_deg } else { 5.0 };
        let n_ra = (360.0 / cell_deg).ceil() as i64;
        let mut index: std::collections::HashMap<i64, u32> = std::collections::HashMap::new();
        let mut sums: Vec<(f64, f64, f64)> = Vec::new(); // (sum ra, sum dec, count) per cell
        let mut cell_of = vec![u32::MAX; n];
        for i in 0..n {
            if !visible[i] {
                continue;
            }
            let key = ((dec[i] + 90.0) / cell_deg).floor() as i64 * n_ra + (ra[i].rem_euclid(360.0) / cell_deg).floor() as i64;
            let c = *index.entry(key).or_insert_with(|| {
                sums.push((0.0, 0.0, 0.0));
                (sums.len() - 1) as u32
            });
            cell_of[i] = c;
            let s = &mut sums[c as usize];
            s.0 += ra[i].rem_euclid(360.0);
            s.1 += dec[i];
            s.2 += 1.0;
        }
        let n_cells = sums.len();
        // per cell: sin(alt) = a + b cos(LST) + c sin(LST), and the lowest sin(alt) that still counts as good
        let (sin_lat, cos_lat) = (geo.lat.to_radians().sin(), geo.lat.to_radians().cos());
        let floor_alt = geo.min_alt + 0.6;
        let mut coef = Vec::with_capacity(n_cells);
        let mut sin_good = Vec::with_capacity(n_cells);
        for &(sr, sd, k) in &sums {
            let (r, d) = ((sr / k).to_radians(), (sd / k).to_radians());
            coef.push((sin_lat * d.sin(), cos_lat * d.cos() * r.cos(), cos_lat * d.cos() * r.sin()));
            let transit = 90.0 - ((sd / k) - geo.lat).abs();
            if transit < floor_alt {
                sin_good.push(2.0); // never good
                continue;
            }
            // airmass term within rel_good of the best (transit) one: bisect the altitude
            let limit = cfg.rel_good * normalized_airmass(transit).powf(geo.airmass_exponent);
            let (mut lo, mut hi) = (floor_alt.min(transit), transit);
            if normalized_airmass(lo).powf(geo.airmass_exponent) <= limit {
                hi = lo;
            } else {
                for _ in 0..40 {
                    let mid = 0.5 * (lo + hi);
                    if normalized_airmass(mid).powf(geo.airmass_exponent) <= limit {
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                }
            }
            sin_good.push(hi.max(floor_alt).to_radians().sin());
        }
        // time nodes: the engine's slots of every night
        let step = if geo.slot_seconds > 0.0 { geo.slot_seconds } else { 900.0 };
        let mut node_sec = Vec::new();
        let mut node_ptr = vec![0usize];
        let mut node_cells = Vec::new();
        let mut night_first_node = Vec::with_capacity(geo.nights.len() + 1);
        for &(start, end) in geo.nights {
            night_first_node.push(node_sec.len());
            let mut t = start;
            while t < end - 1.0 {
                let sec = step.min(end - t);
                let lst = local_sidereal_deg(t + 0.5 * sec, geo.lon);
                let (cl, sl) = (lst.to_radians().cos(), lst.to_radians().sin());
                for c in 0..n_cells {
                    let (a, b, cc) = coef[c];
                    if a + b * cl + cc * sl >= sin_good[c] {
                        node_cells.push(c as u32);
                    }
                }
                node_sec.push(sec);
                node_ptr.push(node_cells.len());
                t += sec;
            }
        }
        night_first_node.push(node_sec.len());
        Urgency {
            cfg,
            cell_of,
            n_cells,
            node_sec,
            sin_good,
            node_ptr,
            node_cells,
            night_first_node,
            cell_band: vec![1.0; n_cells],
            inv: vec![0.0; n_cells],
            night: -1,
        }
    }

    pub fn summary(&self) -> String {
        format!(
            "urgency: {} cells, {} nodes, {} good (node, cell) pairs; alpha {:.2}, clip [{}, {}], cap {}",
            self.n_cells,
            self.node_sec.len(),
            self.node_cells.len(),
            self.cfg.alpha,
            self.cfg.lo,
            self.cfg.hi,
            self.cfg.cap
        )
    }

    /// Recompute the per-cell factors for the night `night` from the targets' remaining demand
    /// (`need[i]`: fibre-seconds target i still needs, 0 when done). Returns a log line.
    pub fn update(&mut self, night: usize, need: &[f64]) -> String {
        self.night = night as i64;
        let mut demand = vec![0.0f64; self.n_cells];
        for (i, &c) in self.cell_of.iter().enumerate() {
            if c != u32::MAX {
                demand[c as usize] += need[i];
            }
        }
        let first = self.night_first_node[night.min(self.night_first_node.len() - 1)];
        let mut inv = vec![0.0f64; self.n_cells];
        let (mut total_sec, total_demand) = (0.0, demand.iter().sum::<f64>());
        for node in first..self.node_sec.len() {
            let cells = &self.node_cells[self.node_ptr[node]..self.node_ptr[node + 1]];
            let dtot: f64 = cells.iter().map(|&c| demand[c as usize]).sum();
            if dtot <= 0.0 {
                continue;
            }
            total_sec += self.node_sec[node];
            let share = self.node_sec[node] / dtot;
            for &c in cells {
                inv[c as usize] += share;
            }
        }
        let u_mean = if total_sec > 0.0 { total_demand / total_sec } else { 0.0 };
        let mut factors = Vec::new();
        for c in 0..self.n_cells {
            self.cell_band[c] = if demand[c] <= 0.0 && self.cfg.idle_lo && u_mean > 0.0 {
                self.cfg.lo
            } else if demand[c] <= 0.0 || u_mean <= 0.0 {
                1.0
            } else if inv[c] <= 0.0 {
                self.cfg.hi // no good time left for this cell: its last chances are now
            } else {
                ((1.0 / inv[c]) / u_mean).powf(self.cfg.alpha).max(self.cfg.lo).min(self.cfg.hi)
            };
            if demand[c] > 0.0 {
                factors.push(self.cell_band[c]);
            }
        }
        self.inv = inv;
        factors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let q = |p: f64| factors.get(((factors.len() as f64 - 1.0) * p).round() as usize).cloned().unwrap_or(1.0);
        format!(
            "urgency night {night}: u_mean {:.3} over {:.0} h; band factor p10 {:.2} p50 {:.2} p90 {:.2} max {:.2} ({} cells with demand)",
            u_mean,
            total_sec / 3600.0,
            q(0.1),
            q(0.5),
            q(0.9),
            q(1.0),
            factors.len()
        )
    }

    /// inv of target i's cell (x fibres = the cell's supply / demand under demand-proportional sharing).
    pub fn inv(&self, i: usize) -> Option<f64> {
        let c = self.cell_of[i];
        if c == u32::MAX || self.night < 0 {
            return None;
        }
        Some(self.inv[c as usize])
    }

    /// Urgency multiplier of target i at altitude `sin_alt` now: the cell's factor times the stock last-chance
    /// factor `stock_last` (1 + urgency / nights_left), capped. None: the target has no cell (never above the limit).
    pub fn factor(&self, i: usize, stock_last: f64, sin_alt: f64) -> Option<f64> {
        let c = self.cell_of[i] as usize;
        if c == u32::MAX as usize || self.night < 0 {
            return None;
        }
        let band = self.cell_band[c];
        let band = if self.cfg.in_window && band > 1.0 && sin_alt < self.sin_good[c] { 1.0 } else { band };
        Some((band * stock_last).min(self.cfg.cap))
    }
}
