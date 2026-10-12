//! Season LP (notes/15): a price for field time per (7-night block x 1 h LST bin) node. Groups = sky cell (5 deg)
//! x T_sat class; value = science still to gain, need = typical field-seconds (fibre-seconds over fibres x fill x
//! use efficiency). A cell may use any node where it is above the limit, at relative quality q = airmass term at
//! transit / airmass term. The LP max sum(value) s.t. node capacities is solved in the dual by damped price steps
//! (softmax over the near-cheapest nodes). Two uses: (1) a target's cell factor = ((cheapest remaining price mu/q
//! of its cell + eps) / (its price now + eps))^alpha in [lo, 1], normalised per decision like the urgency: a cell
//! with cheaper time later (higher, or less contested) yields to one whose time is now; (2) on sparse catalogues
//! the LP price of the node now caps the time price (planner/search.rs). Everything comes from `initialize` and
//! the own results.

use crate::planner::{env_f, env_i};
use crate::skymath::{local_sidereal_deg, normalized_airmass};

const BINS: usize = 24;
const BLOCK: usize = 7;
const CLASSES: usize = 6;

/// Knobs (PRO_SLP_*). mode (PRO_SLP): 0 off, 1 the cell factor replaces the urgency's, 2 multiplies it, 3 no factor;
/// alpha, lo: the factor; lam: the time price is capped at lam x the LP price of the node now (0: no cap), but never
/// below floor x the EMA price where time is scarce: full when the first solve serves <= floor_served - 0.2 of the
/// demand, none at >= floor_served (notes/19); gate: the factor only on the urgency's density gate; cap_gate: the
/// cap only off it.
pub struct SlpCfg {
    pub mode: i64,
    pub alpha: f64,
    pub lo: f64,
    pub lam: f64,
    pub floor: f64,
    pub floor_served: f64,
    pub cap_gate: bool,
    pub iters: usize,
    pub temp: f64,
    pub u0: f64,
}

impl SlpCfg {
    pub fn load() -> SlpCfg {
        SlpCfg {
            mode: env_i("SLP", 2),
            alpha: env_f("SLP_ALPHA", 0.7),
            lo: env_f("SLP_LO", 0.5),
            lam: env_f("SLP_LAM", 1.0),
            floor: env_f("SLP_LAM_FLOOR", 0.5),
            floor_served: env_f("SLP_FLOOR_SERVED", 0.6),
            cap_gate: env_i("SLP_CAP_GATE", 1) != 0,
            iters: env_i("SLP_ITERS", 40).max(1) as usize,
            temp: env_f("SLP_TEMP", 0.15).max(0.01),
            u0: env_f("SLP_U0", 0.6),
        }
    }
}

pub struct SeasonLp {
    pub cfg: SlpCfg,
    /// Factor exponent after the density gate (0: no factor), weight of the price cap, need at the first solve.
    pub alpha: f64,
    pub cap_w: f64,
    pub need0: f64,
    /// Share of the demand the first solve serves (-1 before it).
    pub served0: f64,
    /// Share of the remaining capacity the last solve uses (0 = no demand left; 1 before the first solve).
    pub used_frac: f64,
    cell_of: Vec<u32>,
    class_of: Vec<u8>,
    area: Vec<f64>,
    /// Per cell: the airmass term at transit.
    xt: Vec<f64>,
    /// Per cell, the LST bins above the limit and ln q there.
    bins: Vec<Vec<(usize, f64)>>,
    night_secs: Vec<[f64; BINS]>,
    beta: f64,
    mu: Vec<f64>,
    best: Vec<f64>,
    eps: f64,
    now: usize,
    pub night: i64,
}

impl SeasonLp {
    #[allow(clippy::too_many_arguments)]
    pub fn new(cfg: SlpCfg, alpha: f64, cap_w: f64, lat: f64, lon: f64, min_alt: f64, beta: f64, nights: &[(f64, f64)], slot: f64, ra: &[f64], dec: &[f64], tsat: &[f64], visible: &[bool]) -> SeasonLp {
        let cell_deg = 5.0;
        let n_ra = (360.0 / cell_deg) as i64;
        let mut index = std::collections::HashMap::new();
        let mut sums: Vec<(f64, f64, f64, std::collections::HashSet<(i64, i64)>)> = Vec::new();
        let mut cell_of = vec![u32::MAX; ra.len()];
        let mut class_of = vec![0u8; ra.len()];
        for i in 0..ra.len() {
            if !visible[i] {
                continue;
            }
            let r = ra[i].rem_euclid(360.0);
            let key = ((dec[i] + 90.0) / cell_deg).floor() as i64 * n_ra + (r / cell_deg).floor() as i64;
            let c = *index.entry(key).or_insert_with(|| {
                sums.push((0.0, 0.0, 0.0, Default::default()));
                (sums.len() - 1) as u32
            });
            cell_of[i] = c;
            class_of[i] = ((tsat[i] / 150.0).max(1.0).log2().floor() as usize).min(CLASSES - 1) as u8;
            let s = &mut sums[c as usize];
            s.0 += r;
            s.1 += dec[i];
            s.2 += 1.0;
            s.3.insert((dec[i].floor() as i64, (r * dec[i].to_radians().cos()).floor() as i64));
        }
        let (sl, cl) = (lat.to_radians().sin(), lat.to_radians().cos());
        let mut xt = Vec::new();
        let mut bins = Vec::new();
        for (sr, sd, k, _) in &sums {
            let (r, d) = ((sr / k).to_radians(), (sd / k).to_radians());
            let x_t = normalized_airmass(90.0 - ((sd / k) - lat).abs()).powf(beta);
            let cf = (sl * d.sin(), cl * d.cos() * r.cos(), cl * d.cos() * r.sin());
            let mut row = Vec::new();
            for b in 0..BINS {
                let l = ((b as f64 + 0.5) * 15.0).to_radians();
                let alt = (cf.0 + cf.1 * l.cos() + cf.2 * l.sin()).clamp(-1.0, 1.0).asin().to_degrees();
                if alt >= min_alt + 3.0 {
                    row.push((b, (x_t / normalized_airmass(alt).powf(beta)).ln()));
                }
            }
            xt.push(x_t);
            bins.push(row);
        }
        let area = sums.iter().map(|s| s.3.len().max(1) as f64).collect();
        let step = if slot > 0.0 { slot } else { 900.0 };
        let night_secs = nights
            .iter()
            .map(|&(s, e)| {
                let mut row = [0.0; BINS];
                let mut t = s;
                while t < e - 1.0 {
                    let sec = step.min(e - t);
                    row[((local_sidereal_deg(t + 0.5 * sec, lon) / 15.0) as usize).min(BINS - 1)] += sec;
                    t += sec;
                }
                row
            })
            .collect::<Vec<_>>();
        let nodes = (nights.len() + BLOCK - 1) / BLOCK * BINS;
        let mu = vec![f64::NAN; nodes.max(1)];
        SeasonLp { cfg, alpha, cap_w, need0: -1.0, served0: -1.0, used_frac: 1.0, cell_of, class_of, area, xt, bins, night_secs, beta, mu, best: Vec::new(), eps: 0.0, now: 0, night: -1 }
    }

    /// Solve from night `night` on. `need[i]`: fibre-seconds target i still needs at a typical sky, `value[i]`: the
    /// science it still adds; `unit`: fibre-seconds served per field-second at fill 1 (fibres x use efficiency);
    /// `fibre_area` in deg2; `p_open(n)`: chance night n is open. Warm start from the last prices. Returns a log line.
    pub fn solve(&mut self, night: usize, need: &[f64], value: &[f64], unit: f64, fibre_area: f64, p_open: &dyn Fn(usize) -> f64, iters: usize) -> String {
        self.night = night as i64;
        let temp = self.cfg.temp;
        let nb = self.mu.len() / BINS;
        let b0 = night / BLOCK;
        let mut cap = vec![0.0f64; self.mu.len()];
        for n in night..self.night_secs.len() {
            for b in 0..BINS {
                cap[(n / BLOCK) * BINS + b] += self.night_secs[n][b] * p_open(n);
            }
        }
        let nc = self.area.len();
        let (mut d, mut rho, mut open) = (vec![[0.0f64; CLASSES]; nc], vec![[0.0f64; CLASSES]; nc], vec![0.0f64; nc]);
        for i in 0..need.len() {
            let c = self.cell_of[i];
            if c != u32::MAX && need[i] > 0.0 {
                let (c, k) = (c as usize, self.class_of[i] as usize);
                d[c][k] += need[i];
                rho[c][k] += value[i];
                open[c] += 1.0;
            }
        }
        let mut dens = Vec::new();
        for c in 0..nc {
            let fill = 1.0 - (-1.2 * open[c] * fibre_area / self.area[c]).exp();
            for k in 0..CLASSES {
                if d[c][k] > 0.0 && rho[c][k] > 0.0 {
                    d[c][k] /= unit * fill.max(0.05); // field-seconds at q = 1
                    rho[c][k] = (rho[c][k] / d[c][k]).ln();
                    dens.push(rho[c][k]);
                } else {
                    d[c][k] = 0.0;
                }
            }
        }
        if dens.is_empty() {
            self.best = vec![f64::INFINITY; nc];
            self.used_frac = 0.0;
            return format!("season lp night {night}: no demand");
        }
        dens.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let ln_ref = dens[dens.len() / 2];
        let floor = ln_ref - 8.0;
        for m in self.mu.iter_mut() {
            if m.is_nan() {
                *m = ln_ref;
            }
        }
        let mut load = vec![0.0f64; self.mu.len()];
        let mut e: Vec<(usize, f64, f64)> = Vec::new();
        let mut best = vec![f64::INFINITY; nc];
        let mut served_all = 0.0;
        for it in 0..=iters {
            load.iter_mut().for_each(|x| *x = 0.0);
            served_all = 0.0;
            for c in 0..nc {
                if open[c] <= 0.0 {
                    continue;
                }
                e.clear();
                let mut emin = f64::INFINITY;
                for &(b, lnq) in &self.bins[c] {
                    for blk in b0..nb {
                        let s = blk * BINS + b;
                        if cap[s] > 0.0 {
                            let x = self.mu[s] - lnq;
                            emin = emin.min(x);
                            e.push((s, x, lnq));
                        }
                    }
                }
                best[c] = emin;
                if e.is_empty() {
                    continue;
                }
                let mut served = 0.0;
                for k in 0..CLASSES {
                    if d[c][k] > 0.0 {
                        served += d[c][k] / (1.0 + ((emin - rho[c][k]) / temp).exp());
                    }
                }
                served_all += served;
                let w: f64 = e.iter().map(|x| (-(x.1 - emin) / temp).exp()).sum();
                for &(s, x, lnq) in &e {
                    load[s] += served * (-(x - emin) / temp).exp() / w / lnq.exp();
                }
            }
            if it == iters {
                break;
            }
            let eta = 0.4 * (1.0 - it as f64 / iters as f64) + 0.03;
            for s in 0..self.mu.len() {
                if cap[s] > 0.0 {
                    self.mu[s] = (self.mu[s] + eta * (load[s] / cap[s] - 1.0).clamp(-1.0, 1.0)).max(floor);
                }
            }
        }
        let (mut sum, mut over, mut total, mut used) = (0.0, 0.0, 0.0, 0.0);
        for s in b0 * BINS..self.mu.len() {
            if cap[s] > 0.0 {
                sum += cap[s] * self.mu[s].exp();
                total += cap[s];
                over += (load[s] - cap[s]).max(0.0);
                used += load[s].min(cap[s]);
            }
        }
        self.eps = 0.05 * ln_ref.exp();
        self.best = best.iter().map(|x| x.exp()).collect();
        let demand: f64 = d.iter().flatten().sum();
        self.used_frac = used / total.max(1.0);
        if self.served0 < 0.0 {
            self.served0 = served_all / demand.max(1.0);
        }
        format!(
            "season lp night {night}: demand {:.0} h, capacity {:.0} h (used {:.0} %), served {:.0} %, overload {:.1} %; mean price {:.2} x median density",
            demand / 3600.0,
            total / 3600.0,
            100.0 * used / total.max(1.0),
            100.0 * served_all / demand.max(1.0),
            100.0 * over / total.max(1.0),
            sum / total.max(1.0) / ln_ref.exp()
        )
    }

    pub fn set_now(&mut self, night: usize, lst: f64) {
        self.now = ((night / BLOCK) * BINS + (lst.rem_euclid(360.0) / 15.0) as usize).min(self.mu.len() - 1);
    }

    /// Factor of target i at altitude `sin_alt` now: ((cheapest remaining price + eps) / (price now + eps))^alpha.
    pub fn factor(&self, i: usize, sin_alt: f64) -> Option<f64> {
        let c = *self.cell_of.get(i)? as usize;
        if c == u32::MAX as usize || self.best.is_empty() {
            return None;
        }
        let x = normalized_airmass(sin_alt.clamp(-1.0, 1.0).asin().to_degrees()).powf(self.beta);
        let now = self.mu[self.now].exp() * x / self.xt[c];
        Some(((self.best[c] + self.eps) / (now + self.eps)).powf(self.alpha).clamp(self.cfg.lo, 1.0))
    }

    /// Price of field time now (value per field-second), 0 before the first solve.
    pub fn price_now(&self) -> f64 {
        if self.best.is_empty() {
            0.0
        } else {
            self.mu[self.now].exp()
        }
    }

}
