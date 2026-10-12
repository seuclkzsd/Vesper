//! Planning values of required targets and observation requests (part of the planner): the deadline-aware
//! required value from each target's remaining calendar (stake x P_now x (1 - P_future)), request target
//! values from the Poisson-binomial over the targets' feasibility inside the request window, the pool value,
//! the season supply/demand urgency per sky cell (urgency.rs), and the rule facts the LLM stages see.

use super::*;

impl Planner {
    /// Season supply/demand urgency (urgency.rs). Only where fibres choose among several targets: with fewer
    /// than ~1 target per fibre area the field, not the fibre, is the bottleneck; a season-edge band is then
    /// already observed whenever it is high, and the urgency only pushes it to lower altitude.
    pub(super) fn build_urgency(&mut self) {
        let p = &self.p;
        let gate = ((self.fibre_density - p.urg_dens_lo) / p.urg_dens_span.max(1e-9)).max(0.0).min(1.0);
        if !p.urg_band || !(gate > 0.0) {
            log(&format!("urgency: off ({:.2} targets per fibre area)", self.fibre_density));
            return;
        }
        // urgency only matters when time is short: same scarcity scaling as the time price
        let scarcity_scale = if p.urg_scarcity { 1.0f64.min((self.scarcity / p.scarcity_ref).max(0.2).powf(p.scarcity_power)) } else { 1.0 };
        let cfg = UrgencyConfig {
            cell_deg: p.urg_cell_deg,
            rel_good: p.urg_rel_good,
            alpha: p.urg_alpha * scarcity_scale * gate,
            lo: p.urg_lo,
            hi: p.urg_hi,
            cap: p.urg_cap,
            in_window: p.urg_in_window,
            idle_lo: p.urg_econ,
        };
        let geo = SiteGeometry {
            lat: self.lat,
            lon: self.lon,
            min_alt: self.min_alt,
            airmass_exponent: self.airmass_exponent,
            nights: &self.nights,
            slot_seconds: self.slot_seconds as f64,
        };
        let visible: Vec<bool> = self.hmax.iter().map(|&h| h > 0.0).collect();
        let started = std::time::Instant::now();
        let urg = Urgency::new(cfg, &geo, &self.ra, &self.dec, &visible);
        log(&format!("{} (built in {:.2} s)", urg.summary(), started.elapsed().as_secs_f64()));
        self.urg = Some(urg);
    }

    /// Recompute the urgency factors at the start of a night from the remaining demand: fibre-seconds each
    /// unfinished target still needs (time to saturate at a typical sky, times the share not yet reached).
    /// `urg_econ`: only the demand the rest of the season can afford counts. Targets are ranked by
    /// the value a full exposure still adds per fibre-second (1.2 w (g_max - g) / T, T = min(T_sat, max
    /// exposure)); demand is admitted down that ranking until it fills the remaining supply (fibre-seconds exposed
    /// per night second so far x night seconds left). Leftovers below that price (faint or nearly done targets)
    /// are served in no band, so they must not make a band look short of time.
    pub(super) fn update_urgency(&mut self, night_index: usize) {
        if self.urg.is_none() {
            return;
        }
        let n = self.ids.len();
        let (mut need, mut density) = (vec![0.0; n], vec![0.0; n]);
        for i in 0..n {
            let f = self.factor[i];
            if self.hmax[i] <= 0.0 || f >= DONE_FACTOR || (self.weight[i] <= 0.0 && !self.required[i]) {
                continue;
            }
            let t_sat = self.f0t0 / (self.flux[i].max(1e-3) * TYPICAL_Q);
            let t = self.max_exposure.min(t_sat);
            need[i] = t * (1.0 - f);
            density[i] = if self.required[i] && f < self.req_thr { f64::INFINITY } else { 1.2 * self.weight[i] * ((t / t_sat).min(1.0) - f).max(0.0) / t };
        }
        let (mut price, mut admitted) = (0.0, 1.0);
        if self.p.urg_econ {
            let k = night_index.min(self.nights.len());
            let past: f64 = self.nights[..k].iter().map(|(s, e)| e - s).sum();
            let left: f64 = self.nights[k..].iter().map(|(s, e)| e - s).sum();
            let rate = if night_index >= 3 && past > 0.0 { self.urg_supply / past } else { self.grid.n as f64 };
            let mut order: Vec<usize> = (0..n).filter(|&i| need[i] > 0.0).collect();
            order.sort_by(|&a, &b| density[b].partial_cmp(&density[a]).unwrap_or(std::cmp::Ordering::Equal));
            let (supply, total) = (rate * left, psum(order.iter().map(|&i| need[i])));
            let mut cum = 0.0;
            for &i in &order {
                cum += need[i];
                if cum > supply {
                    price = density[i];
                    break;
                }
            }
            if price > 0.0 {
                for &i in &order {
                    if density[i] < price {
                        need[i] = 0.0;
                    }
                }
                admitted = supply / total.max(1e-9);
            }
        }
        let line = self.urg.as_mut().unwrap().update(night_index, &need);
        if night_index % 7 == 0 {
            log(&format!("{line}; price {price:.2e}/fibre-s admits {:.0} % of the demand", 100.0 * admitted.min(1.0)));
        }
        self.update_complete(night_index);
    }

    /// Complete-or-skip: P(a later exposure supersedes a partial one) = cs_pmax x the cell's supply /
    /// demand (urgency.rs), for targets the useful season supply (x cs_eta) affords (as urg_econ); else 0.
    pub(super) fn update_complete(&mut self, night_index: usize) {
        let n = self.ids.len();
        self.cs_p = vec![0.0; n];
        let p = &self.p;
        let Some(urg) = self.urg.as_ref() else { return };
        if p.cs_pmax <= 0.0 {
            return;
        }
        let (mut need, mut density) = (vec![0.0; n], vec![0.0; n]);
        for i in 0..n {
            let f = self.factor[i];
            if self.hmax[i] <= 0.0 || f >= DONE_FACTOR || self.weight[i] <= 0.0 || (self.required[i] && f < self.req_thr) {
                continue;
            }
            let t_sat = self.f0t0 / (self.flux[i].max(1e-3) * TYPICAL_Q);
            let t = self.max_exposure.min(t_sat);
            need[i] = t * (1.0 - f);
            density[i] = 1.2 * self.weight[i] * ((t / t_sat).min(1.0) - f).max(0.0) / t;
        }
        let k = night_index.min(self.nights.len());
        let past: f64 = self.nights[..k].iter().map(|(s, e)| e - s).sum();
        let left: f64 = self.nights[k..].iter().map(|(s, e)| e - s).sum();
        let rate = if night_index >= 3 && past > 0.0 { self.urg_supply / past } else { self.grid.n as f64 };
        let supply = rate * left * p.cs_eta;
        let mut order: Vec<usize> = (0..n).filter(|&i| need[i] > 0.0).collect();
        order.sort_by(|&a, &b| density[b].partial_cmp(&density[a]).unwrap_or(std::cmp::Ordering::Equal));
        let (mut cum, mut price) = (0.0, 0.0);
        for &i in &order {
            cum += need[i];
            if cum > supply {
                price = density[i];
                break;
            }
        }
        let gate = ((self.fibre_density - p.urg_dens_lo) / p.urg_dens_span.max(1e-9)).max(0.0).min(1.0);
        let fibres = self.grid.n as f64 * p.cs_eta;
        let mut sum = 0.0;
        for &i in order.iter().filter(|&&i| density[i] >= price) {
            self.cs_p[i] = p.cs_pmax * gate * (urg.inv(i).unwrap_or(0.0) * fibres).min(1.0);
            sum += self.cs_p[i];
        }
        if night_index % 7 == 0 {
            log(&format!("complete: night {night_index} price {price:.2e}, sum P_revisit {sum:.0} of {} open", order.len()));
        }
    }

    /// Urgency multiplier of target i at altitude `sin_alt` now (base: 1 + urgency / nights with 20 minutes
    /// above the limit), times the season LP factor (PRO_SLP=1: instead of the urgency's cell factor).
    pub(super) fn urgency_factor(&self, i: usize, night_i: i64, sin_alt: f64) -> f64 {
        let nights_left = (self.last_night[i] - night_i + 1).max(1);
        let stock = 1.0 + self.p.urgency / nights_left as f64;
        let lp = self.slp.as_ref().filter(|lp| lp.alpha > 0.0);
        let urg = if lp.map_or(false, |lp| lp.cfg.mode == 1) { None } else { self.urg.as_ref() };
        let base = urg.and_then(|u| u.factor(i, stock, sin_alt)).unwrap_or(stock);
        match lp.and_then(|lp| lp.factor(i, sin_alt)) {
            Some(m) => (base * m).min(self.p.urg_cap),
            None => base,
        }
    }

    /// Season LP (seasonlp.rs): the cell factor (PRO_SLP=1/2) on the urgency's density gate, the time price cap
    /// off it (ramped with the same gate).
    pub(super) fn build_slp(&mut self) {
        let cfg = SlpCfg::load();
        let gate = ((self.fibre_density - self.p.urg_dens_lo) / self.p.urg_dens_span.max(1e-9)).max(0.0).min(1.0);
        let alpha = if cfg.mode == 1 || cfg.mode == 2 { cfg.alpha } else { 0.0 };
        let cap_w = if cfg.lam > 0.0 { if cfg.cap_gate { 1.0 - gate } else { 1.0 } } else { 0.0 };
        if cfg.mode == 0 || !(alpha > 0.0 || cap_w > 0.0) {
            return;
        }
        let tsat: Vec<f64> = (0..self.ids.len()).map(|i| self.max_exposure.min(self.f0t0 / (self.flux[i].max(1e-3) * TYPICAL_Q))).collect();
        let visible: Vec<bool> = self.hmax.iter().map(|&h| h > 0.0).collect();
        log(&format!("season lp: factor alpha {alpha:.2}, price cap {} x LP price weight {cap_w:.2}", cfg.lam));
        let lp = SeasonLp::new(cfg, alpha, cap_w, self.lat, self.lon, self.min_alt, self.airmass_exponent, &self.nights, self.slot_seconds as f64, &self.ra, &self.dec, &tsat, &visible);
        self.slp = Some(lp);
        self.update_slp(0); // the first (long) solve belongs to the initialisation, not to the first decision's pace
    }

    /// Re-solve the season LP at a night start (warm start; long seasons every few nights). Need and value per
    /// target as in update_urgency; the use efficiency (need served per assigned fibre-second, urg_supply) is
    /// learnt around the prior SLP_U0.
    pub(super) fn update_slp(&mut self, night_index: usize) {
        let n = self.ids.len();
        let (mut need, mut value) = (vec![0.0; n], vec![0.0; n]);
        for i in 0..n {
            let f = self.factor[i];
            if self.hmax[i] <= 0.0 || f >= DONE_FACTOR || self.weight[i] <= 0.0 {
                continue;
            }
            let t_sat = self.f0t0 / (self.flux[i].max(1e-3) * TYPICAL_Q);
            let t = self.max_exposure.min(t_sat);
            need[i] = t * (1.0 - f);
            value[i] = 1.2 * self.weight[i] * ((t / t_sat).min(1.0) - f).max(0.0);
        }
        let total = psum(need.iter().cloned());
        let nf = self.grid.n as f64;
        let prior = 0.05 * nf * psum(self.nights.iter().map(|(s, e)| e - s));
        let stride = (self.nights.len() + 119) / 120;
        let bad = self.forecast_bad_nights.clone();
        let p_open = move |k: usize| if bad.contains(&k) { 0.5 } else { 0.9 };
        let (fibre_area, supply) = (self.grid.glass * self.grid.glass, self.urg_supply);
        let lp = self.slp.as_mut().unwrap();
        let first = lp.need0 < 0.0;
        if first {
            lp.need0 = total;
        }
        lp.night = night_index as i64;
        if !first && night_index % stride != 0 {
            return;
        }
        let u = (((lp.need0 - total).max(0.0) + lp.cfg.u0 * prior) / (supply + prior)).clamp(0.4, 1.5);
        let iters = if first { 5 * lp.cfg.iters } else { lp.cfg.iters };
        let line = lp.solve(night_index, &need, &value, nf * u, fibre_area, &p_open, iters);
        if first || night_index % 7 == 0 {
            log(&format!("{line}; use {u:.2}"));
        }
    }

    /// Deadline-aware required value, part 1 (once): for every required target and night, the best factor one
    /// exposure (up to the longest allowed) can reach that night at quality level 1, from public geometry and the
    /// lunar ephemeris only. Grid of 15 minutes; the whole exposure must stay above the altitude limit, inside
    /// the night; the sky model is averaged over the exposure (trapezoid). Stored as ln(factor / threshold).
    pub(super) fn build_required_outlook(&mut self) {
        if !self.p.req_deadline {
            return;
        }
        let started = std::time::Instant::now();
        const STEP: f64 = 900.0;
        let max_steps = ((self.max_exposure / STEP).floor() as usize).clamp(1, 8);
        let rows: Vec<usize> = (0..self.ids.len()).filter(|&i| self.required[i] && self.hmax[i] > 0.0).collect();
        self.req_row = vec![usize::MAX; self.ids.len()];
        for (r, &i) in rows.iter().enumerate() {
            self.req_row[i] = r;
        }
        self.req_x = vec![vec![f32::NEG_INFINITY; self.nights.len()]; rows.len()];
        let mut model = Vec::new();
        for (k, &(start, end)) in self.nights.iter().enumerate() {
            let points = ((end - start) / STEP).floor() as usize + 1;
            if points < 2 {
                continue;
            }
            let grid: Vec<(f64, f64, Moon)> = (0..points)
                .map(|j| {
                    let t = start + j as f64 * STEP;
                    let lst = local_sidereal_deg(t, self.lon);
                    (lst.to_radians().cos(), lst.to_radians().sin(), Moon::new(t, lst, self.lat, self.lunar_model))
                })
                .collect();
            for (r, &i) in rows.iter().enumerate() {
                // sky model at each grid point, 0 when below the limit
                model.clear();
                let mut any = false;
                for (c, s, moon) in &grid {
                    let sin_alt = self.alt_a[i] + self.alt_b[i] * c + self.alt_c[i] * s;
                    if sin_alt < self.sin_alt_limit {
                        model.push(0.0);
                        continue;
                    }
                    any = true;
                    let alt = sin_alt.min(1.0).asin().to_degrees();
                    model.push(self.sky_model(moon.lunar_factor(self.ra[i], self.dec[i]), alt));
                }
                if !any {
                    continue;
                }
                // best integral of the model (seconds x mean model) over exposures of 1..max_steps grid steps
                let mut best = 0.0f64;
                for j in 0..points {
                    if model[j] <= 0.0 {
                        continue;
                    }
                    let mut interior = 0.0;
                    for m in 1..=max_steps {
                        if j + m >= points || model[j + m] <= 0.0 {
                            break;
                        }
                        if m >= 2 {
                            interior += model[j + m - 1];
                        }
                        best = best.max(STEP * (0.5 * model[j] + interior + 0.5 * model[j + m]));
                    }
                }
                if best > 0.0 {
                    let factor = self.flux[i] * best / self.f0t0;
                    self.req_x[r][k] = (factor / self.req_thr).ln() as f32;
                }
            }
        }
        self.req_xbest = self.req_x.iter().map(|row| row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64).collect();
        log(&format!(
            "required outlook: {} targets x {} nights in {:.2}s",
            rows.len(),
            self.nights.len(),
            started.elapsed().as_secs_f64()
        ));
    }

    /// Deadline-aware required value, part 2 (every night start): P_future of each unfinished required target,
    /// the chance it still reaches the threshold on a later night if it is not done now:
    ///     P_future = kappa * (1 - prod_k (1 - p_k)^w),   p_k = P_open(k) * Phi((x_k + ln s_k - mu) / sigma)
    /// with x_k the target's best reachable ln(factor / threshold) on night k at quality level 1, ln s the night's
    /// quality level (lognormal: mean mu, sd sigma, learnt from the own nightly medians with a prior), an
    /// earthquake's loss fading by exp(-1/4) per night, and w < 1 because neighbouring nights are correlated.
    pub(super) fn update_required_outlook(&mut self, night_index: usize) {
        self.req_night = night_index as i64;
        // close the previous night's quality sample
        if self.night_ln_ratios.len() >= 4 {
            let mut v = std::mem::take(&mut self.night_ln_ratios);
            self.ln_scale_nights.push(median(&mut v));
        }
        self.night_ln_ratios.clear();
        if !self.p.req_deadline || self.req_x.is_empty() {
            return;
        }
        let p = &self.p;
        let n0 = p.req_prior_nights.max(0.0);
        let mu0 = p.req_prior_scale.max(1e-3).ln();
        let n = self.ln_scale_nights.len() as f64;
        let mu = (n0 * mu0 + psum(self.ln_scale_nights.iter().cloned())) / (n0 + n).max(1e-9);
        let var = (n0 * p.req_prior_sigma.powi(2) + psum(self.ln_scale_nights.iter().map(|x| (x - mu).powi(2)))) / (n0 + n).max(1e-9);
        let sigma = (var + 0.1f64.powi(2)).sqrt().clamp(0.15, 0.8);
        // earthquake: tonight's loss relative to the usual level, fading night by night
        let quake_loss = match self.ln_scale_nights.last() {
            Some(&last) if self.quake_active => (1.0 - (last - mu).exp()).clamp(0.0, 0.95),
            _ => 0.0,
        };
        self.req_mu = (mu, sigma);
        // hard: even the best exposure of the season needs a better than typical night
        let hard_x = -p.hrg_frac.max(1e-3).ln() - mu;
        // ... or, for a window that closes early in the season, a looser threshold: the western band
        // transits at dusk on the first nights only, and deferring it meets the season's worst crowding
        let [early_frac, early_share] = p.hrg_early;
        let hard_x_early = if early_frac > 0.0 && early_frac < p.hrg_frac { -early_frac.ln() - mu } else { f64::NEG_INFINITY };
        let early_end = early_share * self.nights.len() as f64;
        let (tail, bias, sd) = self.hrg_cal(self.max_exposure);
        let (mut n_hard, mut n_early) = (0, 0);
        let nights = self.nights.len();
        let ln_open: Vec<f64> = (0..nights)
            .map(|k| if self.forecast_bad_nights.contains(&k) { p.req_p_open_bad } else { p.req_p_open })
            .map(|po| po.clamp(1e-6, 1.0).ln())
            .collect();
        let (mut open_count, mut sum_pf, mut urgent) = (0usize, 0.0f64, 0usize);
        let top = p.req_top_nights;
        let mut ln_terms: Vec<f64> = Vec::new();
        // early-closing targets: (target, later good nights), scaled once every target is known
        let [close_g, close_frac, close_xmax, close_ra] = p.req_close;
        let close_night = close_frac * nights as f64;
        let mut closing: Vec<(usize, usize)> = Vec::new();
        for i in 0..self.ids.len() {
            let r = self.req_row[i];
            if r == usize::MAX {
                continue;
            }
            if !self.req_open(i) {
                self.req_pfuture[i] = 1.0;
                continue;
            }
            let early = self.req_xbest[r] < hard_x_early && (self.last_night[i] as f64) < early_end;
            let hard = p.hrg && (self.req_hard[i] || self.req_xbest[r] < hard_x || early); // once hard, hard until done
            self.req_hard[i] = hard;
            n_hard += usize::from(hard);
            n_early += usize::from(hard && early && self.req_xbest[r] >= hard_x);
            // a hard target's later attempt follows the fitted model of long attempts (bias, spread, closures)
            let (shift, spread, keep) = if hard { (bias, (sigma * sigma + sd * sd).sqrt(), 1.0 - tail) } else { (0.0, sigma, 1.0) };
            let row = &self.req_x[r];
            // ln(1 - p_k) of every later night the target can be reached
            ln_terms.clear();
            let (mut good_later, mut last_good) = (0usize, None);
            for k in night_index + 1..nights {
                let x = row[k];
                if x == f32::NEG_INFINITY {
                    continue;
                }
                let mut z = x as f64 + mu;
                if quake_loss > 0.0 {
                    z += (1.0 - quake_loss * (-((k - night_index) as f64) / 4.0).exp()).max(1e-3).ln();
                }
                let pk = (ln_open[k].exp() * keep * normal_cdf((z - shift) / spread)).min(0.999);
                ln_terms.push((1.0 - pk).ln());
                if pk >= 0.5 {
                    good_later += 1;
                    last_good = Some(k);
                }
            }
            if close_g > 0.0 && self.req_xbest[r] < close_xmax {
                // the window's last good night, among the past nights once no later one is left
                let last_good = last_good.or_else(|| {
                    (0..=night_index.min(nights.saturating_sub(1))).rev().find(|&k| {
                        let x = row[k];
                        x != f32::NEG_INFINITY && ln_open[k].exp() * keep * normal_cdf((x as f64 + mu - shift) / spread) >= 0.5
                    })
                });
                if last_good.map_or(false, |k| (k as f64) < close_night) {
                    closing.push((i, good_later));
                }
            }
            if top > 0 && ln_terms.len() > top {
                // only the best few later nights count: the planner revisits a deferred target only a few times
                ln_terms.select_nth_unstable_by(top - 1, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                ln_terms.truncate(top);
            }
            let ln_miss = p.req_night_weight * psum(ln_terms.iter().cloned()); // ln prod_k (1 - p_k)^w
            let pf = p.req_kappa * (1.0 - ln_miss.exp());
            self.req_pfuture[i] = pf;
            open_count += 1;
            sum_pf += pf;
            if pf < 0.4 {
                urgent += 1;
            }
        }
        // early-closing band: the best few later nights saturate P_future while the window's good nights run out,
        // and the targets of one dusk band share those nights; scale by the good nights left per target in the band
        let mut n_scaled = 0;
        for &(i, good_later) in &closing {
            let band = closing
                .iter()
                .filter(|&&(j, _)| {
                    let d = (self.ra[i] - self.ra[j]).rem_euclid(360.0);
                    d.min(360.0 - d) <= close_ra
                })
                .count();
            let scale = (good_later as f64 / (close_g * band as f64)).min(1.0);
            if scale < 1.0 {
                let pf = self.req_pfuture[i];
                self.req_pfuture[i] = pf * scale;
                sum_pf -= pf * (1.0 - scale);
                urgent += usize::from(pf >= 0.4 && pf * scale < 0.4);
                n_scaled += 1;
            }
        }
        self.vcache = None; // the required values changed
        log(&format!(
            "required outlook night {}: {} open ({} hard{}), mean P_future {:.2}, {} below 0.4; quality ln-mean {:.2} sd {:.2} ({} nights){}",
            night_index,
            open_count,
            n_hard,
            format!(
                "{}{}",
                if n_early > 0 { format!(", {} by the early window", n_early) } else { String::new() },
                if closing.is_empty() { String::new() } else { format!(", {} closing early, {} scaled", closing.len(), n_scaled) }
            ),
            if open_count > 0 { sum_pf / open_count as f64 } else { 0.0 },
            urgent,
            mu,
            sigma,
            self.ln_scale_nights.len(),
            if quake_loss > 0.0 { format!(", earthquake loss {:.2}", quake_loss) } else { String::new() }
        ));
    }

    /// Idle upside applies on this card: a time-rich season (scarcity below PRO_IDLE_SCARCITY).
    pub fn idle_card(&self) -> bool {
        self.p.idle_up[0] > 0.0 && self.scarcity < self.p.idle_up[1]
    }

    /// The season plan has run out of work: its last solve uses under PRO_IDLE_USED of the remaining capacity.
    pub fn season_idle(&self) -> bool {
        self.slp.as_ref().map_or(false, |lp| lp.used_frac < self.p.idle_used)
    }

    /// Remember since when the plain search has found nothing worth its time (NaN = it found something last).
    pub(super) fn note_idle(&self, now: f64, empty: bool) {
        use std::sync::atomic::Ordering::Relaxed;
        let since = f64::from_bits(self.idle_since.load(Relaxed));
        if !empty {
            self.idle_since.store(f64::NAN.to_bits(), Relaxed);
        } else if since.is_nan() {
            self.idle_since.store(now.to_bits(), Relaxed);
        }
    }

    /// Idle as the season running out of work, not a lull between fields (an idle exposure in a mid-season lull
    /// can take the hour before the next field rises): the season plan has no work left for the capacity, and the
    /// plain search has found nothing for PRO_IDLE_HOURS.
    pub fn idle_ready(&self, now: f64) -> bool {
        let since = f64::from_bits(self.idle_since.load(std::sync::atomic::Ordering::Relaxed));
        !since.is_nan() && now - since >= self.p.idle_hours * 3600.0 && self.season_idle()
    }

    /// Futility guard: 0.8 per long failed attempt on a hard target beyond the first PRO_REQ_FUTILE.
    pub(super) fn futile_mult(&self, i: usize) -> f64 {
        let n = self.p.req_futile.0;
        if n > 0 && self.futile[i] > n {
            0.8f64.powi((self.futile[i] - n).min(60) as i32)
        } else {
            1.0
        }
    }

    /// Value of finishing required target i now, per unit chance of success: stake x (1 - P_future).
    pub(super) fn required_stake(&self, i: usize) -> f64 {
        self.req_stake * (1.0 - self.req_pfuture[i]).max(0.0)
    }

    /// (tail, bias, sd) of a t-second attempt on a hard target: P(closure or fault) and the ln-error's centre
    /// and spread (empirical, fitted to long required attempts in simulated runs).
    pub(super) fn hrg_cal(&self, t: f64) -> (f64, f64, f64) {
        let [tail, bias, sd0, sd1] = self.p.hrg_cal;
        let x = t / 3600.0;
        ((tail * x).min(0.9), bias * x, (sd0 * sd0 + (sd1 * x).powi(2)).sqrt().max(1e-3))
    }

    /// Hard required target still open, away from the last nights of its window: the hard-target value applies.
    pub(super) fn hrg_on(&self, i: usize, night_i: i64) -> bool {
        self.req_hard[i] && self.last_night[i] - night_i >= self.p.hrg_end && self.deadline_aware(i) && self.req_open(i)
    }

    /// Required stake x P(success) of a t-second attempt whose predicted factor / threshold is `raw`. Hard
    /// targets: the fitted success curve, an optional gate, and the whole stake in a good window (near
    /// transit, dark, raw(T_max) >= 1.25, i.e. P ~0.85: deferring it only waits for a rarer night).
    pub(super) fn req_attempt(&self, i: usize, raw: f64, t: f64, night_i: i64, good: bool) -> f64 {
        if !self.hrg_on(i, night_i) {
            return self.required_stake(i) * self.p_now(raw, t);
        }
        let (tail, bias, sd) = self.hrg_cal(t);
        let p = (1.0 - tail) * normal_cdf((raw.max(1e-12).ln() - bias) / sd);
        p * if good { self.req_stake } else { self.required_stake(i) }
    }

    /// Turn the active all-or-nothing requests into per-target planning values (see requests.rs).
    ///
    /// The engine's snapshot is the state: `remaining_count` and `completed_target_ids` are recomputed from the valid
    /// ledger at every decision, so a data-loss revision that withdraws a completion reopens that target here.
    pub fn on_requests(&mut self, requests: &[Value], now: f64) {
        let old = std::mem::take(&mut self.request_bonus);
        self.request_terms.clear();
        for cap in std::mem::take(&mut self.request_caps) {
            for &i in &cap.targets {
                self.cap_flag[i] = false;
            }
        }
        self.cap_of.clear();
        // membership of `active`, built once per call (a linear search per request target is slow at 50k targets)
        let mut in_active: Option<HashSet<usize>> = None;
        for request in requests {
            // A request that already met its minimum still appears until its deadline
            // with remaining_count 0; its reward is settled, so it adds no value.
            let remaining = num(request.get("remaining_count").unwrap_or(&request["minimum_completed"])).trunc() as i64;
            if remaining <= 0 {
                continue;
            }
            let completed: HashSet<String> =
                request.get("completed_target_ids").and_then(Value::as_array).into_iter().flatten().map(str_of).collect();
            let reward = num(&request["completion_reward"]);
            let threshold = num(&request["completion_factor_threshold"]);
            let threshold = if threshold > 0.0 && threshold <= 1.0 { threshold } else { 0.5 };
            let parse = |key: &str| request.get(key).and_then(Value::as_str).map(parse_utc).filter(|t| t.is_finite());
            // without a readable window, fall back to "open now, until the end of the survey"
            let issued = parse("issued_at_utc").unwrap_or(f64::MIN);
            let deadline = parse("deadline_utc").unwrap_or(self.survey_end);
            if self.p.req_window && deadline <= now {
                continue;
            }
            let open: Vec<usize> = request
                .get("target_ids")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .map(str_of)
                .filter(|id| !completed.contains(id))
                .filter_map(|id| self.index_of.get(&id).cloned())
                .collect();
            let values: Vec<f64> = if self.p.req_pb {
                let quality = self.typical_quality(now);
                let q: Vec<f64> =
                    open.iter().map(|&i| self.request_feasibility(i, now.max(issued), deadline, threshold, quality)).collect();
                let values = target_values(&q, remaining as usize, reward);
                self.log_request(request, remaining, &q, &values);
                values
            } else {
                vec![reward / remaining as f64; open.len()]
            };
            // LLM request triage: a priority multiplies the planning value (1 = the rule)
            let priority = self.request_priority.get(&str_of(&request["request_id"])).cloned().unwrap_or(1.0);
            if self.p.req_cap && self.p.req_window && !open.is_empty() && reward > 0.0 && priority > 0.0 {
                let id = str_of(&request["request_id"]);
                let value = self.p.request_mult * reward * priority * self.p.request_p_open;
                self.add_request_cap(&id, &open, remaining as usize, threshold, issued, deadline, value, now);
            }
            for (&i, &v) in open.iter().zip(&values) {
                let value = self.p.request_mult * v * priority;
                // Overlapping requests: the marginal rewards add up; each is collectible only inside its own window
                // and at its own threshold.
                *self.request_bonus.entry(i).or_insert(0.0) += value;
                self.request_terms.entry(i).or_default().push(RequestTerm { value, threshold, issued, deadline });
                if self.hmax[i] > 0.0 {
                    let set = in_active.get_or_insert_with(|| self.active.iter().cloned().collect());
                    if set.insert(i) {
                        self.active.push(i);
                    }
                }
            }
        }
        if old != self.request_bonus {
            self.vdirty.extend(old.keys());
            self.vdirty.extend(self.request_bonus.keys());
        }
    }

    /// Request capacity (Params::req_cap): the usable starts of the request's open targets from now on and the
    /// chance that the request can still be completed if it is worked from now (requests.rs `p_capacity`). A target
    /// is usable at a start when it is above the altitude limit (with its margin) and its line of sight is more
    /// likely clear than not under the learnt terrain belief; its need there is the exposure that reaches the
    /// threshold at the typical sky (the Moon and airmass of that moment), and the exposure must end by the night's
    /// end and the deadline.
    #[allow(clippy::too_many_arguments)]
    fn add_request_cap(&mut self, id: &str, open: &[usize], needed: usize, threshold: f64, issued: f64, deadline: f64, value: f64, now: f64) {
        let step = self.p.req_cap_step;
        if !self.cap_geo.contains_key(id) {
            if self.cap_geo.len() > 64 {
                self.cap_geo.clear(); // requests come and go: keep the cache small
            }
            let geo = self.request_geometry(open, issued.max(now), deadline);
            self.cap_geo.insert(id.to_string(), geo);
        }
        let quality = self.typical_quality(now);
        let geo = &self.cap_geo[id];
        let mut targets = Vec::new();
        let mut samples = Vec::new();
        for &i in open {
            let Some(rows) = geo.get(&i) else { continue };
            let list: Vec<CapSample> = rows
                .iter()
                .filter(|g| g.0 + step > now && self.terrain_factor(g.2, g.3) >= 0.5)
                .map(|g| {
                    let model = g.4 / (self.q0 * normalized_airmass(g.2.max(1.0)).powf(self.airmass_exponent));
                    CapSample { t: g.0, end: g.1, need: threshold * self.f0t0 / (self.flux[i].max(1e-9) * model.max(1e-9) * quality) }
                })
                .collect();
            targets.push(i);
            samples.push(list);
        }
        let refs: Vec<&[CapSample]> = samples.iter().map(|v| v.as_slice()).collect();
        let base = p_capacity(&refs, now, step, self.min_exposure, self.max_exposure, needed, 0);
        let key = (needed, (base * 20.0).round() as i64);
        if self.cap_log.get(id) != Some(&key) {
            self.cap_log.insert(id.to_string(), key);
            let usable: Vec<String> = samples.iter().map(|v| format!("{}", v.len())).collect();
            log(&format!("request {id} capacity: {needed} needed, P(capacity) {base:.2}; usable starts [{}]", usable.join(" ")));
        }
        if base <= 0.0 {
            return; // nothing left to protect
        }
        let ci = self.request_caps.len();
        for (bit, &i) in targets.iter().enumerate().take(32) {
            self.cap_of.entry(i).or_default().push((ci, bit as u32));
            self.cap_flag[i] = true;
        }
        targets.truncate(32);
        samples.truncate(32);
        self.request_caps.push(ReqCap { value, needed, threshold, issued, deadline, targets, samples, base });
    }

    /// Window geometry of request targets: every `REQ_CAP_STEP` seconds of the nights inside [from, deadline], the
    /// targets above the altitude limit (with its margin), with the Moon's factor at that moment.
    fn request_geometry(&self, open: &[usize], from: f64, deadline: f64) -> HashMap<usize, Vec<CapGeo>> {
        let step = self.p.req_cap_step;
        let mut out: HashMap<usize, Vec<CapGeo>> = open.iter().map(|&i| (i, Vec::new())).collect();
        let first = self.nights.partition_point(|&(_, e)| e <= from);
        for &(s, e) in self.nights[first..].iter().take_while(|&&(s, _)| s < deadline) {
            let (a, b) = (s.max(from), e.min(deadline).min(self.survey_end));
            let mut t = a;
            while t + self.min_exposure <= b {
                let lst = local_sidereal_deg(t, self.lon);
                let moon = Moon::new(t, lst, self.lat, self.lunar_model);
                for &i in open {
                    let (alt, az) = radec_to_altaz(self.ra[i], self.dec[i], lst, self.lat);
                    if alt >= self.min_alt + ALT_MARGIN_DEG {
                        out.get_mut(&i).unwrap().push((t, b, alt, az, moon.lunar_factor(self.ra[i], self.dec[i])));
                    }
                }
                t += step;
            }
        }
        out
    }

    /// Typical sky quality (realised quality / sky model) over the long history, recomputed at most hourly.
    pub(super) fn typical_quality(&mut self, now: f64) -> f64 {
        let hour = (now / 3600.0).floor() as i64;
        if let Some((h, q)) = self.long_quality {
            if h == hour {
                return q;
            }
        }
        let q = if self.long_ratios.len() >= LONG_MIN {
            let mut ordered: Vec<f64> = self.long_ratios.iter().cloned().collect();
            median(&mut ordered)
        } else {
            TYPICAL_QUALITY
        };
        let q = q.clamp(0.2, 1.5);
        self.long_quality = Some((hour, q));
        q
    }

    /// Log a request when its state changes (new, a completion, a revision, or its odds moving by 10 %).
    pub(super) fn log_request(&mut self, request: &Value, remaining: i64, q: &[f64], values: &[f64]) {
        let id = request.get("request_id").map(str_of).unwrap_or_default();
        let dist = crate::requests::poisson_binomial(q);
        let p_success: f64 = dist.iter().skip(remaining as usize).sum();
        let key = (remaining, q.len(), (p_success * 10.0).round() as i64);
        if self.request_log.get(&id) == Some(&key) {
            return;
        }
        self.request_log.insert(id.clone(), key);
        let fmt = |v: &[f64]| v.iter().map(|x| format!("{x:.2}")).collect::<Vec<_>>().join(" ");
        log(&format!(
            "request {id}: {remaining} more of {} open needed by {}; P(success) {:.2}; q [{}] value [{}]",
            q.len(),
            request.get("deadline_utc").map(str_of).unwrap_or_default(),
            p_success,
            fmt(q),
            values.iter().map(|v| format!("{:.0}", self.p.request_mult * v)).collect::<Vec<_>>().join(" ")
        ));
    }

    /// (required targets, required targets done) for the LLM week review.
    pub fn required_progress(&self) -> (usize, usize) {
        let total = self.required.iter().filter(|&&r| r).count();
        let done = (0..self.ids.len()).filter(|&i| self.required[i] && self.factor[i] >= self.req_thr).count();
        (total, done)
    }

    /// Rule-computed facts about one observation request, for the LLM request triage: time to the deadline,
    /// exposure each open target needs at typical sky, how many open targets are inside their observing window
    /// and can reach the threshold within the longest exposure.
    pub fn request_facts(&self, request: &Value, now: f64) -> Value {
        let deadline = parse_utc(request["deadline_utc"].as_str().unwrap_or(""));
        let threshold = num(&request["completion_factor_threshold"]);
        let completed: HashSet<String> =
            request.get("completed_target_ids").and_then(Value::as_array).into_iter().flatten().map(str_of).collect();
        let current = self.nights.iter().position(|&(_, e)| e > now).unwrap_or(self.nights.len()) as i64;
        let last = self.nights.iter().rposition(|&(s, _)| s < deadline).map_or(-1, |k| k as i64);
        let nights_left = self.nights.iter().filter(|&&(s, e)| e > now && s < deadline).count();
        let (mut open, mut unknown, mut in_window, mut feasible) = (0, 0, 0, 0);
        let mut needs = Vec::new();
        for target in request.get("target_ids").and_then(Value::as_array).into_iter().flatten() {
            let id = str_of(target);
            if completed.contains(&id) {
                continue;
            }
            open += 1;
            let Some(&i) = self.index_of.get(&id) else {
                unknown += 1;
                continue;
            };
            let need = threshold * self.f0t0 / (self.flux[i].max(1e-3) * TYPICAL_Q);
            needs.push(need);
            let window = self.hmax[i] > 0.0 && self.first_night[i] <= last && self.last_night[i] >= current;
            in_window += usize::from(window);
            feasible += usize::from(window && need <= self.max_exposure);
        }
        let remaining = num(request.get("remaining_count").unwrap_or(&request["minimum_completed"])).trunc() as i64;
        let needed = if needs.is_empty() { 0.0 } else { median(&mut needs.clone()) };
        let longest = needs.iter().cloned().fold(0.0, f64::max);
        json!({
            "request_id": str_of(&request["request_id"]),
            "reason": str_of(&request["reason"]).chars().take(300).collect::<String>(),
            "targets": request.get("target_ids").and_then(Value::as_array).map_or(0, |a| a.len()),
            "open_targets": open,
            "unknown_targets": unknown,
            "remaining_needed": remaining,
            "completion_factor_threshold": threshold,
            "completion_reward": num(&request["completion_reward"]),
            "hours_to_deadline": round_to((deadline - now) / 3600.0, 1),
            "nights_to_deadline": nights_left,
            "exposure_needed_s": {"median": needed.round(), "max": longest.round()},
            "max_exposure_s": self.max_exposure,
            "open_targets_in_window": in_window,
            "open_targets_feasible": feasible,
            "rule_feasible": feasible as i64 >= remaining,
        })
    }

    /// Probability that target i can still reach `threshold` inside [from, deadline]. Every run of visibility the
    /// planner itself would use (above the altitude limit with its margin, outside the published terrain sectors,
    /// sampled every REQ_STEP seconds) is one chance. A run succeeds when the sky is good enough for the exposure it
    /// allows (at most the longest exposure, at the best airmass of the run): with the sky lognormal around the
    /// typical quality (sigma REQ_Q_SIGMA over the nights of a window), P = Phi(ln(run / need) / sigma); and it
    /// survives closures with probability REQ_P_OPEN.
    pub(super) fn request_feasibility(&self, i: usize, from: f64, deadline: f64, threshold: f64, quality: f64) -> f64 {
        if self.hmax[i] <= 0.0 || self.flux[i] <= 0.0 {
            return 0.0;
        }
        let step = REQ_STEP;
        let first = self.nights.partition_point(|&(_, e)| e <= from);
        let mut miss = 1.0; // probability that no run completes the target
        for &(s, e) in self.nights[first..].iter().take_while(|&&(s, _)| s < deadline) {
            let (a, b) = (s.max(from), e.min(deadline));
            if b - a < self.min_exposure {
                continue;
            }
            // (seconds, best altitude, chance of a clear line of sight there) of the current run
            let mut run = (0.0f64, f64::MIN, 1.0f64);
            let mut close_run = |run: &mut (f64, f64, f64)| {
                let seconds = run.0.min(self.max_exposure);
                if seconds >= self.min_exposure {
                    let model = 1.0 / (self.q0 * normalized_airmass(run.1.max(1.0)).powf(self.airmass_exponent));
                    let need = threshold * self.f0t0 / (self.flux[i] * model * quality);
                    let good = normal_cdf((seconds / need).ln() / self.p.req_q_sigma) * run.2;
                    miss *= 1.0 - self.p.request_p_open * good;
                }
                *run = (0.0, f64::MIN, 1.0);
            };
            let mut t = a;
            while t < b {
                let (alt, az) = radec_to_altaz(self.ra[i], self.dec[i], local_sidereal_deg(t, self.lon), self.lat);
                let clear = self.terrain_factor(alt, az);
                if alt >= self.min_alt + ALT_MARGIN_DEG && clear > 0.0 {
                    if alt > run.1 {
                        run.1 = alt;
                        run.2 = clear;
                    }
                    run.0 += step.min(b - t);
                } else {
                    close_run(&mut run);
                }
                t += step;
            }
            close_run(&mut run);
        }
        1.0 - miss
    }

    pub(super) fn value(&self, i: usize) -> f64 {
        let f = self.factor[i];
        let damp = 0.6f64.powf(self.misses[i] as f64);
        let request = self.request_bonus.get(&i).cloned().unwrap_or(0.0);
        if self.deadline_aware(i) {
            // deadline-aware required value while not certainly done; plain science afterwards
            if self.req_open(i) {
                return (self.weight[i] * (1.0 - f * f).max(0.0) + self.required_stake(i) * self.futile_mult(i) + request) * damp;
            }
        } else if self.required[i] {
            if f >= self.req_safe {
                return (self.weight[i] * (1.0 - f * f).max(0.0) + request) * damp;
            }
            return (self.weight[i] * (1.0 - f * f) + self.p.required_bonus * (if f < self.req_thr { 1.0 } else { 0.35 }) + request) * damp;
        }
        ((if f >= DONE_FACTOR { 0.0 } else { self.weight[i] * (1.0 - f * f) }) + request) * damp
    }

    /// Required target not yet done: on its certain lower bound in deadline-aware mode, else on the estimate.
    pub(super) fn req_open(&self, i: usize) -> bool {
        self.required[i] && (if self.deadline_aware(i) { self.req_lo[i] } else { self.factor[i] } < self.req_thr)
    }

    /// Deadline-aware value of a required target: on (default) and its outlook computed.
    pub(super) fn deadline_aware(&self, i: usize) -> bool {
        self.p.req_deadline && self.req_night >= 0 && self.req_row.get(i).map_or(false, |&r| r != usize::MAX)
    }

    /// P(factor >= threshold) of an exposure of `t` seconds whose predicted factor / threshold is `raw`.
    /// Lognormal prediction error whose spread grows with the exposure (the weather drifts slot by slot):
    /// sd = sqrt(sd0^2 + (sd1 * t / 3600)^2) (empirical, fitted to required hits in simulated runs).
    pub(super) fn p_now(&self, raw: f64, t: f64) -> f64 {
        let sd = (self.p.req_sigma_now.powi(2) + (self.p.req_sigma_drift * t / 3600.0).powi(2)).sqrt().max(1e-3);
        normal_cdf(raw.max(1e-12).ln() / sd)
    }
}
