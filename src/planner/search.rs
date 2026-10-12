//! One decision of the planner: `Planner::plan` ranks the visible targets, searches candidate fields
//! (value, request and density anchors, mean-shift seeds, an exact cell-edge shift scan, refinement) for the
//! best pointing, fibre assignment, duration and program, then re-chooses the duration at saturation times.
//! `Search` holds the per-decision caches (geometry, gains per program, memo per exposure).

use super::*;

impl Planner {

    /// Return an observe action, or None when nothing useful is up.
    pub fn plan(&mut self, now: f64, night_end: f64, night_index: usize, hours: f64) -> Option<Map<String, Value>> {
        if self.p.partial_discount < 1.0 && night_index as i64 != self.plan_night {
            self.season_plan(now, night_index);
        }
        self.update_scale(hours);
        if night_index as i64 != self.req_night {
            self.update_required_outlook(night_index);
        }
        if self.urg.as_ref().map_or(false, |u| u.night != night_index as i64) {
            self.update_urgency(night_index);
        }
        let lst = local_sidereal_deg(now, self.lon);
        if self.slp.as_ref().map_or(false, |lp| lp.night != night_index as i64) {
            self.update_slp(night_index);
        }
        if let Some(lp) = self.slp.as_mut() {
            lp.set_now(night_index, lst);
        }
        let horizon = night_end.min(self.survey_end);
        let seconds_left = horizon - now;
        if seconds_left < self.min_exposure {
            return None;
        }
        let lst_later = lst + 1800.0 * SIDEREAL_DEG_PER_SECOND;
        let moon = Moon::new(now + 600.0, lst, self.lat, self.lunar_model);
        let scale = self.scale * self.p.plan_factor_safety;
        // the fault reader's band level (stock saturated-hit fit, read_throughput)
        self.fit_band_level(hours);
        // band level now: drift since the last exposure, ALL-sky weather that started or ended
        self.bandf.predict(hours, night_index);
        self.bandf.reweather(&self.weather_now());
        let band_median = self.bandf.median();
        self.band_level = Some(band_median);
        let eff = self.efficiency();
        if self.dbg_band_night != Some(night_index) {
            self.dbg_band_night = Some(night_index);
            log(&format!("band: night {night_index} level {band_median:.3} sd {:.3} efficiency {eff:.3}", self.bandf.sd()));
        }
        let top_mult = self.multipliers.iter().cloned().fold(f64::MIN, f64::max);
        if let Some(e_now) = self.clean_e(hours) {
            let hour = hours.trunc() as i64;
            if self.e_hours.last().map_or(true, |h| h.0 != hour) {
                self.e_hours.push((hour, night_index, Vec::new()));
            }
            self.e_hours.last_mut().unwrap().2.push(e_now);
        }
        let min_up = MIN_VISIBLE_SECONDS.min(seconds_left);
        let level = self.fast_level;
        let n = self.ids.len();
        let min_up_deg = min_up * SIDEREAL_DEG_PER_SECOND;
        // level 0 breadths (pace): 0 = wide (y), 1 = xwide (*_X), 2 = mid (*_MID, the v6 wide search), 3 = narrow
        let (breadth, mid, xw) = (self.breadth.min(3), self.p.mid, self.p.xw);
        let pool_size = match (level, breadth) {
            (0, 0) => self.p.pool,
            (0, 1) => xw[4],
            _ => [mid[4], mid[4] / 2, mid[4] / 4, mid[4] / 8, mid[4] / 8][level.min(4)],
        };
        match self.vcache.take() {
            None => self.vcache = Some((0..n).map(|i| self.value(i)).collect()),
            Some(mut cache) => {
                for &i in &self.vdirty {
                    cache[i] = self.value(i);
                }
                self.vcache = Some(cache);
            }
        }
        self.vdirty.clear();
        let night_i = night_index as i64;
        // sin(alt) = A + B cos(LST) + C sin(LST): no trigonometry per target
        let (c1, s1) = (lst.to_radians().cos(), lst.to_radians().sin());
        let (c2, s2) = ((lst + min_up_deg).to_radians().cos(), (lst + min_up_deg).to_radians().sin());
        let sin_lim = self.sin_alt_limit;
        let mut visible = vec![false; n];
        let mut filler = vec![false; n];
        let mut any_visible = false;
        let mut still_active = Vec::with_capacity(self.active.len());
        let mut proxy: Vec<(f64, usize)> = Vec::new();
        let mut bins: HashMap<(i64, i64), f64> = HashMap::new();
        let mut bin_order: Vec<(i64, i64)> = Vec::new();
        let mut bin_best: HashMap<(i64, i64), (f64, usize)> = HashMap::new();
        // tiling: planning value per target (tile ranking) and the sidereal angle left tonight (hour-angle preference)
        let mut pv: Vec<f64> = if self.tiles.is_empty() { Vec::new() } else { vec![0.0; n] };
        let rem_deg = seconds_left * SIDEREAL_DEG_PER_SECOND;
        {
            let vc = self.vcache.as_ref().unwrap();
            for &i in &self.active {
                let a = self.alt_a[i];
                let sin_alt = a + self.alt_b[i] * c1 + self.alt_c[i] * s1;
                if sin_alt < sin_lim || a + self.alt_b[i] * c2 + self.alt_c[i] * s2 < sin_lim {
                    still_active.push(i); // not up now (or setting soon): keep it, check its value when it rises
                    continue;
                }
                let v = vc[i];
                if v <= 0.0 {
                    if self.p.fillers && self.weight[i] > 0.0 && self.cur[i] < top_mult - 0.01 {
                        // finished, but scored below the top program multiplier: may fill an empty fibre
                        still_active.push(i);
                        filler[i] = true;
                    }
                    continue;
                }
                still_active.push(i);
                visible[i] = true;
                any_visible = true;
                // proxy priority: planning value x a rough airmass factor x urgency (no Moon, no direction)
                let score = v * sin_alt.powf(0.6) * self.urgency_factor(i, night_i, sin_alt) * self.timing_factor(i, lst, rem_deg, night_i, sin_alt);
                proxy.push((score, i));
                if !pv.is_empty() {
                    pv[i] = score;
                }
                if self.p.n_dense > 0 {
                    // plain science still to gain here, binned on the sky at roughly one field size
                    let key = (
                        ((self.ra[i] * self.cos_dec[i]) / self.p.dense_bin_deg).floor() as i64,
                        ((self.dec[i] + 90.0) / self.p.dense_bin_deg).floor() as i64,
                    );
                    let dense_val = self.weight[i] * (1.0 - self.cur[i] / 1.2).max(0.0) * sin_alt.powf(0.6);
                    match bins.get_mut(&key) {
                        Some(total) => *total += dense_val,
                        None => {
                            bins.insert(key, dense_val);
                            bin_order.push(key);
                        }
                    }
                    if dense_val > bin_best.get(&key).map_or(0.0, |b| b.0) {
                        bin_best.insert(key, (dense_val, i));
                    }
                }
            }
        }
        self.active = still_active;
        self.filler = filler;
        if !any_visible {
            return None;
        }

        let durations_all: Vec<f64> = self.level_durations[level.min(3)].clone();
        let mut search = Search {
            pl: self,
            now,
            lst,
            lst_later,
            moon,
            scale,
            band_median,
            night_i,
            base: vec![None; n],
            info: vec![None; n],
            top_mult,
            best_rate: 0.0,
            lam: 0.0,
            durs: Vec::new(),
            gslot: vec![u32::MAX; n],
            gvals: Vec::new(),
            short_floor: self.level_durations[0][0],
            cap_memo: HashMap::new(),
            twin: None,
            touched: Vec::new(),
            rows: Vec::new(),
            marks: (0, 0),
            urg_norm: 1.0,
            tim_norm: 1.0,
            rem_deg,
            night_good: self.p.hrg && !self.ln_scale_nights.is_empty() && (self.p.hrg_good[0] <= -9.0 || (self.ln_scale_nights.len() >= 3 && self.scale.max(1e-9).ln() >= self.req_mu.0 + self.p.hrg_good[0] * self.req_mu.1)),
        };

        let pool_ranked: Vec<(f64, usize)> = if proxy.len() > pool_size {
            // (value, index) pairs are all distinct, so selecting the top pool_size first and sorting only those
            // gives exactly the stock full sort's head (cheaper with tens of thousands of visible targets)
            let mut sorted = proxy;
            let desc = |a: &(f64, usize), b: &(f64, usize)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal);
            if pool_size > 0 && pool_size < sorted.len() {
                sorted.select_nth_unstable_by(pool_size - 1, desc);
            }
            sorted.truncate(pool_size);
            sorted.sort_by(desc);
            sorted
        } else {
            proxy
        };
        if (search.pl.urg.is_some() || search.pl.slp.as_ref().map_or(false, |lp| lp.alpha > 0.0)) && search.pl.p.urg_norm {
            // urgency ranks targets competing for the same time; lambda is learned from urgency-weighted rates, so
            // keep the factors at mean 1 over the candidates on offer now (else boosted hours inflate lambda)
            let (mut boosted, mut plain) = (0.0, 0.0);
            for &(score, i) in &pool_ranked {
                let pl = search.pl;
                let u = pl.urgency_factor(i, night_i, pl.alt_a[i] + pl.alt_b[i] * c1 + pl.alt_c[i] * s1);
                if u > 0.0 {
                    boosted += score;
                    plain += score / u;
                }
            }
            if boosted > 0.0 && plain > 0.0 {
                search.urg_norm = boosted / plain;
            }
        }
        if search.pl.p.timing > 0.0 {
            // like urg_norm: the hour-angle preference only ranks the candidates on offer now (mean 1 over the
            // pool, weighted by value without it), so the learnt time price keeps its level
            let (mut with, mut plain) = (0.0, 0.0);
            for &(score, i) in &pool_ranked {
                let pl = search.pl;
                let sin_alt = pl.alt_a[i] + pl.alt_b[i] * c1 + pl.alt_c[i] * s1;
                let f = pl.timing_factor(i, lst, rem_deg, night_i, sin_alt);
                if f > 0.0 {
                    let without = score / f;
                    with += without * f;
                    plain += without;
                }
            }
            if with > 0.0 && plain > 0.0 {
                search.tim_norm = with / plain;
            }
        }
        let mut pool: Vec<usize> = pool_ranked.into_iter().map(|(_, i)| i).collect();
        search.par_map(&pool.chunks(64).collect::<Vec<_>>(), |_| 0.0, |s, c| c.iter().for_each(|&i| _ = s.exact(i)));
        search.absorb(true);
        pool.retain(|&i| search.base[i].unwrap().is_some());
        if pool.is_empty() {
            return None;
        }

        let mut durations: Vec<f64> = durations_all.iter().cloned().filter(|&t| t <= seconds_left && t >= self.p.min_t).collect();
        if durations.is_empty() {
            durations = vec![seconds_left.trunc().min(self.max_exposure)];
        }
        if self.p.req_window {
            // a request whose deadline falls inside this night: also offer the exposure that ends right at it
            // (never the case when deadlines are night ends, as the official generator makes them)
            for term in self.request_terms.values().flatten() {
                let t = (term.deadline - now).trunc();
                if t >= self.min_exposure && t <= self.max_exposure && t < seconds_left && !durations.contains(&t) {
                    durations.push(t);
                }
            }
            durations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        }
        if let Some(cap) = self.max_t {
            // a confirmation exposure for the fault detector: stay inside one slot (fault.rs), within the card's maximum
            let cap = cap.trunc().max(self.min_exposure).min(seconds_left.trunc()).min(self.max_exposure);
            durations.retain(|&t| t <= cap);
            if durations.last().map_or(true, |&t| t < cap) {
                durations.push(cap);
            }
        }
        let t_long = durations[durations.len() - 1];
        let t_mid = durations[durations.len() / 2];
        // Time cannot be saved for later: its price is never above what the current conditions earn (fast rate).
        // At the season's typical rate a cloudy or degraded hour finds no exposure worth its time, and the search
        // falls to the shortest one, whose partial gains are mostly superseded later (notes/15).
        let rate_ref = if self.p.lam_now > 0.0 && self.rate_now > 0.0 { self.rate_ema.min(self.rate_now) } else { self.rate_ema };
        let mut lam = self.lambda_frac * rate_ref * self.lambda_scale;
        if let Some(lp) = self.slp.as_ref().filter(|lp| lp.cap_w > 0.0 && lp.price_now() > 0.0 && lam > 0.0) {
            // season LP (notes/15): the EMA follows the best rate on offer, 2-3x the season's marginal price on
            // sparse catalogues (and far above it early in any season); the LP price of this node caps it there.
            // Not on dense ones: a low price makes long exposures that leave most fibres saturated (fC-s5 -5 %).
            // Where time is scarce (the season serves well under the demand) the flat LP price undercuts the rate
            // time earns all season long: there the cap stops at floor x the EMA price (notes/19). Where the
            // season serves (almost) everything, the floor only moves gains in time (fD, gX6), so none.
            let scarce = ((lp.cfg.floor_served - lp.served0) / 0.2).clamp(0.0, 1.0);
            let capped = lam.min((lp.cfg.lam * lp.price_now() * self.lambda_scale).max(lp.cfg.floor * scarce * lam));
            lam = lam.powf(1.0 - lp.cap_w) * capped.powf(lp.cap_w);
        }
        let mut rate_decay = false;
        let mut ranked: Vec<(f64, usize)> = Vec::new();
        // lambda prices a whole field; one fibre's share
        let per_fiber = if self.p.lambda_share { self.grid.n.max(1) as f64 } else { 16.0 };
        for &i in &pool {
            let (g1, t1) = search.quick(i, t_long);
            let (g2, t2) = search.quick(i, t_mid);
            let best_net = (g1 - lam * t1 / per_fiber).max(g2 - lam * t2 / per_fiber);
            if best_net > 0.0 {
                ranked.push((best_net, i));
            }
        }
        if ranked.is_empty() {
            rate_decay = true;
            lam = 0.0;
            ranked = pool.iter().map(|&i| (search.quick(i, t_long).0, i)).filter(|r| r.0 > 0.0).collect();
        }
        // idle upside (notes/41): on a time-rich card the season's work can run out months early, and the telescope
        // then waits for whole nights while hundreds of faint targets still sit below their cap. Time is free there,
        // and only an exposure measures tonight's sky: plan as if tonight were a good night (scale x exp(k sd) of
        // the nightly quality), so the best faint targets are re-taken whenever the sky turns out better than before.
        let [idle_k, idle_scarcity] = self.p.idle_up;
        self.note_idle(now, ranked.is_empty());
        if ranked.is_empty()
            && idle_k > 0.0
            && self.scarcity < idle_scarcity
            && self.idle_ready(now)
            && !self.idle_off
            && !self.bad_forecast
            && !self.site_closed()
            && self.weather_now().is_empty()
        {
            let up = (idle_k * self.req_mu.1.clamp(0.0, 0.6)).exp();
            if up > 1.0 {
                search.scale = scale * up;
                ranked = pool.iter().map(|&i| (search.quick(i, t_long).0, i)).filter(|r| r.0 > 0.0).collect();
                if !ranked.is_empty() && self.dbg_idle_night.swap(night_index, std::sync::atomic::Ordering::Relaxed) != night_index {
                    log(&format!("idle upside night {night_index}: {} candidates at scale x {up:.2}", ranked.len()));
                }
            }
        }
        if ranked.is_empty() {
            drop(search);
            if rate_decay {
                self.rate_ema *= 0.9;
            }
            return None;
        }
        search.lam = lam;
        ranked.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        // level 0 is wide, xwide, mid or narrow (the pace's first steps down); levels 1-3 are unchanged
        let n_anchors = [[self.p.n_anchors, xw[0], mid[0], self.p.n_anchors_narrow][breadth], 3, 1, 1][level.min(3)];
        let mut anchors: Vec<usize> = ranked.iter().take(n_anchors).map(|r| r.1).collect();
        // open request targets are anchors too (ranked by science they rarely make the top anchors)
        let mut n_request_anchors = 0;
        if self.p.req_anchors > 0 {
            let mut extra: Vec<(f64, usize)> = Vec::new();
            for &i in self.request_terms.keys() {
                if !visible[i] || anchors.contains(&i) {
                    continue;
                }
                let Some(b) = search.exact(i) else { continue };
                let model = self.sky_model(b.lunar, b.alt);
                let g = [t_long, t_mid].iter().map(|&t| search.request_gain(i, t.min(b.up), model) * b.mult).fold(0.0, f64::max);
                if g > 0.0 {
                    extra.push((g, i));
                }
            }
            // larger gain first; ties to the smaller index (HashMap order must not leak into the decisions)
            extra.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1)));
            n_request_anchors = extra.len().min(self.p.req_anchors);
            anchors.extend(extra.iter().take(n_request_anchors).map(|e| e.1));
        }
        // hard required targets in a good window are anchors too (notes/19), largest stake first
        if search.night_good && self.p.hrg_anchors > 0 && seconds_left >= self.max_exposure {
            let t_max = self.max_exposure;
            let mut extra: Vec<(f64, usize)> = Vec::new();
            for &i in &pool {
                let b = search.base[i].unwrap().unwrap();
                if b.good && !anchors.contains(&i) {
                    extra.push((search.quick(i, t_max).0, i));
                }
            }
            extra.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1)));
            let k = extra.len().min(self.p.hrg_anchors);
            anchors.extend(extra.iter().take(k).map(|e| e.1));
            n_request_anchors += k;
        }
        if self.p.n_dense > 0 && level <= 1 && !bins.is_empty() {
            // also try the densest patches of remaining science: fields with no single outstanding target
            let mut by_value: Vec<(f64, (i64, i64))> = bin_order.iter().map(|k| (bins[k], *k)).collect();
            by_value.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let take = if level == 0 { self.p.n_dense } else { 2 };
            for (_, key) in by_value.into_iter().take(take) {
                // a bin whose science is all done has no best target (stock: a panic, and a 900 s wait)
                let Some(&(_, j)) = bin_best.get(&key) else { continue };
                if !anchors.contains(&j) && search.exact(j).is_some() {
                    anchors.push(j);
                }
            }
        }
        let central = self.central.clone();
        let fibers: Vec<usize> = match level.min(3) {
            0 | 1 => (0..self.grid.n).collect(),
            2 => central.clone(),
            _ => vec![central[0]],
        };
        search.set_durations(&durations);
        let all_k: Vec<usize> = (0..durations.len()).collect();
        let (pitch, fov) = (self.grid.pitch, self.grid.fov);
        // a field's half diagonal: every target that can land on its glass is this close to its centre
        let field_radius = fov / std::f64::consts::SQRT_2 + 0.03;
        // the neighbourhood of an anchor must cover every field it is placed in (stock: 2.1 deg x field scale, which
        // misses the far fibres of a corner placement)
        let cover_radius = |list: &[usize]| {
            let reach = list.iter().map(|&f| {
                let (n, e) = self.grid.fiber_center(f);
                (n * n + e * e).sqrt()
            });
            reach.fold(0.0, f64::max) + field_radius
        };
        let (radius_value, radius_dense) = if self.p.near_cover {
            (cover_radius(&fibers), cover_radius(&central))
        } else {
            (self.p.neighbour_radius_deg, self.p.neighbour_radius_deg)
        };
        // (net, c_alt, c_az, T, pick, total, program)
        let mut best: Option<(f64, f64, f64, f64, Vec<(usize, usize)>, f64, usize)> = None;
        let mut best_near: Vec<usize> = Vec::new();
        let mut source = 0; // geo_stats index of the current best
        // the best distinct pointings (net, c_alt, c_az, T index, total at T) for the shift scan
        let scan_k = match level {
            0 => [self.p.scan_k[0], xw[1], mid[1], self.p.scan_k_narrow][breadth],
            1 => self.p.scan_k[1],
            _ => 0,
        };
        let mut tops: Vec<(f64, f64, f64, usize, f64, usize)> = Vec::new();
        let mut consider = |found: Found, c_alt: f64, c_az: f64, near: &Vec<usize>, best: &mut Option<(f64, f64, f64, f64, Vec<(usize, usize)>, f64, usize)>, best_near: &mut Vec<usize>, durs: &[f64]| {
            if scan_k > 0 && found.total > 0.0 {
                let k = durs.iter().position(|&t| t == found.t).unwrap_or(0);
                let cos_alt = c_alt.to_radians().cos();
                let close = |t: &(f64, f64, f64, usize, f64, usize)| {
                    let d_az = wrap180(t.2 - c_az) * cos_alt;
                    (t.1 - c_alt).abs().max(d_az.abs()) < pitch / 2.0
                };
                match tops.iter().position(close) {
                    Some(pos) if tops[pos].0 >= found.net => {}
                    Some(pos) => tops[pos] = (found.net, c_alt, c_az, k, found.total, found.program),
                    None => tops.push((found.net, c_alt, c_az, k, found.total, found.program)),
                }
                tops.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                tops.truncate(scan_k);
            }
            if best.as_ref().map_or(true, |b| found.net > b.0) {
                *best = Some((found.net, c_alt, c_az, found.t, found.pick, found.total, found.program));
                *best_near = near.clone();
            }
        };
        let (lat, min_alt) = (self.lat, self.min_alt);
        let in_range = |c_alt: f64| min_alt <= c_alt && c_alt <= 89.0;
        let n_value_anchors = anchors.len().min(n_anchors + n_request_anchors);
        // Candidate centres, built and evaluated on PRO_THREADS workers (par_map), then considered in this order: the
        // same result as one thread. (kind, target or tile): value anchors at every fibre (0); density anchors at
        // the central fibres (2: they mark a patch, not a target to centre: a few placements, then refine);
        // mean-shift seeds (1): from the densest patches, then the best anchors, a field-sized kernel climbs to the
        // local peak of remaining gain; tiles (4, DESI-style fixed centres): the best few up now, ranked by the
        // planning value of their targets
        let mut specs: Vec<(usize, usize)> = anchors.iter().enumerate().map(|(r, &a)| (if r < n_value_anchors { 0 } else { 2 }, a)).collect();
        if level == 0 && self.p.seeds > 0 {
            specs.extend(anchors[n_value_anchors..].iter().chain(&anchors[..n_value_anchors]).take(self.p.seeds).map(|&a| (1, a)));
        }
        if level == 0 && !pv.is_empty() && self.p.tile_k > 0 {
            specs.extend(self.best_tiles(&pv, c1, s1, c2, s2).into_iter().map(|k| (4, k)));
        }
        let at = |c: (f64, f64)| in_range(c.0).then(|| (round_to(c.0, 4), pmod(round_to(c.1, 4), 360.0)));
        let ha = |ra: f64| wrap180(lst - ra);
        let (ra, dec, tiles, grid, seed_iters) = (&self.ra, &self.dec, &self.tiles, &self.grid, self.p.seed_iters);
        // (near set, centres, geo_stats source, the best picks at each centre)
        let groups: Vec<(Vec<usize>, Vec<(f64, f64)>, usize, Vec<Option<Found>>)> = search
            .par_map(&specs, |&(kind, i)| ha(if kind == 4 { tiles[i].ra } else { ra[i] }), |s, &(kind, i)| {
                let (near, centres, from) = if kind == 4 {
                    let c = at(radec_to_altaz(tiles[i].ra, tiles[i].dec, lst, lat))?;
                    (s.near_set(tiles[i].ra, tiles[i].dec, field_radius, &visible), vec![c], 4)
                } else if kind == 1 {
                    let a = s.exact(i)?;
                    let near = s.near_set(ra[i], dec[i], 1.2 * fov + 0.05, &visible);
                    let weights: Vec<f64> = near.iter().map(|&j| s.quick(j, t_long).0.max(0.0)).collect();
                    let c = at(s.mean_shift((a.alt, a.az), &near, &weights, seed_iters))?;
                    (near, vec![c], 1)
                } else {
                    let a = s.exact(i)?;
                    let (list, radius): (&[usize], f64) = if kind == 0 { (&fibers, radius_value) } else { (&central, radius_dense) };
                    let centres = list.iter().filter_map(|&f| {
                        let (d_north, d_east) = grid.fiber_center(f);
                        at(shift_altaz(a.alt, a.az, -d_north, -d_east))
                    });
                    (s.near_set(ra[i], dec[i], radius, &visible), centres.collect(), 0)
                };
                let found = centres.iter().map(|&(a, z)| s.evaluate(a, z, &near, &all_k)).collect();
                Some((near, centres, from, found))
            })
            .into_iter()
            .flatten()
            .collect();
        search.absorb(true);
        for (near, centres, from, found) in groups {
            for (&(c_alt, c_az), found) in centres.iter().zip(found) {
                let Some(found) = found else { continue };
                let before = best.as_ref().map(|b| b.0);
                consider(found, c_alt, c_az, &near, &mut best, &mut best_near, &durations);
                if from > 0 && best.as_ref().map(|b| b.0) != before {
                    source = from;
                }
            }
        }
        let lst_of = |c: (f64, f64)| ha(altaz_to_radec(c.0, c.1, lst, lat).0);
        // exact shift scan of the best distinct pointings at their exposure (re-checked exactly before it can win);
        // an accepted shift is scanned again from its own centre (scan_iters rounds; narrow and level 1: one).
        // Each scan only depends on its start: run them in parallel, then take the accepted shifts in order.
        let scan_iters = if level == 0 && breadth < 3 { [self.p.scan_iters, xw[2], mid[2]][breadth] } else { 1 };
        let (scan_radius, cs_final) = (field_radius + pitch / std::f64::consts::SQRT_2 + 0.02, self.p.cs_final);
        let scans = search.par_map(&tops, |t| lst_of((t.1, t.2)), |s, &(net0, alt0, az0, k0, total0, prog0)| {
            let (mut c_alt, mut c_az, mut k, mut total, mut net, mut program) = (alt0, az0, k0, total0, net0, prog0);
            let mut steps = Vec::new();
            for _ in 0..scan_iters {
                let (ra, dec) = altaz_to_radec(c_alt, c_az, lst, lat);
                let near = s.near_set(ra, dec, scan_radius, &visible);
                let Some((dy, dx, _)) = s.shift_scan(c_alt, c_az, &near, k, program, total) else { break };
                let Some((s_alt, s_az)) = at(shift_altaz(c_alt, c_az, dy, dx)) else { break };
                let Some(found) = s.evaluate(s_alt, s_az, &near, &all_k) else { break };
                if found.net <= net + 1e-9 {
                    break;
                }
                (c_alt, c_az, net, total, program) = (s_alt, s_az, found.net, found.total, found.program);
                k = durations.iter().position(|&t| t == found.t).unwrap_or(k);
                steps.push((found, s_alt, s_az, near));
            }
            // where the scan ended (notes/15)
            let fin = (cs_final > 1).then(|| {
                let (ra, dec) = altaz_to_radec(c_alt, c_az, lst, lat);
                (net, c_alt, c_az, durations[k], s.near_set(ra, dec, scan_radius, &visible))
            });
            (steps, fin)
        });
        search.absorb(false);
        let mut finalists: Vec<(f64, f64, f64, f64, Vec<usize>)> = Vec::new();
        for (steps, fin) in scans {
            for (found, s_alt, s_az, near) in steps {
                if best.as_ref().map_or(true, |b| found.net > b.0 + 1e-9) {
                    best = Some((found.net, s_alt, s_az, found.t, found.pick, found.total, found.program));
                    best_near = near;
                    source = 2;
                }
            }
            finalists.extend(fin);
        }
        if best.is_some() && !self.p.refine_steps.is_empty() && level == 0 {
            // local search: nudge the winning pointing to catch targets near the cell edges
            let steps = self.p.refine_steps.clone();
            for _ in 0..[self.p.refine_rounds, xw[3] as i64, mid[3] as i64][breadth.min(2)] {
                let mut improved = false;
                let (b_alt, b_az) = {
                    let b = best.as_ref().unwrap();
                    (b.1, b.2)
                };
                for &(d_north, d_east) in &steps {
                    let (c_alt, c_az) = shift_altaz(b_alt, b_az, d_north, d_east);
                    if !in_range(c_alt) {
                        continue;
                    }
                    let (c_alt, c_az) = (round_to(c_alt, 4), pmod(round_to(c_az, 4), 360.0));
                    let ds: Vec<usize> = if self.p.refine_fixed_t {
                        let t = best.as_ref().unwrap().3;
                        vec![durations.iter().position(|&d| d == t).unwrap_or(0)]
                    } else {
                        all_k.clone()
                    };
                    if let Some(found) = search.evaluate(c_alt, c_az, &best_near, &ds) {
                        if found.net > best.as_ref().unwrap().0 + 1e-9 {
                            best = Some((found.net, c_alt, c_az, found.t, found.pick, found.total, found.program));
                            improved = true;
                            source = 3;
                        }
                    }
                }
                if !improved {
                    break;
                }
            }
        }
        if best.is_some() && search.pl.p.t_breaks && level == 0 {
            // duration: the gain curve of a fibre only bends down where a leading target saturates, so the best
            // duration of this pointing is one of those times (or a slot boundary, or a grid point). Complete-or-skip
            // (notes/15): T is chosen with the pointing, so the best distinct scanned pointings are re-timed too.
            let b = best.as_ref().unwrap();
            let mut runs: Vec<(f64, f64, f64, Vec<usize>)> = vec![(b.1, b.2, b.3, best_near.clone())];
            finalists.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            for (_, a, z, t, near) in finalists {
                if runs.len() < search.pl.p.cs_final && !runs.iter().any(|r| (r.0 - a).abs().max((wrap180(r.1 - z) * a.to_radians().cos()).abs()) < pitch / 2.0) {
                    runs.push((a, z, t, near));
                }
            }
            let slot = search.pl.slot_seconds as f64;
            let to_slot_end = if slot > 0.0 { slot - pmod(now - search.pl.nights[night_index].0, slot) } else { 0.0 };
            // never past a fault confirmation's slot-end cap (max_t)
            let t_cap = search.pl.max_t.map_or(seconds_left, |cap| cap.trunc().max(search.pl.min_exposure).min(seconds_left));
            // the time price keeps its calibration on the grid search: this step only places T given the price
            let rate = search.best_rate;
            let floor = search.short_floor;
            for (run, (b_alt, b_az, b_t, near)) in runs.into_iter().enumerate() {
                let ds = search.duration_candidates(b_alt, b_az, &near, &durations, b_t, to_slot_end, t_cap);
                let (short, long): (Vec<f64>, Vec<f64>) = ds.into_iter().partition(|&t| t < floor);
                // an exposure shorter than the grid (completing targets only) must also earn more than its time price
                for (ds, at_least) in [(long, f64::MIN), (short, 0.0)] {
                    if ds.is_empty() {
                        continue;
                    }
                    search.set_durations(&ds);
                    let all: Vec<usize> = (0..ds.len()).collect();
                    if let Some(found) = search.evaluate(b_alt, b_az, &near, &all) {
                        if found.net > best.as_ref().unwrap().0.max(at_least) + 1e-9 {
                            best = Some((found.net, b_alt, b_az, found.t, found.pick, found.total, found.program));
                            source = if run > 0 { 5 } else { source };
                        }
                    }
                }
            }
            search.best_rate = rate;
        }
        let best_rate_here = search.best_rate;
        let info = std::mem::take(&mut search.info);
        drop(search);
        if best.is_some() {
            self.geo_stats[source] += 1;
            let decided: u64 = self.geo_stats.iter().sum();
            if decided % 500 == 0 {
                log(&format!("geometry: chosen pointing from anchor/seed/scan/refine/tile/re-timed {:?} (level {level})", self.geo_stats));
            }
        }
        if rate_decay {
            self.rate_ema *= 0.9;
        }
        if self.p.lam_now > 0.0 {
            let a = if self.rate_now > 0.0 { self.p.lam_now } else { 1.0 };
            self.rate_now = (1.0 - a) * self.rate_now + a * best_rate_here;
        }
        self.rate_ema = if self.rate_ema > 0.0 {
            (1.0 - self.p.lambda_ema) * self.rate_ema + self.p.lambda_ema * best_rate_here
        } else {
            best_rate_here
        };
        let Some((_, c_alt, c_az, t, pick, total, program)) = best.filter(|b| b.5 > 0.0).or(None) else {
            let hour = hours.trunc() as i64;
            if self.dbg_none_hour != Some(hour) {
                self.dbg_none_hour = Some(hour);
                log(&format!("plan none: ranked={} scale={:.3} lam={:.4}", ranked.len(), self.scale, lam));
            }
            return None;
        };
        let _ = total;
        self.pending = Vec::with_capacity(pick.len());
        for &(fib, j) in &pick {
            let inf = info[j].unwrap();
            let model = inf.m0 + (inf.m1 - inf.m0) * (t / 3600.0).min(1.0);
            let dir_clean = self.direction_factor(inf.alt, inf.az) >= 1.0;
            let sector_clean = self.sector_clean(inf.alt, inf.az);
            self.pending.push((
                j,
                Pred {
                    model,
                    // stock inference of the multiplier of an unsaturated hit (the fault signal E is tuned on it)
                    band_model: model / 0.95,
                    alt: inf.alt,
                    az: inf.az,
                    pred: self.flux[j] * t * model * self.scale / self.f0t0,
                    dir_clean,
                    sector_clean,
                    fiber: Some(fib),
                },
            ));
        }
        self.pending_program = program;
        self.pending_duration = t;
        self.urg_supply += pick.len() as f64 * t;
        self.pending_start = now;
        self.pending_index = self.observe_index;
        self.pending_night = night_index;
        self.pending_weather = self.weather_now();
        self.pending_mixed = false;
        self.pending_end = now + t;
        self.pending_eff = eff;
        // the mount lands at command + offset: command the desired centre minus the learned offset
        let cmd_alt = round_to((c_alt - self.offset.0).max(0.0).min(90.0), 4);
        let mut cmd_az = round_to(pmod(c_az - self.offset.1, 360.0), 4);
        if cmd_az >= 360.0 {
            cmd_az = 0.0;
        }
        self.pending_cmd = Some((cmd_alt, cmd_az));
        let mut sorted_pick = pick.clone();
        sorted_pick.sort();
        let mut assignments = Map::new();
        for (fib, j) in sorted_pick {
            assignments.insert(fib.to_string(), Value::String(self.ids[j].clone()));
        }
        let mut action = Map::new();
        action.insert("action".into(), json!("observe"));
        action.insert("pointing".into(), json!({"alt_deg": cmd_alt, "az_deg": cmd_az}));
        action.insert("assignments".into(), Value::Object(assignments));
        action.insert("duration_seconds".into(), json!(t as i64));
        action.insert("program".into(), json!(PROGRAMS[program]));
        Some(action)
    }
}

/// Per-target geometry at the exposure start: alt, az, lunar, seconds up, gain multiplier, required-stake
/// multiplier (without urgency), unit vector.
#[derive(Clone, Copy)]
struct Base {
    alt: f64,
    az: f64,
    lunar: f64,
    up: f64,
    mult: f64,
    stake_mult: f64,
    u: [f64; 3],
    /// A hard required target in a good window (notes/19).
    good: bool,
}

/// Base plus the sky model at the start and 30 minutes later.
#[derive(Clone, Copy)]
struct Info {
    m0: f64,
    m1: f64,
    up: f64,
    mult: f64,
    stake_mult: f64,
    alt: f64,
    az: f64,
}

/// Best (net, T, program, pick, total) for one pointing.
struct Found {
    net: f64,
    t: f64,
    program: usize,
    pick: Vec<(usize, usize)>,
    total: f64,
}

/// Per-decision caches and closures of `Planner::plan` (python-pro's exact / full / gain / quick / evaluate).
struct Search<'a> {
    pl: &'a Planner,
    now: f64,
    lst: f64,
    lst_later: f64,
    moon: Moon,
    scale: f64,
    /// Posterior median of the band level (cheap ranking; the program choice uses the full posterior).
    band_median: f64,
    night_i: i64,
    /// None = not computed; Some(None) = not usable now.
    base: Vec<Option<Option<Base>>>,
    info: Vec<Option<Info>>,
    top_mult: f64,
    best_rate: f64,
    lam: f64,
    /// Mean urgency factor of this decision's candidate pool (1.0 with the stock urgency).
    urg_norm: f64,
    /// Mean hour-angle preference of the pool (1.0 when off), and the sidereal angle left tonight.
    tim_norm: f64,
    rem_deg: f64,
    /// Tonight's quality is measured (and, with PRO_HRG_TOP_Z > -9, above that z): hard required good windows (notes/19).
    night_good: bool,
    /// This decision's exposure grid, and gain3(j, durs[d]) memoised per target (slot per target, NaN = not yet);
    /// the fourth value is option_cost(j, durs[d]) (NaN = not yet).
    durs: Vec<f64>,
    gslot: Vec<u32>,
    gvals: Vec<[f64; 4]>,
    /// Exposures shorter than this (the stock grid's shortest) only count targets they complete: a short partial
    /// exposure is almost always superseded later, and its linear partial gain would make it look cheap.
    short_floor: f64,
    /// Request capacity penalties by (request, exposure seconds, targets it serves), memoised for this decision.
    cap_memo: HashMap<(usize, i64, u32), f64>,
    /// The second worker of par_map (PRO_THREADS = 2): this search with its own memos, merged back by absorb.
    twin: Option<Box<Search<'a>>>,
    /// Targets whose base this search computed, the target of each gain_k memo row, and (twin) how much of the
    /// other search's lists it has (absorb merges and syncs only those).
    touched: Vec<usize>,
    rows: Vec<usize>,
    marks: (usize, usize),
}

impl<'a> Search<'a> {
    fn exact(&mut self, i: usize) -> Option<Base> {
        if let Some(item) = self.base[i] {
            return item;
        }
        self.touched.push(i);
        let pl = self.pl;
        let ha = pmod(self.lst - pl.ra[i] + 180.0, 360.0) - 180.0;
        let (alt, az) = radec_to_altaz(pl.ra[i], pl.dec[i], self.lst, pl.lat);
        let dirf = pl.direction_factor(alt, az);
        if alt < pl.min_alt || dirf <= 0.0 {
            self.base[i] = Some(None);
            return None;
        }
        let h = pl.hmax[i];
        let up = if h < 180.0 { (h - ha) / SIDEREAL_DEG_PER_SECOND } else { 1e9 };
        // a duty-log closure about to shut this direction: up until it starts
        let up = pl.duty_closing.iter().filter(|c| az_distance(az, c.1) <= 22.5 && alt < c.2).fold(up, |u, c| u.min(c.0));
        let lunar = self.moon.lunar_factor(pl.ra[i], pl.dec[i]);
        let damp = 0.6f64.powf(pl.misses[i] as f64) * 0.8f64.powf(pl.attempts[i] as f64);
        let urgency = pl.urgency_factor(i, self.night_i, alt.to_radians().sin()) / self.urg_norm;
        let timing = pl.timing_factor(i, self.lst, self.rem_deg, self.night_i, alt.to_radians().sin());
        let hard = pl.hrg_on(i, self.night_i);
        let [_, ha_max, moon_min, raw_min] = pl.p.hrg_good;
        let t_max = pl.max_exposure;
        let half = 0.5 * t_max * SIDEREAL_DEG_PER_SECOND;
        // good window: the whole longest exposure up, its middle within ha_max hours of transit, dark, raw(T_max) high
        let good = hard && self.night_good && up >= t_max && (ha + half).abs() <= 15.0 * ha_max && lunar >= moon_min && {
            let (alt_mid, _) = radec_to_altaz(pl.ra[i], pl.dec[i], self.lst + half, pl.lat);
            pl.flux[i] * t_max * pl.sky_model(lunar, alt_mid) * pl.scale / pl.f0t0 / pl.req_thr >= raw_min
        };
        // a hard target's failed attempts were unlucky nights, not a wrong prediction: no 0.8^attempts on its stake
        // ... unless many long attempts stayed well below the threshold (futility guard, notes/41)
        let stake_mult = if hard { 0.6f64.powf(pl.misses[i] as f64) * dirf * pl.futile_mult(i) } else { damp * dirf };
        let item = Base { alt, az, lunar, up, mult: urgency * damp * dirf * timing / self.tim_norm, stake_mult, u: unit_altaz(alt, az), good };
        self.base[i] = Some(Some(item));
        Some(item)
    }

    fn full(&mut self, i: usize) -> Info {
        if let Some(item) = self.info[i] {
            return item;
        }
        let b = self.base[i].unwrap().unwrap();
        let pl = self.pl;
        let (alt2, _) = radec_to_altaz(pl.ra[i], pl.dec[i], self.lst_later, pl.lat);
        let m0 = b.lunar / (pl.q0 * normalized_airmass(b.alt.max(1.0)).powf(pl.airmass_exponent));
        let m1 = b.lunar / (pl.q0 * normalized_airmass(alt2.max(1.0)).powf(pl.airmass_exponent));
        let item = Info { m0, m1, up: b.up, mult: b.mult, stake_mult: b.stake_mult, alt: b.alt, az: b.az };
        self.info[i] = Some(item);
        item
    }

    /// [P(DARK), P(BRIGHT), P(BACKUP)] of a target with this sky model.
    fn p_band(&self, model: f64) -> [f64; 3] {
        if self.pl.p.band_prob {
            self.pl.bandf.p_band(model)
        } else {
            let mut p = [0.0; 3];
            p[self.pl.band(model * self.band_median)] = 1.0;
            p
        }
    }

    /// Expected science gain (score / weight scale times weight) of reaching `reach` under each program: the
    /// program's multiplier if the target's band matches it, the mismatch multiplier otherwise.
    fn science(&self, i: usize, reach: f64, model: f64) -> [f64; 3] {
        let pl = self.pl;
        let probs = self.p_band(model);
        let cur = pl.shaped(pl.cur[i], self.top_mult);
        let miss = (pl.shaped(reach * pl.mismatch, self.top_mult) - cur).max(0.0);
        let mut g = [0.0; 3];
        for k in 0..3 {
            let hit = (pl.shaped(reach * pl.multipliers[k], self.top_mult) - cur).max(0.0);
            g[k] = pl.weight[i] * (probs[k] * hit + (1.0 - probs[k]) * miss);
        }
        g
    }

    /// Expected gain of one exposure of duration t for target i under each program [DARK, BRIGHT, BACKUP].
    fn gain3(&mut self, i: usize, t: f64) -> [f64; 3] {
        let inf = self.full(i);
        if inf.up < t {
            return [0.0; 3];
        }
        let pl = self.pl;
        let p = &pl.p;
        let model = inf.m0 + (inf.m1 - inf.m0) * (t / 3600.0).min(1.0);
        let reach = (pl.flux[i] * t * model * self.scale / pl.f0t0).min(1.0);
        let mut g = if t < self.short_floor && reach < 1.0 { [0.0; 3] } else { self.science(i, reach, model) };
        if reach < p.partial_done && pl.planned[i] && pl.last_night[i] - self.night_i >= p.partial_nights {
            for x in &mut g {
                *x *= p.partial_discount; // it will be completed later: this partial exposure would be wasted
            }
        }
        let (mut extra, stake) = self.required_value(i, t, model); // program-independent: stock required bonus
        extra += self.request_gain(i, t, model); // ... and request bonus
        let keep = 1.0 - self.superseded(i, reach);
        g.map(|x| (x * keep + extra) * inf.mult + stake * inf.stake_mult)
    }

    /// (stock required bonus, deadline-aware required stake) of exposing target i for t seconds at sky model
    /// `model`: one-time values, paid once per required target.
    fn required_value(&self, i: usize, t: f64, model: f64) -> (f64, f64) {
        let pl = self.pl;
        let p = &pl.p;
        let mut extra = 0.0;
        let mut stake = 0.0; // deadline-aware required value (program-independent, not scaled by urgency)
        if pl.deadline_aware(i) && pl.req_open(i) {
            let raw = pl.flux[i] * t * model * pl.scale / pl.f0t0 / pl.req_thr;
            stake = pl.req_attempt(i, raw, t, self.night_i, self.base[i].flatten().map_or(false, |b| b.good));
        } else if pl.required[i] && pl.factor[i] < pl.req_thr {
            let raw = pl.flux[i] * t * model * pl.scale / pl.f0t0 / pl.req_thr;
            let mut bonus = p.required_bonus * pl.p_success(raw);
            let target_best = pl.ideal_model[i];
            if p.req_timing != 0.0 && model < p.req_timing * target_best && pl.last_night[i] - self.night_i >= p.req_timing_nights {
                bonus *= p.req_timing_discount; // a better moment for this target will come
            }
            if pl.bad_forecast && pl.last_night[i] - self.night_i >= p.req_timing_nights {
                bonus *= p.forecast_discount; // tonight is forecast bad over the whole sky
            }
            extra += bonus;
        }
        (extra, stake)
    }

    /// Fibre option cost (PRO_FOV = kappa, notes/43-night-lossA): the part of target i's future surplus that an exposure
    /// of t seconds uses up, to choose between the targets of one fibre. Surplus S = V - lam_f T: V the gain of finishing
    /// it later from its current best at this sky, T the fibre-seconds that takes (<= T_max) at one fibre's share of the
    /// time price; reaching r now leaves max(0, V (1 - r) - lam_f T) for later, so it uses min(S, V r). Times P(its cell
    /// gets later time) as complete-or-skip (the urgency's supply / demand; none without urgency, i.e. on sparse
    /// catalogues) and its density gate. A long exposure then takes the target that needs it, not a bright one that
    /// saturates early and a short exposure could finish later (the over-exposed fibre-seconds with an unfinished
    /// better-matched candidate in the cell). Fields still compare by their gains.
    fn option_cost(&mut self, i: usize, t: f64) -> f64 {
        let pl = self.pl;
        if pl.p.fov <= 0.0 {
            return 0.0;
        }
        let Some(inv) = pl.urg.as_ref().and_then(|u| u.inv(i)) else { return 0.0 };
        let gate = ((pl.fibre_density - pl.p.urg_dens_lo) / pl.p.urg_dens_span.max(1e-9)).clamp(0.0, 1.0);
        let p_later = pl.p.fov * gate * (inv * pl.grid.n as f64 * pl.p.cs_eta).min(1.0);
        let inf = self.full(i);
        if p_later <= 0.0 || inf.up < t {
            return 0.0;
        }
        let model = inf.m0 + (inf.m1 - inf.m0) * (t / 3600.0).min(1.0);
        let rate = pl.flux[i] * model * self.scale / pl.f0t0; // reach per second now
        if rate <= 0.0 {
            return 0.0;
        }
        let t_full = (1.0 / rate).min(pl.max_exposure);
        let cur = pl.shaped(pl.cur[i], self.top_mult);
        let v = pl.weight[i] * (pl.shaped((t_full * rate).min(1.0) * pl.multipliers[0], self.top_mult) - cur).max(0.0) * inf.mult;
        p_later * used_surplus(v, self.lam / pl.grid.n.max(1) as f64 * t_full, rate * t, pl.p.fov_beta)
    }

    /// Complete-or-skip (notes/15): expected share of a partial gain (reach < 1) a later exposure supersedes.
    fn superseded(&self, i: usize, reach: f64) -> f64 {
        let pl = self.pl;
        if pl.cs_p[i] <= 0.0 || reach >= 1.0 || pl.request_terms.contains_key(&i) {
            return 0.0;
        }
        pl.cs_p[i] * (1.0 - reach.max(0.0).powf(pl.p.cs_beta))
    }

    /// Gain with the start-of-exposure sky model only (no second alt/az), best program: for ranking.
    fn quick(&self, i: usize, t: f64) -> (f64, f64) {
        let b = self.base[i].unwrap().unwrap();
        let pl = self.pl;
        let p = &pl.p;
        let t = t.min(b.up);
        if t < pl.min_exposure {
            return (0.0, t);
        }
        let model = pl.sky_model(b.lunar, b.alt);
        let reach = (pl.flux[i] * t * model * self.scale / pl.f0t0).min(1.0);
        let mut g = self.science(i, reach, model).iter().cloned().fold(0.0, f64::max) * (1.0 - self.superseded(i, reach));
        if reach < p.partial_done && pl.planned[i] && pl.last_night[i] - self.night_i >= p.partial_nights {
            g *= p.partial_discount;
        }
        let mut stake = 0.0;
        if pl.deadline_aware(i) && pl.req_open(i) {
            let raw = pl.flux[i] * t * model * pl.scale / pl.f0t0 / pl.req_thr;
            stake = pl.req_attempt(i, raw, t, self.night_i, b.good);
        } else if pl.required[i] && pl.factor[i] < pl.req_thr {
            let raw = pl.flux[i] * t * model * pl.scale / pl.f0t0 / pl.req_thr;
            g += p.required_bonus * pl.p_success(raw);
        }
        g += self.request_gain(i, t, model);
        (g * b.mult + stake * b.stake_mult, t)
    }

    /// Request capacity (Params::req_cap): the value an exposure of t seconds with these fibre picks takes from the
    /// open requests by using up their window. For each request: its value x (P(capacity) if worked from now -
    /// P(capacity) from the end of this exposure, with the request targets it likely completes counted as done).
    /// Zero while the window has room; it bites when the exposure would run into the last starts the request needs
    /// (a late-rising target, or more targets left than exposures that still fit).
    fn cap_penalty(&mut self, t: f64, pick: &[(usize, usize)]) -> f64 {
        let pl = self.pl;
        if pl.request_caps.is_empty() {
            return 0.0;
        }
        let mut masks = [0u32; 8];
        for &(_, j) in pick {
            if !pl.cap_flag[j] {
                continue;
            }
            let inf = self.full(j);
            let model = inf.m0 + (inf.m1 - inf.m0) * (t / 3600.0).min(1.0);
            let reach = pl.flux[j] * t * model * pl.scale / pl.f0t0;
            for &(ci, bit) in &pl.cap_of[&j] {
                let cap = &pl.request_caps[ci];
                let inside = self.now >= cap.issued && self.now + t <= cap.deadline;
                if ci < masks.len() && inside && pl.p_success(reach / cap.threshold.max(1e-6)) >= 0.5 {
                    masks[ci] |= 1 << bit;
                }
            }
        }
        let mut penalty = 0.0;
        for (ci, cap) in pl.request_caps.iter().enumerate().take(masks.len()) {
            let key = (ci, t.round() as i64, masks[ci]);
            if let Some(&p) = self.cap_memo.get(&key) {
                penalty += p;
                continue;
            }
            let mask = masks[ci];
            let rest: Vec<&[CapSample]> =
                cap.samples.iter().enumerate().filter(|(b, _)| mask & (1 << b) == 0).map(|(_, v)| v.as_slice()).collect();
            let served = mask.count_ones() as usize;
            let after = p_capacity(&rest, self.now + t, pl.p.req_cap_step, pl.min_exposure, pl.max_exposure, cap.needed, served);
            let p = cap.value * (cap.base - after).max(0.0);
            self.cap_memo.insert(key, p);
            penalty += p;
        }
        penalty
    }

    /// Request value of exposing target i for t seconds: each request's value times the chance of reaching its
    /// threshold, counted only when the whole exposure lies inside that request's window.
    fn request_gain(&self, i: usize, t: f64, model: f64) -> f64 {
        let pl = self.pl;
        let Some(terms) = pl.request_terms.get(&i) else { return 0.0 };
        let reach = pl.flux[i] * t * model * pl.scale / pl.f0t0;
        let mut g = 0.0;
        for term in terms {
            if pl.p.req_window && !term.window_holds(self.now, t) {
                continue;
            }
            g += term.value * pl.p_success(reach / term.threshold.max(1e-6));
        }
        g
    }

    /// f(item) for every item, in item order. With PRO_THREADS = 2 this search and its twin (on a second thread)
    /// share the items. Every memo is a function of the decision state, so a result does not depend on the worker
    /// that made it: the caller sees exactly the single-thread results (notes/23).
    fn par_map<T: Sync, R: Send>(&mut self, items: &[T], key: impl Fn(&T) -> f64, f: impl Fn(&mut Search<'a>, &T) -> R + Sync) -> Vec<R> {
        if self.pl.p.threads < 2 || items.len() < 2 {
            return items.iter().map(|x| f(self, x)).collect();
        }
        let n = self.info.len();
        let mut twin = self.twin.take().unwrap_or_else(|| {
            let (touched, rows, marks) = (Vec::new(), self.rows.clone(), (self.touched.len(), self.rows.len()));
            let (base, info, durs, gslot, gvals) = (self.base.clone(), vec![None; n], self.durs.clone(), self.gslot.clone(), self.gvals.clone());
            Box::new(Search { base, info, durs, gslot, gvals, cap_memo: HashMap::new(), twin: None, touched, rows, marks, ..*self })
        });
        // the decision state set between calls: time price, normalisers, exposure grid
        (twin.lam, twin.urg_norm, twin.tim_norm) = (self.lam, self.urg_norm, self.tim_norm);
        if twin.durs != self.durs {
            twin.set_durations(&self.durs);
        }
        // items in sky order (key: hour angle): one worker takes them from the east end, the other from the west
        // end, so each mostly scores its own targets (a gain one worker memoised is not computed again by it)
        let mut order: Vec<usize> = (0..items.len()).collect();
        order.sort_by(|&a, &b| key(&items[a]).partial_cmp(&key(&items[b])).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
        let ends = std::sync::Mutex::new((0, items.len()));
        let work = |s: &mut Search<'a>, west: bool| {
            let mut out = Vec::new();
            loop {
                let k = {
                    let mut e = ends.lock().unwrap_or_else(|p| p.into_inner());
                    if e.0 >= e.1 {
                        return out;
                    }
                    if west {
                        e.1 -= 1;
                        e.1
                    } else {
                        e.0 += 1;
                        e.0 - 1
                    }
                };
                out.push((order[k], f(s, &items[order[k]])));
            }
        };
        let mut all = std::thread::scope(|sc| {
            // no second thread (spawn refused): this one takes every item
            let other = std::thread::Builder::new().spawn_scoped(sc, || {
                crate::alloc::cache(true);
                let out = work(&mut twin, true);
                crate::alloc::cache(false);
                out
            })
            .ok();
            let mut all = work(self, false);
            if let Some(h) = other {
                all.extend(h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)));
            }
            all
        });
        self.twin = Some(twin);
        all.sort_by_key(|r| r.0);
        all.into_iter().map(|r| r.1).collect()
    }

    /// Merge what the twin computed into this search (equal values where both have one); `keep`: then give the
    /// twin what this search has (bases and gains), else drop it.
    fn absorb(&mut self, keep: bool) {
        let Some(mut t) = self.twin.take() else { return };
        self.best_rate = self.best_rate.max(t.best_rate);
        for &j in &t.touched {
            if self.base[j].is_none() {
                self.base[j] = t.base[j];
                self.touched.push(j);
            }
        }
        let stride = self.durs.len();
        // gains of the same exposure grid only (the grid is set once per decision, after the first absorb)
        let rows = if t.durs == self.durs { &t.rows[t.marks.1.min(t.rows.len())..] } else { &[] };
        for (r, &j) in rows.iter().enumerate().map(|(r, j)| (r + t.marks.1, j)) {
            self.info[j] = self.info[j].or(t.info[j]);
            let to = self.slot(j);
            for d in 0..stride {
                if self.gvals[to + d][0].is_nan() {
                    self.gvals[to + d] = t.gvals[r * stride + d];
                }
            }
        }
        self.cap_memo.extend(t.cap_memo.drain());
        if keep {
            for &j in &self.touched[t.marks.0..] {
                t.base[j] = self.base[j];
            }
            t.touched.clear();
            t.durs.clone_from(&self.durs);
            t.gslot.copy_from_slice(&self.gslot);
            t.gvals.clone_from(&self.gvals);
            t.rows.clone_from(&self.rows);
            t.marks = (self.touched.len(), self.rows.len());
            self.twin = Some(t);
        }
    }

    /// Start of target j's gain_k memo row.
    fn slot(&mut self, j: usize) -> usize {
        let stride = self.durs.len();
        if self.gslot[j] == u32::MAX {
            self.gslot[j] = (self.gvals.len() / stride) as u32;
            self.rows.push(j);
            self.gvals.extend(std::iter::repeat([f64::NAN; 4]).take(stride));
        }
        self.gslot[j] as usize * stride
    }

    /// gain3(j, durs[d]), memoised for this decision (the same targets are scored at many candidate centres).
    fn gain_k(&mut self, j: usize, d: usize) -> [f64; 3] {
        let at = self.slot(j) + d;
        let cached = self.gvals[at];
        if !cached[0].is_nan() {
            return [cached[0], cached[1], cached[2]];
        }
        let g = self.gain3(j, self.durs[d]);
        self.gvals[at] = [g[0], g[1], g[2], cached[3]];
        g
    }

    /// option_cost(j, durs[d]), memoised like gain_k.
    fn option_k(&mut self, j: usize, d: usize) -> f64 {
        let at = self.slot(j) + d;
        if !self.gvals[at][3].is_nan() {
            return self.gvals[at][3];
        }
        let oc = self.option_cost(j, self.durs[d]);
        self.gvals[at][3] = oc;
        oc
    }

    /// Set this decision's exposure grid (and forget the gain_k memo of the previous one).
    fn set_durations(&mut self, durs: &[f64]) {
        self.durs = durs.to_vec();
        self.gslot.iter_mut().for_each(|s| *s = u32::MAX);
        self.gvals.clear();
        self.rows.clear();
        self.marks.1 = 0;
    }

    /// Usable visible targets (and fillers, which only take an otherwise empty fibre) within `radius` of (ra, dec).
    fn near_set(&mut self, ra: f64, dec: f64, radius: f64, visible: &[bool]) -> Vec<usize> {
        let candidates = self.pl.neighbours(ra, dec, radius);
        candidates.into_iter().filter(|&j| (visible[j] || self.pl.filler[j]) && self.exact(j).is_some()).collect()
    }

    /// The near targets that land on a fibre glass (clear of its edge) at this pointing, grouped by fibre.
    fn cells(&self, c_alt: f64, c_az: f64, near: &[usize]) -> Vec<(usize, Vec<usize>)> {
        let pl = self.pl;
        let frame = TangentFrame::new(c_alt, c_az);
        let mut slot = vec![usize::MAX; pl.grid.n];
        let mut cells: Vec<(usize, Vec<usize>)> = Vec::new();
        for &j in near {
            let b = self.base[j].unwrap().unwrap();
            let Some((dn, de)) = frame.project(&b.u) else { continue };
            let (fib, margin) = pl.grid.classify(dn, de);
            let Some(fib) = fib else { continue };
            if margin < pl.edge_margin(j) {
                continue;
            }
            if slot[fib] == usize::MAX {
                slot[fib] = cells.len();
                cells.push((fib, Vec::new()));
            }
            cells[slot[fib]].1.push(j);
        }
        cells
    }

    /// Best (net, T, program, pick, total) for one pointing, or None. `dsel` indexes the decision's exposure grid
    /// (durs). For every duration and program each fibre takes the target with the largest expected gain; a fibre
    /// without a positive gain may take a filler.
    fn evaluate(&mut self, c_alt: f64, c_az: f64, near: &[usize], dsel: &[usize]) -> Option<Found> {
        let cells = self.cells(c_alt, c_az, near);
        if cells.is_empty() {
            return None;
        }
        let mut found: Option<Found> = None;
        let mut picks: [Vec<(usize, usize)>; 3] = Default::default();
        let fov = self.pl.p.fov > 0.0 && self.pl.urg.is_some();
        for &d in dsel {
            let t = self.durs[d];
            let mut total = [0.0f64; 3];
            // the one-time values inside each total (kept out of the time price: req_rate_excl, rate_excl_req)
            let mut request_part = [0.0f64; 3];
            for p in picks.iter_mut() {
                p.clear();
            }
            for (fib, js) in &cells {
                // (rank, target, gain) of the best regular target and the best filler, per program; the rank is the
                // gain less the target's option cost (0 when off: the gain itself)
                let mut top = [(0.0f64, usize::MAX, 0.0f64); 3];
                let mut fill = [(0.0f64, usize::MAX, 0.0f64); 3];
                for &j in js {
                    let g = self.gain_k(j, d);
                    let filler = self.pl.filler[j];
                    let oc = if fov && !filler { self.option_k(j, d) } else { 0.0 };
                    let best = if filler { &mut fill } else { &mut top };
                    for k in 0..3 {
                        if fov {
                            // only targets with a gain; the largest gain less option cost, ties to the larger index
                            let r = g[k] - oc;
                            if g[k] > 0.0 && (best[k].1 == usize::MAX || r > best[k].0 || (r == best[k].0 && j > best[k].1)) {
                                best[k] = (r, j, g[k]);
                            }
                        } else if g[k] > best[k].0 || (g[k] == best[k].0 && best[k].1 != usize::MAX && j > best[k].1) {
                            // python max((gain, j)): larger gain, ties to the larger index
                            best[k] = (g[k], j, g[k]);
                        }
                    }
                }
                for k in 0..3 {
                    let (_, j, g) = if top[k].1 != usize::MAX { top[k] } else { fill[k] };
                    if j != usize::MAX && g > 0.0 {
                        total[k] += g;
                        picks[k].push((*fib, j));
                        let req = self.pl.p.rate_excl_req && self.pl.required[j];
                        if self.pl.p.req_rate_excl && (req || self.pl.request_terms.contains_key(&j)) {
                            let inf = self.full(j);
                            if inf.up >= t {
                                let model = inf.m0 + (inf.m1 - inf.m0) * (t / 3600.0).min(1.0);
                                let (bonus, stake) = if req { self.required_value(j, t, model) } else { (0.0, 0.0) };
                                request_part[k] += (self.request_gain(j, t, model) + bonus) * inf.mult + stake * inf.stake_mult;
                            }
                        }
                    }
                }
            }
            for k in 0..3 {
                if picks[k].is_empty() {
                    continue;
                }
                // the time price follows the rate of recurring science; a request reward or a required value is paid
                // once, so a field that collects one must not raise the price of every later exposure
                let rate = (total[k] - request_part[k]) / t;
                if rate > self.best_rate {
                    self.best_rate = rate;
                }
                let mut net = total[k] - self.lam * t - self.cap_penalty(t, &picks[k]);
                if self.pl.p.path_jitter != 0 {
                    net += self.pl.p.path_jitter_eps * total[k].abs() * self.jitter(c_alt, c_az, t, k);
                }
                if found.as_ref().map_or(true, |f| net > f.net) {
                    found = Some(Found { net, t, program: k, pick: picks[k].clone(), total: total[k] });
                }
            }
        }
        found
    }

    /// PRO_PATH_JITTER (notes/18): a fixed pseudo-random number in [-1, 1) per (seed, decision, pointing, T, program),
    /// so a pointing evaluated twice in one decision gets the same value.
    fn jitter(&self, c_alt: f64, c_az: f64, t: f64, k: usize) -> f64 {
        let mix = |mut z: u64| {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        let mut h = mix(self.pl.p.path_jitter);
        for x in [self.now.to_bits(), c_alt.to_bits(), c_az.to_bits(), t.to_bits(), k as u64] {
            h = mix(h ^ x);
        }
        (h >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }

    /// Mean shift of a field-sized square kernel (notes/07 #7): move the centre to the gain-weighted mean of the
    /// targets inside the field until it settles (a seed for candidate pointings; `weights[k]` is `near[k]`'s).
    fn mean_shift(&self, start: (f64, f64), near: &[usize], weights: &[f64], iters: usize) -> (f64, f64) {
        let grid = &self.pl.grid;
        let half = grid.fov / 2.0;
        let mut centre = start;
        for _ in 0..iters {
            let frame = TangentFrame::new(centre.0, centre.1);
            let (mut sw, mut sn, mut se) = (0.0, 0.0, 0.0);
            for (k, &j) in near.iter().enumerate() {
                let Some((dn, de)) = frame.project(&self.base[j].unwrap().unwrap().u) else { continue };
                if dn.abs() <= half && de.abs() <= half {
                    sw += weights[k];
                    sn += weights[k] * dn;
                    se += weights[k] * de;
                }
            }
            if sw <= 0.0 {
                break;
            }
            let (dn, de) = (sn / sw, se / sw);
            centre = shift_altaz(centre.0, centre.1, dn, de);
            if dn.abs().max(de.abs()) < grid.pitch / 20.0 {
                break;
            }
        }
        centre
    }

    /// Exact cell-edge shift scan (notes/07 #7, 07b G3). A shift (dx east, dy north) moves target j to (e_j - dx,
    /// n_j - dy); its fibre changes only where that crosses a glass edge +- margin, so within pitch/2 the sum of
    /// best gains at `durs[k]` is piecewise constant on those breakpoints: sweeping dy per dx interval visits every
    /// assignment. Returns the best (dy, dx, total) (ties: wider cell), or None when nothing beats `floor`.
    fn shift_scan(&mut self, c_alt: f64, c_az: f64, near: &[usize], k: usize, program: usize, floor: f64) -> Option<(f64, f64, f64)> {
        let pl = self.pl;
        let grid = &pl.grid;
        let side = grid.side;
        let w = grid.pitch / 2.0;
        let frame = TangentFrame::new(c_alt, c_az);
        // per target: gain, its (row, dy-interval) and (column, dx-interval) pieces inside the window
        let mut gains: Vec<f64> = Vec::new();
        let mut cols: Vec<Vec<(usize, f64, f64)>> = Vec::new();
        let mut rows: Vec<Vec<(usize, f64, f64)>> = Vec::new();
        let pieces = |offset: f64, margin: f64| -> Vec<(usize, f64, f64)> {
            let mut out = Vec::new();
            for idx in 0..side {
                if let Some((lo, hi)) = grid.glass_interval(idx, margin) {
                    // usable when offset - shift lies in [lo, hi]
                    let (a, b) = ((offset - hi).max(-w), (offset - lo).min(w));
                    if a < b {
                        out.push((idx, a, b));
                    }
                }
            }
            out
        };
        for &j in near {
            let Some((dn, de)) = frame.project(&self.base[j].unwrap().unwrap().u) else { continue };
            if dn.abs() > grid.fov / 2.0 + w || de.abs() > grid.fov / 2.0 + w {
                continue;
            }
            let g = self.gain_k(j, k)[program];
            if g <= 0.0 {
                continue;
            }
            let margin = pl.edge_margin(j);
            let (r, c) = (pieces(dn, margin), pieces(de, margin));
            if r.is_empty() || c.is_empty() {
                continue;
            }
            gains.push(g);
            rows.push(r);
            cols.push(c);
        }
        if gains.is_empty() {
            return None;
        }
        let breakpoints = |sets: &[Vec<(usize, f64, f64)>]| -> Vec<f64> {
            let mut p: Vec<f64> = vec![-w, w];
            for set in sets {
                for &(_, a, b) in set {
                    p.push(a);
                    p.push(b);
                }
            }
            p.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            p.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
            p
        };
        let (px, py) = (breakpoints(&cols), breakpoints(&rows));
        // dy events (position, +1 start / -1 end, target, row), sorted by position
        let mut events: Vec<(f64, i32, usize, usize)> = Vec::new();
        for (t, set) in rows.iter().enumerate() {
            for &(r, a, b) in set {
                events.push((a, 1, t, r));
                events.push((b, -1, t, r));
            }
        }
        events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let nf = grid.n;
        let mut members: Vec<Vec<usize>> = vec![Vec::new(); nf];
        let mut cell_max = vec![0.0f64; nf];
        let mut col_of = vec![usize::MAX; gains.len()];
        let mut best: Option<(f64, f64, f64, f64)> = None; // (total, width, dy, dx)
        for xi in 0..px.len() - 1 {
            let dx = 0.5 * (px[xi] + px[xi + 1]);
            let wx = px[xi + 1] - px[xi];
            for (t, set) in cols.iter().enumerate() {
                col_of[t] = set.iter().find(|&&(_, a, b)| a <= dx && dx <= b).map_or(usize::MAX, |p| p.0);
            }
            for m in members.iter_mut() {
                m.clear();
            }
            cell_max.iter_mut().for_each(|v| *v = 0.0);
            let mut total = 0.0;
            let mut e = 0;
            for yi in 0..py.len() - 1 {
                // apply every event at this breakpoint (an interval [a, b] is open on (a, b) between breakpoints)
                while e < events.len() && events[e].0 <= py[yi] + 1e-9 {
                    let (_, kind, t, r) = events[e];
                    e += 1;
                    let c = col_of[t];
                    if c == usize::MAX {
                        continue;
                    }
                    let cell = r * side + c;
                    if kind > 0 {
                        members[cell].push(t);
                        if gains[t] > cell_max[cell] {
                            total += gains[t] - cell_max[cell];
                            cell_max[cell] = gains[t];
                        }
                    } else if let Some(pos) = members[cell].iter().position(|&x| x == t) {
                        members[cell].swap_remove(pos);
                        let top = members[cell].iter().map(|&x| gains[x]).fold(0.0, f64::max);
                        total += top - cell_max[cell];
                        cell_max[cell] = top;
                    }
                }
                let width = wx.min(py[yi + 1] - py[yi]);
                let better = match best {
                    None => true,
                    Some((bt, bw, _, _)) => total > bt + 1e-9 || (total > bt - 1e-9 && width > bw),
                };
                if better {
                    best = Some((total, width, 0.5 * (py[yi] + py[yi + 1]), dx));
                }
            }
        }
        let (total, _, dy, dx) = best?;
        if total <= floor + 1e-9 {
            return None;
        }
        Some((dy, dx, total))
    }

    /// Duration candidates for one pointing: the grid, the next slot boundaries, and the saturation time (with a
    /// margin) of the two leading targets of every fibre at `t_ref`: between those, every fibre's gain is linear
    /// in T, so the best net gain sits on one of them.
    fn duration_candidates(&mut self, c_alt: f64, c_az: f64, near: &[usize], grid: &[f64], t_ref: f64, to_slot_end: f64, t_max: f64) -> Vec<f64> {
        let pl = self.pl;
        let mut out: Vec<f64> = grid.to_vec();
        let slot = pl.slot_seconds as f64;
        if slot > 0.0 && to_slot_end > 0.0 {
            out.extend((0..4).map(|k| to_slot_end + k as f64 * slot));
        }
        for (_, js) in self.cells(c_alt, c_az, near) {
            let mut ranked: Vec<(f64, usize)> = js.iter().map(|&j| (self.gain3(j, t_ref).iter().cloned().fold(0.0, f64::max), j)).collect();
            ranked.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            for &(g, j) in ranked.iter().take(2) {
                if g <= 0.0 {
                    continue;
                }
                let inf = self.full(j);
                // reach(T) = flux * T * model(T) * scale / f0t0 = margin, model(T) = m0 + (m1 - m0) * min(T / 3600, 1)
                let rate = pl.flux[j].max(1e-9) * self.scale.max(1e-9) / pl.f0t0;
                let mut t = pl.p.t_sat_margin / (rate * inf.m0.max(1e-9));
                for _ in 0..4 {
                    t = pl.p.t_sat_margin / (rate * (inf.m0 + (inf.m1 - inf.m0) * (t / 3600.0).min(1.0)).max(1e-9));
                }
                out.push(t.ceil());
            }
        }
        let hi = t_max.min(pl.max_exposure).trunc();
        let mut out: Vec<f64> = out.into_iter().filter(|t| t.is_finite()).map(|t| t.trunc().max(pl.min_exposure).min(hi)).collect();
        out.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        out.dedup();
        out
    }
}

/// The part of a target's future surplus S = v - cost that an exposure reaching `reach` now uses up (option_cost):
/// v the value of finishing it later, cost the fibre time that takes at the time price. Reaching r leaves
/// max(0, v (1 - r) - cost) for later, so beta = 0 uses min(S, v r); beta > 0 uses S r^beta (only near-complete
/// exposures use it up).
pub(crate) fn used_surplus(v: f64, cost: f64, reach: f64, beta: f64) -> f64 {
    let s = v - cost;
    if s <= 0.0 {
        return 0.0;
    }
    let reach = reach.clamp(0.0, 1.0);
    if beta > 0.0 {
        s * reach.powf(beta)
    } else {
        s.min(v * reach)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/option_cost.rs"]
mod tests;
