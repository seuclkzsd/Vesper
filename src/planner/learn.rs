//! Learning from results (part of the planner): unsaturated hits give the quality level (scale) and one
//! exact site-throughput reading per observe for the fault detector (read_throughput), saturated hits
//! bound the band level (band.rs posterior; a grid fit stays as the fault reader's guess), terrain
//! sectors are learnt from hits and zeros (terrain.rs), E = quality level / band level, and hit/miss
//! patterns reveal the pointing offset of Hard-mode cards.

use super::*;

impl Planner {
    /// Part of the recent data was lost: restart the factor estimates. With the per-exposure ledger (default)
    /// exactly: drop the hits of the lost observe-index window, keep each target's best remaining estimate
    /// (with PRO_RESYNC_LEDGER=0: best score / (weight x top multiplier), an under-estimate). A resync without
    /// best_scores is ignored; a target with a ledger hit newer than the resync's snapshot keeps what is known.
    pub(super) fn resync(&mut self, message: &Value) {
        let Some(rows) = message.get("best_scores").and_then(Value::as_array) else {
            log("state_resync without a best_scores list: ignored");
            return;
        };
        let best: HashMap<String, f64> = rows.iter().map(|row| (str_of(&row["target_id"]), num(&row["best_score"]))).collect();
        let top = self.multipliers.iter().cloned().fold(f64::MIN, f64::max);
        let window = message.get("invalidated_window").cloned().unwrap_or(Value::Null);
        let lost = match (window.get("action_index_start"), window.get("action_index_end_exclusive")) {
            (Some(a), Some(b)) if self.p.resync_ledger && !a.is_null() && !b.is_null() => Some((num(a) as i64, num(b) as i64)),
            _ => None,
        };
        let n = self.ids.len();
        let mut kept = vec![-1.0f64; n]; // best remaining ledger factor (-1 = none)
        if let Some((a, b)) = lost {
            self.ledger.retain(|&(_, k, _)| !(a <= k && k < b));
            for &(i, _, f) in &self.ledger {
                kept[i] = kept[i].max(f);
            }
        }
        // ledger hits newer than the resync's snapshot (none for the engine's own, timely resync)
        let snapshot_end = window.get("action_count_at_trigger").filter(|v| v.is_number()).map(|v| num(v) as i64);
        let mut newer = vec![false; n];
        if let Some(at) = snapshot_end {
            for &(i, k, _) in &self.ledger {
                if k >= at {
                    newer[i] = true;
                }
            }
        }
        let mut from_ledger = 0;
        let mut kept_newer = 0;
        for i in 0..n {
            let score = best.get(&self.ids[i]).cloned().unwrap_or(0.0);
            let estimate = if score > 0.0 { (score / (self.weight[i] * top)).min(1.0) } else { 0.0 };
            if newer[i] {
                kept_newer += 1;
                self.factor[i] = self.factor[i].max(estimate).min(1.0);
                if self.weight[i] > 0.0 {
                    self.cur[i] = self.cur[i].max(score / self.weight[i]);
                }
                self.req_lo[i] = self.req_lo[i].max(estimate);
                continue;
            }
            self.factor[i] = if score > 0.0 && lost.is_some() && kept[i] >= 0.0 {
                from_ledger += 1;
                kept[i].max(estimate).min(1.0)
            } else {
                estimate
            };
            self.cur[i] = if self.weight[i] > 0.0 { score / self.weight[i] } else { 0.0 };
            self.req_lo[i] = estimate; // best score / (weight x top multiplier): a lower bound
        }
        self.active = (0..self.ids.len()).filter(|&i| self.hmax[i] > 0.0).collect();
        self.vcache = None;
        // the observe in flight lies after the lost window: it still counts, so let on_result read its hits
        let keep_pending = matches!((lost, self.pending_index), (Some((_, b)), Some(k)) if k >= b);
        if !keep_pending {
            self.pending.clear();
        }
        log(&format!(
            "state_resync: {} targets keep a score ({} factors from the ledger); lost window {:?}; plan rebuilt",
            best.len(),
            from_ledger,
            lost
        ));
        if kept_newer > 0 {
            log(&format!("state_resync: {kept_newer} targets observed after its snapshot kept their state"));
        }
    }

    /// Update factor estimates and the sky-quality estimate from the previous observe.
    pub fn on_result(&mut self, result: &Value, hours: f64) {
        self.reading = None;
        if result.get("action").and_then(Value::as_str) != Some("observe") || self.pending.is_empty() {
            self.pending.clear();
            return;
        }
        let mut hits: HashMap<String, f64> = HashMap::new();
        for hit in result.get("hits").and_then(Value::as_array).into_iter().flatten() {
            hits.insert(str_of(&hit["target_id"]), num(&hit["score"]));
        }
        let any_positive = hits.values().any(|&s| s > 0.0);
        let declared = self.multipliers[self.pending_program];
        let pending = std::mem::take(&mut self.pending);
        // terrain evidence: (target, None = scored 0 | Some(ratio, saturated, predicted factor))
        let mut outcomes: Vec<(usize, Option<(f64, bool, f64)>)> = Vec::new();
        let mut bits: Vec<(f64, usize, bool)> = Vec::new(); // saturated hits: (sky model, program, matched)
        let mut clean_ratios: Vec<f64> = Vec::new(); // unsaturated, direction-clean hits: quality / sky model
        let mut bounds: Vec<f64> = Vec::new(); // saturated hits: quality / sky model is at least this
        for (i, prediction) in &pending {
            let i = *i;
            self.vdirty.insert(i);
            let Some(&score) = hits.get(&self.ids[i]) else {
                self.misses[i] += 1; // a miss: the target did not land on its fibre glass
                continue;
            };
            if score <= 0.0 {
                if any_positive {
                    self.blocked.push((prediction.az, prediction.alt));
                    outcomes.push((i, None));
                }
                continue;
            }
            // score = weight * factor * multiplier; the multiplier is `declared` if the program matched.
            // A saturated hit (factor = 1) shows the multiplier exactly, so it tells whether the sky's
            // program band matched the declared program. Instrument efficiency does not enter the band.
            let multiplier_seen = score / self.weight[i];
            self.cur[i] = self.cur[i].max(multiplier_seen);
            if (multiplier_seen - declared).abs() < 2e-4 {
                push_bounded(&mut self.band_obs, (hours, prediction.model, self.pending_program, true, prediction.dir_clean), 300);
                bits.push((prediction.model, self.pending_program, true));
            } else if (multiplier_seen - self.mismatch).abs() < 2e-4 {
                push_bounded(&mut self.band_obs, (hours, prediction.model, self.pending_program, false, prediction.dir_clean), 300);
                bits.push((prediction.model, self.pending_program, false));
            }
            let factor_if_match = score / (self.weight[i] * declared);
            let factor_if_miss = score / (self.weight[i] * self.mismatch);
            let ratio_match = factor_if_match * self.f0t0 / (self.flux[i] * self.pending_duration * prediction.model);
            let matched = self.band(ratio_match * prediction.band_model) == self.pending_program;
            let factor = if matched { factor_if_match } else { factor_if_miss };
            self.factor[i] = self.factor[i].max(factor.min(1.0));
            self.req_lo[i] = self.req_lo[i].max(factor_if_match.min(factor_if_miss).min(1.0));
            self.ledger.push((i, self.pending_index.unwrap_or(-1), factor.min(1.0)));
            if self.req_open(i) {
                self.attempts[i] += 1; // not enough yet: lower its priority a little for next time
                // futility guard: a long attempt on a hard target that even scaled to the longest exposure stays well
                // below the threshold; a few are unlucky nights, many mean the target is out of reach. Time-rich
                // (idle) cards only, where the time goes to the idle upside.
                let (n, x) = self.p.req_futile;
                if n > 0
                    && self.idle_card()
                    && self.req_hard[i]
                    && self.pending_duration >= 0.5 * self.max_exposure
                    && factor.min(1.0) * (self.max_exposure / self.pending_duration) / self.req_thr < x
                {
                    self.futile[i] += 1;
                    if self.futile[i] == n + 1 {
                        log(&format!("required futility: {} after {} long attempts below {:.2} x threshold", self.ids[i], n + 1, x));
                    }
                }
            }
            outcomes.push((
                i,
                Some((factor * self.f0t0 / (self.flux[i] * self.pending_duration * prediction.model), factor >= 0.97, prediction.pred)),
            ));
            if factor >= 0.999 {
                bounds.push(self.f0t0 / (self.flux[i] * self.pending_duration * prediction.model));
            }
            if factor < 0.97 {
                let ratio = factor * self.f0t0 / (self.flux[i] * self.pending_duration * prediction.model);
                push_bounded(&mut self.samples, (hours, ratio), 24);
                push_bounded(&mut self.all_ratios, ratio, 400);
                if ratio > 0.0 {
                    self.night_ln_ratios.push(ratio.ln());
                }
                push_bounded(&mut self.long_ratios, ratio, LONG_RATIOS);
                if prediction.dir_clean {
                    push_bounded(&mut self.e_ratios, (hours, ratio), 400);
                    clean_ratios.push(ratio);
                }
            }
        }
        if bounds.len() >= 2 {
            // the second highest: one sky-model error cannot set it
            bounds.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            push_bounded(&mut self.sat_bounds, (hours, bounds[1]), 24);
        }
        self.reading = Some(self.read_throughput(&pending, &hits));
        if any_positive && !self.pending_mixed {
            // the band level during this exposure: bounds from saturated hits, quality level / efficiency
            self.bandf.predict(hours, self.pending_night);
            self.bandf.update_bits(&bits);
            if clean_ratios.len() >= 3 {
                self.bandf.update_quality(median(&mut clean_ratios).max(1e-6).ln() - self.pending_eff.ln());
            }
        }
        self.pending_mixed = false;
        self.offset_evidence_update(&pending, &hits);
        if !self.terrain_beliefs.is_empty() {
            self.terrain_evidence(&pending, &outcomes);
        }
        self.update_scale(hours);
    }

    /// Feed the terrain beliefs with the targets of the last observe that lie near a terrain sector, at their
    /// mid-exposure position (a sector's edge cuts a track once, so a track blocked for less than a third of the
    /// exposure is clear at its middle). Clear: scored with a quality ratio at least 2/3 of the exposure's
    /// direction-clean median (or saturated though predicted below 1.5). Blocked: scored 0 while others scored,
    /// no rocket notice out, and inside exactly one label's widest sector.
    pub(super) fn terrain_evidence(&mut self, pending: &[(usize, Pred)], outcomes: &[(usize, Option<(f64, bool, f64)>)]) {
        // reference quality ratio: unsaturated hits of this observe outside every announced / terrain direction
        let mut clean: Vec<f64> = Vec::new();
        let clean_of: HashMap<usize, bool> = pending.iter().map(|(i, p)| (*i, p.dir_clean)).collect();
        for (i, o) in outcomes {
            if let Some((r, false, _)) = o {
                if clean_of.get(i).copied().unwrap_or(false) {
                    clean.push(*r);
                }
            }
        }
        let reference = if clean.len() >= 2 { median(&mut clean) } else { self.scale };
        let rocket = self.notices.iter().any(|(k, _)| k == "rocket_launch");
        let lst_mid = local_sidereal_deg(self.pending_start + 0.5 * self.pending_duration, self.lon);
        let mut changed = false;
        for (i, o) in outcomes {
            let (alt, az) = radec_to_altaz(self.ra[*i], self.dec[*i], lst_mid, self.lat);
            let near: Vec<usize> = (0..self.terrain_beliefs.len()).filter(|&b| self.terrain_beliefs[b].in_cone(alt, az)).collect();
            if near.is_empty() {
                continue;
            }
            match o {
                None => {
                    if !rocket && near.len() == 1 {
                        changed |= self.terrain_beliefs[near[0]].observe(alt, az, true);
                    }
                }
                Some((ratio, saturated, pred)) => {
                    let clear = if *saturated { *pred <= 1.5 } else { *ratio >= 0.67 * reference };
                    if clear {
                        for b in near {
                            changed |= self.terrain_beliefs[b].observe(alt, az, false);
                        }
                    }
                }
            }
        }
        if changed {
            self.vcache = None;
            let text: Vec<String> = self
                .terrain_beliefs
                .iter()
                .map(|b| {
                    let (clean, worst, c, hw, h) = b.summary();
                    if b.collapsed {
                        format!("{:.0}: stock cut", b.label_az)
                    } else {
                        format!(
                            "{:.0}: {} obs, {} uncontradicted, best centre {:+.1} half width {:.1} top {:.1} ({} contradictions)",
                            b.label_az, b.evidence, clean, c, hw, h, worst
                        )
                    }
                })
                .collect();
            log(&format!("terrain beliefs: {}", text.join("; ")));
        }
    }

    /// Site throughput W = efficiency x transparency x sky / seeing seen by the last exposure.
    /// An unsaturated hit has score = weight x f T Q / f0T0 x multiplier, Q = W x G / q0 (G = lunar factor /
    /// airmass^a over the engine's 120 s pieces, public), so it gives W exactly up to the multiplier (declared
    /// if its band matches, else mismatch). Saturated hits show their multiplier, which brackets the band level
    /// B = W G / (q0 eff) >= Q; a hit the bracket allows one multiplier is certain, the others follow the certain
    /// ones (all share W) or the last fitted band level. W = median over the hits.
    pub(super) fn read_throughput(&self, pending: &[(usize, Pred)], hits: &HashMap<String, f64>) -> Reading {
        let start = self.pending_start;
        let duration = self.pending_duration;
        let program = self.pending_program;
        let declared = self.multipliers[program];
        let ratio = if self.mismatch > 0.0 { declared / self.mismatch } else { 1.0 };
        let mut reading = Reading { start, duration, y: None, hits: 0, mismatched: 0, certain: 0, zero: 0, positive: 0, band_lo: 0.0, w_hi: 0.0, band_hi: f64::INFINITY, w_lo: 0.0 };
        let mut usable: Vec<(usize, f64)> = Vec::new(); // (target, score)
        let mut saturated: Vec<(usize, bool)> = Vec::new(); // (target, matched)
        for (i, prediction) in pending {
            let i = *i;
            let Some(&score) = hits.get(&self.ids[i]) else { continue };
            if score <= 0.0 {
                reading.zero += 1;
                continue;
            }
            reading.positive += 1;
            // clean at the start and under the bulletin at its end (a sector event may begin mid-exposure)
            if !prediction.sector_clean || !self.sector_clean(prediction.alt, prediction.az) || self.flux[i] <= 0.0 {
                continue;
            }
            let seen = score / self.weight[i];
            if (seen - declared).abs() < 2e-4 {
                saturated.push((i, true));
            } else if (seen - self.mismatch).abs() < 2e-4 {
                saturated.push((i, false));
            } else if duration > 0.0 {
                usable.push((i, score));
            }
        }
        if usable.is_empty() {
            return reading;
        }
        let pieces = self.exposure_pieces(start, start + duration);
        let total: f64 = psum(pieces.iter().map(|p| p.0));
        if total <= 0.0 {
            return reading;
        }
        // Q per unit W of target i over this exposure
        let model_of = |i: usize| -> f64 {
            let mut g = 0.0;
            for (seconds, lst, moon) in &pieces {
                let (alt, _) = radec_to_altaz(self.ra[i], self.dec[i], *lst, self.lat);
                g += seconds * moon.lunar_factor(self.ra[i], self.dec[i]) / normalized_airmass(alt.max(1.0)).powf(self.airmass_exponent);
            }
            g / total / self.q0
        };
        // band-level bracket [lo, hi) from the saturated hits: band(level x model) matched or not
        let (dark, bright) = (self.band_dark, self.band_bright);
        let (mut lo, mut hi) = (0.0f64, f64::INFINITY);
        for &(i, matched) in &saturated {
            let m = model_of(i);
            if !(m > 0.0) {
                continue;
            }
            match (program, matched) {
                (0, true) => lo = lo.max(dark / m),
                (0, false) => hi = hi.min(dark / m),
                (1, true) => {
                    lo = lo.max(bright / m);
                    hi = hi.min(dark / m);
                }
                (2, true) => hi = hi.min(bright / m),
                (2, false) => lo = lo.max(bright / m),
                _ => {} // BRIGHT mismatched: above or below, no bracket
            }
        }
        if lo >= hi {
            (lo, hi) = (0.0, f64::INFINITY); // inconsistent (pieces or targets span bands): no bracket
        }
        // can some band level in [a, b) put this target in a band that matches (or not) the declared program?
        let band_range = |a: f64, b: f64| -> (bool, bool) {
            if a >= b {
                return (false, false);
            }
            let in_dark = b > dark;
            let in_bright = a < dark && b > bright;
            let in_backup = a < bright;
            let bands = [in_dark, in_bright, in_backup];
            (bands[program], (0..3).any(|k| k != program && bands[k]))
        };
        // per hit: W if matched, W if mismatched, the hypothesis if only one is possible, else the band-fit guess
        struct Hyp {
            w_match: f64,
            w_miss: f64,
            certain: Option<bool>,
            guess: bool,
            calib: bool,
        }
        // terrain cuts a low target's quality, never its band: the efficiency bound uses targets that stay
        // clear of every terrain cone during the exposure
        let lsts = [start, start + 0.5 * duration, start + duration].map(|t| local_sidereal_deg(t, self.lon));
        let terrain_clear = |i: usize| -> bool {
            lsts.iter().all(|&lst| {
                let (alt, az) = radec_to_altaz(self.ra[i], self.dec[i], lst, self.lat);
                !self.terrain_beliefs.iter().any(|b| b.in_cone(alt, az))
            })
        };
        let mut hyps: Vec<Hyp> = Vec::with_capacity(usable.len());
        for &(i, score) in &usable {
            let m = model_of(i);
            if !(m > 0.0) {
                continue;
            }
            let w_match = score / (self.weight[i] * declared) * self.f0t0 / (self.flux[i] * duration * m);
            let w_miss = w_match * ratio;
            // B >= Q: under each hypothesis the band level is at least that hypothesis' W
            let (match_ok, _) = band_range(lo.max(w_match) * m, hi * m);
            let (_, miss_ok) = band_range(lo.max(w_miss) * m, hi * m);
            let certain = match (match_ok, miss_ok) {
                (true, false) => Some(true),
                (false, true) => Some(false),
                _ => None,
            };
            let guess = match self.band_fit {
                Some(level) => self.band(level.max(lo).min(hi) * m) == program,
                None => true,
            };
            hyps.push(Hyp { w_match, w_miss, certain, guess, calib: terrain_clear(i) });
        }
        if hyps.is_empty() {
            return reading;
        }
        // all hits share W: the certain ones fix it, the others take the hypothesis closest to it
        let certain: Vec<f64> =
            hyps.iter().filter_map(|h| h.certain.map(|c| if c { h.w_match } else { h.w_miss })).collect();
        reading.certain = certain.len();
        let anchor = if certain.is_empty() { None } else { Some(geo_median(certain)) };
        let mut ws = Vec::with_capacity(hyps.len());
        for h in &hyps {
            let matched = match (h.certain, anchor) {
                (Some(c), _) => c,
                (None, Some(w)) => (h.w_match / w).ln().abs() <= (h.w_miss / w).ln().abs(),
                (None, None) => h.guess,
            };
            if !matched {
                reading.mismatched += 1;
            }
            ws.push(if matched { h.w_match } else { h.w_miss });
        }
        reading.hits = ws.len();
        reading.w_hi = geo_median(hyps.iter().filter(|h| h.calib).map(|h| if h.certain == Some(true) { h.w_match } else { h.w_miss }).collect());
        reading.w_lo = geo_median(hyps.iter().filter(|h| h.calib).map(|h| if h.certain == Some(false) { h.w_miss } else { h.w_match }).collect());
        let med = geo_median(ws);
        if med > 0.0 {
            reading.y = Some(med.ln());
        }
        // Band-level bounds for the efficiency bounds (fault.rs) from multipliers that read one way only. A hit
        // showing the mismatch multiplier under DARK is saturated (W >= w_alt x declared / mismatch) or unsaturated
        // and matched (g = mismatch / declared, W = w_alt): kept only when even the matched reading of the other
        // hits puts W clearly above w_alt. BACKUP's 1.06 is too close to tell.
        let low_w = geo_median(hyps.iter().map(|h| h.w_match).collect());
        let (mut lo_c, mut hi_c) = (0.0f64, f64::INFINITY);
        for &(i, matched) in &saturated {
            let m = model_of(i);
            if !(m > 0.0) {
                continue;
            }
            match (program, matched) {
                (0, true) => lo_c = lo_c.max(dark / m),
                (0, false) if low_w >= 1.1 * self.f0t0 / (self.flux[i] * duration * m * ratio) => hi_c = hi_c.min(dark / m),
                (1, true) => {
                    lo_c = lo_c.max(bright / m);
                    hi_c = hi_c.min(dark / m);
                }
                (2, true) => hi_c = hi_c.min(bright / m),
                _ => {}
            }
        }
        if lo_c < hi_c {
            (reading.band_lo, reading.band_hi) = (lo_c, hi_c);
        }
        reading
    }

    /// The engine's integration pieces of [start, end): cut at slot boundaries, then every 120 s.
    /// Returns (seconds, local sidereal time at the piece middle, Moon at the piece middle).
    pub(super) fn exposure_pieces(&self, start: f64, end: f64) -> Vec<(f64, f64, Moon)> {
        let slot = if self.slot_seconds > 0 { self.slot_seconds as f64 } else { 900.0 };
        let night_start = self.current_night(start).map_or(start, |n| n.1);
        let mut pieces = Vec::new();
        let mut cursor = start;
        while cursor < end - 1e-6 {
            let boundary = night_start + ((cursor - night_start) / slot + 1e-9).floor() * slot + slot;
            let segment_end = end.min(boundary);
            let mut piece = cursor;
            while piece < segment_end - 1e-6 {
                let piece_end = segment_end.min(piece + 120.0);
                let mid = 0.5 * (piece + piece_end);
                let lst = local_sidereal_deg(mid, self.lon);
                pieces.push((piece_end - piece, lst, Moon::new(mid, lst, self.lat, self.lunar_model)));
                piece = piece_end;
            }
            cursor = segment_end;
        }
        pieces
    }

    // --- pointing offset (Hard-mode cards) ----------------------------------------------------------------

    /// Hard-mode cards add a hidden fixed offset to every pointing (participant guide: actual = command +
    /// (d_alt, d_az)). Each assigned target's hit or miss is evidence; keep a score for each candidate offset
    /// on a grid and adopt the best one once it clearly explains the misses better than no offset.
    pub(super) fn offset_evidence_update(&mut self, pending: &[(usize, Pred)], hits: &HashMap<String, f64>) {
        let Some(cmd) = self.pending_cmd else { return };
        if pending.is_empty() {
            return;
        }
        let rows: Vec<OffsetRow> = pending
            .iter()
            .filter_map(|(i, p)| p.fiber.map(|f| (p.alt, p.az, f, hits.contains_key(&self.ids[*i]))))
            .collect();
        if rows.is_empty() {
            return;
        }
        let missed = rows.iter().filter(|r| !r.3).count() as i64;
        self.offset_misses += missed;
        push_bounded(&mut self.offset_evidence, (cmd, rows.clone()), 400);
        if self.offset_misses < OFFSET_MIN_MISSES {
            return;
        }
        if self.offset_scores.is_none() {
            let evidence: Vec<_> = self.offset_evidence.iter().cloned().collect();
            self.rescore_offsets(&evidence);
        } else {
            self.score_offsets(cmd, &rows);
        }
        self.offset_updates += 1;
        if self.offset_updates % OFFSET_REFINE_EVERY == 1 || OFFSET_REFINE_EVERY <= 1 {
            self.refine_offset();
        }
    }

    pub(super) fn consistent(&self, cmd: (f64, f64), rows: &[OffsetRow], d_alt: f64, d_az: f64) -> i64 {
        let (c_alt, c_az) = (cmd.0 + d_alt, pmod(cmd.1 + d_az, 360.0));
        let mut ok = 0;
        for &(t_alt, t_az, fiber, hit) in rows {
            let fib = tangent_offsets(t_alt, t_az, c_alt, c_az).and_then(|(n, e)| self.grid.classify(n, e).0);
            if (fib == Some(fiber)) == hit {
                ok += 1;
            }
        }
        ok
    }

    pub(super) fn make_offset_grid(&self) -> Vec<(f64, f64)> {
        let (n, step) = (self.offset_steps, self.offset_step);
        let mut grid = Vec::new();
        for i in -n..=n {
            for j in -n..=n {
                grid.push((step * i as f64, step * j as f64));
            }
        }
        grid
    }

    pub(super) fn rescore_offsets(&mut self, evidence: &[((f64, f64), Vec<OffsetRow>)]) {
        self.offset_scores = Some(vec![0; self.offset_grid.len()]);
        for (cmd, past) in evidence {
            self.score_offsets(*cmd, past);
        }
    }

    pub(super) fn score_offsets(&mut self, cmd: (f64, f64), rows: &[OffsetRow]) {
        let add: Vec<i64> = self.offset_grid.iter().map(|&(a, z)| self.consistent(cmd, rows, a, z)).collect();
        if let Some(scores) = self.offset_scores.as_mut() {
            for (s, a) in scores.iter_mut().zip(add) {
                *s += a;
            }
        }
    }

    pub(super) fn best_offset_index(&self) -> usize {
        let scores = self.offset_scores.as_ref().expect("offset scores");
        let mut k = 0;
        for (n, &s) in scores.iter().enumerate() {
            if s > scores[k] {
                k = n;
            }
        }
        k
    }

    pub(super) fn refine_offset(&mut self) {
        let k = self.best_offset_index();
        let (mut base_alt, mut base_az) = self.offset_grid[k];
        let n = self.offset_steps;
        let edge = base_alt.abs().max(base_az.abs()) >= (n as f64 - 0.5) * self.offset_step;
        if edge && (n + 1) as f64 * self.offset_step < self.grid.fov / 2.0 {
            // the best candidate sits on the edge of the grid: the offset may be larger, widen the search
            self.offset_steps = (n as f64 * 1.5) as i64 + 1;
            self.offset_grid = self.make_offset_grid();
            let len = self.offset_evidence.len();
            let recent: Vec<_> = self.offset_evidence.iter().skip(len.saturating_sub(100)).cloned().collect();
            self.rescore_offsets(&recent);
            log(&format!("pointing offset: search widened to +-{:.2} deg", self.offset_steps as f64 * self.offset_step));
            let k = self.best_offset_index();
            (base_alt, base_az) = self.offset_grid[k];
        }
        let len = self.offset_evidence.len();
        let evidence: Vec<_> = self.offset_evidence.iter().skip(len.saturating_sub(OFFSET_FINE_EVIDENCE)).collect();
        let zero: i64 = evidence.iter().map(|(cmd, rows)| self.consistent(*cmd, rows, 0.0, 0.0)).sum();
        let mut scored = Vec::new();
        let fine = self.offset_step / 5.0;
        for i in -6..=6 {
            for j in -6..=6 {
                let (d_alt, d_az) = (base_alt + fine * i as f64, base_az + fine * j as f64);
                let s: i64 = evidence.iter().map(|(cmd, rows)| self.consistent(*cmd, rows, d_alt, d_az)).sum();
                scored.push((s, d_alt, d_az));
            }
        }
        let top = scored.iter().map(|s| s.0).max().unwrap_or(0);
        if top - zero < OFFSET_MARGIN {
            return;
        }
        // several offsets often explain the evidence equally well: take the centre of that set
        let tied: Vec<(f64, f64)> = scored.iter().filter(|s| s.0 == top).map(|s| (s.1, s.2)).collect();
        let count = tied.len() as f64;
        let centre = (
            round_to(psum(tied.iter().map(|t| t.0)) / count, 3),
            round_to(psum(tied.iter().map(|t| t.1)) / count, 3),
        );
        // uncertainty of the estimate: spread of the equally good offsets (azimuth on the sky: x cos alt of the
        // commands), plus the fine grid's own resolution
        let spread = |values: Vec<f64>| {
            let mean = psum(values.iter().cloned()) / count;
            (psum(values.iter().map(|v| (v - mean) * (v - mean))) / count).sqrt()
        };
        let cos_alt = psum(evidence.iter().map(|(cmd, _)| cmd.0.to_radians().cos())) / evidence.len().max(1) as f64;
        let sigma = spread(tied.iter().map(|t| t.0).collect()).max(spread(tied.iter().map(|t| t.1).collect()) * cos_alt)
            + fine / 12f64.sqrt();
        if (centre.0 - self.offset.0).abs() + (centre.1 - self.offset.1).abs() < 0.005 {
            if self.offset != (0.0, 0.0) {
                self.offset_sigma = Some(sigma);
            }
            return;
        }
        self.offset_sigma = Some(sigma);
        let total: usize = evidence.iter().map(|(_, rows)| rows.len()).sum();
        log(&format!(
            "pointing offset: alt {:+.3} az {:+.3} deg explains {}/{} fibre outcomes (no offset: {}; {} equally good grid points, sigma {:.4})",
            centre.0, centre.1, top, total, zero, tied.len(), sigma
        ));
        if self.offset == (0.0, 0.0) {
            self.misses = vec![0; self.ids.len()]; // the misses were the offset, not the targets
            self.vcache = None;
        }
        self.offset = centre;
    }

    /// Band level fitted to the saturated hits of the last BAND_MEMORY_HOURS (grid fit). Only the fault reader
    /// uses it (read_throughput); the program choice uses the band posterior (band.rs).
    pub(super) fn fit_band_level(&mut self, hours: f64) {
        let recent: Vec<BandObs> = self.band_obs.iter().filter(|o| o.0 >= hours - BAND_MEMORY_HOURS).cloned().collect();
        if recent.len() < 4 {
            return;
        }
        let guess = self.scale / 0.95;
        let mut best: Option<((usize, i64), f64)> = None;
        for step in -25i64..31 {
            let w = guess * 1.06f64.powf(step as f64);
            let ok = recent.iter().filter(|o| (self.band(o.1 * w) == o.2) == o.3).count();
            let key = (ok, -step.abs());
            if best.as_ref().map_or(true, |b| key > b.0) {
                best = Some((key, w));
            }
        }
        self.band_fit = best.map(|b| b.1);
    }

    /// Instrument efficiency now: 0.95 (mean jitter) x the fault signal E (clean quality level / band level; 1 when
    /// healthy) of the last three observed hours, however old (a repair clears them). The band posterior reads
    /// the band level as quality level / this.
    pub(super) fn efficiency(&self) -> f64 {
        let start = self.e_hours.len().saturating_sub(3);
        let mut recent: Vec<f64> = self.e_hours[start..].iter().flat_map(|h| h.2.iter().cloned()).collect();
        let e = if recent.len() >= 3 { median(&mut recent) } else { 1.0 };
        0.95 * e.max(0.2).min(1.0)
    }

    /// E = clean-sky quality level / clean-sky band level over the last BAND_MEMORY_HOURS (1 = consistent).
    pub(super) fn clean_e(&self, hours: f64) -> Option<f64> {
        let mut ratios: Vec<f64> = self.e_ratios.iter().filter(|(h, _)| *h >= hours - BAND_MEMORY_HOURS).map(|(_, r)| *r).collect();
        let obs: Vec<&BandObs> = self.band_obs.iter().filter(|o| o.0 >= hours - BAND_MEMORY_HOURS && o.4).collect();
        if ratios.len() < 4 || obs.len() < 4 {
            return None;
        }
        let guess = median(&mut ratios) / 0.95;
        let mut best: Option<((i64, i64), f64)> = None;
        for step in -20i64..31 {
            let w = guess * 1.06f64.powf(step as f64);
            let ok = obs.iter().filter(|o| (self.band(o.1 * w) == o.2) == o.3).count() as i64;
            let key = (ok, -step.abs());
            if best.as_ref().map_or(true, |b| key > b.0) {
                best = Some((key, w));
            }
        }
        Some((guess / best.unwrap().1).min(1.0))
    }

    /// Sky quality now = median of recent samples; fall back to the long-run median when stale.
    pub fn update_scale(&mut self, hours: f64) {
        if self.all_ratios.len() >= 8 {
            let mut ordered: Vec<f64> = self.all_ratios.iter().cloned().collect();
            self.prior_scale = median(&mut ordered);
        }
        let mut recent: Vec<f64> = self.samples.iter().filter(|(w, _)| *w >= hours - SKY_MEMORY_HOURS).map(|(_, r)| *r).collect();
        self.scale = if recent.len() >= 4 {
            median(&mut recent).max(0.05)
        } else if self.p.sat_bound {
            // Long exposures that saturate every target give no quality sample; then the stale long-run level
            // can be far too low (the time price follows it down, exposures grow, nothing unsaturates: a trap).
            let bound = self.sat_bounds.iter().filter(|(w, _)| *w >= hours - SKY_MEMORY_HOURS).map(|(_, b)| *b).fold(0.0, f64::max);
            self.prior_scale.max(bound)
        } else {
            self.prior_scale
        };
    }

    pub(super) fn band(&self, q_band: f64) -> usize {
        if q_band >= self.band_dark {
            0
        } else if q_band >= self.band_bright {
            1
        } else {
            2
        }
    }

    // --- anomaly check ---------------------------------------------------------------------------

    /// After a correct report the instrument is repaired: start the quality estimates afresh.
    pub fn forget_quality_history(&mut self) {
        self.e_hours.clear();
        self.e_ratios.clear();
        self.samples.clear();
        self.sat_bounds.clear();
        self.all_ratios.clear();
        self.long_ratios.clear();
        self.long_quality = None;
        self.prior_scale = 1.0;
    }
}
