//! DESI-style tiling: fixed field centres laid out once at initialize (a hexagonal lattice per pass,
//! passes offset from each other), offered to the search as extra candidate pointings; and the time preference
//! of DESI's next-field selector (observe a field near its best hour angle), per target.

use super::*;

/// A fixed field centre; sin(alt) = a + b cos(LST) + c sin(LST); catalogue targets within the field's equal-area
/// radius (the field turns with the parallactic angle, so a disc is what every orientation covers on average).
pub(super) struct Tile {
    pub ra: f64,
    pub dec: f64,
    a: f64,
    b: f64,
    c: f64,
    pub members: Vec<u32>,
}

impl Planner {
    /// `tile_passes` hexagonal lattices of centres `tile_spacing` x field side apart, pass p shifted by p x
    /// (0.618, 0.382) of the spacing (golden-ratio offsets keep every pass filling the others' gaps). Only lattice
    /// points next to a catalogue target are kept.
    pub(super) fn build_tiles(&mut self) {
        let passes = self.p.tile_passes;
        if passes == 0 || self.grid.fov <= 0.0 {
            return;
        }
        let step = self.p.tile_spacing.max(0.2) * self.grid.fov;
        let row = step * 3f64.sqrt() / 2.0;
        let radius = self.grid.fov / std::f64::consts::PI.sqrt();
        let (sin_lat, cos_lat) = (self.lat.to_radians().sin(), self.lat.to_radians().cos());
        let mut keys: BTreeSet<(usize, i64, i64)> = BTreeSet::new();
        for p in 0..passes {
            let (ox, oy) = ((p as f64 * 0.618_034).fract(), (p as f64 * 0.381_966).fract());
            for &i in &self.active {
                let k0 = (self.dec[i] / row - oy).round() as i64;
                for k in k0 - 1..=k0 + 1 {
                    let dec = (k as f64 + oy) * row;
                    if dec.abs() > 89.0 {
                        continue;
                    }
                    let n_ra = (360.0 * dec.to_radians().cos() / step).ceil().max(1.0);
                    let shift = ox + if k % 2 != 0 { 0.5 } else { 0.0 };
                    let m = (self.ra[i] / (360.0 / n_ra) - shift).round().rem_euclid(n_ra) as i64;
                    keys.insert((p, k, m));
                }
            }
        }
        for (p, k, m) in keys {
            let (ox, oy) = ((p as f64 * 0.618_034).fract(), (p as f64 * 0.381_966).fract());
            let dec = (k as f64 + oy) * row;
            let n_ra = (360.0 * dec.to_radians().cos() / step).ceil().max(1.0);
            let shift = ox + if k % 2 != 0 { 0.5 } else { 0.0 };
            let ra = ((m as f64 + shift) * 360.0 / n_ra).rem_euclid(360.0);
            let members: Vec<u32> = self
                .neighbours(ra, dec, radius)
                .into_iter()
                .filter(|&j| crate::skymath::separation_deg(ra, dec, self.ra[j], self.dec[j]) <= radius)
                .map(|j| j as u32)
                .collect();
            if members.is_empty() {
                continue;
            }
            let (sd, cd) = (dec.to_radians().sin(), dec.to_radians().cos());
            let (a, b, c) = (sin_lat * sd, cos_lat * cd * ra.to_radians().cos(), cos_lat * cd * ra.to_radians().sin());
            self.tiles.push(Tile { ra, dec, a, b, c, members });
        }
        log(&format!("tiling: {} tiles in {passes} passes, spacing {step:.3} deg, radius {radius:.3} deg", self.tiles.len()));
    }

    /// Last night whose observing window contains the target's transit (-1: none), for `timing_factor`.
    pub(super) fn build_transit_nights(&mut self) {
        if self.p.timing <= 0.0 {
            return;
        }
        let spans: Vec<(f64, f64)> =
            self.nights.iter().map(|&(s, e)| (local_sidereal_deg(s, self.lon), (e - s) * SIDEREAL_DEG_PER_SECOND)).collect();
        self.last_transit = vec![-1; self.ra.len()];
        for &i in &self.active {
            if let Some(k) = spans.iter().rposition(|&(l0, span)| pmod(self.ra[i] - l0, 360.0) <= span) {
                self.last_transit[i] = k as i32;
            }
        }
    }

    /// DESI's next-field selector prefers a field near its design hour angle. Per target: (sin alt now / sin of the
    /// best altitude still to come)^(airmass exponent x TIMING), the best still to come being the transit when the
    /// target transits inside a later night's window. A rising band that enters its season yields to targets at
    /// their best now (at transit, setting, or leaving the season). Request targets keep 1 (their window is short).
    /// The search normalises the factors over each decision's candidates (search.rs tim_norm), so they rank fields
    /// and leave the time price alone. Only the season part is used: preferring the best altitude left tonight would
    /// pile rising fields onto the dawn hours.
    pub(super) fn timing_factor(&self, i: usize, _lst: f64, _rem_deg: f64, night_i: i64, sin_alt: f64) -> f64 {
        if self.p.timing <= 0.0 || self.last_transit.is_empty() || sin_alt <= 0.0 || self.request_terms.contains_key(&i) {
            return 1.0;
        }
        let amp = (self.alt_b[i] * self.alt_b[i] + self.alt_c[i] * self.alt_c[i]).sqrt();
        let best = if self.last_transit[i] as i64 > night_i { self.alt_a[i] + amp } else { sin_alt };
        if best <= sin_alt {
            1.0
        } else {
            (sin_alt / best).powf(self.airmass_exponent * self.p.timing)
        }
    }

    /// The `tile_k` tiles with the most planning value (`pv`: proxy per target, 0 when not usable) among those up
    /// now and for the next `min_up_deg` of sidereal time, at least half a spacing apart.
    pub(super) fn best_tiles(&self, pv: &[f64], c1: f64, s1: f64, c2: f64, s2: f64) -> Vec<usize> {
        let mut scored: Vec<(f64, usize)> = Vec::new();
        for (k, t) in self.tiles.iter().enumerate() {
            if t.a + t.b * c1 + t.c * s1 < self.sin_alt_limit || t.a + t.b * c2 + t.c * s2 < self.sin_alt_limit {
                continue;
            }
            let v: f64 = t.members.iter().map(|&j| pv[j as usize]).sum();
            if v > 0.0 {
                scored.push((v, k));
            }
        }
        scored.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let gap = 0.5 * self.p.tile_spacing * self.grid.fov;
        let mut out: Vec<usize> = Vec::new();
        for (_, k) in scored {
            if out.len() >= self.p.tile_k {
                break;
            }
            let t = &self.tiles[k];
            if out.iter().all(|&o| crate::skymath::separation_deg(t.ra, t.dec, self.tiles[o].ra, self.tiles[o].dec) >= gap) {
                out.push(k);
            }
        }
        out
    }
}
