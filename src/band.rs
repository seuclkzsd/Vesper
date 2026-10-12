//! Program band of each target: a grid posterior over the atmospheric band level W.
//!
//! The scorer bands a target by B = (1/T) * integral of D * tau*K*L/(s*X^beta) / q0 dt (v4_scorer.py:248-303):
//! the public sky model of the target (Moon L, airmass X) times an atmospheric level W = tau*K/s that carries the
//! ALL-sky weather but neither the instrument efficiency nor any directional event. So B_j = W * model_j, and the
//! planner needs P(band = p) for each target, not a point estimate of W. It keeps a posterior over ln W on a
//! fixed grid:
//!   - between exposures W drifts (AR(1) per slot within a night, a new night level each night): diffuse;
//!   - an ALL-sky overcast / haze / cold_snap notice that starts (or ends) scales W by its multiplier: shift;
//!   - a saturated hit shows its program multiplier exactly, i.e. whether B fell in the declared band: a soft
//!     one- or two-sided bound on W;
//!   - unsaturated hits measure Q = eff * W * model, so W = (Q / model) / eff, with the efficiency eff taken from
//!     the planner's fault signal. eff is the weak link (an unreported fault or an earthquake lowers it without
//!     notice), so this measurement is a Gaussian plus a shelf above it: W may be higher than it says, never
//!     lower than Q / model (eff <= 1). Where the two disagree, the saturated hits (exact) decide.
//! Each exposure counts as one observation with a small outlier probability (an unannounced closure during the
//! exposure lowers B for that exposure only), so one odd exposure cannot pin the posterior.

use std::collections::BTreeSet;

const LN_LO: f64 = -4.0; // W = 0.018
const STEP: f64 = 0.02;
const N: usize = 320; // up to W = 10.8
/// Within-night log sd of W and its slot-to-slot correlation (reference weather: seeing 6 %, transparency 3 %,
/// sky 3.5 %, phi 0.88).
const SLOT_SIGMA: f64 = 0.08;
const SLOT_PHI: f64 = 0.88;
/// Change of W from one night to the next (night-level AR(1) innovation plus the reset of the slot deviation).
const NIGHT_SIGMA: f64 = 0.16;
/// A gap this long between two updates is a new night (or hours of waiting): use the night spread.
const NIGHT_GAP_HOURS: f64 = 3.0;
/// Soft edge of a band bit: exposure-mean sky model vs the scorer's integral, W drift within the exposure.
const BIT_SIGMA: f64 = 0.05;
/// Spread of ln W measured from unsaturated hits (efficiency jitter, model error).
const QUALITY_SIGMA: f64 = 0.08;
/// Probability that one exposure's saturated hits say nothing about W now (an unannounced closure, model error).
const BITS_OUTLIER: f64 = 0.02;
/// The same for the quality level (any surprise in the efficiency beyond the shelf below).
const QUALITY_OUTLIER: f64 = 0.05;
/// Weight of the shelf above the quality level: the efficiency may be well below the estimate (unreported fault,
/// earthquake), so W may lie anywhere from 5 % to 10x above the level the quality alone says.
const QUALITY_SHELF: f64 = 0.3;
const SHELF_FROM: f64 = 0.05;
const SHELF_TO: f64 = 2.3;
/// ALL-sky, non-closing weather: (event kind, mean ln multiplier of tau*K/s, its sd). Reference weather config:
/// cloudy tau 0.55 K 0.65 s 1.25 at full severity, severity U(0.55, 1) -> tau*K/s in [0.29, 0.53]; smoggy
/// [0.43, 0.64]; cold wave [0.90, 0.94]. Wrong values only cost a few exposures: the data take over.
const ALL_SKY_SHIFTS: [(&str, f64, f64); 3] = [("overcast", -0.93, 0.18), ("haze", -0.64, 0.12), ("cold_snap", -0.08, 0.02)];

/// Standard normal CDF (Abramowitz-Stegun 7.1.26, |error| < 1.5e-7).
pub fn phi(x: f64) -> f64 {
    let z = x.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.3275911 * z);
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let erf = 1.0 - poly * (-z * z).exp();
    if x >= 0.0 {
        0.5 * (1.0 + erf)
    } else {
        0.5 * (1.0 - erf)
    }
}

pub struct BandFilter {
    post: Vec<f64>,
    /// cum[k] = posterior mass below grid cell k (N + 1 entries).
    cum: Vec<f64>,
    ln_dark: f64,
    ln_bright: f64,
    /// Time (hours since the start) and night of the last update.
    hours: Option<f64>,
    night: Option<usize>,
    slot_hours: f64,
    /// ALL-sky weather kinds the posterior is conditioned on.
    weather: BTreeSet<String>,
}

impl BandFilter {
    pub fn new(band_dark: f64, band_bright: f64, slot_seconds: f64) -> BandFilter {
        // broad prior around a typical clear sky (tau*K/s ~ 0.8)
        let (mu, sd) = (0.8f64.ln(), 0.5);
        let post = (0..N).map(|k| (-(Self::x(k) - mu).powi(2) / (2.0 * sd * sd)).exp()).collect();
        let mut f = BandFilter {
            post,
            cum: vec![0.0; N + 1],
            ln_dark: band_dark.max(1e-9).ln(),
            ln_bright: if band_bright > 0.0 { band_bright.ln() } else { f64::NEG_INFINITY },
            hours: None,
            night: None,
            slot_hours: (if slot_seconds > 0.0 { slot_seconds } else { 900.0 }) / 3600.0,
            weather: BTreeSet::new(),
        };
        f.normalize();
        f
    }

    fn x(k: usize) -> f64 {
        LN_LO + k as f64 * STEP
    }

    fn normalize(&mut self) {
        let total: f64 = self.post.iter().sum();
        if !(total > 0.0) || !total.is_finite() {
            self.post = vec![1.0 / N as f64; N];
        } else {
            for p in &mut self.post {
                *p /= total;
            }
        }
        let mut c = 0.0;
        for k in 0..N {
            self.cum[k] = c;
            c += self.post[k];
        }
        self.cum[N] = c;
    }

    /// P(ln W < x): each grid mass spread evenly over its cell.
    fn cdf(&self, x: f64) -> f64 {
        let u = (x - LN_LO) / STEP + 0.5;
        if u <= 0.0 {
            return 0.0;
        }
        if u >= N as f64 {
            return 1.0;
        }
        let i = u as usize;
        self.cum[i] + (u - i as f64) * self.post[i]
    }

    /// [P(DARK), P(BRIGHT), P(BACKUP)] for a target whose band level is W * model.
    pub fn p_band(&self, model: f64) -> [f64; 3] {
        if !(model > 0.0) {
            return [0.0, 0.0, 1.0];
        }
        let lm = model.ln();
        let dark = 1.0 - self.cdf(self.ln_dark - lm);
        let backup = if self.ln_bright.is_finite() { self.cdf(self.ln_bright - lm) } else { 0.0 };
        [dark, (1.0 - dark - backup).max(0.0), backup]
    }

    /// Median of W.
    pub fn median(&self) -> f64 {
        let k = self.cum.partition_point(|&c| c < 0.5).saturating_sub(1).min(N - 1);
        Self::x(k).exp()
    }

    /// Posterior sd of ln W.
    pub fn sd(&self) -> f64 {
        let mean: f64 = (0..N).map(|k| self.post[k] * Self::x(k)).sum();
        (0..N).map(|k| self.post[k] * (Self::x(k) - mean).powi(2)).sum::<f64>().sqrt()
    }

    fn diffuse(&mut self, sd: f64) {
        if !(sd > 0.25 * STEP) {
            return;
        }
        let half = ((3.0 * sd / STEP).ceil() as usize).min(N);
        let kernel: Vec<f64> = (0..=2 * half).map(|i| (-((i as f64 - half as f64) * STEP).powi(2) / (2.0 * sd * sd)).exp()).collect();
        let mut out = vec![0.0; N];
        for (k, &p) in self.post.iter().enumerate() {
            if p < 1e-15 {
                continue;
            }
            let lo = k.saturating_sub(half);
            let hi = (k + half).min(N - 1);
            let mut norm = 0.0;
            for j in lo..=hi {
                norm += kernel[j + half - k];
            }
            for j in lo..=hi {
                out[j] += p * kernel[j + half - k] / norm;
            }
        }
        self.post = out;
        self.normalize();
    }

    fn shift(&mut self, d: f64) {
        let s = d / STEP;
        let mut out = vec![0.0; N];
        for (k, &p) in self.post.iter().enumerate() {
            let u = (k as f64 + s).max(0.0).min((N - 1) as f64);
            let i = u as usize;
            let frac = u - i as f64;
            out[i] += p * (1.0 - frac);
            if i + 1 < N {
                out[i + 1] += p * frac;
            }
        }
        self.post = out;
        self.normalize();
    }

    /// Let W drift from the last update to `hours` (in `night`).
    pub fn predict(&mut self, hours: f64, night: usize) {
        if let Some(last) = self.hours {
            let gap = (hours - last).max(0.0);
            let new_night = self.night.map_or(false, |n| n != night) || gap >= NIGHT_GAP_HOURS;
            let sd = if new_night {
                NIGHT_SIGMA
            } else {
                let slots = gap / self.slot_hours;
                SLOT_SIGMA * (1.0 - SLOT_PHI.powf(2.0 * slots)).max(0.0).sqrt()
            };
            self.diffuse(sd);
        }
        self.hours = Some(hours);
        self.night = Some(night);
    }

    /// The ALL-sky weather now in force: shift W for every kind that started or ended since the last call.
    pub fn reweather(&mut self, kinds: &BTreeSet<String>) {
        for &(kind, mean, sd) in &ALL_SKY_SHIFTS {
            let (was, is) = (self.weather.contains(kind), kinds.contains(kind));
            if was != is {
                self.shift(if is { mean } else { -mean });
                self.diffuse(sd);
            }
        }
        self.weather = kinds.iter().filter(|k| ALL_SKY_SHIFTS.iter().any(|s| s.0 == k.as_str())).cloned().collect();
    }

    fn apply(&mut self, log_like: &[f64], outlier: f64) {
        let top = log_like.iter().cloned().fold(f64::MIN, f64::max);
        if !top.is_finite() {
            return;
        }
        for k in 0..N {
            self.post[k] *= outlier + (1.0 - outlier) * (log_like[k] - top).exp();
        }
        self.normalize();
    }

    /// Saturated hits of one exposure: (sky model, declared program 0/1/2, multiplier matched).
    pub fn update_bits(&mut self, bits: &[(f64, usize, bool)]) {
        if bits.is_empty() {
            return;
        }
        let mut log_like = vec![0.0; N];
        for &(model, program, matched) in bits {
            if !(model > 0.0) {
                continue;
            }
            let lm = model.ln();
            for (k, ll) in log_like.iter_mut().enumerate() {
                let lb = Self::x(k) + lm; // ln B
                let above_dark = phi((lb - self.ln_dark) / BIT_SIGMA);
                let above_bright = if self.ln_bright.is_finite() { phi((lb - self.ln_bright) / BIT_SIGMA) } else { 1.0 };
                let p = match program {
                    0 => above_dark,
                    1 => (above_bright - above_dark).max(0.0),
                    _ => 1.0 - above_bright,
                };
                *ll += (if matched { p } else { 1.0 - p }).max(1e-9).ln();
            }
        }
        self.apply(&log_like, BITS_OUTLIER);
    }

    /// ln W measured from the unsaturated hits of one exposure (quality level / estimated efficiency).
    pub fn update_quality(&mut self, ln_w: f64) {
        if !ln_w.is_finite() {
            return;
        }
        let peak = (1.0 - QUALITY_SHELF) / (QUALITY_SIGMA * (2.0 * std::f64::consts::PI).sqrt());
        let shelf = QUALITY_SHELF / (SHELF_TO - SHELF_FROM);
        let log_like: Vec<f64> = (0..N)
            .map(|k| {
                let d = Self::x(k) - ln_w;
                let mut like = peak * (-d * d / (2.0 * QUALITY_SIGMA * QUALITY_SIGMA)).exp();
                if (SHELF_FROM..SHELF_TO).contains(&d) {
                    like += shelf;
                }
                like.max(1e-300).ln()
            })
            .collect();
        self.apply(&log_like, QUALITY_OUTLIER);
    }
}

#[cfg(test)]
#[path = "../tests/unit/band.rs"]
mod tests;
