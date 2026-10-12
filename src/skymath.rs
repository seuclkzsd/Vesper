//! Sky geometry for the agent: the same formulas the engine uses, so planned hits are real hits.
//!
//! All angles in degrees. Azimuth: 0 = north, 90 = east. Times are UTC seconds since the Unix epoch
//! (f64). Port of `python-pro/skymath.py`.

use serde_json::Value;

pub const SIDEREAL_DEG_PER_SECOND: f64 = 360.98564736629 / 86400.0;

/// Python's float `%` for a positive divisor (the result takes the divisor's sign).
pub fn pmod(x: f64, m: f64) -> f64 {
    let r = x % m;
    if r != 0.0 && r < 0.0 {
        r + m
    } else {
        r
    }
}

/// Python's `round(x, digits)` (correctly rounded decimal).
pub fn round_to(x: f64, digits: usize) -> f64 {
    format!("{:.*}", digits, x).parse().unwrap_or(x)
}

/// Python 3.12+ `sum()` of floats (Neumaier compensated summation).
pub fn psum<I: IntoIterator<Item = f64>>(items: I) -> f64 {
    let mut total = 0.0f64;
    let mut c = 0.0f64;
    for x in items {
        let t = total + x;
        if total.abs() >= x.abs() {
            c += (total - t) + x;
        } else {
            c += (x - t) + total;
        }
        total = t;
    }
    if c != 0.0 && c.is_finite() {
        total += c;
    }
    total
}

pub fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

/// (year, month, day) of a day count since 1970-01-01.
pub fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// ISO-8601 UTC timestamp ("2026-11-01T23:00:00Z", fractions and +HH:MM offsets accepted) -> epoch seconds.
pub fn parse_utc(value: &str) -> f64 {
    let s = value.trim();
    let num = |a: usize, b: usize| -> i64 { s.get(a..b).and_then(|t| t.parse().ok()).unwrap_or(0) };
    let (y, mo, d) = (num(0, 4), num(5, 7), num(8, 10));
    let (h, mi, se) = (num(11, 13), num(14, 16), num(17, 19));
    let mut seconds = (days_from_civil(y, mo, d) * 86400 + h * 3600 + mi * 60 + se) as f64;
    let rest = s.get(19..).unwrap_or("");
    let mut tail = rest;
    if let Some(frac) = rest.strip_prefix('.') {
        let digits: String = frac.chars().take_while(|c| c.is_ascii_digit()).collect();
        if !digits.is_empty() {
            let micros: String = digits.chars().take(6).collect();
            let scale = 10f64.powi(micros.len() as i32);
            seconds += micros.parse::<f64>().unwrap_or(0.0) / scale;
        }
        tail = &frac[digits.len()..];
    }
    if tail.len() >= 6 && (tail.starts_with('+') || tail.starts_with('-')) {
        let sign = if tail.starts_with('-') { -1.0 } else { 1.0 };
        let oh: f64 = tail[1..3].parse().unwrap_or(0.0);
        let om: f64 = tail[4..6].parse().unwrap_or(0.0);
        seconds -= sign * (oh * 3600.0 + om * 60.0);
    }
    seconds
}

fn split(moment: f64) -> (i64, i64, i64, i64, i64, i64) {
    let total = moment.floor() as i64;
    let days = total.div_euclid(86400);
    let secs = total.rem_euclid(86400);
    let (y, m, d) = civil_from_days(days);
    (y, m, d, secs / 3600, (secs % 3600) / 60, secs % 60)
}

pub fn format_utc(moment: f64) -> String {
    let (y, m, d, h, mi, s) = split(moment);
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y, m, d, h, mi, s)
}

/// "YYYY-MM-DD" of the UTC date.
pub fn format_date(moment: f64) -> String {
    let (y, m, d, _, _, _) = split(moment);
    format!("{:04}-{:02}-{:02}", y, m, d)
}

/// "MM-DDTHH" (the fault table's hour stamp).
pub fn format_hour_stamp(moment: f64) -> String {
    let (_, m, d, h, _, _) = split(moment);
    format!("{:02}-{:02}T{:02}", m, d, h)
}

pub fn julian_date(moment: f64) -> f64 {
    moment / 86400.0 + 2440587.5
}

pub fn local_sidereal_deg(moment: f64, longitude_deg: f64) -> f64 {
    let days = julian_date(moment) - 2451545.0;
    pmod(280.46061837 + 360.98564736629 * days + longitude_deg, 360.0)
}

pub fn wrap180(angle: f64) -> f64 {
    pmod(angle + 180.0, 360.0) - 180.0
}

/// Equatorial -> horizontal (alt, az) for a given local sidereal time.
pub fn radec_to_altaz(ra_deg: f64, dec_deg: f64, lst_deg: f64, latitude_deg: f64) -> (f64, f64) {
    let hour_angle = wrap180(lst_deg - ra_deg).to_radians();
    let lat = latitude_deg.to_radians();
    let dec = dec_deg.to_radians();
    let sin_alt = lat.sin() * dec.sin() + lat.cos() * dec.cos() * hour_angle.cos();
    let alt = sin_alt.clamp(-1.0, 1.0).asin();
    let cos_alt = alt.cos().max(1e-12);
    let sin_az = -hour_angle.sin() * dec.cos() / cos_alt;
    let cos_az = (dec.sin() - alt.sin() * lat.sin()) / (cos_alt * lat.cos().max(1e-12));
    (alt.to_degrees(), pmod(sin_az.atan2(cos_az).to_degrees(), 360.0))
}

/// Largest |hour angle| at which a source stays at or above min_alt (0 = never, 180 = always).
pub fn max_hour_angle_deg(dec_deg: f64, latitude_deg: f64, min_alt_deg: f64) -> f64 {
    let lat = latitude_deg.to_radians();
    let dec = dec_deg.to_radians();
    let denominator = lat.cos() * dec.cos();
    if denominator.abs() < 1e-12 {
        return 0.0;
    }
    let value = (min_alt_deg.to_radians().sin() - lat.sin() * dec.sin()) / denominator;
    if value >= 1.0 {
        return 0.0;
    }
    if value <= -1.0 {
        return 180.0;
    }
    value.acos().to_degrees()
}

/// (north, east) gnomonic offsets in degrees of a target from a field centre, or None.
pub fn tangent_offsets(target_alt: f64, target_az: f64, center_alt: f64, center_az: f64) -> Option<(f64, f64)> {
    TangentFrame::new(center_alt, center_az).project(&unit_altaz(target_alt, target_az))
}

/// Unit vector (north, east, up) of a horizontal direction.
pub fn unit_altaz(alt_deg: f64, az_deg: f64) -> [f64; 3] {
    let (alt, az) = (alt_deg.to_radians(), az_deg.to_radians());
    [alt.cos() * az.cos(), alt.cos() * az.sin(), alt.sin()]
}

/// The engine's tangent plane at a field centre (`v4_fiber_map`): projecting a precomputed unit vector costs a few
/// multiplications, so many candidate centres can be tried per decision. Same arithmetic as `tangent_offsets`.
pub struct TangentFrame {
    c: [f64; 3],
    north: [f64; 3],
    east: [f64; 3],
}

impl TangentFrame {
    pub fn new(center_alt: f64, center_az: f64) -> TangentFrame {
        let (calt, caz) = (center_alt.to_radians(), center_az.to_radians());
        TangentFrame {
            c: [calt.cos() * caz.cos(), calt.cos() * caz.sin(), calt.sin()],
            north: [-calt.sin() * caz.cos(), -calt.sin() * caz.sin(), calt.cos()],
            east: [-caz.sin(), caz.cos(), 0.0],
        }
    }

    /// (north, east) offsets in degrees of the direction `t`, or None behind the plane.
    #[inline]
    pub fn project(&self, t: &[f64; 3]) -> Option<(f64, f64)> {
        let depth = t[0] * self.c[0] + t[1] * self.c[1] + t[2] * self.c[2];
        if depth <= 0.0 {
            return None;
        }
        Some((
            ((t[0] * self.north[0] + t[1] * self.north[1] + t[2] * self.north[2]) / depth).to_degrees(),
            ((t[0] * self.east[0] + t[1] * self.east[1] + t[2] * self.east[2]) / depth).to_degrees(),
        ))
    }
}

/// Horizontal -> equatorial (ra, dec) for a given local sidereal time (inverse of `radec_to_altaz`).
pub fn altaz_to_radec(alt_deg: f64, az_deg: f64, lst_deg: f64, latitude_deg: f64) -> (f64, f64) {
    let (alt, az, lat) = (alt_deg.to_radians(), az_deg.to_radians(), latitude_deg.to_radians());
    let sin_dec = (alt.sin() * lat.sin() + alt.cos() * lat.cos() * az.cos()).clamp(-1.0, 1.0);
    let dec = sin_dec.asin();
    // hour angle: sin H = -sin A cos h / cos d, cos H = (sin h - sin d sin phi) / (cos d cos phi)
    let y = -az.sin() * alt.cos();
    let x = (alt.sin() - sin_dec * lat.sin()) / lat.cos().max(1e-12);
    let ha = y.atan2(x).to_degrees();
    (pmod(lst_deg - ha, 360.0), dec.to_degrees())
}

/// The direction d_north / d_east degrees away on the tangent plane at (alt, az). Works near the zenith.
pub fn shift_altaz(alt_deg: f64, az_deg: f64, d_north: f64, d_east: f64) -> (f64, f64) {
    let (alt, az) = (alt_deg.to_radians(), az_deg.to_radians());
    let point = [alt.cos() * az.cos(), alt.cos() * az.sin(), alt.sin()];
    let north = [-alt.sin() * az.cos(), -alt.sin() * az.sin(), alt.cos()];
    let east = [-az.sin(), az.cos(), 0.0];
    let (dn, de) = (d_north.to_radians(), d_east.to_radians());
    let mut v = [0.0; 3];
    for k in 0..3 {
        v[k] = point[k] + dn * north[k] + de * east[k];
    }
    let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let (x, y, z) = (v[0] / norm, v[1] / norm, v[2] / norm);
    (z.clamp(-1.0, 1.0).asin().to_degrees(), pmod(y.atan2(x).to_degrees(), 360.0))
}

/// Reference layout the stock constants were tuned for: 4 x 4 fibres of 0.4 deg2, no gap.
pub const REF_GLASS_DEG: f64 = 0.632_455_532_033_675_9; // sqrt(0.4)

/// n x n square fibres; fibre 0 bottom-left, rows along +alt, columns along +az.
///
/// Geometry follows the engine's `challenge/v4_fiber_map.FiberGrid`: side = isqrt(n_fibers), glass =
/// sqrt(fiber_area_deg2), pitch = glass + gap_deg, fov = side * pitch, and with gap 0 every in-field point is on
/// glass. The published (1e-6-rounded) `glass_side_deg` / `pitch_deg` / `fov_side_deg` are used when they agree
/// with those values (as in the stock agent); otherwise the values derived from the area win.
#[derive(Clone)]
pub struct FiberGrid {
    pub side: usize,
    pub n: usize,
    pub glass: f64,
    pub pitch: f64,
    pub fov: f64,
    pub gap: f64,
}

fn isqrt(n: usize) -> usize {
    let mut s = (n as f64).sqrt() as usize;
    while s * s > n {
        s -= 1;
    }
    while (s + 1) * (s + 1) <= n {
        s += 1;
    }
    s
}

impl FiberGrid {
    pub fn new(instrument: &Value) -> FiberGrid {
        let f = |k: &str| instrument.get(k).and_then(Value::as_f64).unwrap_or(0.0);
        let mut n = f("n_fibers").max(0.0) as usize;
        let mut side = isqrt(n);
        if n == 0 || side * side != n {
            // the engine only allows perfect squares; trust grid_side if n_fibers is unusable
            side = (f("grid_side").max(1.0)) as usize;
            n = side * side;
        }
        let area = f("fiber_area_deg2");
        let gap = f("gap_deg").max(0.0);
        let (glass, pitch, fov) = if area > 0.0 {
            let glass = area.sqrt();
            let pitch = glass + gap;
            let fov = side as f64 * pitch;
            // the published values are the same numbers rounded to 1e-6 deg; the stock agent used them (and a
            // different last digit moves near-zenith pointings), so keep them when they agree with the engine
            let close = |key: &str, exact: f64| {
                let v = f(key);
                if v > 0.0 && (v - exact).abs() <= 2e-6 * exact.max(1.0) { v } else { exact }
            };
            (close("glass_side_deg", glass), close("pitch_deg", pitch), close("fov_side_deg", fov))
        } else {
            let glass = f("glass_side_deg");
            let pitch = if f("pitch_deg") > 0.0 { f("pitch_deg") } else { glass + gap };
            let fov = if f("fov_side_deg") > 0.0 { f("fov_side_deg") } else { side as f64 * pitch };
            (glass, pitch, fov)
        };
        FiberGrid { side, n, glass, pitch, fov, gap }
    }

    /// Fibres nearest the field centre: the central 2 x 2 block for an even side (5, 6, 9, 10 on 4 x 4). For an
    /// odd side the centre fibre first, then (odd_ring) its four neighbours, so a density anchor still gets
    /// several placements around the field centre.
    pub fn central_fibers(&self, odd_ring: bool) -> Vec<usize> {
        let s = self.side;
        if s <= 1 {
            return vec![0];
        }
        if s % 2 == 1 {
            let c = (s / 2) * s + s / 2;
            return if odd_ring { vec![c, c - s, c - 1, c + 1, c + s] } else { vec![c] };
        }
        let (a, b) = (s / 2 - 1, s / 2);
        vec![a * s + a, a * s + b, b * s + a, b * s + b]
    }

    /// Ratio of this layout's field side to the reference 4 x 4 / 0.4 deg2 field (1.0 on that layout).
    pub fn fov_scale(&self) -> f64 {
        snap_one(self.fov / (4.0 * REF_GLASS_DEG))
    }

    /// Ratio of this layout's fibre (glass) side to the reference 0.4 deg2 fibre (1.0 on that layout).
    pub fn glass_scale(&self) -> f64 {
        snap_one(self.glass / REF_GLASS_DEG)
    }

    /// Ratio of this layout's fibre pitch to the reference pitch (1.0 on that layout).
    pub fn pitch_scale(&self) -> f64 {
        snap_one(self.pitch / REF_GLASS_DEG)
    }

    pub fn fiber_center(&self, fiber: usize) -> (f64, f64) {
        let (row, col) = (fiber / self.side, fiber % self.side);
        let middle = (self.side as f64 - 1.0) / 2.0;
        ((row as f64 - middle) * self.pitch, (col as f64 - middle) * self.pitch)
    }

    /// Offsets along one axis at which a point lies on the glass of row (or column) k with at least `margin` to
    /// its edges: the same test as `classify` (|offset - centre| <= glass/2 - margin, inside the field).
    pub fn glass_interval(&self, k: usize, margin: f64) -> Option<(f64, f64)> {
        let centre = (k as f64 - (self.side as f64 - 1.0) / 2.0) * self.pitch;
        let half = self.fov / 2.0;
        let lo = (centre - self.glass / 2.0 + margin).max(-half);
        let hi = (centre + self.glass / 2.0 - margin).min(half);
        if lo < hi {
            Some((lo, hi))
        } else {
            None
        }
    }

    /// (fiber id or None when not on glass, margin in degrees to the glass edge).
    pub fn classify(&self, d_north: f64, d_east: f64) -> (Option<usize>, f64) {
        let half = self.fov / 2.0;
        if d_north.abs() > half || d_east.abs() > half {
            return (None, -1.0);
        }
        let middle = self.side as f64 / 2.0;
        let last = self.side as i64 - 1;
        let row = ((d_north / self.pitch + middle).floor() as i64).clamp(0, last) as usize;
        let col = ((d_east / self.pitch + middle).floor() as i64).clamp(0, last) as usize;
        let fiber = row * self.side + col;
        let (c_north, c_east) = self.fiber_center(fiber);
        let margin = self.glass / 2.0 - (d_north - c_north).abs().max((d_east - c_east).abs());
        if margin >= 0.0 {
            (Some(fiber), margin)
        } else if self.gap == 0.0 {
            // the engine: with gap 0 the cells tile the field, every in-field point is glass (a rounding
            // sliver at a shared edge belongs to the cell the floor() picked)
            (Some(fiber), 0.0)
        } else {
            (None, margin)
        }
    }
}

/// 1.0 when x is within 1e-6 of 1 (so the reference layout reproduces the stock constants bit for bit).
fn snap_one(x: f64) -> f64 {
    if (x - 1.0).abs() < 1e-6 {
        1.0
    } else {
        x
    }
}

pub fn normalized_airmass(alt_deg: f64) -> f64 {
    if alt_deg <= 0.0 {
        return f64::INFINITY;
    }
    let zenith = 90.0 - alt_deg;
    let raw = 1.0 / (zenith.to_radians().cos() + 0.50572 * (96.07995 - zenith).powf(-1.6364));
    raw / (1.0 / (1.0 + 0.50572 * 96.07995f64.powf(-1.6364)))
}

pub fn sun_radec(moment: f64) -> (f64, f64) {
    let days = julian_date(moment) - 2451545.0;
    let mean_longitude = pmod(280.460 + 0.9856474 * days, 360.0);
    let anomaly = pmod(357.528 + 0.9856003 * days, 360.0).to_radians();
    let longitude = pmod(mean_longitude + 1.915 * anomaly.sin() + 0.020 * (2.0 * anomaly).sin(), 360.0).to_radians();
    let obliquity = (23.439 - 0.0000004 * days).to_radians();
    (
        pmod((obliquity.cos() * longitude.sin()).atan2(longitude.cos()).to_degrees(), 360.0),
        (obliquity.sin() * longitude.sin()).asin().to_degrees(),
    )
}

pub fn moon_radec(moment: f64) -> (f64, f64) {
    let days = julian_date(moment) - 2451545.0;
    let mean_longitude = pmod(218.316 + 13.176396 * days, 360.0).to_radians();
    let anomaly = pmod(134.963 + 13.064993 * days, 360.0).to_radians();
    let arg_latitude = pmod(93.272 + 13.229350 * days, 360.0).to_radians();
    let longitude = mean_longitude + 6.289f64.to_radians() * anomaly.sin();
    let latitude = 5.128f64.to_radians() * arg_latitude.sin();
    let obliquity = (23.439 - 0.0000004 * days).to_radians();
    let x = longitude.cos() * latitude.cos();
    let y = longitude.sin() * latitude.cos() * obliquity.cos() - latitude.sin() * obliquity.sin();
    let z = longitude.sin() * latitude.cos() * obliquity.sin() + latitude.sin() * obliquity.cos();
    (pmod(y.atan2(x).to_degrees(), 360.0), z.asin().to_degrees())
}

pub fn separation_deg(ra1: f64, dec1: f64, ra2: f64, dec2: f64) -> f64 {
    let (r1, d1, r2, d2) = (ra1.to_radians(), dec1.to_radians(), ra2.to_radians(), dec2.to_radians());
    let cosine = d1.sin() * d2.sin() + d1.cos() * d2.cos() * (r1 - r2).cos();
    cosine.clamp(-1.0, 1.0).acos().to_degrees()
}

/// The public lunar model from the score config.
#[derive(Clone, Copy)]
pub struct LunarModel {
    pub maximum_penalty: f64,
    pub altitude_exponent: f64,
    pub angular_decay_scale_deg: f64,
}

/// Moon state at one instant; lunar_factor() is the public lunar model.
#[derive(Clone, Copy)]
pub struct Moon {
    pub ra: f64,
    pub dec: f64,
    pub illumination: f64,
    pub alt: f64,
    model: LunarModel,
}

impl Moon {
    pub fn new(moment: f64, lst_deg: f64, latitude_deg: f64, model: LunarModel) -> Moon {
        let (ra, dec) = moon_radec(moment);
        let (sun_ra, sun_dec) = sun_radec(moment);
        let illumination = (1.0 - separation_deg(sun_ra, sun_dec, ra, dec).to_radians().cos()) / 2.0;
        let (alt, _) = radec_to_altaz(ra, dec, lst_deg, latitude_deg);
        Moon { ra, dec, illumination, alt, model }
    }

    pub fn lunar_factor(&self, ra_deg: f64, dec_deg: f64) -> f64 {
        if self.alt <= 0.0 {
            return 1.0;
        }
        let separation = separation_deg(ra_deg, dec_deg, self.ra, self.dec);
        let penalty = self.model.maximum_penalty
            * self.illumination
            * self.alt.to_radians().sin().powf(self.model.altitude_exponent)
            * (-separation / self.model.angular_decay_scale_deg).exp();
        (1.0 - penalty).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
#[path = "../tests/unit/skymath.rs"]
mod tests;
