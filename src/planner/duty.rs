//! Duty-log facts in force: the decoder's view (`crate::dutylog`) with the LLM duty-log stage's additions
//! (`view_in`, `set_view`). Drops are reported at their time (exposures end there); flat tests and stops are waited
//! out; closures shut sectors; terrain sets clear altitudes; urgent-request notices end exposures at their issue.

const RETRY: [f64; 3] = [0.0, 900.0, 3600.0];

/// Compass azimuths of clockwise sectors of 8-point directions (0 = N .. 7 = NW).
pub fn compass_of(sectors: &[(u8, u8)]) -> Vec<f64> {
    let mut out: Vec<f64> = Vec::new();
    for &(a, b) in sectors {
        let n = (b as usize + 8 - a as usize) % 8;
        out.extend((0..=n).map(|k| 45.0 * ((a as usize + k) % 8) as f64));
    }
    out.dedup();
    out
}

#[derive(Clone, Debug)]
struct Drop {
    known: f64,
    at: f64,
    partial: bool,
    sure: bool, // an engineering notice / observed drop, not a lone guess
    seen: bool, // a colleague saw it and asks for the report
    gone: f64,  // known time of its cancellation (INF: none)
    tries: usize,
    settled: bool,
}

/// A decoder's card-level view (UTC seconds): faults (onset, partial, observed, sure: may be paid), flat tests,
/// stops, closures (start, end, compass azimuths from `compass_of`, closed below; rain left out), terrain (compass
/// azimuth, clear above), urgent issue times, explained drops (shaking, cloud).
#[derive(Default)]
pub struct ViewIn {
    pub faults: Vec<(f64, bool, bool, bool)>,
    pub flats: Vec<(f64, f64)>,
    pub stops: Vec<(f64, f64)>,
    pub closures: Vec<(f64, f64, Vec<f64>, f64)>,
    pub terrain: Vec<(f64, f64)>,
    pub urgent: Vec<f64>,
    pub explained: Vec<(f64, f64)>,
}

/// The decoder's view (`crate::dutylog`) and the LLM duty-log stage's additions `model` (facts the rules did not
/// find: fault times, never paid; stops outside the decoder's dead time; wind/launch/thick-cloud closures; terrain
/// of directions the decoder has none for). Drops the log explains: an earthquake's time, or the time of the drop a
/// colleague asked about when the answer is cloud, -1 h to +3 h.
pub fn view_in(log: &crate::dutylog::DutyLog, model: &crate::llm::duty_log::Facts) -> ViewIn {
    use crate::dutylog::{Dirs, DutyFact, Expect, Wx};
    let v = &log.view;
    let f = |t: i64| t as f64;
    let flats: Vec<(f64, f64)> = v.flats.values().flatten().map(|&(a, b)| (f(a), f(b))).collect();
    let mut stops: Vec<(f64, f64)> = v.dead.iter().map(|&(a, b)| (f(a), f(b))).filter(|w| !flats.contains(w)).collect();
    let mut faults: Vec<(f64, bool, bool, bool)> = v.faults.iter().map(|x| (f(x.at), x.expect == Expect::Partial, x.observed, true)).collect();
    for &(t, announced) in &model.faults {
        if !faults.iter().any(|x| (x.0 - t).abs() < 61.0) && !v.cancelled.iter().any(|&c| (f(c) - t).abs() < 61.0) {
            faults.push((t, false, !announced, false));
        }
    }
    stops.extend(model.stops.iter().filter(|s| !v.dead.iter().any(|&(a, b)| f(a) < s.1 && s.0 < f(b))));
    let mut closures: Vec<(f64, f64, Vec<f64>, f64)> = v.closures.iter().filter_map(|(a, b, d, w, below)| match d {
        Dirs::Sectors(s) if !w.observe() && *w != Wx::Rain => Some((f(*a), f(*b), compass_of(s), below.map_or(85.0, |x| x as f64))),
        _ => None,
    }).collect();
    closures.extend(model.closures.iter().map(|(a, b, d)| (*a, *b, d.iter().map(|&k| 45.0 * k as f64).collect(), 85.0)));
    let mut terrain: Vec<(f64, f64)> = v.terrain_clear.iter().enumerate().filter_map(|(k, c)| c.map(|c| (45.0 * k as f64, c as f64))).collect();
    terrain.extend(model.terrain.iter().filter(|(d, _)| v.terrain_clear[*d].is_none()).map(|&(d, h)| (45.0 * d as f64, h + 1.5)));
    let mut asked = None;
    let explained = log.facts.iter().filter_map(|x| match &x.fact {
        DutyFact::DropQuestion { at } => {
            asked = *at;
            None
        }
        DutyFact::Earthquake { at } => *at,
        DutyFact::DropExplainedCloud => Some(asked.unwrap_or(x.issued_at)),
        _ => None,
    });
    ViewIn {
        faults,
        flats,
        stops,
        closures,
        terrain,
        urgent: v.urgent.iter().filter_map(|u| u.0.map(f)).collect(),
        explained: explained.map(|t| (f(t) - 3600.0, f(t) + 3.0 * 3600.0)).collect(),
    }
}

#[derive(Default)]
pub struct Duty {
    drops: Vec<Drop>,
    dead: Vec<(f64, f64, f64, bool, i8)>, // (known, start, end, stop, probe: 0 / 1 confirmed / -1 refuted)
    closures: Vec<(f64, f64, f64, Vec<f64>, f64)>, // (known, start, end, compass azimuths, below altitude)
    terrain: Vec<(f64, f64, f64)>,                 // (known, compass azimuth, top)
    explained: Vec<(f64, f64, f64)>,               // (known, start, end): drops that are weather or shaking
    urgent: Vec<(f64, f64)>,                       // (known, expected issue)
    pub paid_false: i64,
    pub in_flight: Option<usize>,
}

impl Duty {
    pub fn active(&self) -> bool {
        !(self.drops.is_empty() && self.dead.is_empty() && self.closures.is_empty() && self.terrain.is_empty())
    }

    /// The decoder's resolved view at `now` (it is recomputed from every message so far): drops it no longer
    /// lists are gone from now on, new ones are known from now; report bookkeeping of the others is kept.
    pub fn set_view(&mut self, now: f64, v: ViewIn) {
        for d in self.drops.iter_mut().filter(|d| d.gone > now) {
            if !v.faults.iter().any(|f| (f.0 - d.at).abs() < 60.0) {
                d.gone = now;
            }
        }
        for (at, partial, seen, sure) in v.faults {
            match self.drops.iter_mut().find(|d| (d.at - at).abs() < 60.0) {
                Some(d) => {
                    d.gone = f64::INFINITY;
                    d.seen |= seen; // an announced drop a colleague then saw
                }
                // a model-only drop learned over an hour late (its answer arrives in the background): one try
                None => {
                    let tries = if !sure && at < now - 3600.0 { RETRY.len() - 1 } else { 0 };
                    self.drops.push(Drop { known: now, at, partial, sure, seen, gone: f64::INFINITY, tries, settled: false })
                }
            }
        }
        let old = std::mem::take(&mut self.dead);
        let probe = |a: f64, b: f64| old.iter().find(|w| w.1 == a && w.2 == b).map_or(0, |w| w.4);
        self.dead = v.flats.iter().map(|&(a, b)| (now, a, b, false, 0)).chain(v.stops.iter().map(|&(a, b)| (now, a, b, true, probe(a, b)))).collect();
        self.closures = v.closures.into_iter().map(|(a, b, az, below)| (now, a, b, az, below)).collect();
        self.terrain = v.terrain.into_iter().map(|(az, clear)| (now, az, clear - 1.5)).collect();
        self.urgent = v.urgent.into_iter().map(|t| (now, t)).collect();
        self.explained = v.explained.into_iter().map(|(a, b)| (now, a, b)).collect();
    }

    fn live(&self, d: &Drop, now: f64) -> bool {
        !d.settled && d.known <= now && d.gone > now
    }

    /// The first drop the log calls for reporting now: (index, reason, paid). `repaired`: the last correct
    /// report; `free`: free false reports left; `dropped(r, t)`: a reading after t is well below the level before r.
    /// A drop is reported at its time, again 15 and 60 min later while false reports are free. A sure drop (an
    /// engineering notice, or one a colleague saw) is paid for until a duty-log report was a paid false one; then
    /// only when the readings show it against the level before the first drop due since the repair (an unreported
    /// one keeps the instrument down, so the readings before a later drop are low too), at most two paid false.
    /// Partial drops end with their night.
    pub fn report_due(&mut self, now: f64, repaired: f64, free: i64, dropped: impl Fn(f64, f64) -> bool) -> Option<(usize, String, bool)> {
        let first = self.drops.iter().filter(|d| d.known <= now && d.gone > now && d.at > repaired && d.at <= now).fold(f64::INFINITY, |m, d| m.min(d.at));
        for (k, d) in self.drops.iter_mut().enumerate() {
            if d.settled || d.known > now || d.gone <= now {
                continue;
            }
            if repaired >= d.at || now > d.at + if d.partial { 12.0 * 3600.0 } else { 3.0 * 86400.0 } || d.tries >= RETRY.len() {
                d.settled = true;
                continue;
            }
            if now < d.at + RETRY[d.tries] {
                continue;
            }
            if free >= 1 {
                return Some((k, format!("duty log: instrument drop at {} (try {})", crate::skymath::format_utc(d.at), d.tries + 1), false));
            }
            if d.tries == 0 && d.sure && (self.paid_false == 0 || self.paid_false < 2 && dropped(first.min(d.at), d.at)) {
                return Some((k, format!("duty log: instrument drop at {} (paid)", crate::skymath::format_utc(d.at)), true));
            }
            if d.tries > 0 {
                d.settled = true;
            }
        }
        None
    }

    /// A due drop waits for a reading that confirms it (no free report left): keep the next exposure short.
    pub fn awaiting(&self, now: f64) -> bool {
        self.drops.iter().any(|d| self.live(d, now) && d.tries == 0 && d.sure && d.at <= now && now < d.at + 7200.0)
    }

    /// Enough drops announced (5) that a drop in the readings is more likely the instrument than the weather.
    pub fn dense(&self, now: f64) -> bool {
        self.drops.iter().filter(|d| d.known <= now).count() >= 5
    }

    /// A live drop within `win` seconds of `now` (its own report handles it).
    pub fn drop_near(&self, now: f64, win: f64) -> bool {
        self.drops.iter().any(|d| self.live(d, now) && (d.at - now).abs() < win)
    }

    /// A live drop due within `ahead` seconds from now (its own report is coming).
    pub fn drop_due_within(&self, now: f64, ahead: f64) -> bool {
        self.drops.iter().any(|d| self.live(d, now) && d.at > now && d.at < now + ahead)
    }

    pub fn sent(&mut self, k: usize) {
        if let Some(d) = self.drops.get_mut(k) {
            d.tries += 1;
        }
        self.in_flight = Some(k);
    }

    /// The engine's answer to a report (any source): a correct one repairs every drop up to now.
    pub fn on_report(&mut self, correct: bool, paid: bool, now: f64) {
        let mine = self.in_flight.take().is_some();
        if correct {
            self.drops.iter_mut().filter(|d| d.at <= now).for_each(|d| d.settled = true);
        } else if paid && mine {
            self.paid_false += 1;
        }
    }

    /// The earliest moment after `now` an exposure should end at: a drop's time, a dead window's start, an
    /// announced urgent request's issue.
    pub fn next_boundary(&self, now: f64) -> Option<f64> {
        let drops = self.drops.iter().filter(|d| self.live(d, now) && d.at > now).map(|d| d.at);
        let dead = self.dead.iter().filter(|w| w.0 <= now && w.1 > now && w.4 >= 0).map(|w| w.1);
        let urgent = self.urgent.iter().filter(|u| u.0 <= now && u.1 > now).map(|u| u.1);
        drops.chain(dead).chain(urgent).fold(None, |m: Option<f64>, t| Some(m.map_or(t, |m| m.min(t))))
    }

    /// The dead window in force at `now`: (index, end, stop, probe state).
    pub fn dead_at(&self, now: f64) -> Option<(usize, f64, bool, i8)> {
        self.dead.iter().enumerate().filter(|(_, w)| w.0 <= now && w.1 <= now && now < w.2 && w.4 >= 0).map(|(k, w)| (k, w.2, w.3, w.4)).max_by(|a, b| a.1.total_cmp(&b.1))
    }

    /// The result of the probe exposure inside stop `k`: anything scored refutes the stop.
    pub fn probed(&mut self, k: usize, scored: bool) {
        if let Some(w) = self.dead.get_mut(k) {
            w.4 = if scored { -1 } else { 1 };
        }
    }

    /// Does [a, b) touch dead time known at `now` (no detector reading from it)?
    pub fn overlaps_dead(&self, a: f64, b: f64, now: f64) -> bool {
        self.dead.iter().any(|w| w.0 <= now && w.4 >= 0 && w.1 < b && a < w.2)
    }

    /// No detector report now: dead time, or a drop the log explains as weather or shaking.
    pub fn quiet(&self, now: f64) -> bool {
        self.dead_at(now).is_some() || self.explained.iter().any(|e| e.0 <= now && e.1 <= now && now < e.2)
    }

    /// Closed (compass azimuth, below altitude) pairs for exposures starting now (closures within 15 min). A
    /// closure with no altitude of its own (85 deg default) closes its sector up to the zenith: the engine tests
    /// every target's own azimuth, and the zdl truth closes wind and thick cloud to 90 deg.
    pub fn closed(&self, now: f64) -> Vec<(f64, f64)> {
        let mut out: Vec<(f64, f64)> = Vec::new();
        for c in self.closures.iter().filter(|c| c.0 <= now && c.1 < now + 900.0 && now < c.2) {
            out.extend(c.3.iter().map(|&az| (az, if c.4 >= 85.0 { 90.5 } else { c.4 })));
        }
        out
    }

    /// Closures that start between 15 min and `horizon` seconds from now: (seconds to the start, compass azimuth,
    /// below altitude). A target in one is up only until then (an exposure crossing in integrates nothing after).
    pub fn closing(&self, now: f64, horizon: f64) -> Vec<(f64, f64, f64)> {
        let mut out: Vec<(f64, f64, f64)> = Vec::new();
        for c in self.closures.iter().filter(|c| c.0 <= now && c.1 >= now + 900.0 && c.1 < now + horizon) {
            out.extend(c.3.iter().map(|&az| (c.1 - now, az, if c.4 >= 85.0 { 90.5 } else { c.4 })));
        }
        out
    }

    /// Terrain tops (compass azimuth, top) known at `now`: per direction the lowest statement.
    pub fn terrain(&self, now: f64) -> Vec<(f64, f64)> {
        let mut out: Vec<(f64, f64)> = Vec::new();
        for &(_, az, top) in self.terrain.iter().filter(|t| t.0 <= now) {
            match out.iter_mut().find(|e| e.0 == az) {
                Some(e) => e.1 = e.1.min(top),
                None => out.push((az, top)),
            }
        }
        out
    }

    pub fn has_terrain(&self) -> bool {
        !self.terrain.is_empty()
    }
}

#[cfg(test)]
#[path = "../../tests/unit/duty.rs"]
mod tests;
