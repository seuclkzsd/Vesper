//! Planner of the pro agent: pointing, fibre assignment, duration and program in one search.
//! Files: mod.rs (constants, state, catalogue precomputation, messages, sky and direction helpers), values.rs
//! (planning values of required targets and observation requests), search.rs (`Planner::plan` and its
//! per-decision search), learn.rs (learning from results). Derived from the organizers' example planner (see NOTICE).
//!
//! Per decision (plan):
//! 1. Visible targets with something left to gain are ranked with a cheap proxy; the best POOL go on.
//! 2. The gain of one exposure of duration T for target i under program p is
//!        weight * E_band[reach(T) * multiplier(p, or the mismatch value) - best so far]   (P(band): band.rs)
//!      + required stake * P_now(T) * (1 - P_future)   (deadline-aware: the target's remaining calendar)
//!      + request value * P(request target reaches its threshold), only for exposures inside the request window
//!    the science part times an urgency factor for targets with few nights left. Request values come from the
//!    Poisson-binomial over the request's targets (requests.rs, values.rs).
//! 3. Candidate fields: the N_ANCHORS best targets and up to REQ_ANCHORS open request targets centred on each
//!    fibre, the N_DENSE densest patches of remaining science and mean-shift seeds; each anchor's neighbourhood
//!    covers every field it is placed in, and the best pointings get an exact cell-edge shift scan. For each
//!    duration and program every fibre gets the target with the largest expected gain; the winning (pointing,
//!    duration, program) maximises  total gain - lambda * T  (lambda = time price, scaled by the card's time
//!    scarcity), and is then refined by small pointing shifts.
//! 4. Duration: on the final pointing the exposure is re-chosen among the saturation times (with a margin) of the
//!    leading targets of every fibre (where the gain curve bends down), the next slot boundaries and the coarse
//!    grid. Below the grid's shortest exposure only targets the exposure completes count.
//! 5. A fibre left empty may take a finished target that can still gain its program multiplier (filler).
//! 6. The command is the chosen centre minus the learned pointing offset (Hard-mode cards).
//!
//! Learning from results (learn.rs): unsaturated hits give the quality level (scale) and one exact site-throughput
//! reading per observe for the fault detector (fault.rs); saturated hits show the program multiplier exactly and
//! so bound the band level (band.rs keeps a posterior over it); terrain sectors are learnt from the own hits
//! (terrain.rs); hit/miss patterns reveal the pointing offset. The planner only uses the public catalogue, the
//! public score formula, bulletins, forecasts and its own results.

use serde_json::{json, Map, Value};
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

use crate::band::BandFilter;
use crate::log;
use crate::requests::{normal_cdf, p_capacity, target_values, CapSample, RequestTerm};
use crate::terrain::TerrainBelief;
use crate::seasonlp::{SeasonLp, SlpCfg};
use crate::urgency::{SiteGeometry, Urgency, UrgencyConfig};
use crate::skymath::{
    local_sidereal_deg, max_hour_angle_deg, normalized_airmass, parse_utc, pmod, psum, radec_to_altaz, round_to,
    shift_altaz, tangent_offsets, unit_altaz, wrap180, altaz_to_radec, FiberGrid, LunarModel, Moon, TangentFrame, SIDEREAL_DEG_PER_SECOND,
};

mod learn; // learning from results: quality level, throughput reading, band level, terrain, E, pointing offset
mod search; // one decision: the search for pointing, fibres, duration and program
mod tiling; // fixed field centres (DESI-style passes) and the hour-angle preference
mod values; // planning values of required targets and observation requests

/// `PRO_<NAME>` environment override of a float constant.
pub fn env_f(name: &str, default: f64) -> f64 {
    std::env::var(format!("PRO_{name}")).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(default)
}

/// `PRO_<NAME>` environment override of an integer constant.
pub fn env_i(name: &str, default: i64) -> i64 {
    std::env::var(format!("PRO_{name}")).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(default)
}

const TYPICAL_Q: f64 = 0.6;
const DURATIONS: [f64; 11] = [300.0, 450.0, 600.0, 750.0, 900.0, 1200.0, 1500.0, 1800.0, 2400.0, 3000.0, 3600.0];
const LEVEL_DURATIONS_1: [f64; 7] = [300.0, 600.0, 900.0, 1200.0, 1800.0, 2400.0, 3600.0];
const LEVEL_DURATIONS_2: [f64; 4] = [450.0, 900.0, 1800.0, 3600.0];
const LEVEL_DURATIONS_3: [f64; 2] = [900.0, 1800.0];
const MIN_VISIBLE_SECONDS: f64 = 600.0;
const ALT_MARGIN_DEG: f64 = 0.6; // keep targets this far above the altitude limit
const REQUIRED_SAFE_FACTOR: f64 = 0.62; // a required target counts as safe at this estimated factor (for threshold 0.5)
const DONE_FACTOR: f64 = 0.95; // other targets are done at this factor
// --- pointing offset (Hard-mode cards: a fixed, unannounced offset; its size is not published) ---
const OFFSET_STEPS: i64 = 16; // coarse grid half-width in steps of pitch/12; widens by half on an edge hit
const OFFSET_MIN_MISSES: i64 = 6; // start estimating after this many assigned-but-missed targets
const OFFSET_REFINE_EVERY: i64 = 10; // observes between fine searches
const OFFSET_FINE_EVIDENCE: usize = 150; // observes used by the fine search
const OFFSET_MARGIN: i64 = 8; // adopt an offset only if it explains this many more outcomes
// --- learning ---
const SKY_MEMORY_HOURS: f64 = 2.0; // quality samples older than this are stale
const BAND_MEMORY_HOURS: f64 = 2.0; // saturated hits used to fit the band level
const REQ_STEP: f64 = 300.0; // sampling step of request-window visibility (s)
const LONG_RATIOS: usize = 4000; // quality samples kept for the typical sky (several weeks of hits)
const TYPICAL_QUALITY: f64 = 0.8; // typical sky before LONG_MIN samples exist
const LONG_MIN: usize = 50;
const CLOSED_KINDS: [&str; 2] = ["rain", "storm"];
const SKY_WEATHER_KINDS: [&str; 5] = ["rain", "storm", "overcast", "haze", "cold_snap"];
const BLOCKING_KINDS: [&str; 2] = ["terrain_obstruction", "rocket_launch"];
// LLM week plan `avoid` sectors (extra_avoid): value factor below this altitude within this azimuth of the sector
// centre; the live bulletin already avoids the event (x0.35), so only a running exposure is at risk: a ~5 % nudge.
const AVOID_FACTOR: f64 = 0.95;
const AVOID_HALF_WIDTH_DEG: f64 = 45.0;
const AVOID_BELOW_ALT_DEG: f64 = 60.0;

fn direction_az(direction: &str) -> Option<f64> {
    Some(match direction {
        "N" => 0.0,
        "NE" => 45.0,
        "E" => 90.0,
        "SE" => 135.0,
        "S" => 180.0,
        "SW" => 225.0,
        "W" => 270.0,
        "NW" => 315.0,
        _ => return None,
    })
}

/// ALL-sky weather kinds (rain, storm, overcast, haze, cold_snap) of one bulletin.
fn all_sky_kinds(bulletin: &Value) -> BTreeSet<String> {
    bulletin
        .get("notices")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|n| n.get("direction").and_then(Value::as_str) == Some("ALL"))
        .filter_map(|n| n.get("event_kind").and_then(Value::as_str))
        .filter(|k| SKY_WEATHER_KINDS.contains(k))
        .map(str::to_string)
        .collect()
}

fn az_distance(a: f64, b: f64) -> f64 {
    wrap180(a - b).abs()
}

/// Tunable constants (each one overridable with `PRO_<NAME>`).
struct Params {
    lambda_frac: f64,
    lambda_ema: f64,
    scarcity_ref: f64,
    scarcity_power: f64,
    n_anchors: usize,
    n_dense: usize,
    dense_bin_deg: f64,
    refine_rounds: i64,
    refine_fixed_t: bool,
    refine_steps: Vec<(f64, f64)>,
    pool: usize,
    neighbour_radius_deg: f64,
    edge_margin_deg: f64,
    min_t: f64,
    plan_factor_safety: f64,
    required_bonus: f64,
    req_p_lo: f64,
    req_p_hi: f64,
    request_mult: f64,
    req_pb: bool,       // request value: Poisson-binomial over the targets' feasibility (else reward / remaining)
    req_window: bool,   // a request counts only exposures wholly inside its [issued, deadline] window
    req_anchors: usize, // every decision also centres fields on up to this many open request targets
    req_rate_excl: bool, // the time price (rate_ema) ignores request rewards
    req_q_sigma: f64,   // spread (log) of the sky quality over a request window
    request_p_open: f64, // chance that a run of visibility inside a request window is not lost to closures
    req_cap: bool,      // request capacity: an exposure that leaves too little window for the rest of a request pays for it
    req_cap_step: f64,  // sampling step (s) of the usable starts of request targets
    forecast_discount: f64,
    req_timing: f64,
    req_timing_nights: i64,
    req_timing_discount: f64,
    urgency: f64,
    partial_discount: f64,
    partial_done: f64,
    partial_nights: i64,
    plan_util: f64,
    plan_q: f64,
    kappa: f64,
    band_prob: bool,     // program from P(band) of every target (else from the posterior median band)
    fillers: bool,       // empty fibres may take finished targets short only of their program multiplier
    t_breaks: bool,      // re-choose the duration among saturation times and slot boundaries on the final pointing
    t_sat_margin: f64,   // a saturation-time candidate is the time to reach this factor (estimated quality)
    resync_ledger: bool, // state_resync: factors from the exposure ledger (else best score / weight / top multiplier)
    lambda_share: bool,  // rank anchors with lambda / n_fibers (else / 16, the reference 4 x 4 grid)
    odd_ring: bool,      // odd grids: density anchors also on the centre fibre's 4 neighbours
    // season supply/demand urgency (urgency.rs)
    urg_band: bool,      // per-cell supply/demand factor clip((u_c / u_mean)^alpha, lo, hi)
    urg_alpha: f64,
    urg_lo: f64,
    urg_hi: f64,
    urg_cap: f64,        // cap of cell/LP factor x base 1 + urgency / nights_left (the base alone: uncapped)
    urg_rel_good: f64,   // good altitude: airmass term within this factor of the cell's best (transit)
    urg_cell_deg: f64,
    urg_scarcity: bool,  // scale alpha with the card's time scarcity, like the time price
    urg_norm: bool,      // normalise the factors to mean 1 over each decision's candidate pool
    urg_dens_lo: f64,    // alpha x clamp((candidates per fibre - lo) / span, 0, 1): off on sparse catalogues
    urg_dens_span: f64,
    urg_in_window: bool, // a cell's boost only while the target is at a good altitude (discounts always)
    urg_econ: bool,      // demand only from targets the remaining supply can afford (market-clearing price)
    // --- deadline-aware required value (penalty x P_now x (1 - P_future)) ---
    req_deadline: bool,     // 1: required bonus from the target's remaining calendar; 0: REQUIRED_BONUS and discounts
    req_kappa: f64,         // P_future never above this (competition, correlated weather, faults)
    req_night_weight: f64,  // weight of one future night in P_future (nights are correlated, phi_night ~0.8)
    req_sigma_now: f64,     // sd of ln(g) predicted for a short exposure starting now
    req_sigma_drift: f64,   // ... plus this much per hour of exposure (in quadrature)
    req_prior_scale: f64,   // prior median of a night's quality level (scale) before any night is measured
    req_prior_sigma: f64,   // prior sd of ln(nightly scale)
    req_prior_nights: f64,  // weight of the prior, in nights
    req_p_open: f64,        // chance a future night is open at the target's best window
    req_p_open_bad: f64,    // ... on a night forecast with bad weather over the whole sky
    req_stake_mult: f64,    // stake = penalty x this (1.6 matches the fixed REQUIRED_BONUS, 80 for a 50 penalty);
                            // a lower stake accepts a few more misses for more science
    req_top_nights: usize,  // P_future counts only this many best later nights (0: all)
    req_close: [f64; 4],    // early-closing band: [g, frac, xmax, ra]: an open required target whose last good night
                            // (p_k >= 0.5) is before frac x the season and whose xbest < xmax has P_future scaled by
                            // min(1, later good nights / (g x open such targets within +-ra deg of RA)); g 0: off
    idle_up: [f64; 2],      // idle upside: [k, scarcity below which it applies]; k 0 = off
    idle_hours: f64,        // ... only after the plain search found nothing for this long
    idle_used: f64,         // ... and once the season LP uses under this share of the remaining capacity
    req_futile: (i64, f64), // hard required futility guard: after n long attempts whose measured factor,
                            // scaled to the longest exposure, stayed below x times the threshold, the stake decays
                            // 0.8 per further such attempt (n 0: off)
    // --- hard required targets: t_half at the typical sky > hrg_frac x longest exposure ---
    hrg: bool,
    hrg_frac: f64,
    hrg_early: [f64; 2], // a target whose window closes before this share of the season (the western band at dusk) is
                         // hard already at t_half > hrg_early[0] x longest exposure (0 or >= hrg_frac: off)
    hrg_cal: [f64; 4],  // success of a long attempt: (1 - tail T/3600) Phi((ln raw - bias T/3600) / sd(T)), sd^2 = sd0^2 + (sd1 T/3600)^2
    hrg_end: i64,       // a target's last nights keep the plain deadline-aware value (any attempt may be the last)
    hrg_good: [f64; 4], // full stake when tonight's quality is above this z of the nightly spread (<= -9: any measured night), |HA mid| < h, Moon factor >=, raw(T_max) >=
    hrg_anchors: usize,
    terrain_learn: bool,    // terrain sectors: P(clear) learnt from own hits (terrain.rs) instead of the hard cut
    // --- field geometry ---
    near_cover: bool,    // each anchor's neighbourhood covers every field it is placed in (else NEIGHBOUR_RADIUS_DEG)
    margin_abs: f64,     // edge margin (deg) without pointing misses or once the offset is learned; <= 0: default
    scan_k: [usize; 2],  // best distinct pointings given an exact cell-edge shift scan, at search levels 0 and 1
    scan_iters: usize,   // scan rounds per pointing at search level 0 (each from the last accepted shift); 1 above
    // level 0 xwide and mid (the pace's steps down): anchors, scan_k, scan_iters, refine_rounds, pool
    xw: [usize; 5],
    mid: [usize; 5],
    // level 0 narrow (the pace's step before level 1): anchors, scan pointings, 1 scan round
    n_anchors_narrow: usize,
    scan_k_narrow: usize,
    seeds: usize,        // mean-shift seeds (level 0): field-sized kernel from the densest patches / best anchors
    seed_iters: usize,   // mean-shift iterations per seed
    rate_excl_req: bool, // the time price also leaves out required values (paid once, like request rewards)
    lam_now: f64,        // the price follows min(slow, fast EMA of this weight) achievable rate (0: slow only)
    sat_bound: bool,     // no fresh quality sample: quality level >= the bound saturated hits show
    // --- complete-or-skip ---
    cs_final: usize,     // re-time this many scanned pointings at saturation times (1: the winner only)
    cs_pmax: f64,        // P_revisit ceiling (0: off)
    cs_eta: f64,         // useful share of the exposed fibre-seconds
    cs_beta: f64,        // partial gain x (1 - P_revisit (1 - reach^beta))
    fov: f64,            // fibre option cost weight (search.rs option_cost; 0: off)
    fov_beta: f64,       // ... used share of the surplus: 0 min(S, V reach) / S, else S reach^beta
    // --- tiling (tiling.rs) ---
    tile_passes: usize,  // offset passes of fixed field centres (0: no tiles)
    tile_k: usize,       // best tiles offered to the search per decision (level 0)
    tile_spacing: f64,   // centre spacing / field side
    timing: f64,         // hour-angle preference exponent (0: off; tiling.rs timing_factor)
    // --- research only: PRO_PATH_JITTER=<seed> moves every candidate's net by up to +-eps x its gain, so
    // near-ties (within eps) fall the other way and the run takes another, equally intended path; 0 = off (exact)
    path_jitter: u64,
    threads: usize,      // search workers per decision: the platform gives 2 CPUs and charges min(turn, CPU)
    path_jitter_eps: f64,
}

/// What the default constants scale with on a card that is not the reference 4 x 4 / 0.4 deg2 layout.
/// Every factor is exactly 1.0 on the reference layout and the default score config, so the defaults
/// apply there unscaled.
struct Geo {
    fov_scale: f64,   // field side / reference field side
    glass_scale: f64, // fibre side / reference fibre side
    pitch_scale: f64, // fibre pitch / reference pitch
    penalty_scale: f64, // required penalty per missing target / 50
}

impl Params {
    /// Distances (`PRO_NEIGHBOUR_RADIUS_DEG`, `PRO_DENSE_BIN_DEG`, `PRO_EDGE_MARGIN_DEG`, `PRO_REFINE`) are given
    /// for the reference layout and scaled with the card's field / fibre size; `PRO_REQUIRED_BONUS` is given for
    /// a 50-point required penalty and scaled with the card's penalty.
    fn load(geo: &Geo) -> Params {
        // PRO_GEO_SCALE=0: the reference absolute distances on every layout
        let geo_on = env_i("GEO_SCALE", 1) != 0;
        // the edge margin guards against pointing errors, an absolute angle: shrink it with small fibres
        // (0.02 deg would cost 30 % of a 0.245 deg fibre), never widen it beyond the reference value
        let margin_scale = geo.glass_scale.min(1.0);
        let geo = &Geo {
            fov_scale: if geo_on { geo.fov_scale } else { 1.0 },
            glass_scale: if geo_on { margin_scale } else { 1.0 },
            pitch_scale: if geo_on { geo.pitch_scale } else { 1.0 },
            penalty_scale: geo.penalty_scale,
        };
        let refine = env_f("REFINE", 0.1) * geo.pitch_scale;
        let mut refine_steps = Vec::new();
        if refine > 0.0 {
            for dn in [-1.0, 0.0, 1.0] {
                for de in [-1.0, 0.0, 1.0] {
                    if dn != 0.0 || de != 0.0 {
                        refine_steps.push((dn * refine, de * refine));
                    }
                }
            }
        }
        Params {
            lambda_frac: env_f("LAMBDA_FRAC", 0.68),
            lambda_ema: env_f("LAMBDA_EMA", 0.03),
            scarcity_ref: env_f("SCARCITY_REF", 0.86),
            scarcity_power: env_f("SCARCITY_POWER", 0.7),
            n_anchors: env_i("N_ANCHORS", 168).max(0) as usize,
            n_dense: env_i("N_DENSE", 20).max(0) as usize,
            dense_bin_deg: env_f("DENSE_BIN_DEG", 2.5) * geo.fov_scale,
            refine_rounds: env_i("REFINE_ROUNDS", 12),
            refine_fixed_t: env_i("REFINE_FIXED_T", 1) != 0,
            refine_steps,
            pool: env_i("POOL", 3600).max(1) as usize,
            neighbour_radius_deg: env_f("NEIGHBOUR_RADIUS_DEG", 2.1) * geo.fov_scale,
            edge_margin_deg: env_f("EDGE_MARGIN_DEG", 0.04) * geo.glass_scale,
            min_t: env_i("MIN_T", 0) as f64,
            plan_factor_safety: env_f("PLAN_FACTOR_SAFETY", 1.0),
            required_bonus: env_f("REQUIRED_BONUS", 80.0) * geo.penalty_scale,
            req_p_lo: env_f("REQ_P_LO", 0.95),
            req_p_hi: env_f("REQ_P_HI", 1.35),
            request_mult: env_f("REQUEST_MULT", 4.0),
            req_pb: env_i("REQ_PB", 1) != 0,
            req_window: env_i("REQ_WINDOW", 1) != 0,
            req_anchors: env_i("REQ_ANCHORS", 8).max(0) as usize,
            req_rate_excl: env_i("REQ_RATE_EXCL", 1) != 0,
            req_q_sigma: env_f("REQ_Q_SIGMA", 0.5).max(0.05),
            request_p_open: env_f("REQUEST_P_OPEN", 0.9),
            req_cap: env_i("REQ_CAP", 1) != 0,
            req_cap_step: env_f("REQ_CAP_STEP", 120.0).max(30.0),
            forecast_discount: env_f("FORECAST_DISCOUNT", 0.2),
            req_timing: env_f("REQ_TIMING", 0.85),
            req_timing_nights: env_i("REQ_TIMING_NIGHTS", 5),
            req_timing_discount: env_f("REQ_TIMING_DISCOUNT", 0.3),
            urgency: env_f("URGENCY", 2.5),
            partial_discount: env_f("PARTIAL_DISCOUNT", 1.0),
            partial_done: env_f("PARTIAL_DONE", 0.9),
            partial_nights: env_i("PARTIAL_NIGHTS", 3),
            plan_util: env_f("PLAN_UTIL", 0.55),
            plan_q: env_f("PLAN_Q", 0.7),
            kappa: env_f("KAPPA", 1.0),
            band_prob: env_i("BAND_PROB", 1) != 0,
            fillers: env_i("FILLERS", 1) != 0,
            t_breaks: env_i("T_BREAKS", 1) != 0,
            t_sat_margin: env_f("T_SAT_MARGIN", 1.05),
            resync_ledger: env_i("RESYNC_LEDGER", 1) != 0,
            lambda_share: env_i("LAMBDA_SHARE", 1) != 0,
            odd_ring: env_i("ODD_RING", 1) != 0,
            urg_band: env_i("URG_BAND", 1) != 0,
            urg_alpha: env_f("URG_ALPHA", 0.9),
            urg_lo: env_f("URG_LO", 0.7),
            urg_hi: env_f("URG_HI", 2.0),
            urg_cap: env_f("URG_CAP", 2.5),
            urg_rel_good: env_f("URG_REL_GOOD", 1.15),
            urg_cell_deg: env_f("URG_CELL_DEG", 5.0),
            urg_scarcity: env_i("URG_SCARCITY", 1) != 0,
            urg_norm: env_i("URG_NORM", 1) != 0,
            urg_dens_lo: env_f("URG_DENS_LO", 0.8),
            urg_dens_span: env_f("URG_DENS_SPAN", 0.6),
            urg_in_window: env_i("URG_IN_WINDOW", 1) != 0,
            urg_econ: env_i("URG_ECON", 1) != 0,
            req_deadline: env_i("REQ_DEADLINE", 1) != 0,
            req_kappa: env_f("REQ_KAPPA", 0.8),
            req_night_weight: env_f("REQ_NIGHT_WEIGHT", 0.35),
            req_sigma_now: env_f("REQ_SIGMA_NOW", 0.1),
            req_sigma_drift: env_f("REQ_SIGMA_DRIFT", 0.2),
            req_prior_scale: env_f("REQ_PRIOR_SCALE", 0.6),
            req_prior_sigma: env_f("REQ_PRIOR_SIGMA", 0.3),
            req_prior_nights: env_f("REQ_PRIOR_NIGHTS", 5.0),
            req_p_open: env_f("REQ_P_OPEN", 0.9),
            req_p_open_bad: env_f("REQ_P_OPEN_BAD", 0.5),
            req_stake_mult: env_f("REQ_STAKE_MULT", 0.8),
            req_top_nights: env_i("REQ_TOP_NIGHTS", 5).max(0) as usize,
            req_close: [
                env_f("REQ_CLOSE_G", 3.0).max(0.0),
                env_f("REQ_CLOSE_FRAC", 0.6),
                env_f("REQ_CLOSE_XMAX", 1.6),
                env_f("REQ_CLOSE_RA", 10.0),
            ],
            req_futile: (env_i("REQ_FUTILE", 3).max(0), env_f("REQ_FUTILE_X", 0.8)),
            idle_up: [env_f("IDLE_UP", 1.5).max(0.0), env_f("IDLE_SCARCITY", 0.3)],
            idle_hours: env_f("IDLE_HOURS", 0.0).max(0.0),
            idle_used: env_f("IDLE_USED", 0.5),
            hrg: env_i("HRG", 1) != 0,
            hrg_frac: env_f("HRG_FRAC", 0.85),
            hrg_early: [env_f("HRG_FRAC_EARLY", 0.7), env_f("HRG_EARLY_FRAC", 0.6)],
            hrg_cal: [env_f("HRG_TAIL", 0.1), env_f("HRG_BIAS", 0.04), env_f("HRG_SD0", 0.05), env_f("HRG_SD1", 0.087)],
            hrg_end: env_i("HRG_END", 3),
            hrg_good: [env_f("HRG_TOP_Z", -9.0), env_f("HRG_HA", 1.5), env_f("HRG_MOON", 0.95), env_f("HRG_RAW", 1.25)],
            hrg_anchors: env_i("HRG_ANCHORS", 4).max(0) as usize,
            terrain_learn: env_i("TERRAIN_LEARN", 1) != 0,
            near_cover: env_i("NEAR_COVER", 1) != 0,
            margin_abs: env_f("MARGIN_ABS", 0.004).max(0.0),
            scan_k: [env_i("SCAN_K", 48).max(0) as usize, env_i("SCAN_K1", 1).max(0) as usize],
            scan_iters: env_i("SCAN_ITERS", 5).max(1) as usize,
            xw: [("N_ANCHORS", 112), ("SCAN_K", 32), ("SCAN_ITERS", 4), ("REFINE_ROUNDS", 10), ("POOL", 1800)]
                .map(|(name, v)| env_i(&format!("{name}_X"), v).max(1) as usize),
            mid: [("N_ANCHORS", 56), ("SCAN_K", 16), ("SCAN_ITERS", 3), ("REFINE_ROUNDS", 6), ("POOL", 900)]
                .map(|(name, v)| env_i(&format!("{name}_MID"), v).max(1) as usize),
            n_anchors_narrow: env_i("N_ANCHORS_NARROW", 12).max(0) as usize,
            scan_k_narrow: env_i("SCAN_K_NARROW", 4).max(0) as usize,
            seeds: env_i("SEEDS", 20).max(0) as usize,
            seed_iters: env_i("SEED_ITERS", 5).max(1) as usize,
            rate_excl_req: env_i("RATE_EXCL_REQ", 1) != 0,
            lam_now: env_f("LAM_NOW", 0.45).max(0.0).min(1.0),
            sat_bound: env_i("SAT_BOUND", 1) != 0,
            cs_final: env_i("CS_FINAL", 4).max(1) as usize,
            cs_pmax: env_f("CS_PMAX", 0.75).max(0.0).min(1.0),
            cs_eta: env_f("CS_ETA", 0.6).max(0.05),
            cs_beta: env_f("CS_BETA", 2.0).max(0.1),
            fov: env_f("FOV", 0.5).max(0.0),
            fov_beta: env_f("FOV_BETA", 0.0).max(0.0),
            tile_passes: env_i("TILE_PASSES", 3).clamp(0, 8) as usize,
            tile_k: env_i("TILE_K", 8).max(0) as usize,
            tile_spacing: env_f("TILE_SPACING", 0.98),
            timing: env_f("TIMING", 1.0).max(0.0),
            path_jitter: env_i("PATH_JITTER", 0).max(0) as u64,
            threads: env_i("THREADS", 2).clamp(1, 2) as usize,
            path_jitter_eps: env_f("PATH_JITTER_EPS", 1e-3).abs(),
        }
    }
}

/// A JSON number (or numeric string) as f64.
pub fn num(v: &Value) -> f64 {
    match v {
        Value::Number(n) => n.as_f64().unwrap_or(0.0),
        Value::String(s) => s.trim().parse().unwrap_or(0.0),
        Value::Bool(b) => f64::from(u8::from(*b)),
        _ => 0.0,
    }
}

fn truthy(v: &Value) -> bool {
    match v {
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().unwrap_or(0.0) != 0.0,
        Value::String(s) => !s.is_empty(),
        Value::Null => false,
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

fn str_of(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    values[values.len() / 2]
}

/// Median of positive values (geometric mean of the middle pair for an even count); 0 when empty.
fn geo_median(mut values: Vec<f64>) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] * values[n / 2]).sqrt()
    }
}

pub const PROGRAMS: [&str; 3] = ["DARK", "BRIGHT", "BACKUP"];

/// Prediction for one assigned target of the observe in flight.
#[derive(Clone)]
struct Pred {
    model: f64,
    band_model: f64,
    alt: f64,
    az: f64,
    pred: f64,
    dir_clean: bool,
    /// outside every announced sector event however wide it may be (fault readings)
    sector_clean: bool,
    fiber: Option<usize>,
}

/// What one observe showed about the site throughput (see `Planner::read_throughput`).
#[derive(Clone, Debug)]
pub struct Reading {
    pub start: f64,
    pub duration: f64,
    /// log W: median over clean unsaturated hits (None: no such hit)
    pub y: Option<f64>,
    pub hits: usize,
    pub mismatched: usize,
    /// hits whose multiplier was certain (only one band possible)
    pub certain: usize,
    /// assigned targets that landed on their fibre and scored 0 / more than 0
    pub zero: usize,
    pub positive: usize,
    /// lower bound of the band level tau*K/s (no efficiency) from the saturated hits' multipliers (0: none)
    pub band_lo: f64,
    /// W if every hit not certainly matched was mismatched (the largest W the multipliers allow)
    pub w_hi: f64,
    /// upper bound of the band level (INFINITY: none) and W if every hit not certainly mismatched was matched
    pub band_hi: f64,
    pub w_lo: f64,
}

/// (hours, model, program, matched, direction-clean) of a saturated hit.
type BandObs = (f64, f64, usize, bool, bool);
/// (alt, az, fiber, hit) of one assigned target.
type OffsetRow = (f64, f64, usize, bool);

/// One open request for the capacity check (Params::req_cap; requests.rs `p_capacity`).
pub(super) struct ReqCap {
    /// Planning value of the whole request: REQUEST_MULT x reward x priority x P(no closure).
    value: f64,
    needed: usize,
    threshold: f64,
    issued: f64,
    deadline: f64,
    /// Open targets and their usable starts from now on (needs at the typical sky).
    targets: Vec<usize>,
    samples: Vec<Vec<CapSample>>,
    /// P(capacity) when the request is worked from now (nothing served by the coming exposure).
    base: f64,
}

/// Window geometry of a request target at one sampled start: (start, latest end = night end or deadline, alt, az,
/// lunar factor).
pub(super) type CapGeo = (f64, f64, f64, f64, f64);

pub struct Planner {
    p: Params,
    lat: f64,
    lon: f64,
    pub min_alt: f64,
    pub nights: Vec<(f64, f64)>,
    /// The calendar's night_date of each night (local evening date; forecasts name nights by it).
    pub night_dates: Vec<String>,
    /// Observing slots per night (calendar slot_count).
    pub night_slots: Vec<usize>,
    survey_end: f64,
    pub slot_seconds: i64,
    grid: FiberGrid,
    /// Fibres nearest the field centre (density anchors, fast search levels).
    central: Vec<usize>,
    /// Exposure grid per search level, inside the card's [min, max] exposure range.
    level_durations: Vec<Vec<f64>>,
    pub min_exposure: f64,
    pub max_exposure: f64,
    /// Required targets: completion threshold (score config) and the factor at which one counts as safe.
    req_thr: f64,
    req_safe: f64,
    f0t0: f64,
    q0: f64,
    airmass_exponent: f64,
    band_dark: f64,
    band_bright: f64,
    multipliers: [f64; 3],
    mismatch: f64,
    lunar_model: LunarModel,
    pub ids: Vec<String>,
    index_of: HashMap<String, usize>,
    ra: Vec<f64>,
    dec: Vec<f64>,
    flux: Vec<f64>,
    weight: Vec<f64>,
    pub required: Vec<bool>,
    cos_dec: Vec<f64>,
    hmax: Vec<f64>,
    factor: Vec<f64>,
    cur: Vec<f64>,
    rate_ema: f64,
    /// Fast EMA of the achievable rate (Params::lam_now).
    rate_now: f64,
    /// (hour, night, [E samples]): quality level / band level.
    pub e_hours: Vec<(i64, usize, Vec<f64>)>,
    e_ratios: VecDeque<(f64, f64)>,
    misses: Vec<i64>,
    vcache: Option<Vec<f64>>,
    vdirty: HashSet<usize>,
    attempts: Vec<i64>,
    /// Long attempts on a hard required target that stayed well below the threshold (Params::req_futile).
    futile: Vec<i64>,
    active: Vec<usize>,
    alt_a: Vec<f64>,
    alt_b: Vec<f64>,
    alt_c: Vec<f64>,
    sin_alt_limit: f64,
    cells: HashMap<i64, Vec<(f64, usize)>>,
    cell_ras: HashMap<i64, Vec<f64>>,
    first_night: Vec<i64>,
    last_night: Vec<i64>,
    ideal_model: Vec<f64>,
    scarcity: f64,
    lambda_frac: f64,
    pub scale: f64,
    prior_scale: f64,
    samples: VecDeque<(f64, f64)>,
    /// (hours, quality ratio lower bound) of observes whose hits saturated: a saturated hit only bounds the quality.
    sat_bounds: VecDeque<(f64, f64)>,
    all_ratios: VecDeque<f64>,
    /// Quality ratios over a much longer span (typical sky for request feasibility), and their cached median.
    long_ratios: VecDeque<f64>,
    long_quality: Option<(i64, f64)>,
    pending_cmd: Option<(f64, f64)>,
    pub bad_forecast: bool,
    offset: (f64, f64),
    offset_evidence: VecDeque<((f64, f64), Vec<OffsetRow>)>,
    offset_scores: Option<Vec<i64>>,
    offset_step: f64,
    offset_steps: i64,
    offset_grid: Vec<(f64, f64)>,
    offset_misses: i64,
    offset_updates: i64,
    /// Uncertainty (deg) of the adopted pointing offset: spread of the offsets that explain the evidence equally well.
    offset_sigma: Option<f64>,
    /// Where the chosen pointing came from: anchor placement, mean-shift seed, shift scan, refine nudge, tile, re-timed.
    geo_stats: [u64; 6],
    pub band_level: Option<f64>,
    /// Last band level fitted to >= 4 saturated hits (the band carries no instrument efficiency, so a fault
    /// leaves it alone; the quality-based guess scale / 0.95 does not; fit_band_level).
    band_fit: Option<f64>,
    band_obs: VecDeque<BandObs>,
    /// Posterior over the atmospheric band level (program choice).
    bandf: BandFilter,
    /// Finished targets visible now that could still gain their program multiplier (fillers of empty fibres).
    filler: Vec<bool>,
    pending: Vec<(usize, Pred)>,
    pending_program: usize,
    pending_duration: f64,
    pending_start: f64,
    /// Site throughput read from the last observe (fault detector input), taken by main.rs.
    pub reading: Option<Reading>,
    /// Longest duration allowed for the next exposure (a slot-contained confirmation exposure), set by main.rs.
    pub max_t: Option<f64>,
    /// No idle upside while a duty log is being followed: its reports need the short reaction time of
    /// slot-long waits.
    pub idle_off: bool,
    /// Night of the observe in flight, the ALL-sky weather it was planned under, and whether that weather (or a
    /// site closure) changed while it ran (then its hits say little about the band level now).
    pending_night: usize,
    pending_weather: BTreeSet<String>,
    pending_mixed: bool,
    pending_end: f64,
    /// Instrument efficiency assumed for the observe in flight (band level = quality level / efficiency).
    pending_eff: f64,
    blocked: Vec<(f64, f64)>,
    /// (event kind, direction) of the current bulletin, terrain excluded.
    pub notices: BTreeSet<(String, String)>,
    terrain: BTreeSet<String>,
    pub extra_avoid: BTreeSet<String>,
    /// duty log (rule decoder + LLM duty-log stage): (label azimuth, true top altitude) of terrain sectors.
    pub terrain_top: Vec<(f64, f64)>,
    /// duty-log closures in force (compass azimuth +-22.5 deg, closed below this altitude): wind, launch, thick cloud, bright low sky
    pub duty_closed: Vec<(f64, f64)>,
    /// duty-log closures starting within the longest exposure: (seconds to the start, compass azimuth, below altitude)
    pub duty_closing: Vec<(f64, f64, f64)>,
    /// Multiplier on the time price lambda for the week (LLM week plan; 1 = the rule).
    pub lambda_scale: f64,
    /// Priority of each observation request by id (LLM request triage; missing = 1, the rule).
    pub request_priority: HashMap<String, f64>,
    pub fast_level: usize,
    /// Level 0 breadth chosen by the pace: 0 = wide (N_ANCHORS, SCAN_K, ...), 1 = xwide (*_X), 2 = mid (*_MID), 3 = narrow.
    pub breadth: usize,
    /// Open request targets: summed request value (planning value) and each request's term (value, window).
    request_bonus: HashMap<usize, f64>,
    request_terms: HashMap<usize, Vec<RequestTerm>>,
    /// Last logged (remaining, open targets, rounded P(success)) of each request.
    request_log: HashMap<String, (i64, usize, i64)>,
    /// Request capacity (Params::req_cap): one entry per open request, the requests each open target belongs to
    /// (index into request_caps, bit of the target there), a flag per target for the search's inner loop, and the
    /// window geometry of each request's targets (built once per request).
    request_caps: Vec<ReqCap>,
    cap_of: HashMap<usize, Vec<(usize, u32)>>,
    cap_flag: Vec<bool>,
    cap_geo: HashMap<String, HashMap<usize, Vec<CapGeo>>>,
    /// Last logged (needed, rounded P(capacity) x 20) of each request.
    cap_log: HashMap<String, (usize, i64)>,
    density_order: Vec<usize>,
    planned: Vec<bool>,
    plan_night: i64,
    dbg_none_hour: Option<i64>,
    dbg_idle_night: std::sync::atomic::AtomicUsize,
    /// Since when the plain search has found nothing (f64 bits, NaN = it found something at the last decision).
    idle_since: std::sync::atomic::AtomicU64,
    dbg_band_night: Option<usize>,
    /// (target, observe index, estimated factor) of every positive hit, for an exact state_resync.
    ledger: Vec<(usize, i64, f64)>,
    /// The engine's observe_action_index of the coming decision, and of the observe in flight.
    pub observe_index: Option<i64>,
    pending_index: Option<i64>,
    /// Season supply/demand urgency per sky cell (None: the base last-chance urgency alone).
    urg: Option<Urgency>,
    /// Catalogue targets per fibre patrol area (targets per deg2 of footprint x fibre area); gates the urgency.
    fibre_density: f64,
    /// Fibre-seconds exposed so far (assigned fibres x duration): the supply rate the urgency's demand price uses.
    urg_supply: f64,
    /// Season LP (seasonlp.rs; None: off).
    slp: Option<SeasonLp>,
    /// Complete-or-skip: P(a later exposure supersedes a partial one) per target (values.rs).
    cs_p: Vec<f64>,
    // --- deadline-aware required value (Params::req_deadline) ---
    /// Stake of an unfinished required target: penalty per missing target (score config) x Params::req_stake_mult.
    req_stake: f64,
    /// Calendar row of each required target (usize::MAX: none).
    req_row: Vec<usize>,
    /// Per row and night: ln(best factor reachable that night at quality level 1 / threshold); -inf when the
    /// target is never up for a whole exposure that night.
    req_x: Vec<Vec<f32>>,
    /// P(the target still reaches the threshold on a later night), per target; refreshed at every night start.
    req_pfuture: Vec<f64>,
    /// Lower bound of each target's best factor: a hit's score / weight is factor x multiplier, and when the
    /// sky band is misjudged the multiplier is the declared one, not 1.0 (or the reverse); the lower of the two
    /// readings is certain. A required target counts as done only on this bound.
    req_lo: Vec<f64>,
    req_night: i64,
    /// Hard required targets, each row's best ln(factor / threshold) at quality 1, and the nightly
    /// quality's ln-mean and sd (update_required_outlook).
    req_hard: Vec<bool>,
    req_xbest: Vec<f64>,
    req_mu: (f64, f64),
    /// ln(quality ratio) samples of the current night, and ln(median quality level) of each past night.
    night_ln_ratios: Vec<f64>,
    ln_scale_nights: Vec<f64>,
    /// Night indices forecast bad over the whole sky (set by main at every night start).
    pub forecast_bad_nights: HashSet<usize>,
    /// An earthquake notice is in the current bulletin (set by main).
    pub quake_active: bool,
    /// One belief per terrain notice direction (Params::terrain_learn).
    terrain_beliefs: Vec<TerrainBelief>,
    /// Fixed field centres (tiling.rs) and, per target, the last night whose window holds its transit.
    tiles: Vec<tiling::Tile>,
    last_transit: Vec<i32>,
}

fn push_bounded<T>(deque: &mut VecDeque<T>, item: T, maxlen: usize) {
    if deque.len() == maxlen {
        deque.pop_front();
    }
    deque.push_back(item);
}

/// The base exposure grid `stock` restricted to the card's [min, max] exposure range (plus its limits; a longer maximum
/// adds longer steps).
fn duration_grid(stock: &[f64], min_exposure: f64, max_exposure: f64) -> Vec<f64> {
    let stock_max = DURATIONS[DURATIONS.len() - 1];
    let mut d: Vec<f64> = stock.iter().cloned().filter(|&t| t >= min_exposure && t <= max_exposure).collect();
    if max_exposure > stock_max {
        // longer exposures allowed: extend in steps of x1.25 up to the maximum
        let mut t = stock_max;
        loop {
            t = (t * 1.25).round();
            if t >= max_exposure {
                break;
            }
            d.push(t);
        }
        d.push(max_exposure);
    } else if stock.iter().any(|&t| t > max_exposure) {
        d.push(max_exposure); // the grid was cut by a shorter maximum: keep the longest allowed exposure
    }
    if min_exposure > stock[0] && stock.iter().any(|&t| t < min_exposure) {
        d.push(min_exposure); // ... and the shortest one when it was cut by a longer minimum
    }
    if d.is_empty() {
        d.push(min_exposure);
    }
    d.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    d.dedup();
    d
}

/// Catalogue targets per fibre area (targets per deg2 of footprint x fibre area), and the footprint area: from
/// the footprint polygons (shoelace in RA x sin dec), else the occupied 1 x 1 deg cells.
fn fibre_density(init: &Value, ra: &[f64], dec: &[f64], fibre_area: f64) -> (f64, f64) {
    let cells: std::collections::HashSet<(i64, i64)> =
        ra.iter().zip(dec).map(|(&r, &d)| (d.floor() as i64, (r.rem_euclid(360.0) * d.to_radians().cos()).floor() as i64)).collect();
    let binned = cells.len() as f64;
    let mut poly = 0.0;
    for comp in init["footprint"].as_array().into_iter().flatten() {
        let v: Vec<(f64, f64)> = comp["vertices"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|p| Some((p.get(0)?.as_f64()?, p.get(1)?.as_f64()?.to_radians().sin())))
            .collect();
        let mut x = Vec::with_capacity(v.len());
        for (k, &(r, _)) in v.iter().enumerate() {
            x.push(if k == 0 { r } else { x[k - 1] + wrap180(r - v[k - 1].0) });
        }
        let twice: f64 = (0..v.len()).map(|k| x[k] * v[(k + 1) % v.len()].1 - x[(k + 1) % v.len()] * v[k].1).sum();
        poly += 0.5 * twice.abs() * 180.0 / std::f64::consts::PI;
    }
    let area = if poly > 0.5 * binned && poly < 2.0 * binned { poly } else { binned };
    (ra.len() as f64 * fibre_area / area.max(1.0), area)
}

impl Planner {
    pub fn new(init: &Value) -> Planner {
        let site = &init["site"];
        let lat = num(&site["latitude_deg"]);
        let lon = num(&site["longitude_deg"]);
        let min_alt = num(&site["minimum_altitude_deg"]);
        let survey = &init["survey"];
        let nights: Vec<(f64, f64)> = survey["nights"]
            .as_array()
            .map(|a| {
                a.iter()
                    .map(|n| (parse_utc(n["observing_start_utc"].as_str().unwrap_or("")), parse_utc(n["observing_end_utc"].as_str().unwrap_or(""))))
                    .collect()
            })
            .unwrap_or_default();
        let night_dates: Vec<String> = survey["nights"]
            .as_array()
            .map(|a| a.iter().map(|n| n.get("night_date").and_then(Value::as_str).unwrap_or("").to_string()).collect())
            .unwrap_or_default();
        let survey_end = parse_utc(survey["end_utc"].as_str().unwrap_or(""));
        let slot_seconds = num(&survey["slot_seconds"]) as i64;
        // observing slots of each night (the engine's slot list: faults are placed by index into it)
        let night_slots: Vec<usize> = survey["nights"]
            .as_array()
            .map(|a| {
                a.iter()
                    .zip(&nights)
                    .map(|(n, &(s, e))| match n.get("slot_count").map(num) {
                        Some(c) if c >= 1.0 => c as usize,
                        _ => ((e - s) / (if slot_seconds > 0 { slot_seconds as f64 } else { 900.0 })).ceil().max(1.0) as usize,
                    })
                    .collect()
            })
            .unwrap_or_default();
        let instrument = &init["instrument"];
        let grid = FiberGrid::new(instrument);
        let mut min_exposure = num(&instrument["exposure"]["min_duration_seconds"]).trunc();
        let mut max_exposure = num(&instrument["exposure"]["max_duration_seconds"]).trunc();
        if !(min_exposure > 0.0) {
            min_exposure = 60.0;
        }
        if !(max_exposure >= min_exposure) {
            max_exposure = min_exposure.max(3600.0);
        }
        let score = &init["scoring"];
        let opt = |v: &Value, default: f64| if v.is_null() { default } else { num(v) };
        let req_thr = opt(&score["required"]["observed_factor_threshold"], 0.5);
        let req_thr = if req_thr > 0.0 && req_thr <= 1.0 { req_thr } else { 0.5 };
        let req_penalty = opt(&score["required"]["penalty_per_missing"], 50.0).abs();
        let geo = Geo {
            fov_scale: grid.fov_scale(),
            glass_scale: grid.glass_scale(),
            pitch_scale: grid.pitch_scale(),
            penalty_scale: if req_penalty > 0.0 { req_penalty / 50.0 } else { 1.0 },
        };
        let p = Params::load(&geo);
        let p_req_stake_mult = p.req_stake_mult;
        // 0.62 at the usual threshold 0.5; the same headroom ratio elsewhere (capped below DONE_FACTOR)
        let req_safe = if req_thr == 0.5 { REQUIRED_SAFE_FACTOR } else { (REQUIRED_SAFE_FACTOR / 0.5 * req_thr).min(DONE_FACTOR) };
        let central = grid.central_fibers(p.odd_ring);
        let level_durations: Vec<Vec<f64>> = [&DURATIONS[..], &LEVEL_DURATIONS_1[..], &LEVEL_DURATIONS_2[..], &LEVEL_DURATIONS_3[..]]
            .iter()
            .map(|d| duration_grid(d, min_exposure, max_exposure))
            .collect();
        log(&format!(
            "planner: grid {}x{} pitch {:.4} fov {:.3} deg; radius {:.3} bin {:.3} margin {:.4} refine {:.4}; central {:?}; exposure [{}, {}]; required thr {} bonus {:.1}",
            grid.side, grid.side, grid.pitch, grid.fov, p.neighbour_radius_deg, p.dense_bin_deg, p.edge_margin_deg,
            p.refine_steps.first().map_or(0.0, |s| s.0.abs()), central, min_exposure, max_exposure, req_thr, p.required_bonus
        ));
        let f0t0 = num(&score["flux_zero_point"]) * num(&score["exposure_zero_point_seconds"]);
        let q0 = num(&score["q0"]);
        let airmass_exponent = num(&score["airmass_exponent"]);
        let program = &score["program"];
        let multipliers = [
            num(&program["multipliers"]["DARK"]),
            num(&program["multipliers"]["BRIGHT"]),
            num(&program["multipliers"]["BACKUP"]),
        ];
        let lm = &score["lunar_model"];
        let lunar_model = LunarModel {
            maximum_penalty: num(&lm["maximum_penalty"]),
            altitude_exponent: num(&lm["altitude_exponent"]),
            angular_decay_scale_deg: num(&lm["angular_decay_scale_deg"]),
        };

        let columns: Vec<String> = init["targets"]["columns"].as_array().map(|a| a.iter().map(str_of).collect()).unwrap_or_default();
        let col = |name: &str| columns.iter().position(|c| c == name).unwrap_or(usize::MAX);
        let (c_id, c_ra, c_dec, c_flux, c_w, c_req) =
            (col("target_id"), col("ra_deg"), col("dec_deg"), col("feature_flux"), col("science_weight"), col("required"));
        let empty = Vec::new();
        let rows = init["targets"]["rows"].as_array().unwrap_or(&empty);
        let cell = |row: &Value, c: usize| row.get(c).cloned().unwrap_or(Value::Null);
        let ids: Vec<String> = rows.iter().map(|r| str_of(&cell(r, c_id))).collect();
        let index_of = ids.iter().enumerate().map(|(i, id)| (id.clone(), i)).collect();
        let ra: Vec<f64> = rows.iter().map(|r| num(&cell(r, c_ra))).collect();
        let dec: Vec<f64> = rows.iter().map(|r| num(&cell(r, c_dec))).collect();
        let flux: Vec<f64> = rows.iter().map(|r| num(&cell(r, c_flux))).collect();
        let weight: Vec<f64> = rows.iter().map(|r| num(&cell(r, c_w))).collect();
        let required: Vec<bool> = rows.iter().map(|r| truthy(&cell(r, c_req))).collect();
        let n = rows.len();
        let sin_dec: Vec<f64> = dec.iter().map(|d| d.to_radians().sin()).collect();
        let cos_dec: Vec<f64> = dec.iter().map(|d| d.to_radians().cos()).collect();
        let sin_lat = lat.to_radians().sin();
        let cos_lat = lat.to_radians().cos();
        let hmax: Vec<f64> = dec.iter().map(|&d| max_hour_angle_deg(d, lat, min_alt + ALT_MARGIN_DEG)).collect();
        let active: Vec<usize> = (0..n).filter(|&i| hmax[i] > 0.0).collect();
        let alt_a = sin_dec.iter().map(|s| sin_lat * s).collect();
        let alt_b = (0..n).map(|i| cos_lat * cos_dec[i] * ra[i].to_radians().cos()).collect();
        let alt_c = (0..n).map(|i| cos_lat * cos_dec[i] * ra[i].to_radians().sin()).collect();
        // best sky model a target can ever get: at transit, Moon down (used to time faint required targets)
        let ideal_model = dec
            .iter()
            .map(|&d| 1.0 / (q0 * normalized_airmass((90.0 - (d - lat).abs()).max(1.0)).powf(airmass_exponent)))
            .collect();
        // Time scarcity: fibre-seconds the catalogue needs (typical sky) / night seconds on offer. A time-rich
        // season should price telescope time lower than a tight one.
        let night_seconds = psum(nights.iter().map(|(s, e)| e - s));
        let need = psum(flux.iter().map(|&f| max_exposure.min(f0t0 / (f.max(1e-3) * TYPICAL_Q)))) / grid.n as f64;
        let scarcity = need / night_seconds.max(1.0);
        let lambda_frac = p.lambda_frac * 1.0f64.min((scarcity / p.scarcity_ref).max(0.2).powf(p.scarcity_power));
        log(&format!("planner: scarcity {:.2}, time price fraction {:.2}", scarcity, lambda_frac));
        let (fibre_density, footprint_area) = fibre_density(init, &ra, &dec, grid.glass * grid.glass);
        log(&format!("planner: {:.2} targets per fibre area ({} targets on {:.0} deg2)", fibre_density, n, footprint_area));
        log(&format!(
            "planner: geometry near_cover {} margin_abs {} scan_k {:?} seeds {}x{}",
            p.near_cover, p.margin_abs, p.scan_k, p.seeds, p.seed_iters
        ));
        let mut density_order: Vec<usize> = (0..n).collect();
        // stable sort by -w*flux (ties keep catalogue order)
        density_order.sort_by(|&a, &b| (-weight[a] * flux[a]).partial_cmp(&(-weight[b] * flux[b])).unwrap_or(std::cmp::Ordering::Equal));
        let offset_step = grid.pitch / 12.0;
        let mut planner = Planner {
            lat,
            lon,
            min_alt,
            night_dates,
            night_slots,
            survey_end,
            slot_seconds,
            central,
            level_durations,
            min_exposure,
            max_exposure,
            req_thr,
            req_safe,
            f0t0,
            q0,
            airmass_exponent,
            band_dark: num(&program["bands"]["DARK"]),
            band_bright: num(&program["bands"]["BRIGHT"]),
            multipliers,
            mismatch: num(&program["mismatch_multiplier"]),
            lunar_model,
            ids,
            index_of,
            factor: vec![0.0; n],
            cur: vec![0.0; n],
            rate_ema: 0.0,
            rate_now: 0.0,
            e_hours: Vec::new(),
            e_ratios: VecDeque::new(),
            misses: vec![0; n],
            vcache: None,
            vdirty: HashSet::new(),
            attempts: vec![0; n],
            futile: vec![0; n],
            active,
            alt_a,
            alt_b,
            alt_c,
            sin_alt_limit: (min_alt + ALT_MARGIN_DEG).to_radians().sin(),
            cells: HashMap::new(),
            cell_ras: HashMap::new(),
            first_night: Vec::new(),
            last_night: Vec::new(),
            ideal_model,
            scarcity,
            lambda_frac,
            scale: 1.0,
            prior_scale: 1.0,
            samples: VecDeque::new(),
            sat_bounds: VecDeque::new(),
            all_ratios: VecDeque::new(),
            long_ratios: VecDeque::new(),
            long_quality: None,
            pending_cmd: None,
            bad_forecast: false,
            offset: (0.0, 0.0),
            offset_evidence: VecDeque::new(),
            offset_scores: None,
            offset_step,
            offset_steps: OFFSET_STEPS,
            offset_grid: Vec::new(),
            offset_misses: 0,
            offset_updates: 0,
            offset_sigma: None,
            geo_stats: [0; 6],
            band_level: None,
            band_fit: None,
            band_obs: VecDeque::new(),
            bandf: BandFilter::new(num(&program["bands"]["DARK"]), num(&program["bands"]["BRIGHT"]), slot_seconds as f64),
            filler: vec![false; n],
            pending: Vec::new(),
            pending_program: 2,
            pending_duration: 0.0,
            pending_start: 0.0,
            reading: None,
            max_t: None,
            pending_night: 0,
            pending_weather: BTreeSet::new(),
            pending_mixed: false,
            pending_end: 0.0,
            pending_eff: 0.95,
            blocked: Vec::new(),
            notices: BTreeSet::new(),
            terrain: BTreeSet::new(),
            extra_avoid: BTreeSet::new(),
            terrain_top: Vec::new(),
            duty_closed: Vec::new(),
            duty_closing: Vec::new(),
            lambda_scale: 1.0,
            request_priority: HashMap::new(),
            fast_level: 0,
            breadth: 0,
            request_bonus: HashMap::new(),
            request_terms: HashMap::new(),
            request_log: HashMap::new(),
            request_caps: Vec::new(),
            cap_of: HashMap::new(),
            cap_flag: vec![false; n],
            cap_geo: HashMap::new(),
            cap_log: HashMap::new(),
            density_order,
            planned: vec![false; n],
            plan_night: -1,
            dbg_none_hour: None,
            dbg_idle_night: std::sync::atomic::AtomicUsize::new(usize::MAX),
            idle_off: false,
            idle_since: std::sync::atomic::AtomicU64::new(f64::NAN.to_bits()),
            dbg_band_night: None,
            ledger: Vec::new(),
            observe_index: None,
            pending_index: None,
            urg: None,
            fibre_density,
            urg_supply: 0.0,
            slp: None,
            cs_p: vec![0.0; n],
            req_stake: (if req_penalty > 0.0 { req_penalty } else { 50.0 }) * p_req_stake_mult,
            req_row: Vec::new(),
            req_x: Vec::new(),
            req_pfuture: vec![0.0; n],
            req_lo: vec![0.0; n],
            req_night: -1,
            req_hard: vec![false; n],
            req_xbest: Vec::new(),
            req_mu: (0.0, 1.0),
            night_ln_ratios: Vec::new(),
            ln_scale_nights: Vec::new(),
            forecast_bad_nights: HashSet::new(),
            quake_active: false,
            terrain_beliefs: Vec::new(),
            tiles: Vec::new(),
            last_transit: Vec::new(),
            ra,
            dec,
            flux,
            weight,
            required,
            cos_dec,
            hmax,
            nights,
            grid,
            p,
        };
        planner.build_index();
        planner.build_tiles();
        planner.build_windows();
        planner.build_transit_nights();
        planner.build_required_outlook();
        planner.offset_grid = planner.make_offset_grid();
        planner.build_urgency();
        planner.build_slp();
        planner
    }

    // --- precomputation --------------------------------------------------------------------------

    fn build_index(&mut self) {
        self.cells.clear();
        for &i in &self.active {
            self.cells.entry(self.dec[i].floor() as i64).or_default().push((self.ra[i], i));
        }
        for band in self.cells.values_mut() {
            band.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        }
        self.cell_ras = self.cells.iter().map(|(k, band)| (*k, band.iter().map(|(ra, _)| *ra).collect())).collect();
    }

    fn neighbours(&self, ra: f64, dec: f64, radius: f64) -> Vec<usize> {
        let mut found = Vec::new();
        let cos_dec = (89.0f64.min(dec.abs() + radius)).to_radians().cos().max(0.05);
        let width = radius / cos_dec;
        for key in ((dec - radius).floor() as i64)..=((dec + radius).floor() as i64) {
            let Some(band) = self.cells.get(&key) else { continue };
            if band.is_empty() {
                continue;
            }
            let ras = &self.cell_ras[&key];
            let mut spans = vec![(ra - width, ra + width)];
            if spans[0].0 < 0.0 {
                spans = vec![(0.0, spans[0].1), (spans[0].0 + 360.0, 360.0)];
            } else if spans[0].1 >= 360.0 {
                spans = vec![(spans[0].0, 360.0), (0.0, spans[0].1 - 360.0)];
            }
            for (low, high) in spans {
                let a = ras.partition_point(|&x| x < low);
                let b = ras.partition_point(|&x| x <= high);
                for k in a..b.max(a) {
                    found.push(band[k].1);
                }
            }
        }
        found
    }

    /// First and last night on which each target has at least 20 minutes above the limit.
    fn build_windows(&mut self) {
        let need = 20.0 * 60.0 * SIDEREAL_DEG_PER_SECOND;
        let spans: Vec<(f64, f64)> =
            self.nights.iter().map(|&(s, e)| (local_sidereal_deg(s, self.lon), (e - s) * SIDEREAL_DEG_PER_SECOND)).collect();
        self.first_night = vec![self.nights.len() as i64; self.ra.len()];
        self.last_night = vec![-1; self.ra.len()];
        for &i in &self.active {
            let h = self.hmax[i];
            for (k, &(l0, span)) in spans.iter().enumerate() {
                let overlap = if h >= 180.0 {
                    span
                } else {
                    let a = pmod(self.ra[i] - h - l0, 360.0);
                    (span.min(a + 2.0 * h) - a).max(0.0) + span.min(a - 360.0 + 2.0 * h).max(0.0)
                };
                if overlap >= need {
                    if self.first_night[i] > k as i64 {
                        self.first_night[i] = k as i64;
                    }
                    self.last_night[i] = k as i64;
                }
            }
        }
    }

    // --- messages and results --------------------------------------------------------------------

    pub fn on_messages(&mut self, messages: &[Value], latest_bulletin: &Value) {
        for message in messages {
            let kind = message.get("record_type").and_then(Value::as_str).unwrap_or("");
            if kind == "bulletin" && message.get("initial").map_or(false, truthy) {
                for notice in message.get("notices").and_then(Value::as_array).into_iter().flatten() {
                    if notice.get("event_kind").and_then(Value::as_str) == Some("terrain_obstruction") {
                        let direction = notice.get("direction").map(str_of).unwrap_or_default();
                        if let Some(label) = direction_az(&direction) {
                            if self.p.terrain_learn && self.terrain.insert(direction.clone()) {
                                self.terrain_beliefs.push(TerrainBelief::new(label));
                            }
                        }
                        self.terrain.insert(direction);
                    }
                }
            } else if kind == "state_resync" {
                self.resync(message);
            }
            if kind == "bulletin" && !self.pending.is_empty() {
                // a bulletin issued while the observe in flight ran: did the ALL-sky weather change under it?
                let issued = parse_utc(message.get("issued_at_utc").and_then(Value::as_str).unwrap_or(""));
                if issued < self.pending_end - 1.0 && all_sky_kinds(message) != self.pending_weather {
                    self.pending_mixed = true;
                }
            }
        }
        self.notices = latest_bulletin
            .get("notices")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter(|n| n.get("event_kind").and_then(Value::as_str) != Some("terrain_obstruction"))
            .map(|n| (n.get("event_kind").map(str_of).unwrap_or_default(), n.get("direction").map(str_of).unwrap_or_default()))
            .collect();
    }

    /// ALL-sky weather kinds (rain, storm, overcast, haze, cold_snap) of the current bulletin.
    fn weather_now(&self) -> BTreeSet<String> {
        self.notices.iter().filter(|(k, d)| d == "ALL" && SKY_WEATHER_KINDS.contains(&k.as_str())).map(|(k, _)| k.clone()).collect()
    }

    pub fn site_closed(&self) -> bool {
        self.notices.iter().any(|(k, d)| CLOSED_KINDS.contains(&k.as_str()) && d == "ALL")
    }

    /// Kinds of the current ALL-sky notices (fault detector segments).
    pub fn all_sky_kinds(&self) -> BTreeSet<String> {
        self.notices.iter().filter(|(_, d)| d == "ALL").map(|(k, _)| k.clone()).collect()
    }

    // --- planning --------------------------------------------------------------------------------

    pub fn current_night(&self, now: f64) -> Option<(usize, f64, f64)> {
        self.nights.iter().enumerate().find(|(_, &(s, e))| s <= now && now < e).map(|(k, &(s, e))| (k, s, e))
    }

    /// Fibres of the grid, and whether an id is in the catalogue (the output guard, guard.rs).
    pub fn n_fibers(&self) -> usize {
        self.grid.n
    }

    /// Search workers per decision (PRO_THREADS).
    pub fn threads(&self) -> usize {
        self.p.threads
    }

    pub fn is_target(&self, id: &str) -> bool {
        self.index_of.contains_key(id)
    }

    pub fn next_night_start(&self, now: f64) -> Option<f64> {
        self.nights.iter().map(|&(s, _)| s).find(|&s| s > now)
    }

    /// P(clear of every terrain sector) under the learnt beliefs.
    fn terrain_clear(&self, alt: f64, az: f64) -> f64 {
        let mut clear = 1.0;
        for b in &self.terrain_beliefs {
            // the duty log's ridge top of this sector: clear above it
            if self.terrain_top.iter().any(|&(label, top)| az_distance(label, b.label_az) < 1.0 && alt >= top + 1.5 && b.top_blocked < top + 1.5) {
                continue;
            }
            clear *= 1.0 - b.p_blocked(alt, az);
        }
        clear
    }

    /// Inside the assumed worst case of a published terrain sector (below 50 deg within 60 deg of its compass point)?
    fn terrain_blocked(&self, alt: f64, az: f64) -> bool {
        alt < 50.0 && self.terrain.iter().filter_map(|d| direction_az(d)).any(|daz| az_distance(az, daz) <= 60.0)
    }

    /// Chance of a clear line of sight past the published terrain sectors: P(clear) learnt from the own hits
    /// (terrain.rs; below 0.05 counts as blocked), or the fixed hard cut (PRO_TERRAIN_LEARN=0).
    fn terrain_factor(&self, alt: f64, az: f64) -> f64 {
        if self.p.terrain_learn {
            let clear = self.terrain_clear(alt, az);
            if clear < 0.05 {
                0.0
            } else {
                clear
            }
        } else if self.terrain_blocked(alt, az) {
            0.0
        } else {
            1.0
        }
    }

    fn direction_factor(&self, alt: f64, az: f64) -> f64 {
        let terrain = self.terrain_factor(alt, az);
        if terrain <= 0.0 {
            return 0.0;
        }
        self.weather_direction_factor(alt, az) * terrain
    }

    /// Directional bulletins, avoided sectors and learned blocked spots (everything but the terrain sectors).
    fn weather_direction_factor(&self, alt: f64, az: f64) -> f64 {
        // directions a duty-log closure shuts now (rule decoder + LLM duty-log stage)
        if self.duty_closed.iter().any(|&(c, below)| az_distance(az, c) <= 22.5 && alt < below) {
            return 0.0;
        }
        let mut factor: f64 = 1.0;
        for (kind, direction) in &self.notices {
            let Some(daz) = direction_az(direction) else { continue };
            let near = az_distance(az, daz) <= 67.5;
            if BLOCKING_KINDS.contains(&kind.as_str()) && near && alt < 62.0 {
                return 0.0;
            }
            if near && alt < 75.0 {
                factor = factor.min(0.35);
            }
        }
        // sectors the LLM week plan avoids tonight (forecast weather): a soft nudge for low pointings only, milder
        // than a live bulletin notice, because a forecast says the event touches the night, not when
        for direction in &self.extra_avoid {
            if let Some(daz) = direction_az(direction) {
                if az_distance(az, daz) <= AVOID_HALF_WIDTH_DEG && alt < AVOID_BELOW_ALT_DEG {
                    factor = factor.min(AVOID_FACTOR);
                }
            }
        }
        let start = self.blocked.len().saturating_sub(40);
        for &(blocked_az, blocked_alt) in &self.blocked[start..] {
            if az_distance(az, blocked_az) <= 12.0 && alt <= blocked_alt + 3.0 {
                factor = factor.min(0.2);
            }
        }
        factor
    }

    /// Distance (deg) target i must keep from its fibre's glass edge: MARGIN_ABS (float noise) on an exact pointing;
    /// on a Hard-mode card the default margin until the offset is learned, then max(MARGIN_ABS, its uncertainty).
    /// Never wider than the default margin; a target that missed keeps the default per-target widening.
    fn edge_margin(&self, i: usize) -> f64 {
        let stock = self.p.edge_margin_deg * (0.5 + 1.5 * self.misses[i] as f64);
        if self.p.margin_abs <= 0.0 {
            return stock;
        }
        let stock_clean = self.p.edge_margin_deg * 0.5;
        let base = match self.offset_sigma {
            Some(sigma) => self.p.margin_abs.max(sigma).min(stock_clean),
            None if self.offset_misses == 0 => self.p.margin_abs.min(stock_clean),
            None => stock_clean,
        };
        if self.misses[i] > 0 {
            base.max(stock)
        } else {
            base
        }
    }

    /// Outside every announced directional event, however wide (up to 120 deg of azimuth below 80 deg around the
    /// named 8-point direction, reference weather config).
    fn sector_clean(&self, alt: f64, az: f64) -> bool {
        !self.notices.iter().any(|(_, d)| direction_az(d).map_or(false, |daz| az_distance(az, daz) <= 82.5 && alt < 82.0))
    }

    /// Which targets will the season complete? Fill the remaining useful fibre-time with targets in
    /// order of value density (w * flux); the rest are fillers whose partial exposures are worth taking.
    fn season_plan(&mut self, now: f64, night_index: usize) {
        self.plan_night = night_index as i64;
        let mut cap = psum(self.nights.iter().filter(|&&(_, e)| e > now).map(|&(s, e)| (e - s.max(now)).max(0.0)));
        cap *= self.grid.n as f64 * self.p.plan_util;
        self.planned = vec![false; self.ids.len()];
        let mut n = 0;
        for &i in &self.density_order {
            if cap <= 0.0 {
                break;
            }
            if self.hmax[i] <= 0.0 || self.factor[i] >= DONE_FACTOR {
                continue;
            }
            cap -= self.max_exposure.min(self.f0t0 / (self.flux[i].max(1e-3) * self.p.plan_q));
            self.planned[i] = true;
            n += 1;
        }
        log(&format!("season plan: {n} targets to complete"));
    }

    fn sky_model(&self, lunar: f64, alt: f64) -> f64 {
        lunar / (self.q0 * normalized_airmass(alt.max(1.0)).powf(self.airmass_exponent))
    }

    fn shaped(&self, v: f64, top_mult: f64) -> f64 {
        // Convex value of score/weight v: finishing a target beats half-doing it twice, because only
        // its best exposure counts (a partial exposure is wasted if the target is redone later).
        if self.p.kappa != 1.0 {
            top_mult * (v / top_mult).powf(self.p.kappa)
        } else {
            v
        }
    }

    fn p_success(&self, raw: f64) -> f64 {
        ((raw - self.p.req_p_lo) / (self.p.req_p_hi - self.p.req_p_lo)).max(0.0).min(1.0)
    }
}
