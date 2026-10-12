# Vesper

**An LLM-assisted telescope-survey scheduling agent, written in Rust.** Vesper is the final version that team
**starwing** submitted to the [GOSIM 2026 Agentic Observer Hackathon](https://create.gosim.org/survey26/) (survey26,
`participant-agent-protocol-v4`).

[中文说明见 README.zh.md](README.zh.md)

*Vesper* is the evening star, the first light in the western sky at dusk. The name fits the agent's most important
finding: on the A-type cards, the early-season dusk band in the west (RA 300–360°) decides who wins. Its faint QSOs
are only observable for the first few weeks of the season. They have to be shot early, or they are lost.

This repository contains exactly the code of the submitted final version (platform version `222b7453`). Building it
reproduces the submitted binary byte for byte (see [Reproduce the submission](#reproduce-the-submission)).

## What the agent does

Each run is one simulated observing season on a multi-fibre spectroscopic telescope. The engine sends a decision request
for every step. The agent answers `observe` (pointing, fibre-to-target assignment, exposure time, program), `wait`,
`report` (an instrument fault) or `finish`. The score combines:

- the science value of each target's best exposure, with program-band multipliers;
- penalties for required targets left unobserved;
- rewards for paid observation requests completed before their deadline;
- fault reports: correct reports earn points, and false reports are penalised after two free ones;
- a uniformity term.

On the "realistic" cards, the zero-reward requests also carry a staff duty log. This is a multilingual chat with
codewords, Caesar-shifted and Morse-coded lines, time zones, corrections and rumours. It announces guide-camera faults
to report, flat tests and maintenance stops, closures and terrain.

Vesper is a rule-based search planner with six LLM stages layered on top. Each stage runs by default. Its output is
validated and clamped to a rule trust region, and the stage falls back to the rules on any failure.

## Results

**Final ranking: 2nd of 102 ranked teams.** The final ranking comes from the hidden cards E–H. The organizers ran
each team's final version on them with its model key, and published the results on the
[final results page](https://create.gosim.org/survey26/platform/leaderboard/final-hidden):

| Rank | Team | **Total (mean of E–H)** | E | F | G | H |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Pleiades | 33,575.79 | 37,542.8 | 31,005.2 | 31,412.0 | 34,343.3 |
| **2** | **starwing (Vesper)** | **33,051.03** | 37,193.9 | 30,722.2 | 30,307.0 | 33,981.0 |
| 3 | 迷星叫 | 32,598.75 | 37,312.4 | 30,206.4 | 29,441.2 | 33,435.0 |

Online phase of the competition, final version, rules-only evaluation (no model):

| Card | A | B | C | D | **A–D mean** | A1 | B1 | C1 | D1 | **super** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Score | 27,636.8 | 44,686.4 | 29,013.9 | 33,255.5 | **33,648.2** | 37,312.7 | 57,196.7 | 28,747.4 | 61,648.5 | **174,842.9** |

The super board weights the cards as 0.2 × (A–D sum) + 0.8 × (A1–D1 sum). A1–D1 are the "realistic" cards with duty
logs. When the online phase closed, the team was 2nd on both the A–D board and the super board.

How the agent improved over the online phase (A–D mean on the platform):

| Version | A–D mean |
|---|---:|
| Organizers' Python example (with Kimi) | 29,298 |
| Organizers' Rust example (with Kimi) | 29,945 |
| First merged improvements | 31,847 |
| Dense search, season LP, complete-or-skip, tiling | 33,103 |
| Two-thread wide search, duty-log decoder | 33,456 – 33,561 |
| Version at the close of the online phase | 33,597 |
| **Vesper (final)** | **33,648** |

## How it works

**Planner (`src/planner/`).** Every decision runs one search. It picks the pointing, the fibres, the exposure time and
the program, and maximises `gain - lambda * T`. The time price `lambda` comes from the season's achievable rate. The
search is supported by:

- a season LP of field time (`src/seasonlp.rs`);
- sky-cell urgency (`src/urgency.rs`);
- P(band) per target (`src/band.rs`);
- deadline-aware values for required targets and observation requests (`src/planner/values.rs`,
  `src/requests.rs`). For required targets whose observing window closes early in the season, this also covers how
  many good nights they have left and how many targets compete with them in the same band.

When a fibre chooses its target inside a cell, it accounts for the option cost of using up a target that could still
be completed later (`option_cost` in `src/planner/search.rs`, `PRO_FOV`).

**Faults (`src/fault.rs`).** A change-point detector watches each exposure's throughput reading.

**Duty log (`src/planner/dutylog/`, `src/planner/duty.rs`).** A rule decoder turns the duty log into facts. Announced
faults are then:

- reported at their announced time;
- flat tests and stops are waited out;
- closed sectors, and closures due within the longest exposure, are avoided;
- ridge tops are applied as terrain limits.

A fault the log missed is still reported once the readings stay below 1 % of normal throughput for 35 min.

**Guard and pacing.** Every action passes an output guard (`src/guard.rs`). Pacing (`pace` in `src/main.rs`) keeps
within the 900 s CPU budget.

### LLM stages

| Stage | Where (file : function) | Trigger (default on) | What the model's output decides | Validated / clamped | Fallback |
|---|---|---|---|---|---|
| **Natural-language understanding** | `src/llm/request_triage.rs : triage`; `src/llm/week_plan.rs : WeekPlanner::plan`; `src/llm/duty_log.rs : ask` | a new paying request; a new 7-day forecast; new duty-log messages | each request's priority from its free-text reason; the week's night plan from the forecast notices; what the duty-log lines mean where the rule decoder is unsure | priority in [1, 1.5] when the rules judge a request feasible, else [0.5, 1.5]; duty-log facts checked field by field | priority 1; the decoder's facts |
| **Data parsing** | `src/llm/duty_log.rs` → `DutyLog::resolve` → `src/planner/duty.rs : view_in`; `src/llm/fault_judge.rs : FaultJudge::judge` | new duty-log lines; a rule fault report; a new forecast | duty-log text becomes structured facts (`submit_duty_facts`) that drive fault reports, stop waits, closed sectors and terrain limits; the fault detector's state is read for report-now or wait | a model fault time is reported only while false reports are free; closures ≤ 16 h, ≤ 3 sectors | rule decision |
| **Task planning** | `src/llm/week_plan.rs` → `src/main.rs : night_advice` | a new 7-day forecast | sets each night of the forecast week: `bad_night`, `avoid` sectors, the week's time price `lambda_scale` | `bad_night` only on the rule's bad nights; `lambda_scale` within [0.98, 1.02] | rule plan |
| **Action decision-making** | `src/llm/fault_judge.rs` → `src/main.rs : maybe_report`; `request_triage.rs` → `Planner::on_requests` | every fault report the rules decide on; new requests | whether the `report` goes out now or after one confirmation exposure; which targets the priorities favour | can delay a rule report by one exposure, never drop or start one | the rule's report |
| **Tool calling** | `src/llm/client.rs : Endpoint::request` | every LLM question | the model answers by calling the offered function (`tools` / `tool_calls`) | arguments checked against the schema and the trust region | text JSON; then rules |
| **Plan adaptation** | `week_plan.rs` with `src/main.rs : week_context` | each planned forecast | re-plans the coming week from the review of the last period | as task planning | as task planning |

**Quota and robustness (`src/llm/client.rs`).** The hard budget is `LLM_MAX_CALLS`: 16 HTTP requests per run by
default, retries included. Cards with a duty log get 24 more, for questions asked in the background so that no
decision waits on them. The client also:

- retries at most once;
- backs off on HTTP 429 (Retry-After; a non-finite value is ignored);
- opens a circuit breaker after 401/403, a quota message, or repeated failures;
- bounds every wait. The connect timeout is bounded too.

Without a key, or with `OBSERVER_MODEL_DISABLED=1`, the agent runs the rules alone.

## Build and run

```bash
# static binary, as submitted
CC_x86_64_unknown_linux_musl=gcc cargo build --release --locked --target x86_64-unknown-linux-musl
# -> target/x86_64-unknown-linux-musl/release/rust-pro
```

The binary speaks the v4 JSON-lines protocol on stdin/stdout. Run it with the organizers' local runner from the
challenge repository, or on the platform.

`observer.project.json` builds from source in `rust:1-bookworm`. The submitted ZIP instead shipped the prebuilt static
binary as `bin/rust-pro`, with an `observer.project.json` of:

```json
{"schema_version": "observer-project-v1", "protocol": "jsonl-v4", "image": "debian:bookworm-slim",
 "build": [], "run": ["./bin/rust-pro"], "working_directory": ".",
 "environment": {"PRO_PACE_SAFETY": "2.0", "PRO_FOV": "0.3"}}
```

### Configuration

| Variable | Meaning |
|---|---|
| `OPENAI_API_KEY` (or `KIMI_API_KEY`) | model key. Without it, the agent runs rules only. |
| `OPENAI_BASE_URL` | default `https://api.kimi.com/coding/v1`; any OpenAI-compatible endpoint |
| `OPENAI_MODEL` | model name |
| `LLM_MAX_CALLS`, `LLM_TIMEOUT` | call budget (16) and per-question timeout (40 s) |
| `PRO_PACE_SAFETY=2.0` | **final setting.** Pacing factor. The agent's CPU estimate counts both search threads, so at the default 0.75 it narrowed its search while using under 30 % of the budget. |
| `PRO_FOV=0.3` | **final setting.** Strength of the fibre option cost (default 0.5). |
| `PRO_<NAME>` | any planner constant (see `Params::load` in `src/planner/mod.rs`); `PRO_FIXED_LEVEL=0` pins the search level |

## Reproduce the submission

```bash
git clone <repository URL> Vesper && cd Vesper
CC_x86_64_unknown_linux_musl=gcc cargo build --release --locked --target x86_64-unknown-linux-musl
sha256sum target/x86_64-unknown-linux-musl/release/rust-pro
# e4afbcfa3d9315003b0e948a6ff0ef87c7afe259dfeb7e1b33f490be9da51349
# = bin/rust-pro of the submitted ZIP (rustc 1.97.0, target x86_64-unknown-linux-musl)
```

Run it with `PRO_PACE_SAFETY=2.0 PRO_FOV=0.3`, as in the submission.

## Scope of this repository

This repository holds the submitted version only. The development tooling is not included: unit tests, a local LLM
stub, a faster byte-identical referee, the cluster evaluation harness, the synthetic card generator, the duty-log gold
set, and the experiment notes. The source still declares `#[cfg(test)]` modules under `tests/unit/`, exactly as
submitted, so `cargo test` does not build here. Comments such as `notes/23` refer to those internal notes.

## License

Code written by the team is licensed under the [Apache License 2.0](LICENSE). The agent started from the organizers'
Rust example, licensed [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). The portions derived from it
remain under that license; see [NOTICE](NOTICE).

The task cards, simulated data and evaluation code are the organizers' materials. Please cite the
GOSIM 2026 Agentic Observer Hackathon (https://create.gosim.org/survey26/).
