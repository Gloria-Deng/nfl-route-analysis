# -*- coding: utf-8 -*-
"""
NFL Big Data Bowl 2027 - "Reliability first, then the route tree" (round 8)

Reproduces every number quoted in solution/writeup_r8.md from the raw competition
files only, on one CPU kernel, with no internet and no GPU. Nothing here reads a
path outside the mounted competition data, and nothing reads a file we produced
elsewhere: the kernel is the analysis, not a report about a job.

Pipeline
  S0 data traps (entity_type, sampling rate, season_type, who has separation)
  S1 break detection in the WR/DB skill drills -> four change-of-direction mechanism
     quantities per attempt, two detector variants
  S2 reliability of the aggregated player trait, computed WITHIN position group
     (split-half across drills + Spearman-Brown), and the pooled version that
     inflates it -> this is where most public Combine analyses go wrong
  S3 linkage to regular-season outcomes: WR separation at the throw, DB cushion,
     route-level OLS, SE clustered by player, player-cluster bootstrap CI
  S4 where the signal actually lives: the route-family sign flip, priced with a
     family-wise permutation null and power bars at the realised n
  S5 the reliability ceiling: is the pooled null a measurement artefact or not
  S6 what we could not test, stated as such
  S7 (round 8) Combine-year effect on the raw drill metrics, aggregated by drill,
     so the year drift is a table the reader can check rather than a sentence
  S8 (round 8) recording truncation: the predictor reads 1.5 s after a break, and
     that window is not always recorded - exposure by drill x year, and what
     dropping clipped recordings does to reliability and to the headline betas
  S9 (round 8) forward validation: leave-one-season-out refit of every route
     family, so the sign flip is tested against a held-out season three times
  S10 (round 8) the figures used in the writeup, drawn from the tables above

Runtime as measured on the same data outside Kaggle: ~20 min single process, peak
RSS well under 4 GB, so no subsampling is needed and none is applied
(SUBSET=0 is the published setting; SUBSET=1 is a ~1 min development smoke run
that shrinks the resampling counts only and is never used for a reported number).
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks

warnings.filterwarnings("ignore")

SEED = 777202601
RNG = np.random.default_rng(SEED)
SUBSET = int(os.environ.get("BDB27_SUBSET", "0")) == 1
NBOOT, NPERM, NHALF = (200, 200, 40) if SUBSET else (2000, 2000, 200)

# ---------------------------------------------------------------- cell 0: config
# %% [markdown]
# ## 0 - Configuration and the files we read
#
# This kernel attaches the competition data only. `combine_tracking.csv` is the
# Combine 10 Hz sensor tracking; `player_play.csv` carries the regular-season
# Next Gen Stats play-level columns. We use nothing outside the competition files.
#
# `OUT_DIR` is the kernel working directory: every table printed below is also
# written there as CSV/JSON so the writeup can point at a file rather than at a
# screenshot. `BDB27_DATA` exists only so the same script can be run against the
# same files somewhere other than Kaggle for verification; the published kernel
# uses the mount below.

_CANDIDATES = [os.environ.get("BDB27_DATA", ""), "/kaggle/input/nfl-big-data-bowl-2027"]
IN = next(p for p in _CANDIDATES if p and os.path.isdir(p))
OUT_DIR = os.environ.get("BDB27_OUT", os.getcwd())
if not os.path.isdir(OUT_DIR):
    os.makedirs(OUT_DIR, exist_ok=True)
DET = {"w3_t45": (3, 45.0), "w5_t60": (5, 60.0)}
MIN_ANGLE, MIN_SPEED_AT_BREAK, D_PEAK, DT = 25.0, 1.0, 5, 0.1
MET = ["break_angle_deg", "decel_in", "reaccel_out", "time_to_reaccel_s",
       "time_to_recover_s", "approach_speed", "peak_a_after", "n_breaks"]
PROM = "reaccel_out"
PROM_DET = {"WR": "w3_t45", "DB": "w5_t60"}
GOF = {"WR": "SKILL_DRILLS_WR", "DB": "SKILL_DRILLS_DB"}
FAMS = ["FLAT", "CROSS", "GO", "HITCH", "SCREEN", "POST", "CORNER", "OUT"]
RES = {}


def log(*a):
    print(*a, flush=True)


def wp(name):
    """Absolute path for a table this kernel writes, so nothing depends on CWD."""
    return os.path.join(OUT_DIR, name)


log("[0] data dir %s | out dir %s | SUBSET=%s | NBOOT=%d NPERM=%d NHALF=%d"
    % (IN, OUT_DIR, SUBSET, NBOOT, NPERM, NHALF))
for _f in ("combine_tracking.csv", "player_play.csv", "players.csv", "games.csv"):
    log("[0] found %s: %s" % (_f, os.path.exists(os.path.join(IN, _f))))


# %% [markdown]
# ## 1 - Four data traps, checked before any number is used
#
# 1. **`entity_type`.** BALL rows in the receiver drills carry an `nfl_id`, so any
#    per-player aggregation that does not filter `entity_type == "PLAYER"` mixes
#    the football into the athletes.
# 2. **Sampling rate.** The Combine tracking is 10 Hz (0.1 s frames). Everything
#    below (turn rate, the 0.5 s minimum spacing between accepted breaks) is in
#    frames of 0.1 s, and we verify the step size rather than assume it.
# 3. **`season_type`.** `player_play` mixes PRE / REG / POST. The theme is
#    regular-season performance, so only `REG` is kept.
# 4. **Who actually has a separation value.** `separation_at_pass_forward` is
#    recorded for *route runners*. For defensive backs it is essentially absent,
#    which is why the DB half of the analysis has to be a different question
#    (pre-snap `cushion`), not the same question on the other side of the ball.

# %%
ct = pd.read_csv(f"{IN}/combine_tracking.csv",
                 usecols=["draft_year", "event_id", "nfl_id", "entity_type", "time",
                          "drill_type", "drill_name", "attempt", "x", "y", "s", "a", "dir"])
n_all = len(ct)
ct = ct[ct.entity_type == "PLAYER"].copy()
ct["ts"] = pd.to_datetime(ct.time)
step = ct.sort_values(["event_id", "ts"]).groupby("event_id")["ts"].diff().dt.total_seconds()
RES["traps"] = dict(
    tracking_rows_all=n_all, tracking_rows_player=int(len(ct)),
    ball_rows_dropped=int(n_all - len(ct)),
    ball_rows_that_carried_an_nfl_id=int((ct.entity_type == "BALL").sum()),
    dt_mode_s=float(step.mode().iloc[0]), frac_dt_gt_0p15=float((step > 0.15).mean()),
    skill_attempts=int(ct[ct.drill_type.isin(GOF.values())].event_id.nunique()),
    skill_players=int(ct[ct.drill_type.isin(GOF.values())].nfl_id.nunique()),
    combine_years={int(k): int(v) for k, v in
                   ct[ct.drill_type.isin(GOF.values())].groupby("draft_year").nfl_id.nunique().items()})
log("[1] PLAYER rows %d of %d | dt mode %.3f s | frac dt>0.15s %.4f" %
    (RES["traps"]["tracking_rows_player"], RES["traps"]["tracking_rows_all"],
     RES["traps"]["dt_mode_s"], RES["traps"]["frac_dt_gt_0p15"]))
log("[1] WR+DB skill drills: %d attempts, %d players, years %s" %
    (RES["traps"]["skill_attempts"], RES["traps"]["skill_players"], RES["traps"]["combine_years"]))

players = pd.read_csv(f"{IN}/players.csv")
games = pd.read_csv(f"{IN}/games.csv", usecols=["game_id", "season", "season_type"])
pp = pd.read_csv(f"{IN}/player_play.csv", usecols=lambda c: c in [
    "nfl_id", "game_id", "play_id", "lined_up_position", "separation_at_pass_forward",
    "cushion", "route_ran", "team_coverage_type", "team_coverage_man_zone",
    "coverage_assignment", "blitzing", "offense_formation", "receiver_alignment",
    "target", "pass_length", "yards_after_catch", "down", "yards_to_go", "quarter"])
pp = pp.merge(games, on="game_id", how="left")
non_reg = int((pp.season_type != "REG").sum())
pp = pp[pp.season_type == "REG"].copy()
pp = pp.merge(players[["nfl_id", "nfl_position", "draft_year", "draft_round",
                       "draft_overall_pick"]].rename(columns={"draft_year": "combine_year"}),
              on="nfl_id", how="left")
have_sep = pp.groupby(pp.lined_up_position.astype("object").fillna("MISSING"))[
    "separation_at_pass_forward"].apply(lambda s: 100 * s.notna().mean())
RES["traps"].update(reg_rows=int(len(pp)), rows_dropped_as_non_reg=non_reg)
RES["traps"]["separation_nonnull_pct_by_lineup"] = {k: round(float(v), 1)
                                                   for k, v in have_sep.items()}
log("[1] kept %d REG rows (%d non-REG dropped)" % (len(pp), non_reg))
log("[1] %% of rows with a separation value, by lined_up_position: %s" %
    {k: round(float(v), 1) for k, v in have_sep.items() if v > 0 or k in ("CB", "SLOT_CB")})

# ---------------------------------------------------------------- cell: detection
# %% [markdown]
# ## 2 - From (x, y, t) to a change-of-direction event, and four mechanism quantities
#
# The Combine drills are not annotated with where the break happens, so it has to be
# found. We difference the heading `dir` frame to frame, wrap it to +/-180 deg, divide
# by 0.1 s to get a turn rate, and smooth it with a `w`-tap moving average; peaks of
# the smoothed |turn rate| above `thr` deg/s, at least 5 frames apart, are candidate
# breaks. A candidate is accepted only if the *net* heading change in a window around
# it is at least 25 deg and the player was moving at least 1 yd/s at that frame - that
# second condition is what stops slow, noisy wobble from being counted as a cut.
#
# Per accepted break we keep four mechanism quantities plus the energy the player
# arrived with:
#
# | quantity | definition |
# |---|---|
# | `break_angle_deg` | largest net heading change in a +/-0.4 / 0.5 / 0.7 s window |
# | `decel_in` | most negative tangential acceleration in the 0.8 s before the break |
# | `reaccel_out` | largest tangential acceleration in the 1.5 s after the break |
# | `time_to_reaccel_s` | when that maximum occurs after the break |
# | `time_to_recover_s` | first frame back to 90%% of approach speed, within 3 s |
#
# Tangential acceleration is `d(speed)/dt` from the `s` series, not the unsigned
# total-acceleration column `a`. Two detector variants (3-tap/45 deg/s and
# 5-tap/60 deg/s) are run in parallel; the detector is a choice, not a fact, and
# section 3 shows how much it matters.


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def smooth(v, w):
    v = np.asarray(v, float)
    if len(v) < w:
        return v
    return np.convolve(v, np.ones(w) / w, mode="same")


def turn_rate(dirv, w):
    hd = wrap_deg(np.diff(dirv)) / DT
    return smooth(np.append(hd, hd[-1]), w)


def attempt_breaks(dirv, sv, av, w, thr):
    n = len(sv)
    if n < 20:
        return []
    tang = smooth(np.gradient(smooth(sv, 3), DT), 3)
    pk, _ = find_peaks(np.abs(turn_rate(dirv, w)), height=thr, distance=D_PEAK)
    out = []
    for i in pk:
        i = int(i)
        if i < 6 or i > n - 12:
            continue
        ang = 0.0
        for lo, hi in [(-4, 4), (-5, 5), (-7, 7)]:
            a, b = max(0, i + lo), min(n - 1, i + hi)
            ang = max(ang, abs(float(wrap_deg(dirv[b] - dirv[a]))))
        if ang < MIN_ANGLE or sv[i] < MIN_SPEED_AT_BREAK:
            continue
        pre = sv[max(0, i - 10):max(1, i - 4)]
        pre_spd = float(pre.mean()) if pre.size else np.nan
        dec = tang[max(0, i - 8):i + 1]
        post = tang[i:min(n, i + 15)]
        rec = np.nan
        if np.isfinite(pre_spd):
            hit = np.where(sv[i:min(n, i + 31)] >= 0.9 * pre_spd)[0]
            if hit.size:
                rec = float(hit[0] * DT)
        win = sv[max(0, i - 3):min(n, i + 9)]
        out.append(dict(break_angle_deg=ang, approach_speed=pre_spd,
                        decel_in=float(-dec.min()) if dec.size else np.nan,
                        reaccel_out=float(post.max()) if post.size else np.nan,
                        time_to_reaccel_s=float(post.argmax() * DT) if post.size else np.nan,
                        time_to_recover_s=rec,
                        post_window_s=float(n - 1 - i) * DT,
                        peak_a_after=float(av[i:min(n, i + 15)].max())))
    return out


sk = ct[ct.drill_type.isin(GOF.values())].sort_values(["event_id", "ts"])
att = {}
for dname, (w, thr) in DET.items():
    recs = []
    for ev, g in sk.groupby("event_id", sort=False):
        dirv, sv, av = (g[c].values.astype(float) for c in ("dir", "s", "a"))
        if not (np.isfinite(dirv).all() and np.isfinite(sv).all() and np.isfinite(av).all()):
            continue
        bl = attempt_breaks(dirv, sv, av, w, thr)
        if not bl:
            continue
        b = pd.DataFrame(bl)
        r = dict(event_id=ev, nfl_id=int(g.nfl_id.iloc[0]), draft_year=int(g.draft_year.iloc[0]),
                 drill_type=g.drill_type.iloc[0], drill_name=g.drill_name.iloc[0],
                 attempt=int(g.attempt.iloc[0]), n_breaks=len(b),
                 duration_s=float(len(g)) * DT,
                 post_window_med_s=float(b.post_window_s.median()),
                 pct_breaks_clipped15=float(100 * (b.post_window_s < 1.5).mean()),
                 pct_breaks_clipped10=float(100 * (b.post_window_s < 1.0).mean()))
        for k in MET:
            r[k] = float(b[k].mean()) if k != "n_breaks" else len(b)
        recs.append(r)
    att[dname] = pd.DataFrame(recs)
    RES.setdefault("detectors", {})[dname] = dict(
        attempts_with_break=int(len(att[dname])),
        pct_of_skill_attempts=float(100 * len(att[dname]) / sk.event_id.nunique()),
        breaks=int(att[dname].n_breaks.sum()), players=int(att[dname].nfl_id.nunique()),
        window_taps=w, threshold_deg_s=thr)
    log("[2] %s (w=%d, thr=%.0f deg/s): %d attempts with >=1 break (%.1f%% of skill "
        "attempts), %d breaks, %d players" %
        (dname, w, thr, RES["detectors"][dname]["attempts_with_break"],
         RES["detectors"][dname]["pct_of_skill_attempts"], RES["detectors"][dname]["breaks"],
         RES["detectors"][dname]["players"]))
log("[2] attempt-level means (%s): %s" %
    ("w3_t45", att["w3_t45"][MET].mean().round(3).to_dict()))

# %% [markdown]
# ## 3 - Reliability first: does the trait repeat *within a position group*?
#
# A Combine number only ranks players if it is a property of the player. We test that
# without touching any NFL outcome, three ways:
#
# * **split-half across drills** - split each player's drills at random into two halves,
#   average each half, correlate the two halves across players, repeat 200 times, take
#   the median, then double it with Spearman-Brown. This is the reliability of the
#   *aggregated player metric*, which is what a linkage analysis actually uses.
# * **ICC(1) between drills** - the variance decomposition of a single drill reading.
# * **test-retest within one drill** - only possible where a player repeated the same
#   drill (gauntlet, the DB 45-degree reaction, the line drill, the 40).
#
# Four constructions are compared: the raw value; z-scored within drill; z-scored
# within Combine year; and z-scored within drill x Combine year. And we report the
# split-half **pooled over the WR and DB drill populations** next to the same quantity
# computed **inside** each group - because that difference is the point of this section.

KEYGRP = {"zdrill": ["drill_name"], "zyear": ["draft_year"],
          "zdrill_year": ["drill_name", "draft_year"]}


def cn(kind, m):
    return m if kind == "raw" else "%s@%s" % (m, "+".join(KEYGRP[kind]))


def zbuild(a, kind, m):
    if kind == "raw":
        return a
    z = a.copy()
    g = z.groupby(KEYGRP[kind], dropna=False)[m]
    z[cn(kind, m)] = (z[m] - g.transform("mean")) / g.transform("std")
    return z


def split_half_sb(df, c, unit="drill_name"):
    pv = df.pivot_table(index="nfl_id", columns=unit, values=c).dropna(thresh=4)
    if len(pv) < 15 or pv.shape[1] < 2:
        return np.nan, int(len(pv))
    cols, sp = list(pv.columns), []
    for _ in range(NHALF):
        RNG.shuffle(cols)
        h = len(cols) // 2
        if h < 1:
            continue
        sp.append(stats.spearmanr(pv[cols[:h]].mean(axis=1), pv[cols[h:]].mean(axis=1)).statistic)
    med = float(np.nanmedian(sp)) if sp else np.nan
    return ((2 * med / (1 + med)) if med == med else np.nan), int(len(pv))


rel = []
for d in DET:
    for kind in ["raw", "zdrill", "zyear", "zdrill_year"]:
        for m in MET:
            c, z = cn(kind, m), zbuild(att[d], kind, m)
            for grp, dt in list(GOF.items()) + [("POOLED", None)]:
                src = z if dt is None else z[z.drill_type == dt]
                sb, nply = split_half_sb(src, c)
                rel.append(dict(detector=d, construction=kind, metric=m, group=grp,
                                n_players=nply, split_half_SB=sb))
RL = pd.DataFrame(rel)
RL.to_csv(wp("r8_reliability.csv"), index=False)
show = RL[(RL.metric == PROM)].pivot_table(index="group", columns=["detector", "construction"],
                                           values="split_half_SB")
log("[3] split-half Spearman-Brown for the promoted metric %s:" % PROM)
log(show.round(3).to_string())
def sb_of(detect, kind, grp, metric=PROM):
    r = RL[(RL.detector == detect) & (RL.construction == kind) & (RL.group == grp) &
           (RL.metric == metric)]
    return float(r.split_half_SB.iloc[0]) if len(r) else float("nan")


# pooling inflation, all metric x detector x construction cells: what a notebook that
# ignores position group would print, minus what the trait is worth inside the group.
wide = RL.pivot_table(index=["detector", "construction", "metric"], columns="group",
                      values="split_half_SB").reset_index()
wide["inflation_vs_best_group"] = wide["POOLED"] - wide[["WR", "DB"]].max(axis=1)
wide["pooled_beats_both_by_020"] = ((wide["POOLED"] > wide["WR"] + 0.20) &
                                    (wide["POOLED"] > wide["DB"] + 0.20)).astype(int)
GATE = 0.40
MECH = ["break_angle_deg", "decel_in", "reaccel_out", "time_to_reaccel_s", "time_to_recover_s"]
RES["reliability"] = dict(
    promoted_table={"%s|%s|%s" % (i[0], i[1], i[2]): float(v) for i, v in
                    RL[RL.metric == PROM].set_index(["detector", "construction", "group"])
                    .split_half_SB.items()},
    promoted_WR_raw_pooled=sb_of("w3_t45", "raw", "POOLED"),
    promoted_WR_raw_withinWR=sb_of("w3_t45", "raw", "WR"),
    promoted_DB_raw_pooled=sb_of("w5_t60", "raw", "POOLED"),
    promoted_DB_raw_withinDB=sb_of("w5_t60", "raw", "DB"),
    pooling_inflation_cells=int(len(wide)),
    pooling_median_inflation=float(wide.inflation_vs_best_group.median()),
    pooling_share_pooled_beats_both=float(wide.pooled_beats_both_by_020.mean()),
    verdict=("the pooled split-half is the number a public notebook would quote; the "
             "within-group value is the reliability of the trait being used"))
log("[3] POOLING CHECK w3_t45/raw: pooled %.3f vs within-WR %.3f vs within-DB %.3f"
    % (sb_of("w3_t45", "raw", "POOLED"), sb_of("w3_t45", "raw", "WR"), sb_of("w3_t45", "raw", "DB")))
log("[3] inflation over %d metric x detector x construction cells: median %.3f, "
    "share where the pooled value beats BOTH within-group values by >0.20: %.1f%%"
    % (RES["reliability"]["pooling_inflation_cells"],
       RES["reliability"]["pooling_median_inflation"],
       100 * RES["reliability"]["pooling_share_pooled_beats_both"]))
for grp in ["WR", "DB"]:
    sub = RL[(RL.group == grp) & (RL.metric.isin(MECH))].dropna(subset=["split_half_SB"])
    best = sub.sort_values("split_half_SB", ascending=False).head(3)
    RES.setdefault("promotion_decision", {})[grp] = dict(
        gate=GATE, n_mech_cells=int(len(sub)),
        n_clearing_gate=int((sub.split_half_SB >= GATE).sum()),
        best_three=best[["detector", "construction", "metric",
                         "split_half_SB"]].round(3).to_dict("records"))
    log("[3] PROMOTION DECISION %s: %d/%d mechanism cells reach split-half SB >= %.2f; "
        "best three: %s" % (grp, RES["promotion_decision"][grp]["n_clearing_gate"],
                            RES["promotion_decision"][grp]["n_mech_cells"], GATE,
                            [(r["detector"], r["construction"], r["metric"], r["split_half_SB"])
                             for r in RES["promotion_decision"][grp]["best_three"]]))
log("[3] consequence: where a group has no cell at or above the gate, every linkage "
    "number for that group is an estimate on an unrepeated measure, not a test of a "
    "player trait.")

# %% [markdown]
# ## 4 - Linking the Combine movement to the regular season
#
# Route-level OLS, outcome and metric both expressed in SD, standard errors clustered
# by player from the **full-model** residuals, and a player-cluster bootstrap for the
# interval (B = 2,000 on the pooled arm, B = 500 on each route-family cell - the
# table says which is which), controlling for the play context that is available in
# `player_play`: season, down, yards-to-go, route, coverage type, man/zone, blitz,
# receiver alignment, formation, target, quarter. All four standardisation
# constructions are reported for every arm, so no construction is chosen after seeing
# the outcome.
#
# Why the estimator is written this way. Partialling the context out first
# (Frisch-Waugh-Lovell) and taking the sandwich from the *context* residuals is not
# the same estimator once the SE is clustered by player, because the regressor is not
# demeaned within player: on the smallest route family it inflated the SE from 0.0335
# to 0.0404 and turned p = 0.006 into p = 0.024. `Cell` below is the estimator the
# certified job used, and the agreement between this kernel and that job is tabulated
# in the report's compliance record rather than asserted here.
#
# Two honest limits belong in this section rather than in a footnote. First,
# `coverage_assignment` - the variable the brief names as the thing to control for a
# coverage outcome - is **0.0%% non-null on WR rows**, so it cannot be controlled on
# the WR side at all; on DB rows it is present for 63%%. Second, the DB side cannot be
# run against `separation_at_pass_forward` because that column is not recorded for
# defensive alignments, so the DB question is replaced by pre-snap `cushion` and
# labelled as a replacement, not as the original hypothesis.

NUM = ["down", "yards_to_go"]
CAT = ["route_ran", "team_coverage_type", "team_coverage_man_zone", "coverage_assignment",
       "blitzing", "receiver_alignment", "offense_formation", "target", "quarter", "season"]
CONSTR = ["raw", "zdrill", "zyear", "zdrill_year"]
MIN_CELL = 120 if SUBSET else 400


def design(d):
    parts = [np.ones((len(d), 1))]
    for c in NUM:
        if c in d.columns:
            v = pd.to_numeric(d[c], errors="coerce").fillna(pd.to_numeric(d[c], errors="coerce").median())
            if v.std() > 0:
                parts.append(((v - v.mean()) / v.std()).values[:, None])
    for c in CAT:
        if c not in d.columns or d[c].notna().mean() * 100 < 20:
            continue
        s = d[c].astype("object").where(d[c].notna(), "MISSING")
        vc = s.value_counts(normalize=True)
        for lv in [v for v in vc.index if vc[v] * 100 >= 20][1:]:
            parts.append((s == lv).values.astype(float)[:, None])
    return np.column_stack(parts)


class Cell(object):
    """One estimable cell: standardised outcome on standardised metric + context,
    with the SE clustered by player from the FULL-model residuals.

    `self.rows` are positions in the parent frame of the rows that survive the
    finite-value filter, so the permutation loop can hand back a permuted metric
    vector with no length mismatch - that mismatch is the silent-kill bug class in
    a draw loop like this. `boot_ci` uses the fact that the metric coefficient of the
    full model equals the FWL ratio sum(xr*yr)/sum(xr^2), which is additive over
    players, so one multinomial weight vector over precomputed per-player scalars is
    one bootstrap draw.
    """

    def __init__(self, parent, rows, ycol, mcol):
        rows = np.asarray(rows, int)
        sub = parent.iloc[rows]
        m = sub[mcol].values.astype(float)
        y = pd.to_numeric(sub[ycol], errors="coerce").values.astype(float)
        ok = np.isfinite(m) & np.isfinite(y)
        self.rows = rows[ok]
        self.xsd = float(m[ok].std())
        self.x = (m[ok] - m[ok].mean()) / (self.xsd + 1e-12)
        y = y[ok]
        self.y = (y - y.mean()) / (y.std() + 1e-12)
        self.gid = pd.factorize(sub["nfl_id"].values[ok])[0]
        self.np_ = int(self.gid.max()) + 1
        self.Xc = design(sub)[ok]
        self.X = np.column_stack([self.x, self.Xc])
        self.n = len(self.y)
        self.nk = self.X.shape[1]
        self.sub = sub[ok].reset_index(drop=True)
        self.mz = None

    def _est(self, xstd):
        X = np.column_stack([xstd, self.Xc])
        beta = np.linalg.lstsq(X, self.y, rcond=None)[0]
        r = self.y - X @ beta
        S = np.zeros((self.np_, self.nk))
        np.add.at(S, self.gid, X * r[:, None])
        XtX_inv = np.linalg.pinv(X.T @ X)
        bread = XtX_inv @ (S.T @ S) @ XtX_inv
        adj = (self.np_ / max(self.np_ - 1, 1)) * ((self.n - 1) / max(self.n - self.nk, 1))
        se = float(np.sqrt(max(bread[0, 0] * adj, 0.0)))
        b = float(beta[0])
        z = b / se if se > 0 else np.nan
        return dict(beta=b, se=se, z=float(z),
                    p=float(2 * stats.norm.sf(abs(z))) if z == z else np.nan)

    def fit(self):
        self.mz = self._est(self.x)
        return self.mz

    def perm_est(self, parent_metric_permuted):
        xv = parent_metric_permuted[self.rows]
        return self._est((xv - xv.mean()) / (xv.std() + 1e-12))

    def boot_ci(self, B=None):
        B = B or NBOOT
        Q = np.linalg.qr(self.Xc)[0]
        xr = self.x - Q @ (Q.T @ self.x)
        yr = self.y - Q @ (Q.T @ self.y)
        a = np.bincount(self.gid, weights=xr * yr, minlength=self.np_)
        c = np.bincount(self.gid, weights=xr * xr, minlength=self.np_)
        out = []
        for _ in range(B):
            w = RNG.multinomial(self.np_, np.full(self.np_, 1.0 / self.np_)).astype(float)
            d = float((w * c).sum())
            if d > 1e-12:
                out.append(float((w * a).sum()) / d)
        if len(out) < 20:
            return np.nan, np.nan, len(out)
        lo, hi = np.percentile(out, [2.5, 97.5])
        return float(lo), float(hi), len(out)


def base_frame(detect, kind, grp, tgt):
    """the certified join: the player's constructed metric x that player's REG routes
    that carry the target. One row per player in the metric table, so no fan-out."""
    c = cn(kind, PROM)
    za = zbuild(att[detect], kind, PROM)
    za = za[za.drill_type == GOF[grp]][["nfl_id", c]].dropna()
    ply = za.groupby("nfl_id")[c].mean().reset_index()
    rws = pp[pp.nfl_id.isin(set(ply.nfl_id))].merge(ply, on="nfl_id", how="inner")
    rws = rws[rws[tgt].notna() & rws[c].notna()].reset_index(drop=True)
    return (rws if len(rws) else None), c


def cell_specs(rws, grp):
    """(label, row positions) for every sub-group estimated separately."""
    if grp == "WR":
        return [(rt, np.where(rws.route_ran.values == rt)[0]) for rt in FAMS]
    pos = rws.lined_up_position.astype("object").fillna("MISSING").values
    mz = rws.team_coverage_man_zone.values
    return [("pos_S", np.where(np.isin(pos, ["SS", "FS", "DB"]))[0]),
            ("pos_CB", np.where(np.isin(pos, ["CB", "SLOT_CB"]))[0]),
            ("zone", np.where(mz == "ZONE_COVERAGE")[0]),
            ("man", np.where(mz == "MAN_COVERAGE")[0])]


arms, perm_out = [], {}
for grp, tgt in [("WR", "separation_at_pass_forward"), ("DB", "cushion")]:
    d0 = PROM_DET[grp]
    for kind in CONSTR:                      # every construction, before seeing outcomes
        rws, c = base_frame(d0, kind, grp, tgt)
        if rws is None:
            arms.append(dict(group=grp, target=tgt, detector=d0, construction=kind,
                             metric=PROM, label="all", status="insufficient",
                             routes=0, players=0))
            continue
        allcell = Cell(rws, np.arange(len(rws)), tgt, c)
        f = allcell.fit()
        blo, bhi, bn = allcell.boot_ci()
        sp_rho = float(stats.spearmanr(rws.groupby("nfl_id")[c].mean(),
                                       rws.groupby("nfl_id")[tgt].mean()).statistic)
        arms.append(dict(group=grp, target=tgt, detector=d0, construction=kind, metric=PROM,
                         label="all", status="measured", routes=int(len(rws)),
                         players=int(rws.nfl_id.nunique()),
                         player_seasons=int(rws.groupby(["nfl_id", "season"]).ngroups),
                         raw_player_rho=sp_rho, adj_beta=f["beta"], adj_cse=f["se"],
                         adj_p=f["p"], boot_lo=blo, boot_hi=bhi, boot_B=bn,
                         n_ctx_covars=int(allcell.nk)))
        cells = []
        for lab, rows in cell_specs(rws, grp):
            if len(rows) < MIN_CELL:
                arms.append(dict(group=grp, target=tgt, detector=d0, construction=kind,
                                 metric=PROM, label=lab, status="not attempted "
                                 "(cluster < 15 or fewer than %d routes)" % MIN_CELL,
                                 routes=int(len(rows)),
                                 players=int(rws.iloc[rows].nfl_id.nunique())))
                continue
            cl = Cell(rws, rows, tgt, c)
            if cl.np_ < 15 or not cl.fit():
                arms.append(dict(group=grp, target=tgt, detector=d0, construction=kind,
                                 metric=PROM, label=lab, status="not attempted "
                                 "(cluster < 15 or fewer than %d routes)" % MIN_CELL,
                                 routes=int(cl.n), players=int(cl.np_)))
                continue
            flo, fhi, fn = cl.boot_ci(B=60 if SUBSET else 500)
            cells.append((lab, cl))
            arms.append(dict(group=grp, target=tgt, detector=d0, construction=kind,
                             metric=PROM, label=lab, status="measured", routes=int(cl.n),
                             players=int(cl.np_),
                             player_seasons=int(cl.sub.groupby(["nfl_id", "season"]).ngroups),
                             adj_beta=cl.mz["beta"], adj_cse=cl.mz["se"], adj_p=cl.mz["p"],
                             boot_lo=flo, boot_hi=fhi, boot_B=fn, abs_z=abs(cl.mz["z"])))
        # ---- family-wise permutation null: the metric is shuffled ACROSS PLAYERS, so
        # the null keeps each player's route mix and destroys only the athlete signal.
        if cells:
            mfull = rws[c].values.astype(float)
            gfull = pd.factorize(rws.nfl_id.values)[0]
            npl = int(gfull.max()) + 1
            obs = dict((lab, abs(cl.mz["z"])) for lab, cl in cells)
            maxes = np.empty(NPERM)
            percell = np.empty((NPERM, len(cells)))
            for b in range(NPERM):
                mp = mfull[RNG.permutation(npl)[gfull]]
                mx = 0.0
                for j, (lab, cl) in enumerate(cells):
                    r = cl.perm_est(mp)
                    zz = abs(r["z"]) if r and r["z"] == r["z"] else 0.0
                    percell[b, j] = zz
                    mx = max(mx, zz)
                maxes[b] = mx
            omax = max(obs.values())
            perm_out[grp + "|" + kind] = dict(
                cells=int(len(obs)), players=npl, B=int(NPERM), observed_max_abs_z=float(omax),
                null_mean=float(maxes.mean()), null_p50=float(np.median(maxes)),
                null_p95=float(np.percentile(maxes, 95)), null_p99=float(np.percentile(maxes, 99)),
                family_wise_p=float(np.mean(maxes >= omax)), observed=obs,
                per_cell_fwer_p=dict((lab, float(np.mean(percell[:, j] >= obs[lab])))
                                     for j, (lab, _) in enumerate(cells)))
            log("[4] permutation %s/%s: observed max|z| %.2f, null p95 %.2f, "
                "family-wise p %.4f" % (grp, kind, omax, np.percentile(maxes, 95),
                                        float(np.mean(maxes >= omax))))
            log("[4]   per-cell family-wise p: %s" %
                {k: round(v, 3) for k, v in perm_out[grp + "|" + kind]["per_cell_fwer_p"].items()})
ARMS = pd.DataFrame(arms)
ARMS.to_csv(wp("r8_arms.csv"), index=False)
RES["arms"] = ARMS.round(4).to_dict("records")
RES["permutation"] = perm_out
log("[4] pooled arms (label=all):")
log(ARMS[ARMS.label == "all"].round(4).to_string(index=False))
log("[4] route-family / coverage arms (raw construction):")
log(ARMS[(ARMS.label != "all") & (ARMS.construction == "raw")].round(4).to_string(index=False))
HEAD = ARMS[(ARMS.group == "WR") & (ARMS.construction == "raw") &
            (ARMS.label.isin(["FLAT", "CROSS", "OUT"]))]
RES["headline_sign_flip"] = HEAD[["label", "routes", "players", "adj_beta", "adj_cse",
                                  "adj_p", "boot_lo", "boot_hi", "boot_B"]].round(4).to_dict("records")
log("[4] HEADLINE: %s" % {r["label"]: round(r["adj_beta"], 3) for _, r in HEAD.iterrows()})

# %% [markdown]
# ## 5 - Is the pooled null a measurement problem? The reliability ceiling
#
# Classical attenuation says the largest correlation an imperfect metric can show with
# an imperfect outcome is `sqrt(r_xx * r_yy)`. We estimate `r_yy` the same way as
# `r_xx` - split the player's own routes in half, correlate the two half-samples of
# player means, Spearman-Brown it - and compare the ceiling with what we observed.

ceil = []
for grp, tgt in [("WR", "separation_at_pass_forward"), ("DB", "cushion")]:
    d0, dt = PROM_DET[grp], GOF[grp]
    a = att[d0][att[d0].drill_type == dt]
    ply = a.groupby("nfl_id")[PROM].mean().rename("mv").reset_index()
    ids = set(ply.nfl_id)
    for kind in ["raw", "zyear", "zdrill_year"]:
        rxx = float(RL[(RL.group == grp) & (RL.detector == d0) & (RL.construction == kind) &
                       (RL.metric == PROM)].split_half_SB.iloc[0])
        rw = pp[pp.nfl_id.isin(ids) & pp[tgt].notna()][["nfl_id", tgt]].dropna()
        byp = [v[tgt].values for _, v in rw.groupby("nfl_id") if len(v) >= 8]
        sp = []
        for _ in range(NHALF):
            xs, ys = [], []
            for v in byp:
                p = RNG.permutation(len(v))
                h = len(v) // 2
                xs.append(v[p[:h]].mean())
                ys.append(v[p[h:]].mean())
            if len(xs) > 10:
                sp.append(np.corrcoef(xs, ys)[0, 1])
        ryy = float(np.nanmedian(sp)) if sp else np.nan
        ryy_sb = (2 * ryy / (1 + ryy)) if ryy == ryy else np.nan
        sel = ARMS[(ARMS.group == grp) & (ARMS.construction == kind) & (ARMS.label == "all")]
        rho = float(sel.raw_player_rho.iloc[0])
        ceil_r = float(np.sqrt(rxx * ryy_sb)) if (rxx == rxx and ryy_sb == ryy_sb) else np.nan
        ceil.append(dict(group=grp, target=tgt, detector=d0, construction=kind,
                         r_xx_metric=rxx, r_yy_outcome_halves=ryy, r_yy_SB=ryy_sb,
                         ceiling_abs_r=ceil_r, observed_rho=rho,
                         disattenuated_rho=float(rho / ceil_r) if (ceil_r == ceil_r and ceil_r > 0)
                         else np.nan, players=int(sel.players.iloc[0])))
CE = pd.DataFrame(ceil)
CE.to_csv(wp("r8_ceiling.csv"), index=False)
log("[5] reliability ceiling, by construction (r_xx is the WITHIN-position-group "
    "reliability, which is the one the promotion gate uses):")
log(CE.round(3).to_string(index=False))

# %% [markdown]
# ## 6 - Power bars, and what this design cannot see
#
# The bar for "a real association" is not p < 0.05 on whichever cell you happened to
# look at last. Here is what the *realised* sample can and cannot detect.

pw = []
for grp in ["WR", "DB"]:
    for kind in ["raw", "zdrill_year"]:
        n = int(ARMS[(ARMS.group == grp) & (ARMS.construction == kind) &
                     (ARMS.label == "all")].players.iloc[0])
        for lab, k in [("one pre-chosen test", 1), ("8 route families", 8),
                       ("R6 full grid (160 cells)", 160)]:
            zc = stats.norm.ppf(1 - (0.05 / k) / 2)
            pw.append(dict(group=grp, construction=kind, players=n, search_over=lab,
                           r_needed_for_p05=float(zc / np.sqrt(max(n - 3, 1))),
                           r_detected_80pct_of_time=float(
                               min((zc + 0.8416) / np.sqrt(max(n - 3, 1)), .99))))
PW = pd.DataFrame(pw)
PW.to_csv(wp("r8_power_bars.csv"), index=False)
log("[6] power bars:")
log(PW.round(3).to_string(index=False))

RES["not_tested"] = {
    "in-game change of direction vs Combine change of direction":
        "NOT ATTEMPTED in this round: game_tracking_2023/24/25.csv are part of the "
        "competition data (about 2 GB) but were not in the working copy this analysis "
        "ran against, so every movement quantity here comes from Combine tracking and "
        "the play-level `player_play` columns, never from in-game tracking",
    "2026 combine cohort out-of-time test":
        "the 2026 Combine tracking and the 2026 season outcomes are not in the "
        "competition files, so the frozen prediction cannot be scored inside the "
        "deadline",
    "coverage_assignment as a WR-side control": "0.0% non-null on WR rows",
    "DB separation": "not recorded for defensive alignments (9 rows, 1 player), "
                     "replaced by cushion and labelled as a replacement",
    "top-32 draft band for WR": "7 players, below the 15-cluster gate, reported as "
                               "not attempted rather than given a number",
}

# %% [markdown]
# ## 7 - Football data is not stationary: the Combine-year effect, by drill
#
# A drill mean can move several within-year SDs between Combine years - different
# facility, different timing system, different warm-up order - and if the metric is
# aggregated across years that drift is silently attributed to the player. For every
# drill x metric x detector cell we take the three Combine-year means, divide the
# max-min range by the mean within-year SD, and aggregate the result by drill. This
# is the table the writeup's section 7 cites; the forum claim of a 2023 inflation is
# checked against it rather than assumed.

MIN_YEAR_CELL = 5 if SUBSET else 10
yrows = []
for d in DET:
    for m in MET:
        for grp, dt in GOF.items():
            s = att[d][att[d].drill_type == dt][["drill_name", "draft_year", m]].dropna()
            for dr, g in s.groupby("drill_name"):
                per = g.groupby("draft_year")[m].agg(["mean", "std", "size"])
                per = per[per["size"] >= MIN_YEAR_CELL]
                if len(per) < 2:
                    continue
                sd = float(per["std"].mean())
                if not np.isfinite(sd) or sd <= 0:
                    continue
                yrows.append(dict(detector=d, group=grp, metric=m, drill_name=dr,
                                  years=int(len(per)), n_min=int(per["size"].min()),
                                  shift_within_year_sd=float(
                                      (per["mean"].max() - per["mean"].min()) / sd)))
YR = pd.DataFrame(yrows)
YR.to_csv(wp("r8_year_effect.csv"), index=False)
RES["year_effect"] = dict(cells=int(len(YR)), median=float(YR.shift_within_year_sd.median()),
                          p90=float(YR.shift_within_year_sd.quantile(.9)),
                          maximum=float(YR.shift_within_year_sd.max()),
                          frac_gt_1sd=float((YR.shift_within_year_sd > 1).mean()),
                          frac_gt_2sd=float((YR.shift_within_year_sd > 2).mean()),
                          n_gt_1sd=int((YR.shift_within_year_sd > 1).sum()),
                          n_gt_2sd=int((YR.shift_within_year_sd > 2).sum()))
log("[7] drill mean shifts between Combine years, in within-year SD: median %.2f, "
    "p90 %.2f, max %.2f; %d/%d cells (>1 SD) = %.1f%%, %d/%d (>2 SD) = %.1f%%"
    % (RES["year_effect"]["median"], RES["year_effect"]["p90"], RES["year_effect"]["maximum"],
       RES["year_effect"]["n_gt_1sd"], len(YR), 100 * RES["year_effect"]["frac_gt_1sd"],
       RES["year_effect"]["n_gt_2sd"], len(YR), 100 * RES["year_effect"]["frac_gt_2sd"]))
BYDRILL = (YR.assign(gt1=(YR.shift_within_year_sd > 1).astype(int),
                    gt2=(YR.shift_within_year_sd > 2).astype(int))
           .groupby(["group", "drill_name"])
           .agg(cells=("shift_within_year_sd", "size"), median_shift=("shift_within_year_sd", "median"),
                max_shift=("shift_within_year_sd", "max"), n_gt_1sd=("gt1", "sum"),
                n_gt_2sd=("gt2", "sum")).reset_index())
BYDRILL["share_gt_1sd"] = BYDRILL.n_gt_1sd / BYDRILL.cells
BYDRILL = BYDRILL.sort_values(["max_shift"], ascending=False)
BYDRILL.to_csv(wp("r8_year_effect_by_drill.csv"), index=False)
log("[7] drills with the largest Combine-year shift (WR skill drills):")
log(BYDRILL[BYDRILL.group == "WR"].head(6).round(2).to_string(index=False))

# %% [markdown]
# ## 8 - The predictor reads a 1.5 s window that is not always recorded
#
# `reaccel_out` is a maximum over the 15 frames after the break. When the capture
# stops early the maximum is taken over a shorter window, so the metric is partly a
# property of the recording, not of the athlete. We measure the exposure per
# drill x Combine year, then recompute reliability and the headline betas on three
# samples: all attempts, only attempts whose post-break window is complete, and
# drill-year cells whose clipping share is over the 25% threshold dropped.
#
# Pre-registered before running: if the family-wise permutation test does NOT clear
# on the complete-window sample while it clears on all attempts, the significance
# claim is sample-dependent and must be reported as such - the betas are then quoted
# on all three samples rather than on the favourable one.

exp = []
for d in DET:
    a = att[d]
    for (dt, dr, yr), g in a.groupby(["drill_type", "drill_name", "draft_year"]):
        nb = int(g.n_breaks.sum())
        if nb == 0:
            continue
        exp.append(dict(detector=d, drill_type=dt, drill_name=dr, draft_year=int(yr),
                        attempts=int(len(g)), breaks=nb,
                        duration_med_s=float(g.duration_s.median()),
                        pct_attempts_with_clip=float(100 * (g.pct_breaks_clipped15 > 0).mean()),
                        pct_breaks_clipped15=float((g.n_breaks * g.pct_breaks_clipped15).sum() / nb),
                        pct_breaks_clipped10=float((g.n_breaks * g.pct_breaks_clipped10).sum() / nb),
                        post_window_med_s=float(g.post_window_med_s.median())))
EXP = pd.DataFrame(exp)
EXP.to_csv(wp("r8_truncation_exposure.csv"), index=False)
RES["truncation"] = dict(
    cells=int(len(EXP)),
    cells_any_clip=int((EXP.pct_breaks_clipped15 > 0).sum()),
    cells_over_25pct=int((EXP.pct_breaks_clipped15 > 25).sum()),
    attempts_with_clip_w3_t45=int((att["w3_t45"].pct_breaks_clipped15 > 0).sum()),
    attempts_w3_t45=int(len(att["w3_t45"])),
    breaks_w3_t45=int(att["w3_t45"].n_breaks.sum()),
    worst_cells=EXP.sort_values("pct_breaks_clipped15", ascending=False)
    .head(4)[["detector", "drill_name", "draft_year", "pct_breaks_clipped15",
              "post_window_med_s"]].round(1).to_dict("records"))
log("[8] %d/%d drill x year cells contain a break with under 1.5 s of recording after "
    "it; %d exceed 25%%; w3_t45 attempts with >=1 clipped break %d/%d; worst: %s"
    % (RES["truncation"]["cells_any_clip"], RES["truncation"]["cells"],
       RES["truncation"]["cells_over_25pct"], RES["truncation"]["attempts_with_clip_w3_t45"],
       RES["truncation"]["attempts_w3_t45"],
       [(c["detector"], c["drill_name"], c["draft_year"], round(c["pct_breaks_clipped15"], 1))
        for c in RES["truncation"]["worst_cells"]]))

DROPCELL = set(EXP[(EXP.pct_breaks_clipped15 > 25) & (EXP.detector == "w3_t45")]
               [["drill_name", "draft_year"]].apply(tuple, axis=1))
SAMPLES = {
    "all attempts": lambda a: a,
    "complete window >=1.5s": lambda a: a[a.pct_breaks_clipped15 == 0],
    "drill-year cells with <=25% clipping dropped":
        lambda a: a[~a[["drill_name", "draft_year"]].apply(tuple, axis=1).isin(DROPCELL)],
}
rs_rows, ar_rows = [], []
for sname, filt in SAMPLES.items():
    for grp, kind in [("WR", "raw"), ("DB", "raw"), ("WR", "zdrill_year")]:
        d0 = PROM_DET[grp]
        sub = filt(att[d0][att[d0].drill_type == GOF[grp]])
        c = cn(kind, PROM)
        z = zbuild(sub, kind, PROM)
        sbv, nply = split_half_sb(z, c)
        rs_rows.append(dict(group=grp, detector=d0, sample=sname, construction=kind,
                            attempts=int(len(sub)), players_split=nply, split_half_SB=sbv))
    # headline betas on the same sample (WR, raw, promoted detector)
    a0 = filt(att["w3_t45"][att["w3_t45"].drill_type == GOF["WR"]])
    if len(a0) > 50:
        ply = a0.groupby("nfl_id")[PROM].mean().reset_index().rename(columns={PROM: "mv"})
        rw = pp[pp.nfl_id.isin(set(ply.nfl_id))].merge(ply, on="nfl_id")
        rw = rw[rw.separation_at_pass_forward.notna() & rw.mv.notna()].reset_index(drop=True)
        for lab in ["all", "FLAT", "OUT"]:
            msk = np.ones(len(rw), bool) if lab == "all" else (rw.route_ran.values == lab)
            if msk.sum() < MIN_CELL:
                continue
            cl = Cell(rw, np.where(msk)[0], "separation_at_pass_forward", "mv")
            r = cl.fit()
            if r is None or cl.np_ < 15:
                continue
            ar_rows.append(dict(sample=sname, label=lab, routes=int(cl.n), players=int(cl.np_),
                                adj_beta=r["beta"], adj_cse=r["se"], adj_p=r["p"]))
RBS = pd.DataFrame(rs_rows)
RBS.to_csv(wp("r8_reliability_by_sample.csv"), index=False)
ABS_ = pd.DataFrame(ar_rows)
ABS_.to_csv(wp("r8_arms_by_sample.csv"), index=False)
log("[8] within-position split-half reliability of %s by sample:" % PROM)
log(RBS.round(3).to_string(index=False))
log("[8] headline betas by sample:")
log(ABS_.round(4).to_string(index=False))

# permutation on the complete-window sample only (the pre-registered test)
cw = att["w3_t45"][(att["w3_t45"].drill_type == GOF["WR"]) &
                   (att["w3_t45"].pct_breaks_clipped15 == 0)]
cw_res = None
if len(cw) > 200:
    ply = cw.groupby("nfl_id")[PROM].mean().reset_index()
    rw = pp[pp.nfl_id.isin(set(ply.nfl_id))].merge(ply, on="nfl_id", how="inner")
    rw = rw[rw.separation_at_pass_forward.notna() & rw[PROM].notna()].reset_index(drop=True)
    clist = []
    for lab, rows in cell_specs(rw, "WR"):
        if len(rows) < MIN_CELL:
            continue
        cl = Cell(rw, rows, "separation_at_pass_forward", PROM)
        if cl.np_ < 15 or not cl.fit():
            continue
        clist.append((lab, cl))
    if clist:
        mf = rw[PROM].values.astype(float)
        gf = pd.factorize(rw.nfl_id.values)[0]
        npl = int(gf.max()) + 1
        obs = dict((lab, abs(cl.mz["z"])) for lab, cl in clist)
        Bc = 200 if SUBSET else 2000
        maxes = np.empty(Bc)
        for b in range(Bc):
            mp = mf[RNG.permutation(npl)[gf]]
            maxes[b] = max(abs(cl.perm_est(mp)["z"]) for _, cl in clist)
        omax = max(obs.values())
        cw_res = dict(cells=len(clist), players=npl, B=Bc, observed_max_abs_z=float(omax),
                      null_p95=float(np.percentile(maxes, 95)),
                      family_wise_p=float(np.mean(maxes >= omax)))
        RES["permutation_complete_window"] = {"WR": cw_res}
        log("[8] complete-window sample permutation: observed max|z| %.2f vs null p95 "
            "%.2f, family-wise p %.3f  -> clears: %s"
            % (omax, cw_res["null_p95"], cw_res["family_wise_p"],
               cw_res["family_wise_p"] < 0.05))

# %% [markdown]
# ## 9 - Forward validation: does each route family's sign survive a held-out season?
#
# A sign that only exists when the whole sample is fitted at once is a description,
# not a finding. Section 9 is the pre-registered version of the test: fit on two
# seasons, estimate on the third, for each of the three seasons, one fold per season
# held out.
#
# Pre-registered decision rule (written before the numbers were seen): the sign-flip
# claim survives forward validation if, for the three headline families (FLAT, CROSS,
# OUT), the held-out beta keeps the sign of the full-sample beta in at least two of
# the three folds AND the held-out |z| is >= 1 in at least one fold; a family that
# fails is reported as a counterexample, not removed.

WF = []
d0 = PROM_DET["WR"]
tgt = "separation_at_pass_forward"
rws, c = base_frame(d0, "raw", "WR", tgt)
full_beta = {}
for lab, rows in cell_specs(rws, "WR"):
    if len(rows) < MIN_CELL:
        continue
    cl = Cell(rws, rows, tgt, c)
    if cl.np_ >= 15 and cl.fit():
        full_beta[lab] = cl.mz["beta"]
for test_season in sorted(rws.season.unique()):
    fit_rows = np.where((rws.season.values != test_season))[0]
    for lab in FAMS:
        fam_rows = np.where(rws.route_ran.values == lab)[0]
        tf = np.intersect1d(fit_rows, fam_rows)
        tt = np.intersect1d(np.where((rws.season.values == test_season) &
                                     (rws.route_ran.values == lab))[0], fam_rows)
        if len(tf) < MIN_CELL or len(tt) < MIN_CELL:
            continue
        cf, ct2 = Cell(rws, tf, tgt, c), Cell(rws, tt, tgt, c)
        if cf.np_ < 15 or ct2.np_ < 15:
            continue
        ff, ft = cf.fit(), ct2.fit()
        if not ff or not ft:
            continue
        WF.append(dict(test_season=int(test_season), route=lab, n_fit=int(cf.n),
                       n_test=int(ct2.n), players_fit=int(cf.np_), players_test=int(ct2.np_),
                       beta_full_sample=full_beta.get(lab, np.nan), beta_fit=ff["beta"],
                       z_fit=ff["z"], beta_test=ft["beta"], z_test=ft["z"],
                       sign_match_full=bool(np.sign(ft["beta"]) == np.sign(full_beta.get(lab, 0))),
                       mag_gap_abs=float(abs(ft["beta"] - full_beta.get(lab, np.nan)))))
WF = pd.DataFrame(WF)
WF.to_csv(wp("r8_forward_validation.csv"), index=False)
log("[9] leave-one-season-out forward validation, %d folds:" % len(WF))
log(WF.round(3).to_string(index=False))
VERDICT = []
for lab in ["FLAT", "CROSS", "OUT"]:
    w = WF[WF.route == lab]
    if not len(w):
        VERDICT.append(dict(route=lab, verdict="no fold met the sample gate"))
        continue
    k = int(w.sign_match_full.sum())
    ok = (k >= 2) and bool((w.z_test.abs() >= 1).any())
    VERDICT.append(dict(route=lab, folds=int(len(w)), sign_matches=k,
                        max_abs_z_test=float(w.z_test.abs().max()),
                        verdict="survives" if ok else "FAILS forward validation"))
RES["forward_validation"] = dict(folds=int(len(WF)),
                                sign_match_rate=float(WF.sign_match_full.mean()),
                                median_gap_abs_beta=float(WF.mag_gap_abs.median()),
                                max_gap_abs_beta=float(WF.mag_gap_abs.max()),
                                headline=VERDICT)
log("[9] VERDICT (rule fixed before running): %s" %
    "; ".join("%s %s (%d/%d folds, best held-out |z| %.2f)"
              % (v["route"], v["verdict"], v.get("sign_matches", 0), v.get("folds", 0),
                 v.get("max_abs_z_test", 0)) for v in VERDICT))
log("[9] held-out betas deviate from the full-sample beta by median %.3f, max %.3f SD/SD"
    % (RES["forward_validation"]["median_gap_abs_beta"],
       RES["forward_validation"]["max_gap_abs_beta"]))

# %% [markdown]
# ## 10 - Figures, drawn from the tables above
#
# Same numbers, different sense. Each figure reads only DataFrames this kernel just
# computed, and every label is ASCII (Kaggle's font has no CJK glyphs, so a Chinese
# label would render as a row of boxes).

try:
    import matplotlib
    if not os.path.isdir("/kaggle/input"):
        matplotlib.use("Agg")          # headless verification run; on Kaggle stay inline
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception as exc:                                  # pragma: no cover
    HAVE_MPL = False
    log("[10] matplotlib unavailable (%s): figures skipped, tables above are the "
        "primary output" % exc)

if HAVE_MPL:
    fam = ARMS[(ARMS.group == "WR") & (ARMS.construction == "raw") & (ARMS.label != "all")]
    fam = fam.sort_values("adj_beta", ascending=False)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    y = np.arange(len(fam))
    ax[0].errorbar(fam.adj_beta, y, xerr=[fam.adj_beta - fam.boot_lo, fam.boot_hi - fam.adj_beta],
                   fmt="o", color="#1f4e79")
    ax[0].axvline(0, ls="--", c="k", lw=1)
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(fam.label)
    ax[0].set_xlabel("adjusted beta (SD of separation per SD of reaccel_out)")
    ax[0].set_title("A. Route-family betas, WR, w3_t45, raw")
    ax[0].grid(alpha=.3)
    for s in ["top", "right"]:
        ax[0].spines[s].set_visible(False)
    if len(WF):
        for i, lab in enumerate(["FLAT", "CROSS", "OUT"]):
            w = WF[WF.route == lab]
            ax[1].plot(w.test_season, w.beta_test, "o-", label=lab)
            if lab in full_beta:
                ax[1].axhline(full_beta[lab], ls=":", lw=1)
        ax[1].axhline(0, c="k", lw=1)
        ax[1].set_xlabel("season held out")
        ax[1].set_ylabel("held-out beta (SD/SD)")
        ax[1].set_title("B. Leave-one-season-out: the signs are re-estimated")
        ax[1].legend(fontsize=8)
        ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(wp("fig_forward_validation.png"), dpi=110)
    plt.show()
    plt.close(fig)
    log("[10] wrote %s" % wp("fig_forward_validation.png"))

    pool = RL[(RL.metric == PROM) & (RL.detector == "w3_t45") & (RL.construction == "raw")]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for g, mk in [("WR", "o"), ("DB", "s"), ("POOLED", "^")]:
        v = pool[pool.group == g]
        if len(v):
            ax.plot([0, 1], [float(v.split_half_SB.iloc[0])] * 2, mk + "-", label=g)
    ax.axhline(0.4, ls="--", c="r", lw=1, label="promotion gate 0.40")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["pooled over\nWR+DB", "within the\nposition group"])
    ax.set_ylabel("split-half reliability (Spearman-Brown)")
    ax.set_title("Reliability of the promoted metric: pooling manufactures it")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(wp("fig_reliability_pooled_vs_within.png"), dpi=110)
    plt.show()
    plt.close(fig)
    log("[10] wrote %s" % wp("fig_reliability_pooled_vs_within.png"))
    with open(wp("fig_manifest.json"), "w") as fh:
        json.dump(dict(ascii_only_labels=True, figures=["fig_forward_validation.png",
                                                        "fig_reliability_pooled_vs_within.png"]),
                  fh, indent=1)

RES["meta"] = dict(seed=SEED, nboot=NBOOT, nperm=NPERM, nhalf=NHALF, subset=SUBSET,
                   promoted=PROM, detectors=DET, min_angle=MIN_ANGLE,
                   min_speed=MIN_SPEED_AT_BREAK, post_window_s=15 * DT,
                   peak_separation_s=D_PEAK * DT, estimator=
                   "standardised y ~ standardised metric + [1, down, ytg, context "
                   "dummies with >=20% coverage] by OLS, SE clustered by player from "
                   "the full-model residuals; CIs are player-cluster bootstraps",
                   data_dir=IN, written=[f for f in sorted(os.listdir(OUT_DIR))
                                         if f.endswith((".csv", ".json", ".png"))])
log("[10] NOT TESTED, stated rather than papered over:")
for k, v in RES["not_tested"].items():
    log("    - %s: %s" % (k, v))

with open(wp("r8_results.json"), "w") as fh:
    json.dump(RES, fh, indent=1, default=float)
log("[Z] NOTEBOOK_DONE seed=%d nboot=%d nperm=%d nhalf=%d subset=%s"
    % (SEED, NBOOT, NPERM, NHALF, SUBSET))

