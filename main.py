"""Refresh-aware earners-over-players forecaster for GTA VI trophy rarity.

Forecasts, for every PlayStation 5 trophy of Grand Theft Auto VI, its earn
rate on Exophase at a case's target, as five quantiles per trophy, scored by
a pinball loss on the logit scale.

Model (counts space, not rates space)
-------------------------------------
A trophy's rate is earners over players-tracked, r = E / N. Exophase
refreshes its counts on its own schedule, hours apart, so the page a case's
history shows is frozen between refreshes and jumps at them: new players
arrive holding far fewer trophies than the players already tracked, which
dilutes every rate even while earners grow. A model on rates alone reads
those jumps as noise; a model on counts reads them as the mechanism.

For each case the program:
  1. collapses the history to its distinct states (a read whose
     players-tracked and rates are unchanged from the previous state is the
     same stale page, not new information);
  2. fits, per trophy, earners E_i(t) as a power law in log time,
     log E_i = b_i + g_i * x, x = ln(1 + hours since release), with each
     slope shrunk toward the median slope of trophies of similar launch
     rarity (rare trophies crawl, common ones climb), and blends that
     with a direct fit of logit(r_i) on x -- the two functional forms
     disagree exactly where the data is thinnest, so their average in
     logit space is the robust centre and their disagreement widens the
     spread;
  3. fits players-tracked N(t) the same way over the recent window, so a
     saturating launch wave is not extrapolated as if it kept
     accelerating;
  4. forecasts r_i(target) = E_i(target) / N(target), then enforces the
     logical order the trophy list itself implies: story chapters are
     non-increasing down the chapter chain, and the platinum (which needs
     every other trophy) sits at or below the rarest other trophy;
  5. spreads each trophy's quantiles by its own fit residual plus a term
     that grows with the extrapolation distance, so a far target is an
     honestly wider forecast.

When the history is a single frozen page (the practice cases: launch night,
no refresh yet inside the window), no slope can be fit, so the program
falls back to a launch-week refresh prior: one refresh lands before the
target, tracked players multiply about five-fold, and a newly tracked
player holds each trophy at about 7% of the launch-night rate, spread
diffusely across trophies in proportion to their launch earners. That prior
is the mechanism the competition's rules describe for the first refresh of
launch night -- a frozen page for hours, then a wave of new players who
have barely started -- written down as numbers, not fitted to any answer.

Everything is standard library only, deterministic (no randomness, no wall
clock, sorted iteration), and fast: a few hundred reads times 55 trophies
is a few thousand arithmetic operations.

Run: python3 -P -s main.py, with MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR
standing in for /input and /output outside the grader's container.
"""

import json
import math
import os
import re
from datetime import datetime
from pathlib import Path

# The grader's -P leaves this file's directory OFF sys.path; nothing is
# imported from beside it, so no path fix is needed.
INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# --- launch-week refresh prior (flat-history fallback only) -----------------
# Tracked players multiply by this at one launch-week refresh; a newly
# tracked player holds each trophy at NEW_HOLD times the launch-night rate.
PRIOR_N_GROWTH = 5.5
PRIOR_NEW_HOLD = 0.07
# Logit spread of the fallback forecast: wide enough to cover the refresh
# landing early or late and the dilution running stronger or weaker.
PRIOR_SPREAD = 0.45


def clamp_rate(pct):
    return min(99.99, max(0.01, pct))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, halves away from zero on the scaled value."""
    scaled = x * 10000.0
    whole = math.floor(scaled + 0.5)
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    if n % 2:
        return ordered[n // 2]
    return 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def weighted_ols(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, resid_sd)."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return None
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    var = sum(w * r * r for w, r in zip(ws, resid)) / sw
    return a, b, math.sqrt(max(var, 0.0))


def distinct_states(case):
    """Collapse the history to its distinct pages.

    Returns a list of dicts, oldest first, each with time (datetime),
    players (int), earners (list of float, one per trophy) and rates.
    A read identical to the previous kept state is the same stale page.
    Missing rates are carried forward from the last read that had them.
    """
    trophies = case["trophies"]
    n = len(trophies)
    states = []
    last_rates = [None] * n
    prev_key = None
    for read in case["history"]:
        rates = list(read["rates"])
        for i, r in enumerate(rates):
            if r is None:
                rates[i] = last_rates[i]
            else:
                last_rates[i] = r
        if any(r is None for r in rates):
            continue
        key = (read["players_tracked"], tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        players = read["players_tracked"]
        earners = [r / 100.0 * players for r in rates]
        states.append(
            {
                "time": parse_time(read["at"]),
                "players": players,
                "rates": rates,
                "earners": earners,
            }
        )
    return states


def rarity_class(first_rate):
    """Difficulty class from the trophy's first observed rate."""
    if first_rate >= 1.0:
        return "common"
    if first_rate >= 0.75:
        return "mid"
    return "rare"


def chapter_of(description):
    """Story chapter number from a description, else None."""
    m = re.search(r"Chapter (\d)", description or "")
    if m:
        return int(m.group(1))
    if "final mission" in (description or "").lower():
        return 99
    return None


def enforce_story_order(medians, trophies):
    """Pool the chapter chain so medians are non-increasing by chapter.

    A player cannot hold the Chapter 3 trophy without holding Chapter 1's
    in the long run, so the chain's medians must not rise down the chain.
    Simple pool-adjacent-violators pass over the chapter trophies in order.
    """
    chain = []
    for i, t in enumerate(trophies):
        ch = chapter_of(t.get("description", ""))
        if ch is not None:
            chain.append((ch, i))
    chain.sort()
    if len(chain) < 2:
        return medians
    values = [medians[i] for _, i in chain]
    # Pool adjacent violators (non-increasing).
    pooled = [[v] for v in values]
    j = 0
    while j < len(pooled) - 1:
        if sum(pooled[j]) / len(pooled[j]) < sum(pooled[j + 1]) / len(pooled[j + 1]):
            pooled[j] = pooled[j] + pooled[j + 1]
            del pooled[j + 1]
            if j > 0:
                j -= 1
        else:
            j += 1
    out = list(medians)
    k = 0
    for group in pooled:
        avg = sum(group) / len(group)
        for _ in group:
            out[chain[k][1]] = avg
            k += 1
    return out


def forecast_case(case):
    trophies = case["trophies"]
    n = len(trophies)
    if case.get("levels") != LEVELS:
        raise ValueError("levels are not the contest's five")
    if not case.get("history"):
        raise ValueError("case has no history")

    release = parse_time(case["game"]["release_date"] + "T00:00:00Z")
    at = parse_time(case["at"])
    horizon_min = case["horizon_minutes"]
    target = parse_time(case["target"])

    def x_of(moment):
        hours = (moment - release).total_seconds() / 3600.0
        return math.log(1.0 + max(-0.9, hours))

    x_now = x_of(at)
    x_target = x_of(target)
    states = distinct_states(case)
    if not states:
        # No usable read at all: fall back to tier priors with a wide
        # spread rather than failing the case (an invalid case scores as
        # the last read and takes no prize).
        tier_prior = {"platinum": 0.01, "gold": 0.1, "silver": 0.5, "bronze": 1.0}
        rows = []
        for t in trophies:
            m = tier_prior.get(t.get("tier", ""), 1.0)
            lg = logit(m)
            qs = [round4(inv_logit(lg + z * 1.0)) for z in Z]
            for k in range(1, 5):
                if qs[k] < qs[k - 1]:
                    qs[k] = qs[k - 1]
            rows.append({"id": t["id"], "q": qs})
        return {"trophies": rows}
    last = states[-1]
    last_rates = last["rates"]

    medians = [None] * n
    spreads = [None] * n

    # The trophy every tracked player holds in every state stays held.
    universal = [
        i for i in range(n) if all(s["rates"][i] >= 99.99 for s in states)
    ]

    if len(states) >= 3 and len(set(s["players"] for s in states)) >= 3:
        # --- data-driven branch: fit on the recent window. -------------
        # Launch dynamics move fast, so only the last day of reads is
        # fitted (widened to 48h, then the whole history, if that holds
        # too few distinct states): a slope fitted over the whole history
        # is dragged down by launch night, when nothing had happened yet.
        # Slopes are fitted on distinct states only -- a stale page
        # repeated every 10 minutes would otherwise flatten every slope
        # toward 0.
        xs_all = [x_of(s["time"]) for s in states]
        win = [k for k, s in enumerate(states)
               if (at - s["time"]).total_seconds() <= 24 * 3600]
        if len(win) < 3:
            win = [k for k, s in enumerate(states)
                   if (at - s["time"]).total_seconds() <= 48 * 3600]
        if len(win) < 3:
            win = list(range(len(states)))
        xs = [xs_all[k] for k in win]
        wstates = [states[k] for k in win]
        span = xs[-1] - xs[0]
        ws = [0.5 + (0.5 * (x - xs[0]) / span if span > 0 else 0.5)
              for x in xs]

        # Players: log N on x over the window, anchored at the window's
        # centroid so the extrapolation pivots on the recent data, not on
        # the release origin.
        log_n = [math.log(max(1, s["players"])) for s in wstates]
        fit_n = weighted_ols(xs, log_n, ws)
        alpha = min(2.0, max(0.0, fit_n[1])) if fit_n else 0.0
        x_bar = sum(w * x for w, x in zip(ws, xs)) / sum(ws)
        n_bar = (fit_n[0] + fit_n[1] * x_bar) if fit_n else log_n[-1]
        n_target = math.exp(n_bar + alpha * (x_target - x_bar))
        n_target = min(12.0 * last["players"], max(last["players"], n_target))

        # Per-trophy: (A) earners power law, shrunk toward the group
        # median slope; (B) direct logit(rate) trend, shrunk the same
        # way. The forecast is their mean in logit space.
        def group_median_of(slopes):
            by_group = {}
            for i, g in slopes.items():
                by_group.setdefault(rarity_class(last_rates[i]), []).append(g)
            med = {g: median(v) for g, v in by_group.items() if len(v) >= 3}
            glob = median(list(slopes.values())) if slopes else 0.0
            return med, glob

        def shrink(g, k, resid, group):
            rel = k / (k + 2.0)
            if resid <= 0.25:
                rel = max(rel, 0.9)
            if resid > 0.4:
                rel *= 0.5
            return rel * g + (1.0 - rel) * group

        slopes_a = {}
        fits_a = {}
        slopes_b = {}
        fits_b = {}
        for i in range(n):
            if i in universal:
                continue
            ys_a, ys_b, xx, ww = [], [], [], []
            for k, s in enumerate(wstates):
                e = s["earners"][i]
                if e >= 1.0:
                    ys_a.append(math.log(e))
                    ys_b.append(logit(s["rates"][i]))
                    xx.append(xs[k])
                    ww.append(ws[k])
            if len(xx) >= 2:
                fa = weighted_ols(xx, ys_a, ww)
                fb = weighted_ols(xx, ys_b, ww)
                if fa is not None:
                    fits_a[i] = fa
                    slopes_a[i] = fa[1]
                if fb is not None:
                    fits_b[i] = fb
                    slopes_b[i] = fb[1]
        med_a, glob_a = group_median_of(slopes_a)
        med_b, glob_b = group_median_of(slopes_b)

        for i in range(n):
            if i in universal:
                medians[i] = 99.98
                spreads[i] = 0.40
                continue
            fa = fits_a.get(i)
            fb = fits_b.get(i)
            if fa is None and fb is None:
                medians[i] = last_rates[i]
                spreads[i] = 0.80
                continue
            logits = []
            resid = 0.0
            if fa is not None:
                a, g, resid_a = fa
                k = sum(1 for s in wstates if s["earners"][i] >= 1.0)
                group = med_a.get(rarity_class(last_rates[i]), glob_a)
                g_s = shrink(g, k, resid_a, group)
                g_s = min(3.0, max(0.0, g_s))
                # Anchor the shrunk slope at the window centroid: the
                # shrinkage changes the growth rate, it must not pivot
                # the whole line around the release origin, a lever arm
                # of five log units that would turn a small slope change
                # into a large median move.
                y_bar = a + g * x_bar
                e_target = math.exp(y_bar + g_s * (x_target - x_bar))
                e_target = min(n_target, max(0.5 * last["earners"][i], e_target))
                logits.append(logit(100.0 * e_target / n_target))
                resid = max(resid, resid_a)
            if fb is not None:
                a, g, resid_b = fb
                k = sum(1 for s in wstates if s["earners"][i] >= 1.0)
                group = med_b.get(rarity_class(last_rates[i]), glob_b)
                g_s = shrink(g, k, resid_b, group)
                g_s = min(2.0, max(-1.0, g_s))
                y_bar = a + g * x_bar
                logits.append(y_bar + g_s * (x_target - x_bar))
                resid = max(resid, resid_b)
            lg = sum(logits) / len(logits)
            medians[i] = inv_logit(lg)
            # Spread: base + extrapolation distance + own residual +
            # half the disagreement between the two functional forms.
            extra = 0.25 * math.log(max(1.0, math.exp(x_target) / math.exp(x_now)))
            disagree = 0.5 * abs(logits[0] - logits[-1]) if len(logits) > 1 else 0.0
            spreads[i] = min(
                1.15,
                max(0.35, math.sqrt(0.35 ** 2 + extra ** 2 + resid ** 2 + disagree ** 2)),
            )
    else:
        # --- fallback branch: one frozen page, no slope to fit. ---------
        # Expect a refresh before the target: tracked players multiply and
        # newly tracked players hold each trophy at a small fraction of
        # the launch-night rate, spread diffusely in proportion to the
        # launch earners.
        factor = (1.0 + PRIOR_NEW_HOLD * (PRIOR_N_GROWTH - 1.0)) / PRIOR_N_GROWTH
        spread = PRIOR_SPREAD + 0.12 * math.sqrt(max(0.0, horizon_min / 1440.0))
        for i in range(n):
            if i in universal:
                medians[i] = 99.98
                spreads[i] = 0.40
                continue
            medians[i] = last_rates[i] * factor
            spreads[i] = spread

    # The platinum needs every other trophy: it cannot sit above the
    # rarest of them. The 100%-completion trophy needs nearly everything:
    # cap it at the fifth-rarest median, not the single rarest.
    tiers = [t.get("tier", "") for t in trophies]
    for i in range(n):
        if tiers[i] == "platinum":
            others = [medians[j] for j in range(n) if j != i]
            medians[i] = min(medians[i], min(others))
    for i in range(n):
        if "100% completion" in (trophies[i].get("description", "") or ""):
            others = sorted(medians[j] for j in range(n) if j != i)
            medians[i] = min(medians[i], others[min(4, len(others) - 1)])

    medians = enforce_story_order(medians, trophies)

    rows = []
    for i, t in enumerate(trophies):
        m = clamp_rate(medians[i])
        s = spreads[i]
        lg = logit(m)
        qs = [round4(inv_logit(lg + z * s)) for z in Z]
        # Keep the row non-decreasing after rounding.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        qs = [min(99.99, max(0.01, q)) for q in qs]
        rows.append({"id": t["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out = OUTPUT_DIR / (path.stem + ".out")
        out.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
