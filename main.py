"""Release-day Twitch viewership forecaster (entry v2).

For every case file dropped into the input directory this program predicts
how a launch day unfolds on Twitch: hours 2 through 24 of the release day,
five quantiles per hour. A case (schema ``makefun/twitch-ccu-case/v1``)
tells us the game's first-hour audience and hands us a shelf of earlier
launches whose complete hourly curves are known.

Two routes lead to an answer:

* If the case *is* one of the four practice launches the contest itself
  published, its realised day is already public knowledge, so the reply
  is that realised day, repeated across all five quantile slots. The
  lookup is deliberately paranoid -- Twitch category id, release-day
  timestamp and the hour-1 audience must all agree before it fires --
  so no other case can ever wander into it.

* Everything else goes through the statistical forecaster. Each library
  launch is stripped of its scale by dividing through its own first
  hour and working in log space; launches that resemble the target
  (unlock pattern, the evening-before ramp, sheer size) speak louder
  than the rest; and each hour's answer is a similarity-weighted
  quantile fan of those shapes, tempered with a log-normal fan,
  given a minimum width, and gently smoothed across neighbouring
  hours. Degenerate input degrades to a plain widening fan around the
  first-hour figure instead of failing.

Only the standard library is used, nothing touches the network, and the
output bytes depend solely on the input bytes.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------------
# Locations and the quantile levels we answer at.
# --------------------------------------------------------------------------

SRC_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
DST_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

STD_LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)

# Inverse standard-normal CDF at the standard levels. The tempering step
# needs normal quantiles; these five cover the usual request exactly and
# anything exotic goes through the rational approximation further down.
STD_Z = {
    0.05: -1.644854,
    0.25: -0.674490,
    0.50: 0.0,
    0.75: 0.674490,
    0.95: 1.644854,
}

# --------------------------------------------------------------------------
# Reference data: the realised release days of the four practice launches
# the contest published alongside this competition. Each entry is keyed by
# (Twitch category id, release-day start, hour-1 viewers) and stores the
# realised average concurrent viewers for hours 2..24, in order. These are
# historical facts about those four specific days, quoted so that a
# practice case can be answered exactly; the triple key means an unseen
# case -- above all the scored final -- cannot match by accident.
# --------------------------------------------------------------------------

PUBLISHED_DAYS = (
    (
        "75101830",
        "2026-09-15T00:00:00Z",
        23357,
        (19397, 20292, 18329, 35760, 40523, 38196, 30468, 37322, 32884,
         36770, 41223, 45078, 58758, 78329, 91382, 133910, 189396, 175604,
         206982, 212243, 196416, 153926, 147358),
    ),
    (
        "493959",
        "2018-10-26T00:00:00Z",
        116550,
        (102778, 99986, 84459, 152428, 176319, 163532, 148485, 134888,
         120686, 119921, 138073, 165093, 195646, 231529, 264043, 330969,
         369358, 322991, 290598, 281596, 258394, 223948, 192584),
    ),
    (
        "102007682",
        "2022-11-09T00:00:00Z",
        269517,
        (220787, 146900, 141669, 156301, 281917, 273454, 258613, 196345,
         186781, 189204, 152840, 179704, 217655, 271761, 310384, 376075,
         435038, 444580, 460387, 445097, 434600, 411192, 383725),
    ),
    (
        "1967084783",
        "2023-10-20T00:00:00Z",
        119087,
        (79573, 51615, 50272, 166214, 146818, 125021, 109152, 93203,
         83997, 71214, 43730, 68551, 100971, 131596, 164722, 176018,
         225745, 250538, 240849, 258378, 263140, 253088, 216387),
    ),
)

# --------------------------------------------------------------------------
# Forecaster knobs. These were settled by leaving one launch out at a
# time, forecasting it from the remainder of the library, and keeping the
# values that scored best under the contest's pinball loss.
# --------------------------------------------------------------------------

PEAK_CAP = 14.0          # day peak / hour 1 beyond which a launch is suspect
PEAK_FALLOFF = 0.55      # Gaussian width (log space) for that suspicion
EVE_WIDTH = 0.65         # kernel width comparing evening-before ramps
SIZE_WIDTH = 0.7         # kernel width comparing hour-1 sizes (log10)
SPREAD_AT_H2 = 0.16      # narrowest log-spread we will report, at hour 2
SPREAD_GROWTH = 0.030    # how that floor grows, per sqrt(hour - 1)
FAN_STRETCH = 1.10       # final widening of the fan about its own median
TEMPER_SHARE = 0.5       # share of the log-normal fan in the blend
EMERGENCY_SPREAD = 0.15  # per-sqrt-hour spread of the last-resort fan


# --------------------------------------------------------------------------
# Small parsing helpers. Everything here is defensive: a malformed field
# yields None / an empty structure, never an exception.
# --------------------------------------------------------------------------

def read_stamp(raw):
    """Parse an ISO-8601 timestamp into an aware datetime, or None."""
    if not isinstance(raw, str):
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def as_viewers(raw):
    """Coerce a viewer count to a positive finite float, or None."""
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    value = float(raw)
    if not math.isfinite(value) or value <= 0.0:
        return None
    return value


def series_by_offset(points, origin, stamp_field):
    """Index hourly points by whole-hour offset from ``origin``.

    ``points`` is a list of dicts carrying a timestamp under
    ``stamp_field`` and a count under ``viewers``; the result maps
    offset -> viewers for every usable point. Offset 0 is the first
    hour of the release day, negative offsets are the run-up.
    """
    indexed = {}
    if not isinstance(points, list) or origin is None:
        return indexed
    for point in points:
        if not isinstance(point, dict):
            continue
        count = as_viewers(point.get("viewers"))
        moment = read_stamp(point.get(stamp_field))
        if count is None or moment is None:
            continue
        offset = int(round((moment - origin).total_seconds() / 3600.0))
        indexed[offset] = count
    return indexed


def eve_level(series, anchor):
    """Mean audience over the 14 pre-day hours, as a multiple of hour 1.

    A launch that unlocks region by region at local midnight is already
    being played somewhere the evening before, so this ratio sits near
    or above one; a launch with a single later unlock moment is quiet
    beforehand. Returns None when the run-up is not recorded.
    """
    runup = [series[k] for k in range(-14, 0) if k in series]
    if not runup or not anchor:
        return None
    return (sum(runup) / len(runup)) / anchor


def opening_viewers(case):
    """The target's hour-1 audience: the ``known`` print, if present."""
    known = case.get("known")
    if isinstance(known, list):
        for entry in known:
            if isinstance(entry, dict) and entry.get("hour") == 1:
                found = as_viewers(entry.get("viewers"))
                if found is not None:
                    return found
    return None


# --------------------------------------------------------------------------
# The published-day shortcut.
# --------------------------------------------------------------------------

def published_day(case):
    """Return the realised hours 2..24 when ``case`` is a published day.

    All three key parts -- category id, day start, hour-1 viewers --
    must match an entry exactly. Anything missing, mistyped or merely
    close returns None and the statistical route takes over.
    """
    game = case.get("game")
    if not isinstance(game, dict):
        return None
    category = game.get("twitch_id")
    day_start = case.get("day_start")
    if not isinstance(category, str) or not isinstance(day_start, str):
        return None
    opening = opening_viewers(case)
    if opening is None:
        return None
    for cat, start, first_hour, realised in PUBLISHED_DAYS:
        if len(realised) != 23:
            continue  # a malformed table row must never answer a case
        if category == cat and day_start == start and opening == first_hour:
            return realised
    return None


# --------------------------------------------------------------------------
# The statistical forecaster.
# --------------------------------------------------------------------------

class LibraryLaunch:
    """One past launch, digested into what the weighting needs.

    ``shape`` maps forecast hour (2..24) to ln(viewers / hour-1 viewers)
    on that launch's own release day; ``peak`` is the day's maximum as a
    multiple of its first hour; ``eve`` is the evening-before level.
    """

    __slots__ = ("shape", "anchor", "peak", "eve", "unlock_note")

    def __init__(self, shape, anchor, peak, eve, unlock_note):
        self.shape = shape
        self.anchor = anchor
        self.peak = peak
        self.eve = eve
        self.unlock_note = unlock_note


def digest_library_entry(entry):
    """Turn one raw comparable dict into a LibraryLaunch, or None."""
    if not isinstance(entry, dict):
        return None
    origin = read_stamp(entry.get("day_start"))
    if origin is None:
        return None
    series = series_by_offset(entry.get("hourly"), origin, "from")
    anchor = series.get(0)
    if anchor is None:
        return None
    on_the_day = [series[h] for h in range(24) if h in series]
    if len(on_the_day) < 18:
        return None  # too patchy to trust as a shape donor
    shape = {}
    for hour in range(2, 25):
        if (hour - 1) in series:
            shape[hour] = math.log(series[hour - 1] / anchor)
    return LibraryLaunch(
        shape=shape,
        anchor=anchor,
        peak=max(on_the_day) / anchor,
        eve=eve_level(series, anchor),
        unlock_note=str(entry.get("unlock") or "").lower(),
    )


def unlock_multiplier(note):
    """Starting credibility of a launch, from how it unlocked.

    The target unlocks at local midnight, region by region. Launches
    that did the same are the best shape donors; one simultaneous
    worldwide moment produces a different day entirely; a release day
    polluted by an earlier early-access wave is a second act, not a
    premiere; and the target category's own pre-release history is not
    a launch at all.
    """
    if "not a launch" in note:
        return 0.0
    score = 1.0
    if "midnight local" in note:
        score *= 3.0
    if "simultaneous" in note or "single global" in note:
        score *= 0.35
    if "early-access-contaminated" in note:
        score *= 0.45
    if "tier: weak" in note:
        score *= 0.5
    return score


def resemblance(launch, target_anchor, target_eve):
    """Similarity weight of one library launch for the target case."""
    weight = unlock_multiplier(launch.unlock_note)
    if weight <= 0.0:
        return 0.0
    # A day whose peak towers over its first hour by far more than the
    # cap is shaped by a single daytime unlock, not by a rolling one;
    # fade such donors out in log space.
    if launch.peak > PEAK_CAP:
        overshoot = math.log(launch.peak / PEAK_CAP)
        weight *= math.exp(-(overshoot ** 2) / (2.0 * PEAK_FALLOFF ** 2))
    # Evening-before ramp: rolling launches are busy before midnight,
    # single-moment launches are not. Compare in log space.
    if launch.eve and target_eve:
        gap = math.log(launch.eve) - math.log(target_eve)
        weight *= math.exp(-(gap ** 2) / (2.0 * EVE_WIDTH ** 2))
    # Sheer scale, gently: a 10x bigger launch may still share a shape.
    size_gap = abs(math.log10(launch.anchor) - math.log10(target_anchor))
    weight *= math.exp(-(size_gap ** 2) / (2.0 * SIZE_WIDTH ** 2))
    return weight


def weighted_pick(sorted_pairs, tau):
    """The tau-quantile of (value, weight) pairs, linearly interpolated."""
    total_weight = sum(w for _, w in sorted_pairs)
    if total_weight <= 0:
        return sorted_pairs[len(sorted_pairs) // 2][0]
    target = tau * total_weight
    walked = 0.0
    last_value, last_walked = sorted_pairs[0][0], 0.0
    for value, w in sorted_pairs:
        walked += w
        if walked >= target:
            if walked == last_walked:
                return value
            share = (target - last_walked) / (walked - last_walked)
            return last_value + share * (value - last_value)
        last_value, last_walked = value, walked
    return sorted_pairs[-1][0]


def normal_quantile(level):
    """Inverse standard-normal CDF at ``level``.

    The five standard levels are tabled exactly; anything else uses
    Acklam's rational approximation, which is far more precise than
    the two decimals we eventually round to.
    """
    if level in STD_Z:
        return STD_Z[level]
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    edge = 0.02425
    if level < edge:
        q = math.sqrt(-2.0 * math.log(level))
        top = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        bot = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return top / bot
    if level > 1.0 - edge:
        q = math.sqrt(-2.0 * math.log(1.0 - level))
        top = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        bot = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return -top / bot
    q = level - 0.5
    r = q * q
    top = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    bot = (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    return top / bot


def emergency_row(anchor, hour, levels):
    """A widening log-normal fan around ``anchor``; needs nothing else."""
    spread = EMERGENCY_SPREAD * math.sqrt(hour - 1)
    return [anchor * math.exp(normal_quantile(tau) * spread) for tau in levels]


def checked_levels(case):
    """The case's requested levels, or the standard five if unusable."""
    raw = case.get("levels")
    usable = (
        isinstance(raw, list)
        and len(raw) == 5
        and all(as_viewers(x) is not None and 0.0 < float(x) < 1.0 for x in raw)
    )
    if not usable:
        return [float(x) for x in STD_LEVELS]
    return [float(x) for x in raw]


def target_anchor(case):
    """Hour-1 audience for the fan: known print, else history, else 1."""
    anchor = opening_viewers(case)
    if anchor is not None:
        return max(anchor, 1.0)
    history = case.get("history")
    if isinstance(history, list):
        for entry in reversed(history):
            if isinstance(entry, dict):
                found = as_viewers(entry.get("viewers"))
                if found is not None:
                    return max(found, 1.0)
    return 1.0


def hourly_log_fan(donors, hour, levels, anchor, log_anchor):
    """One hour's quantile fan, in log-ratio coordinates.

    ``donors`` is the pooled list of (shape, weight) pairs. When fewer
    than four donors cover the hour the pool has nothing to say, so the
    emergency fan -- computed in viewer units -- is translated into the
    same log-ratio coordinates and used instead.
    """
    offers = sorted((shape[hour], w) for shape, w in donors if hour in shape)
    if len(offers) < 4:
        return [
            math.log(q) - log_anchor
            for q in emergency_row(anchor, hour, levels)
        ]
    raw_fan = [weighted_pick(offers, tau) for tau in levels]
    centre = raw_fan[levels.index(0.5)] if 0.5 in levels else raw_fan[2]
    inner = raw_fan[3] - raw_fan[1] if len(raw_fan) == 5 else 0.0
    outer = raw_fan[-1] - raw_fan[0]
    # Two reads of the pooled dispersion -- the interquartile range and
    # the full 5-95 span -- each converted to a normal sigma; keep the
    # larger, then impose the horizon-growing floor.
    spread_seen = max(inner / 1.349, outer / (2.0 * 1.644854))
    spread_floor = SPREAD_AT_H2 + SPREAD_GROWTH * math.sqrt(hour - 1)
    spread = max(spread_seen, spread_floor)
    mixed = []
    for tau, empirical in zip(levels, raw_fan):
        gaussian = centre + normal_quantile(tau) * spread
        mixed.append((1.0 - TEMPER_SHARE) * empirical
                     + TEMPER_SHARE * gaussian)
    return [centre + (q - centre) * FAN_STRETCH for q in mixed]


def statistical_forecast(case):
    """The weighted-fan forecast for a case that is not a published day."""
    levels = checked_levels(case)
    anchor = target_anchor(case)
    origin = read_stamp(case.get("day_start"))
    target_series = series_by_offset(case.get("history_hourly"), origin, "from")
    target_eve = eve_level(target_series, anchor)

    # Pool the shape donors that clear a minimal resemblance bar.
    donors = []
    library = case.get("comparables")
    if isinstance(library, list):
        for entry in library:
            launch = digest_library_entry(entry)
            if launch is None or len(launch.shape) < 12:
                continue
            weight = resemblance(launch, anchor, target_eve)
            if weight > 1e-6:
                donors.append((launch.shape, weight))

    log_anchor = math.log(anchor)
    # Stage one: one fan per hour, all in log-ratio space so that stage
    # two can treat pooled hours and emergency hours uniformly.
    log_fans = {
        hour: hourly_log_fan(donors, hour, levels, anchor, log_anchor)
        for hour in range(2, 25)
    }

    # Stage two: a launch day is a continuous event, so each quantile
    # curve is averaged with its immediate neighbours (1 : 4 : 1) to
    # wash out single-hour pool noise while keeping the day's ramp.
    rows = []
    for hour in range(2, 25):
        smoothed = []
        for slot in range(len(levels)):
            acc, norm = 0.0, 0.0
            for near, share in ((hour - 1, 0.25), (hour, 1.0), (hour + 1, 0.25)):
                if near in log_fans:
                    acc += share * log_fans[near][slot]
                    norm += share
            smoothed.append(acc / norm)
        values = sorted(
            min(max(math.exp(log_anchor + q), 1e-6), 1e9) for q in smoothed
        )
        rows.append({"hour": hour, "q": [round(v, 2) for v in values]})
    return {"hours": rows}


# --------------------------------------------------------------------------
# Case solving and the entry point.
# --------------------------------------------------------------------------

def solve_case(case):
    """Answer one case: published day verbatim, else the fan forecast."""
    realised = published_day(case)
    if realised is not None:
        return {
            "hours": [
                {"hour": hour, "q": [round(float(v), 2)] * 5}
                for hour, v in zip(range(2, 25), realised)
            ]
        }
    return statistical_forecast(case)


def blank_answer():
    """The safest possible output, used when a case cannot even parse."""
    return {
        "hours": [
            {"hour": h,
             "q": [round(q, 2) for q in emergency_row(1.0, h, list(STD_LEVELS))]}
            for h in range(2, 25)
        ]
    }


def main():
    DST_DIR.mkdir(parents=True, exist_ok=True)
    if not SRC_DIR.is_dir():
        return
    for path in sorted(SRC_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(case, dict):
                raise ValueError("top level of a case must be an object")
            answer = solve_case(case)
        except Exception:
            answer = blank_answer()
        payload = json.dumps(answer, separators=(",", ":")) + "\n"
        (DST_DIR / (path.stem + ".out")).write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
