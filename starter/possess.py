"""Possession from the tactical tracking, as a state machine.

The published baseline gives the ball to whoever is closest, and calls the frame dead
whenever the tracker lost the ball. Most of those gaps are occlusions during live play,
so the label flickers and one possession becomes many. This program keeps the last team
through a loose ball or a gap, and only changes label on evidence:

  - the other team is clearly closest and within CONTROL_M for SWITCH_FRAMES, or
  - the ball is outside the pitch / isolated and nearly stopped, or
  - a Smart Tagging set piece is about to restart (a short dead pad before it).

Every label is written by this file. Nothing here is assigned by watching the video.
Constants are the same on every frame. The half boundaries and the second-half kickoff
are the published clock in DATA.md and FAQ.md, not labels.

    python starter/possess.py --self-check
    python starter/possess.py
    python evaluation/validate_submission.py evaluation/results/possess_v2.csv
    python evaluation/run_evaluation.py

The public 5 minutes are a pipeline check after the file exists. They are not a
tuning set: do not search these constants to climb that board.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from baseline import (  # noqa: E402
    FIELD_SEP, GROUP_SEP, MISSING, RESULTS_DIR, team_of_each_track, write_submission,
)
from load_smart_tagging import load as load_smart_tagging  # noqa: E402

# Metres. A player this close is controlling the ball; farther is a loose ball, and a
# loose ball stays with the last team. One frame inside the radius is a touch, not a
# change of possession, so the other team has to hold it for SWITCH_FRAMES.
CONTROL_M = 2.0
SWITCH_FRAMES = 12  # 0.48 s at 25 fps
# The challenger must be this much closer than the current side before a switch run
# starts. Equal races keep the last team; they never default to home.
SWITCH_MARGIN_M = 0.5

# Outside the pitch by this margin, sustained, is a stoppage. A single frame past the
# line is tracker noise. Once the state is dead, a following occlusion stays dead,
# because the rule for a gap is "keep the current label", and the current label is D.
OFF_MARGIN_M = 1.0
OFF_HOLD = 5

# Ball nearly stopped and nobody near it: in-play stoppage signal (keeper hold, free
# kick wait) that the off-pitch rule misses.
STOP_SPEED = 0.8  # m/s
STOP_ISOLATION_M = 4.0
STOP_HOLD = 25  # 1 s

# Distance-weighted k nearest players. Weight is exp(-d / tau), so a player at
# 6 m counts, but much less than a player at 1 m. A cluster of teammates can
# outweigh one closer opponent. The result is an equivalent distance the rest
# of the state machine already understands.
KNN_K = 3
KNN_TAU = 2.0

# Seconds before a Smart Tagging SET PIECES start are treated as dead. Throw-ins are
# Team=N/A; the tag is used for timing only, never for who restarts. A long pad
# after the tag starts eats live play — the tags last tens of seconds.
SET_PIECE_CODE = "set pieces"
SET_PIECE_PRE_FRAMES = 75  # 3 s at 25 fps

# Linear fill for ball gaps no longer than this. Longer gaps are occlusions:
# they do not invent a ball, and they do not become D by themselves.
GAP_FILL = 25  # 1 s

# FAQ.md: the second half is graded from 110808 and the ball is not in play until this
# kickoff. Those frames are dead. The first half is graded from its own kickoff.
SECOND_HALF_KICKOFF = 113742

HOME, AWAY, DEAD = 0, 1, 2
CODE_TO_LABEL = {HOME: "H", AWAY: "A", DEAD: "D"}


def fill_short_gaps(values, max_gap):
    """Linear interpolation across NaN runs of length 1..max_gap.

    A run that touches either end of the array is left missing: there is no second
    sample to interpolate toward. Long runs are left missing on purpose.
    """
    out = values.copy()
    finite = np.isfinite(values)
    n = len(values)
    i = 0
    while i < n:
        if finite[i]:
            i += 1
            continue
        start = i
        while i < n and not finite[i]:
            i += 1
        if start == 0 or i == n:
            continue
        gap = i - start
        if gap > max_gap:
            continue
        left = values[start - 1]
        right = values[i]
        step = (right - left) / (gap + 1)
        out[start:i] = left + step * np.arange(1, gap + 1)
    return out


def _parse_xy_speed(group):
    """Normalised (x, y, speed) from an 'x,y,speed' group, or None when missing."""
    if not group or group[0] == MISSING or group[0] == "n":
        return None
    comma = group.find(",")
    if comma < 0:
        return None
    comma2 = group.find(",", comma + 1)
    if comma2 < 0:
        return None
    try:
        x = float(group[:comma])
        y = float(group[comma + 1:comma2])
        speed = float(group[comma2 + 1:])
    except ValueError:
        return None
    return x, y, speed


def set_piece_dead_mask(n_frames, events, pre_frames=SET_PIECE_PRE_FRAMES):
    """True on frames just before each SET PIECES start. Timing only, no Team tag."""
    mask = np.zeros(n_frames, dtype=bool)
    n_events = 0
    for event in events:
        if (event.get("code") or "").strip().lower() != SET_PIECE_CODE:
            continue
        start = int(event["start_frame"])
        n_events += 1
        lo = max(0, start - pre_frames)
        hi = min(n_frames, start)
        if lo < hi:
            mask[lo:hi] = True
    return mask, n_events


def read_tracking(raw_path, team_of_track, n_frames, pitch, knn=False):
    """Ball position/speed in metres, and nearest home/away distance per frame.

    Missing samples stay NaN for the ball and +inf for a distance, which the state
    machine treats as 'no evidence' rather than as dead.
    """
    length, width = pitch
    n_slots = len(team_of_track)
    ball_x = np.full(n_frames, np.nan)
    ball_y = np.full(n_frames, np.nan)
    ball_speed = np.full(n_frames, np.nan)
    player_x = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    player_y = np.full((n_frames, n_slots), np.nan, dtype=np.float32)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            try:
                video_frame = int(parts[0]) - 1
            except ValueError:
                continue
            if video_frame < 0 or video_frame >= n_frames:
                continue

            ball = _parse_xy_speed(parts[2])
            if ball is not None:
                ball_x[video_frame] = ball[0] * length
                ball_y[video_frame] = ball[1] * width
                ball_speed[video_frame] = ball[2]

            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if slot >= n_slots:
                    break
                xy = _parse_xy_speed(group)
                if xy is None:
                    continue
                player_x[video_frame, slot] = xy[0] * length
                player_y[video_frame, slot] = xy[1] * width

    observed = int(np.isfinite(ball_x).sum())
    ball_x = fill_short_gaps(ball_x, GAP_FILL)
    ball_y = fill_short_gaps(ball_y, GAP_FILL)
    ball_speed = fill_short_gaps(ball_speed, GAP_FILL)
    filled = int(np.isfinite(ball_x).sum()) - observed

    if knn:
        home_dist, away_dist = knn_distances(
            ball_x, ball_y, player_x, player_y, team_of_track)
    else:
        home_dist, away_dist = nearest_distances(
            ball_x, ball_y, player_x, player_y, team_of_track)
    return ball_x, ball_y, ball_speed, home_dist, away_dist, observed, filled


def nearest_distances(ball_x, ball_y, player_x, player_y, team_of_track):
    """Per-frame distance in metres to the closest player of each team.

    Frames with no ball, or a team with nobody tracked, get +inf.
    """
    delta_x = player_x - ball_x[:, None]
    delta_y = player_y - ball_y[:, None]
    dist = np.hypot(delta_x, delta_y)
    dist[~np.isfinite(dist)] = np.inf

    home = np.where(team_of_track == HOME, dist, np.inf).min(axis=1)
    away = np.where(team_of_track == AWAY, dist, np.inf).min(axis=1)
    return home, away


def _effective_distance(dist, k, tau):
    """One distance per frame from the k nearest columns, weighted by exp(-d/tau)."""
    n_slots = dist.shape[1]
    if n_slots == 0:
        return np.full(dist.shape[0], np.inf)
    k_use = min(k, n_slots)
    nearest = np.partition(dist, k_use - 1, axis=1)[:, :k_use]
    weights = np.exp(-nearest / tau)
    weights[~np.isfinite(nearest)] = 0.0
    score = weights.sum(axis=1)
    effective = np.full(dist.shape[0], np.inf)
    ok = score > 1e-8
    effective[ok] = np.maximum(0.0, -tau * np.log(score[ok]))
    return effective


def knn_distances(ball_x, ball_y, player_x, player_y, team_of_track, k=KNN_K, tau=KNN_TAU):
    """Per-frame equivalent distance to each team from its k nearest players."""
    dist = np.hypot(player_x - ball_x[:, None], player_y - ball_y[:, None])
    dist[~np.isfinite(dist)] = np.inf
    home = _effective_distance(dist[:, team_of_track == HOME], k, tau)
    away = _effective_distance(dist[:, team_of_track == AWAY], k, tau)
    return home, away


def _challenger_runs(state, home_dist, away_dist, control_m, switch_margin_m, touch_m=None):
    """Which side, if any, is winning a control race this frame.

    The current possessor only needs to stay inside the control radius. Stealing
    the ball is a touch: the challenger must be inside touch_m, not merely the
    closer body. A presser outside that radius does not take it. From dead,
    either side can open inside the wider control radius. Ties keep the last team.
    """
    if touch_m is None:
        touch_m = control_m
    home_close = home_dist <= control_m
    away_close = away_dist <= control_m
    home_touch = home_dist <= touch_m
    away_touch = away_dist <= touch_m

    if state == HOME:
        if away_touch and away_dist + switch_margin_m < home_dist:
            return None, AWAY
        if home_close and (home_dist <= away_dist + switch_margin_m or not away_touch):
            return HOME, None
        return None, None
    if state == AWAY:
        if home_touch and home_dist + switch_margin_m < away_dist:
            return None, HOME
        if away_close and (away_dist <= home_dist + switch_margin_m or not home_touch):
            return AWAY, None
        return None, None

    # Dead: either side can open with a clear lead.
    if home_close and home_dist + switch_margin_m < away_dist:
        return None, HOME
    if away_close and away_dist + switch_margin_m < home_dist:
        return None, AWAY
    if home_close and not away_close:
        return None, HOME
    if away_close and not home_close:
        return None, AWAY
    return None, None


def label_frames(ball_x, ball_y, ball_speed, home_dist, away_dist, half_blocks, pitch,
                 force_dead=None, kickoff_2nd=SECOND_HALF_KICKOFF, control_m=CONTROL_M,
                 switch_frames=SWITCH_FRAMES, switch_margin_m=SWITCH_MARGIN_M,
                 off_margin_m=OFF_MARGIN_M, off_hold=OFF_HOLD,
                 stop_speed=STOP_SPEED, stop_isolation_m=STOP_ISOLATION_M,
                 stop_hold=STOP_HOLD, touch_m=None):
    """H/A/D/X for one video. X everywhere outside the graded halves.

    Each half starts dead, so a label cannot bleed out of halftime. Inside a half
    the machine holds its label until control or a stoppage says otherwise.
    """
    length, width = pitch
    n_frames = len(ball_x)
    labels = np.full(n_frames, "X", dtype="<U1")
    if force_dead is None:
        force_dead = np.zeros(n_frames, dtype=bool)
    finite = np.isfinite(ball_x) & np.isfinite(ball_y)
    speed = np.where(np.isfinite(ball_speed), ball_speed, np.inf)
    nearest = np.minimum(home_dist, away_dist)
    off = np.zeros(n_frames, dtype=bool)
    off[finite] = (
        (ball_x[finite] < -off_margin_m) | (ball_x[finite] > length + off_margin_m) |
        (ball_y[finite] < -off_margin_m) | (ball_y[finite] > width + off_margin_m)
    )
    isolated_stop = finite & (~off) & (speed <= stop_speed) & (nearest >= stop_isolation_m)

    for half, first, last in half_blocks:
        state = DEAD
        home_run = 0
        away_run = 0
        off_run = 0
        stop_run = 0
        start = first
        if half == 2 and first <= kickoff_2nd <= last:
            labels[first:kickoff_2nd] = "D"
            start = kickoff_2nd

        for i in range(start, last + 1):
            if force_dead[i]:
                state = DEAD
                home_run = away_run = off_run = stop_run = 0
                labels[i] = "D"
                continue

            if not finite[i]:
                home_run = away_run = off_run = stop_run = 0
            elif off[i]:
                off_run += 1
                stop_run = 0
                home_run = away_run = 0
                if off_run >= off_hold:
                    state = DEAD
            elif isolated_stop[i]:
                stop_run += 1
                off_run = 0
                home_run = away_run = 0
                if stop_run >= stop_hold:
                    state = DEAD
            else:
                off_run = stop_run = 0
                keeper, challenger = _challenger_runs(
                    state, home_dist[i], away_dist[i], control_m, switch_margin_m,
                    touch_m)
                if keeper == HOME:
                    home_run += 1
                    away_run = 0
                elif keeper == AWAY:
                    away_run += 1
                    home_run = 0
                elif challenger == HOME:
                    home_run += 1
                    away_run = 0
                elif challenger == AWAY:
                    away_run += 1
                    home_run = 0
                else:
                    home_run = away_run = 0

                if home_run >= switch_frames and state != HOME:
                    state = HOME
                elif away_run >= switch_frames and state != AWAY:
                    state = AWAY
            labels[i] = CODE_TO_LABEL[state]
    return labels


def spell_summary(labels, fps):
    """(possession count, mean duration in seconds) over H and A runs."""
    if len(labels) == 0:
        return 0, 0.0
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(labels)]))
    durations = [(end - start) / float(fps)
                 for start, end in zip(starts, ends)
                 if labels[start] in ("H", "A")]
    if not durations:
        return 0, 0.0
    return len(durations), float(np.mean(durations))


def self_check():
    """The machine on a toy pitch, with no match data and no ground truth."""
    length, width = 105.0, 68.0
    n = 220
    ball_x = np.full(n, np.nan)
    ball_y = np.full(n, np.nan)
    ball_speed = np.full(n, np.nan)
    home_dist = np.full(n, np.inf)
    away_dist = np.full(n, np.inf)
    force_dead = np.zeros(n, dtype=bool)
    blocks = [(1, 10, 159), (2, 170, 210)]

    # Half 1: home controls, a gap must not split it, then away takes it clearly.
    ball_x[10:40] = 50
    ball_y[10:40] = 34
    ball_speed[10:40] = 5.0
    home_dist[10:40] = 1.0
    away_dist[10:40] = 8.0
    ball_x[40:55] = np.nan  # 15-frame occlusion
    ball_x[55:90] = 52
    ball_y[55:90] = 34
    ball_speed[55:90] = 5.0
    home_dist[55:90] = 8.0
    away_dist[55:90] = 1.0

    # Equal race while away has it: stay away (no home default from a tie).
    ball_x[70:82] = 52
    ball_y[70:82] = 34
    ball_speed[70:82] = 5.0
    home_dist[70:82] = 1.0
    away_dist[70:82] = 1.0

    # Outside the pitch long enough becomes dead; following gap stays dead.
    ball_x[90:96] = length + 3
    ball_y[90:96] = 34
    ball_speed[90:96] = 2.0
    home_dist[90:96] = 1.0

    # Leave dead with home control, then an isolated stopped ball returns to dead.
    ball_x[100:100 + SWITCH_FRAMES + 5] = 50
    ball_y[100:100 + SWITCH_FRAMES + 5] = 34
    ball_speed[100:100 + SWITCH_FRAMES + 5] = 5.0
    home_dist[100:100 + SWITCH_FRAMES + 5] = 1.0
    away_dist[100:100 + SWITCH_FRAMES + 5] = 8.0
    stop_at = 100 + SWITCH_FRAMES + 5
    ball_x[stop_at:stop_at + STOP_HOLD + 5] = 50
    ball_y[stop_at:stop_at + STOP_HOLD + 5] = 34
    ball_speed[stop_at:stop_at + STOP_HOLD + 5] = 0.2
    home_dist[stop_at:stop_at + STOP_HOLD + 5] = 6.0
    away_dist[stop_at:stop_at + STOP_HOLD + 5] = 6.0

    # Set-piece pad forces D even if someone is close.
    force_dead[150:158] = True
    home_dist[150:158] = 0.5
    away_dist[150:158] = 8.0
    ball_x[150:158] = 50
    ball_y[150:158] = 34
    ball_speed[150:158] = 5.0

    labels = label_frames(
        ball_x, ball_y, ball_speed, home_dist, away_dist, blocks, (length, width),
        force_dead=force_dead, kickoff_2nd=185)

    assert labels[0] == "X" and labels[165] == "X", (labels[0], labels[165])
    assert labels[10] == "D"
    assert labels[10 + SWITCH_FRAMES - 1] == "H"
    assert set(labels[40:55]) == {"H"}, set(labels[40:55])
    assert labels[55 + SWITCH_FRAMES - 1] == "A"
    assert set(labels[70:82]) == {"A"}, set(labels[70:82])  # equal race keeps away
    assert labels[93] == "A"
    assert labels[94] == "D"
    assert labels[96] == "D"
    assert labels[100 + SWITCH_FRAMES - 1] == "H"
    assert labels[stop_at + STOP_HOLD - 1] == "D"
    assert set(labels[150:158]) == {"D"}
    assert set(labels[170:185]) == {"D"}
    assert labels[185] == "D"

    mask, n_events = set_piece_dead_mask(100, [
        {"code": "SET PIECES", "start_frame": 40, "end_frame": 70},
        {"code": "BUILD UP", "start_frame": 10, "end_frame": 20},
    ], pre_frames=10)
    assert n_events == 1
    assert mask[30:40].all() and not mask[40] and not mask[29]

    filled = fill_short_gaps(np.array([0.0, np.nan, np.nan, 6.0, np.nan, np.nan, np.nan, 1.0]), 2)
    assert filled[1] == 2.0 and filled[2] == 4.0, filled
    assert np.isnan(filled[4]), filled

    # A presser outside the touch radius does not steal. A player on the ball does.
    n2 = 80
    px = np.full(n2, 50.0)
    py = np.full(n2, 34.0)
    ps = np.full(n2, 3.0)
    phd = np.full(n2, 1.0)
    pad = np.full(n2, 8.0)
    phd[30:55] = 1.9
    pad[30:55] = 1.6  # closer, but not a touch
    phd[55:80] = 2.5
    pad[55:80] = 0.6
    pressed = label_frames(
        px, py, ps, phd, pad, [(1, 0, n2 - 1)], (length, width),
        kickoff_2nd=10**9, touch_m=1.2)
    assert pressed[SWITCH_FRAMES] == "H"
    assert set(pressed[30:55]) == {"H"}, set(pressed[30:55])
    assert pressed[55 + SWITCH_FRAMES - 1] == "A"

    # Three teammates at 2.2 m outweigh one opponent at 1.5 m. One player at 6 m
    # is still about 6 m: distance decays, it does not get a hard vote.
    mates = _effective_distance(np.array([[2.2, 2.2, 2.2, 20.0]]), 3, 2.0)[0]
    presser = _effective_distance(np.array([[1.5, 20.0, 20.0]]), 3, 2.0)[0]
    assert mates < presser, (mates, presser)
    lone = _effective_distance(np.array([[6.0, 30.0]]), 3, 2.0)[0]
    assert abs(lone - 6.0) < 0.05, lone

    print("self-check ok")


def build(game=GAME, data_dir=None, touch_m=None, knn=False):
    """Labels for the whole tactical video."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    pitch = GAMES[game]["pitch"]
    print("reading {} ...".format(os.path.basename(raw_path)))
    ball_x, ball_y, ball_speed, home_dist, away_dist, observed, filled = read_tracking(
        raw_path, team_of_each_track(meta_path), n_frames, pitch, knn=knn)
    if knn:
        print("  distances are k={}, tau={} m weighted".format(KNN_K, KNN_TAU))
    print("  ball observed on {} frames, short gaps filled {}".format(observed, filled))

    events = load_smart_tagging(game, data_dir)
    force_dead, n_set = set_piece_dead_mask(n_frames, events)
    print("  set-piece dead pads: {} events, {} frames".format(n_set, int(force_dead.sum())))

    finite = np.isfinite(ball_x) & np.isfinite(ball_y)
    length, width = pitch
    outside = finite & (
        (ball_x < -OFF_MARGIN_M) | (ball_x > length + OFF_MARGIN_M) |
        (ball_y < -OFF_MARGIN_M) | (ball_y > width + OFF_MARGIN_M))
    print("  ball x [{:.1f}, {:.1f}]  y [{:.1f}, {:.1f}]  outside {} frames".format(
        np.nanmin(ball_x), np.nanmax(ball_x), np.nanmin(ball_y), np.nanmax(ball_y),
        int(outside.sum())))
    labels = label_frames(
        ball_x, ball_y, ball_speed, home_dist, away_dist, half_video_blocks(game), pitch,
        force_dead=force_dead, touch_m=touch_m)
    return labels


def main():
    parser = argparse.ArgumentParser(description="Write a possession submission from the ATD.")
    parser.add_argument("--self-check", action="store_true",
                        help="run the toy-pitch checks and exit")
    parser.add_argument("--touch", type=float, default=None,
                        help="metres. A challenger must be this close to steal the ball")
    parser.add_argument("--knn", action="store_true",
                        help="weight the 3 nearest players of each team by distance")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.out is None:
        if args.knn:
            name = "possess_v6.csv"
        elif args.touch:
            name = "possess_v4.csv"
        else:
            name = "possess_v2.csv"
        args.out = os.path.join(RESULTS_DIR, name)

    self_check()
    if args.self_check:
        return

    labels = build(touch_m=args.touch, knn=args.knn)
    path = write_submission(args.out, labels)
    counts = {label: int(np.sum(labels == label)) for label in ("H", "A", "D", "X")}
    possessions, mean_s = spell_summary(labels, GAMES[GAME]["fps"])
    print("H {H:>6}  A {A:>6}  D {D:>6}  X {X:>6}".format(**counts))
    print("possessions {}   mean {:.1f} s".format(possessions, mean_s))
    print("-> {}".format(os.path.relpath(path, ROOT)))
    print("\nNow check the file, and only then the public window:")
    print("  python evaluation/validate_submission.py {}".format(os.path.relpath(path, ROOT)))
    print("  python evaluation/run_evaluation.py")


if __name__ == "__main__":
    main()
