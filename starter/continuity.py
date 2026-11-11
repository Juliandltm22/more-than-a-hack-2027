"""A tracking-only possession state machine with temporal continuity.

It combines nearest-team evidence with a geometric out-of-play signal. A team keeps
possession through loose balls and missing observations; a turnover needs sustained,
clearer control by the other team. A missing ball alone never means dead ball.

The constants below are fixed, interpretable starting heuristics. This program does not
read the public ground truth or score candidate variants.

    python starter/continuity.py
    python evaluation/validate_submission.py evaluation/candidates/continuity_candidate.csv
"""

import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from games import (  # noqa: E402
    GAME, GAMES, anchor_video_frame, half_video_blocks, require, source_frames,
)
from baseline import AWAY, HOME, NOBODY, team_of_each_track, write_submission  # noqa: E402

# Control and turnover evidence, in metres and seconds.
CONTROL_DISTANCE_M = 1.5
TURNOVER_MARGIN_M = 0.35
TURNOVER_SECONDS = 0.5
ACQUIRE_SECONDS = 0.12

# A ball clearly beyond the pitch for this long is out of play.
OUTSIDE_MARGIN_M = 0.25
OUTSIDE_SECONDS = 0.16

DEAD = 2
UNKNOWN = NOBODY
LABELS = {HOME: "H", AWAY: "A", DEAD: "D"}
FIELD_SEP = ":"
GROUP_SEP = ";"
SECOND_HALF_KICKOFF = anchor_video_frame(GAMES[GAME]["kickoff_2nd"]) + 1


def _parse_group(group):
    """Return normalized (x, y, speed), allowing speed to be missing when position exists."""
    try:
        fields = group.split(",", 2)
        x, y = float(fields[0]), float(fields[1])
    except (AttributeError, IndexError, TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    try:
        speed = float(fields[2]) if len(fields) > 2 else np.nan
    except ValueError:
        speed = np.nan
    if not math.isfinite(speed):
        speed = np.nan
    return x, y, speed


def read_tracking(raw_path, team_of_track, n_frames, pitch):
    """Read ball position/speed and nearest-player distance per team, in metres."""
    length, width = pitch
    ball_x = np.full(n_frames, np.nan, dtype=np.float32)
    ball_y = np.full(n_frames, np.nan, dtype=np.float32)
    home_dist = np.full(n_frames, np.inf, dtype=np.float32)
    away_dist = np.full(n_frames, np.inf, dtype=np.float32)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\r\n").split(FIELD_SEP)
            if len(parts) != 3:
                continue
            try:
                frame = int(parts[0]) - 1
            except ValueError:
                continue
            if frame < 0 or frame >= n_frames:
                continue

            ball = _parse_group(parts[2])
            if ball is None:
                continue

            bx, by = ball[0] * length, ball[1] * width
            ball_x[frame], ball_y[frame] = bx, by
            nearest_home = np.inf
            nearest_away = np.inf

            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if slot >= len(team_of_track):
                    break
                player = _parse_group(group)
                if player is None:
                    continue
                dx = player[0] * length - bx
                dy = player[1] * width - by
                distance = math.hypot(dx, dy)
                if team_of_track[slot] == HOME:
                    nearest_home = min(nearest_home, distance)
                else:
                    nearest_away = min(nearest_away, distance)

            home_dist[frame] = nearest_home
            away_dist[frame] = nearest_away

    return ball_x, ball_y, home_dist, away_dist


def _team_candidate(state, home_dist, away_dist):
    """Return a team with plausible control, applying a margin when stealing possession."""
    if state in (DEAD, UNKNOWN):
        if min(home_dist, away_dist) > CONTROL_DISTANCE_M:
            return UNKNOWN
        if abs(home_dist - away_dist) < TURNOVER_MARGIN_M:
            return UNKNOWN
        return HOME if home_dist < away_dist else AWAY

    current_distance = home_dist if state == HOME else away_dist
    other = AWAY if state == HOME else HOME
    other_distance = away_dist if other == AWAY else home_dist
    if (other_distance <= CONTROL_DISTANCE_M and
            other_distance + TURNOVER_MARGIN_M < current_distance):
        return other
    return state if current_distance <= CONTROL_DISTANCE_M else UNKNOWN


def decode(ball_x, ball_y, home_dist, away_dist, half_blocks, pitch, fps):
    """Decode observations into H/A/D/X, resetting state between the two halves."""
    length, width = pitch
    n_frames = len(ball_x)
    labels = ["X"] * n_frames
    outside_frames = max(1, int(math.ceil(fps * OUTSIDE_SECONDS)))
    turnover_frames = max(1, int(math.ceil(fps * TURNOVER_SECONDS)))
    acquire_frames = max(1, int(math.ceil(fps * ACQUIRE_SECONDS)))

    for _half, first, last in half_blocks:
        state = UNKNOWN
        outside_run = 0
        candidate = UNKNOWN
        candidate_start = first
        candidate_run = 0

        start = first
        if _half == 2:
            kickoff = min(max(SECOND_HALF_KICKOFF, first), last + 1)
            labels[first:kickoff] = ["D"] * (kickoff - first)
            start = kickoff

        for frame in range(start, last + 1):
            x, y = float(ball_x[frame]), float(ball_y[frame])
            hdist, adist = float(home_dist[frame]), float(away_dist[frame])
            visible = math.isfinite(x) and math.isfinite(y)
            if visible:
                outside = (
                    x < -OUTSIDE_MARGIN_M or x > length + OUTSIDE_MARGIN_M or
                    y < -OUTSIDE_MARGIN_M or y > width + OUTSIDE_MARGIN_M
                )
                if outside:
                    outside_run += 1
                    candidate = UNKNOWN
                    candidate_run = 0
                    if state in (HOME, AWAY):
                        # Preserve a known team through a short tracking blip. A
                        # confirmed outside run is backfilled with D below.
                        labels[frame] = LABELS[state]
                    if outside_run >= outside_frames:
                        dead_start = frame - outside_run + 1
                        state = DEAD
                        labels[dead_start:frame + 1] = ["D"] * (outside_run)
                    elif state == DEAD:
                        labels[frame] = "D"
                    continue

                outside_run = 0
                # DEAD comes only from sustained outside-ball evidence. A visible
                # in-bounds ball ends that stoppage; reacquire the team normally.
                if state == DEAD:
                    state = UNKNOWN
                    candidate = UNKNOWN
                    candidate_run = 0
            else:
                outside_run = 0
                candidate = UNKNOWN
                candidate_run = 0
                if state == DEAD:
                    labels[frame] = "D"
                elif state in (HOME, AWAY):
                    # Ball observations are missing on about 41% of graded frames.
                    # An occlusion alone is no evidence of a turnover or stoppage.
                    labels[frame] = LABELS[state]
                continue

            observation = _team_candidate(state, hdist, adist)
            required = acquire_frames if state == UNKNOWN else turnover_frames

            if observation == UNKNOWN:
                candidate = UNKNOWN
                candidate_run = 0
                if state in (HOME, AWAY):
                    labels[frame] = LABELS[state]
                continue

            if observation == state:
                candidate = UNKNOWN
                candidate_run = 0
                labels[frame] = LABELS[state]
                continue

            if observation == candidate:
                candidate_run += 1
            else:
                candidate = observation
                candidate_start = frame
                candidate_run = 1

            if candidate_run >= required:
                state = observation
                labels[candidate_start:frame + 1] = [LABELS[state]] * candidate_run
                candidate = UNKNOWN
                candidate_run = 0
            elif state in (HOME, AWAY):
                labels[frame] = LABELS[state]

    return labels


def print_evidence_summary(ball_x, ball_y, home_dist, away_dist, half_blocks, pitch):
    """Print aggregate signal coverage without exposing any frame-level match data."""
    length, width = pitch
    nearest = np.minimum(home_dist, away_dist)
    for half, first, last in half_blocks:
        block = slice(first, last + 1)
        x = ball_x[block]
        y = ball_y[block]
        distance = nearest[block]
        visible = np.isfinite(x) & np.isfinite(y)
        outside = visible & (
            (x < -OUTSIDE_MARGIN_M) | (x > length + OUTSIDE_MARGIN_M) |
            (y < -OUTSIDE_MARGIN_M) | (y > width + OUTSIDE_MARGIN_M)
        )
        controlled = visible & np.isfinite(distance) & (distance <= CONTROL_DISTANCE_M)
        total = last - first + 1
        print(
            "half {} evidence: frames {}  ball {:5.1f}%  outside {:5.1f}%  "
            "controlled {:5.1f}%".format(
                half,
                total,
                100.0 * int(visible.sum()) / total,
                100.0 * int(outside.sum()) / total,
                100.0 * int(controlled.sum()) / total,
            )
        )


def build(data_dir=None):
    """Build a full-length tactical-video submission from the provided tracking."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, data_dir)
    team_of_track = team_of_each_track(meta_path)
    n_frames = source_frames("tactical", GAME)
    pitch = GAMES[GAME]["pitch"]
    evidence = read_tracking(raw_path, team_of_track, n_frames, pitch)
    ball_x, ball_y, home_dist, away_dist = evidence
    blocks = half_video_blocks(GAME)
    print_evidence_summary(
        ball_x, ball_y, home_dist, away_dist, blocks, pitch,
    )
    return decode(
        ball_x, ball_y, home_dist, away_dist,
        blocks, pitch, GAMES[GAME]["fps"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=None,
                        help="match data root (defaults to ./data or MTAH_DATA_DIR)")
    parser.add_argument(
        "--out",
        default=os.path.join(ROOT, "evaluation", "candidates", "continuity_candidate.csv"),
        help="where to write the frame,label CSV",
    )
    args = parser.parse_args()

    labels = build(data_dir=args.data_dir)
    path = write_submission(args.out, labels)
    print("-> {}".format(os.path.relpath(path, ROOT)))
    print("Validate the exact file before submitting:")
    print("  python evaluation/validate_submission.py {}".format(os.path.relpath(path, ROOT)))


if __name__ == "__main__":
    main()
