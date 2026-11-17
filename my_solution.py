import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from starter.baseline import build, write_submission  # noqa: E402

# Graded halves, in video frames (DATA.md): (first frame, last frame), inclusive.
FIRST_HALF = (11185, 90651)
SECOND_HALF = (110808, 190197)
SECOND_HALF_KICKOFF = 113742  # frames before this in the 2nd half are all D (FAQ.md)

# How many observed frames in a row the other team must be nearest before we
# switch. 25 frames = 1 second. Chosen by football reasoning, NOT tuned on the
# public ground truth (RULES.md forbids that).
N_SWITCH = 75  # 3 seconds

OUT_PATH = os.path.join(ROOT, "evaluation", "results", "my_solution_v1.csv")


def sticky_labels(codes, start, end, n_switch):
    """Return an array of 'H'/'A' (or 'X' if nothing is ever seen) for frames start..end.

    codes: per-frame team of the nearest player: 0 = home, 1 = away, -1 = ball missing.
    """
    length = end - start + 1
    out = np.full(length, "X", dtype="<U1")
    current = None   # team currently in possession (0 or 1)
    streak = 0       # consecutive observed frames where the OTHER team is nearest

    for i in range(length):
        c = int(codes[start + i])
        if c != -1:                      # ball was seen on this frame
            if current is None:
                current = c              # first sighting of the half
            elif c == current:
                streak = 0
            else:
                streak += 1
                if streak >= n_switch:   # other team has been nearest long enough
                    current = c
                    streak = 0
        # missing ball (c == -1): change nothing, keep the current team
        if current is not None:
            out[i] = "H" if current == 0 else "A"

    # Frames before the very first sighting stay 'X'; back-fill them with the first label.
    seen = np.where(out != "X")[0]
    if len(seen) > 0:
        out[: seen[0]] = out[seen[0]]
    return out


def main():
    codes = build()                      # reads data/tactical/... (takes a minute or two)
    n_frames = len(codes)
    labels = np.full(n_frames, "X", dtype="<U1")   # X everywhere to start

    for start, end in (FIRST_HALF, SECOND_HALF):
        labels[start:end + 1] = sticky_labels(codes, start, end, N_SWITCH)

    # Second half is graded from 110808 but kicks off at 113742: those frames are dead ball.
    labels[SECOND_HALF[0]:SECOND_HALF_KICKOFF] = "D"

    path = write_submission(OUT_PATH, labels)
    counts = {k: int(np.sum(labels == k)) for k in "HADX"}
    changes = int(np.sum(labels[1:] != labels[:-1]))
    print("wrote", os.path.relpath(path, ROOT))
    print("label counts:", counts)
    print("number of label changes (lower = fewer tiny possessions):", changes)
    print("\nNext:")
    print("  python evaluation/validate_submission.py", os.path.relpath(path, ROOT))
    print("  python evaluation/run_evaluation.py")


if __name__ == "__main__":
    main()