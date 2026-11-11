"""Guards for the public window. Not a tuning loop.

possess_v2 misses 9 of 17 possessions at IoU 0.5. Reading them showed three causes:

  - the nearest tracked player is the other team for the whole spell
  - the right team, but the spell starts or ends far enough that IoU falls under 0.5
  - the ball is on the pitch but several metres from every player, so the machine
    stays dead or keeps the previous team

Giving that distant player the ball (possess_v5) did not recover the misses. It
raised the possession count from 18 to 26. Marking long set-piece openings dead
(possess_v3) labeled live home possessions as D.

This test fails those two regressions. It does not demand that all 17 spells match.
The public window is a check that the file is still the same kind of pipeline,
not a target to climb.

    python starter/test_pipeline.py
    python starter/test_pipeline.py evaluation/results/possess_v3.csv
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

GT_PATH = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
DEFAULT_PRED = os.path.join(ROOT, "evaluation", "results", "possess_v2.csv")

# possess_v2 sits at 18 possessions and fragmentation 1.05. v5 and the raw
# baseline sit at 1.43 and above. v3 is the file that called true home frames dead.
MAX_FRAG = 1.25
MAX_SPELLS = 22
MIN_SPELLS = 14


def check(pred_path):
    """Return a list of failure strings. Empty means the file still looks like v2."""
    gt = pm.label_codes(pm.read_label_csv(GT_PATH))
    pred = pm.label_codes(pm.read_label_csv(pred_path))
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pred, runs)
    _, _, frag = pm.fragmentation(gt, filled, runs)
    n_pred = pm.possession_stats(filled, runs)["n_spells"]
    home_called_dead = int(np.sum((gt == pm.H) & (pred == pm.D)))

    failures = []
    if frag > MAX_FRAG:
        failures.append("fragmentation {:.2f} is above {:.2f}".format(frag, MAX_FRAG))
    if not MIN_SPELLS <= n_pred <= MAX_SPELLS:
        failures.append("possession count {} is outside {}-{}".format(
            n_pred, MIN_SPELLS, MAX_SPELLS))
    if home_called_dead:
        failures.append("{} true home frames were labeled dead".format(home_called_dead))
    return failures


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PRED
    if not os.path.isfile(path):
        raise SystemExit("missing submission: {}".format(path))
    failures = check(path)
    if failures:
        raise SystemExit("\n".join(failures))
    print("ok  {}".format(os.path.relpath(path, ROOT)))


if __name__ == "__main__":
    main()
