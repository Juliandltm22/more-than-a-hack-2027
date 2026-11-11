# Possession pipeline

This is the pipeline that writes the submission. It is a state machine, not a trained model. The scored file is `evaluation/results/possess_v2.csv`: 199,319 rows, header `frame,label`, labels `H`, `A`, `D`, and `X`.

## Run it

Python 3.8 or newer. Install `numpy` and `pandas` from `requirements.txt`. Unpack the match so the tactical tracking and Smart Tagging files sit in `data/tactical/`. Do not commit that folder.

```bash
python starter/possess.py
python evaluation/validate_submission.py evaluation/results/possess_v2.csv
python starter/test_pipeline.py
python evaluation/run_evaluation.py
```

`python starter/possess.py` regenerates `possess_v2.csv`. The validator must report 199,319 rows. `starter/test_pipeline.py` fails the file if possessions shatter or if true home frames are labeled dead. The scorer compares every CSV in `evaluation/results/` with the public 5 minutes only.

## What it does

For each frame of the tactical video:

1. Read the ball and the players from the tactical tracking, in metres, with a team on every track slot.
2. If the ball is missing for less than a second, fill the gap in a straight line. A longer gap keeps the current label. A missing ball is not a dead ball.
3. The team within 2 m of the ball is in control. Farther than that, the ball is loose and stays with the last team.
4. The other team takes over only after they are at least 0.5 m closer for 12 frames (about half a second). A tie keeps the last team.
5. The label is `D` when the ball stays off the pitch, when it is nearly stopped and nobody is within 4 m, during the 3 seconds before each Smart Tagging `SET PIECES` start, and on the published frames before the second-half kickoff.
6. Outside the two halves the label is `X`. Each half starts dead, so a label cannot cross halftime.

## What went in

- Tactical tracking: `BAR-ATH_tactical_atd.xml` and `BAR-ATH_tactical_atd.txt`.
- Smart Tagging set pieces, for the 3-second dead pad only. The team tag is not used.

Dropped: both videos, broadcast tracking, broadcast homography, tactical homography, and the other Smart Tagging phases.

No training data. No labels from this match were used to choose the constants.

## Files

| file | role |
|---|---|
| `starter/possess.py` | the state machine; writes `possess_v2.csv` |
| `starter/test_pipeline.py` | rejects a file that fragments possessions or labels home spells dead |
| `TECHNICAL.md` | the short note for the submission form |

The match bundle and `possess_v2.csv` are gitignored. Upload the CSV on the form. This repository only has the code that rebuilds it.
