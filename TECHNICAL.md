# Possession submission

Rename this file to `<team_name>.md` before upload. The scored file is `possess_v2.csv` (199,319 rows, `frame,label`, labels `H` `A` `D` `X`).

## Approach

A state machine reads the tactical tracking and writes one label per video frame. It is not a trained model. The published baseline gives the ball to the nearest player and calls the frame dead whenever the ball is missing. Most missing-ball frames are occlusions during live play, so that rule cuts one possession into many. This machine keeps the last team through a loose ball or a gap, and changes label only on evidence.

There is no training data. No labels from this match were used to fit constants. The public 5-minute ground truth was used only to check that the file is valid and that possessions are not shattered.

## Assets

Used:

- Tactical automatic tracking (`BAR-ATH_tactical_atd.xml` and `BAR-ATH_tactical_atd.txt`): ball and player positions, in metres, and which team each track slot belongs to.
- Tactical Smart Tagging, set pieces only. The 3 seconds before each `SET PIECES` start are labeled dead. The team tag is not used. Throw-ins are `Team = N/A`, and the intervals are phases, not possessions.

Dropped:

- Both videos.
- Broadcast tracking and broadcast homography. That feed is a different timeline.
- Tactical homography. The tracking is already in pitch coordinates.
- The other Smart Tagging phases. Copied as labels, they disagree with the nearest player too often.

## Possession rules

- A player within 2 m is controlling the ball. Farther than that, the ball is loose and stays with the last team.
- The other team takes the ball only if they are at least 0.5 m closer and hold that for 12 frames (about half a second). A tie keeps the last team. It does not default to home.
- A missing ball does not become dead. Short gaps, up to 1 second, are filled in a straight line. Longer gaps keep the current label.
- The ball is dead when it stays outside the pitch, when it is nearly stopped and nobody is within 4 m, during the 3 seconds before a set piece, and on the published pre-kickoff frames of the second half (110808–113741).
- Outside the two graded halves the label is `X`. Each half starts dead, so a label cannot cross halftime.

## Limitations

On the public window the file has 18 possessions against 17, mean length 14.8 s against 14.2 s. Nine spells still miss an IoU of 0.5. In several of those the nearest tracked player is the other team for the whole spell. Following that player more loosely, or voting the three nearest, added spells and scored worse. Dead-ball recall is about half of the true dead frames. Raising it by marking long set-piece openings dead labeled live home possessions as dead.

The public window has 17 possessions. It is not a ranking of these variants.

## Reproduction

Python 3.8 or newer, with `numpy` and `pandas` (`requirements.txt`). Unpack the match bundle so `data/tactical/` contains the tactical tracking and Smart Tagging files. Do not commit that bundle.

```bash
python starter/possess.py
python evaluation/validate_submission.py evaluation/results/possess_v2.csv
python starter/test_pipeline.py
python evaluation/run_evaluation.py
```

`python starter/possess.py` regenerates `evaluation/results/possess_v2.csv`. The validator must report 199,319 rows. `test_pipeline.py` checks that the file has not started fragmenting possessions or labeling home spells dead. The scorer prints the public 5-minute board only.
