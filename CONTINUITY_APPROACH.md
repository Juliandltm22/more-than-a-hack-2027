# Temporal-continuity possession candidate

## Summary

This candidate turns the supplied tactical tracking into one `H`, `A`, `D`, or `X`
label for every tactical-video frame. It starts from nearest-team evidence but models
possession as a persistent state: brief ball occlusions, loose balls, and isolated
tracking changes do not automatically create a new possession.

The method is deterministic and has no training phase. It does not read either the
public or private ground truth when generating a submission.

## Assets used

- Tactical ATD metadata, to map track slots to the home or away team.
- Tactical ATD tracking, for ball and player positions.
- Published pitch dimensions, half boundaries, and second-half kickoff alignment from
  the starter repository.

The tactical and broadcast videos, homographies, broadcast tracking, and Smart Tagging
are not used by this version.

## State machine

Player and ball coordinates are converted from normalized pitch coordinates to metres.
For every visible ball observation, the program measures the nearest home and away player.

- A team is considered in control when its nearest player is within 1.5 m of the ball.
- Initial control requires 0.12 seconds of consistent evidence.
- A turnover requires the other team to be at least 0.35 m closer and to hold that
  advantage for 0.5 seconds.
- Missing ball observations retain an established team state. Missing data alone does
  not imply a stoppage or turnover.
- A ball at least 0.25 m beyond the pitch for 0.16 seconds confirms `D`. Shorter outside
  runs retain the existing team and are overwritten with `D` only if confirmed.
- The first visible in-bounds ball ends an outside-ball stoppage, after which team control
  must be acquired normally.
- Published graded frames before the second-half kickoff are `D`.
- Frames outside the two graded halves are `X`. Uncertain frames immediately after a
  stoppage remain `X` until control is established.

These constants are fixed football and tracking heuristics. They were not selected by a
search over the public five-minute labels.

## Reproduction

Place the licensed tactical tracking files at:

```text
data/tactical/BAR-ATH_tactical_atd.xml
data/tactical/BAR-ATH_tactical_atd.txt
```

Then run:

```bash
python3 -m pip install -r requirements.txt
python3 starter/continuity.py
python3 evaluation/validate_submission.py evaluation/candidates/continuity_candidate.csv
```

The candidate file is generated under `evaluation/candidates/` and is ignored by Git.

## Verification policy

The public five-minute ground truth was used after the method was frozen to check frame
alignment, output behavior, and gross possession fragmentation. It was not used as
training data or as a parameter-selection loop. The final frozen check reported 16
predicted possessions against 17 public possessions, with spellF1 0.3879. That sample is
too small to establish a full-match improvement over the published baseline.

## Limitations

- Dead ball is detected only from sustained outside-pitch tracking and the published
  second-half kickoff interval. In-pitch free kicks and other stoppages are often missed.
- Only the nearest player for each team is considered. Challenges, aerial balls, keeper
  control, and tracker identity changes can produce misleading distances.
- The ball is absent on about 41% of graded frames. Continuity handles these gaps but can
  carry the wrong team through a real transition that occurs entirely inside one.
- The public check covers only 17 possessions, so its ranking is noisy and should not be
  treated as evidence of full-match performance.

