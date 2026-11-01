# Building an RL environment that chooses which Wi-Fi samples to label

*Draft for author review. The implementation and this draft were AI-assisted.
Nothing has been published.*

A Wi-Fi activity classifier can work differently when the receiver changes.
One way to adapt it is to label a few recordings from the target receiver.
That leaves a practical question: **which recordings should we label first?**

I built a small Gymnasium environment around that question. The policy gets a
fixed labeling budget and chooses examples one at a time. A fixed downstream
classifier learns from the requested labels. At the end, the environment
rewards the resulting accuracy improvement on separate recording events.

The first learned policy did improve over using source labels alone, but it
performed worse than random selection and two simple heuristics. That result
is the starting point for this prototype.

## What the policy does

An episode begins with a classifier trained on 480 labeled source recordings
and a query pool of 24 target recordings. Each recording is a 128 × 56 CSI
amplitude window. The policy sees 27 summaries of each query window, the
classifier's eight activity scores, and any labels it has already requested.

It gets 16 actions. Each action picks an unqueried recording. The environment
reveals the recorded activity—such as jumping or squatting—and updates the
classifier. The next decision can therefore depend on what earlier requests
revealed.

The classifier stays fixed as an algorithm: weighted RBF kernel ridge on
one-hot activity labels. The policy controls the labeled subset. Query-pool
normalization stays fixed throughout each episode. Features summarize overall
level, motion energy, and the motion envelope's temporal shape without using
subcarrier identity. This invariance is useful to test; it does not establish
that receiver effects disappear.

## Where reward comes from

The environment has another 24 recordings from the target receiver, with
activity labels that the policy does not see. They are separate physical
recording events from the query pool.

Every intermediate reward is zero. After the sixteenth request:

```text
reward = final reward-set accuracy − source-only reward-set accuracy
```

For example, improving accuracy from 25% to 50% gives reward `0.25`. An
unhelpful sequence can produce a negative reward. Every strategy gets the
same number of labels, so this version does not need a separate label-cost
penalty or a decision about when to stop.

No optimal query sequence is needed to score a policy. The answer key is the
dataset's recorded activities; reward is a direct comparison between the
classifier's predictions and those labels. Tests check that changing hidden
answers cannot affect policy inputs before they are revealed, and that reward
labels cannot alter preterminal observations or rewards.

## Keeping synchronized recordings together

The recordings come from the public [wise-har dataset](https://doi.org/10.4121/48e1331b-91df-4e1a-906b-b0a0e9d16c74.v3).
Different receivers observed the same physical events. Randomly splitting
receiver rows could therefore put one event in both training and evaluation.

Instead, I split by recording event across every receiver. Policy training
uses repetition 20 as its query pool and 21 for reward, on receivers A–E.
Evaluation uses repetitions 23 and 24, on A–G. Repetition 22 is unused, and
source examples come from repetitions 0–19. Receivers F/G are excluded from
policy training.

The loader checks the fixture and event-manifest hashes before reconstructing
labels. This standalone project carries its own source and calibration fixtures; it
never opens the earlier classification task's hidden test data or requires its
repository at runtime.

These precautions prevent the specified partitions from sharing physical
events. The release was already inspected during development, so this is a
development evaluation rather than a prospective holdout. The data also
contains one participant in one office. It cannot establish transfer to new
rooms or people.

## The first experiment

I trained one small policy for 400 episodes using terminal-return REINFORCE
with a learned value baseline. The network shares an encoder across candidate
recordings and pools context over the set, so reordering rows reorders its
action scores. The frozen policy then selects its largest legal action score.

All methods received the same 16-label budget. Results average seven receiver
episodes over five pool-shuffle seeds:

| Query strategy | Final accuracy | Gain over source only |
| --- | ---: | ---: |
| Source only | 14.88% | — |
| Random | 40.83% | +25.95 percentage points |
| Smallest top-two score margin | 42.26% | +27.38 points |
| Farthest-first feature diversity | 42.86% | +27.98 points |
| Learned policy, first run | 33.33% | +18.45 points |

There are only 24 unique physical reward events, observed across seven
receivers. The seeds repeat these recordings. They are not independent new
experiments, and this table is not evidence that one RL algorithm is generally
better or worse than active-learning heuristics. There was one training seed
and no evaluation-driven hyperparameter search.

The useful result so far is a working, testable environment with a meaningful
baseline gap. Cached kernel updates match full classifier refits to numerical
precision, which makes short CPU training runs inexpensive.

## How this could fit a capstone

The engineering question is straightforward: can a CSI system adapt with fewer
human labels by choosing its calibration examples carefully?

This environment supplies the offline experiment. A physical project could
add CSI acquisition and an activity-classification pipeline, then use the
environment's query interface as a calibration step. Random and uncertainty
selection provide comparisons even if the learned policy remains weaker.

The next learning experiment should use a validation protocol fixed before
tuning, compare multiple training seeds, and examine which examples each
strategy selects. New-room claims would require recordings from those rooms.
The current recordings already support implementing and debugging the
label-selection loop.

## Running a terminal agent through Harbor

The same environment is also packaged as an interactive Harbor task. A
terminal agent uses `csi-calibration state` and `csi-calibration query INDEX`
to inspect the pool and request labels. A private service enforces the query
budget; a separate verifier replays the service's saved action history and
computes the accuracy gain. The agent does not submit a replacement classifier
or its own score.

A scripted Harbor check on one receiver-F episode completed 16 queries and
improved accuracy from 12.5% to 58.3%, giving reward 0.4583. That check used
the margin heuristic and no language-model calls. It validates the interactive
packaging, separate from the seven-receiver learned-policy experiment above.
Two subsequent Qwen3-8B attempts did not complete the 16-query task; neither
was a training run. Their evidence is linked from the project README. Setup
and Harbor evidence are in
[HARBOR.md](HARBOR.md).

Code, setup, commands, and raw results are linked from the [project README](README.md).
