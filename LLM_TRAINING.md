# LLM training and contribution milestones

The target is a reproducible LLM reinforcement-learning example that could be
useful to OpenEnv maintainers. Nothing has been submitted or accepted.
Use OpenEnv's existing Harbor integration for training work. Keep the current
Harbor runner, private service, separate grader, and Verifiers adapter.

## Current evidence

- The Gymnasium environment and small Torch-policy trainer work. The first
  learned policy trails random and heuristic selection.
- Scripted Harbor, OpenEnv, and Verifiers lifecycle checks work. They contain
  no model tokens and cannot be used as evidence of LLM training.
- Two Qwen3-8B inference attempts were incomplete. Their native API traces
  contain no sampled token IDs.
- Compact CLI observations, explicit completion instructions, a CPU size
  audit, and fresh scripted Harbor checks are implemented. No LLM has yet
  validated the updated interface.

## Next gates, in order

1. **Complete a real episode.** Freeze the model revision, chat template,
   task bundle, and limits. Require exactly 16 distinct accepted actions and
   an independently replayed verifier reward. Report context use, latency,
   and cost. Lifecycle `ok=true` alone is insufficient.
2. **Validate training capture through the native OpenEnv path.** Preserve
   engine-sampled token IDs, behavior log probabilities, loss masks, and the
   correspondence between turns, requests, and the verified episode reward.
   Do not reconstruct training tokens by retokenizing saved text. The
   observation-size audit is a display diagnostic, not training capture.
3. **Prove optimizer and recovery behavior.** Run the existing framework's
   trainer with actual task rewards. Record finite losses, gradients, changed
   weights, policy synchronization, and checkpoint/resume equivalence. Save
   optimizer, scheduler, RNG, and training-progress state as well as weights.
   Artificial nonzero advantages establish an optimizer diagnostic only.
4. **Measure the trained policy.** Compare the same frozen base model, trained
   checkpoint, random, margin, and diversity baselines under the same 16-query
   budget. Keep training on A–E repetitions 20/21 and freeze evaluation on
   repetitions 23/24. Report incomplete episodes as failures, not dropped
   examples. Log completion rate separately from accuracy gain.
5. **Propose a scoped upstream contribution.** Present the measured example
   and any reusable fixes to maintainers before preparing a large PR. A new
   generic Harbor adapter would duplicate existing support. Maintainer review
   determines whether an example, regression fix, or separate project fits.

Every experiment must have a new output directory, pinned configuration,
source hashes, commands, and a candid statement of what passed and failed.
Keep previous runs intact. A positive RL improvement is a measured outcome,
not a prerequisite to reporting an honest experiment.

## Limits on claims

The data is public development data from one participant and one office.
Evaluation has 24 unique physical reward events observed across receivers;
shuffle seeds reuse those events. Repeated training can memorize this finite
task. Successful training would demonstrate this pipeline, not general
active-learning superiority or transfer to new people or rooms.

No GPU provisioning, model trials, training, external messages, publication,
or submission is part of the local interface check. A further model run needs
its own explicit authorization under the project's original no-trials limit.
