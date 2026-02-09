# Status Report: Slither.io Gen2 Bot Training

## Analysis Results
- **Date:** 2026-02-09
- **Total Episodes:** 7,555
- **Average Reward (Last 1000):** ~488.51
- **Average Steps (Last 1000):** ~57.16 (Very Short)
- **Reward Trend (Slope):** -0.0739 (Regressing)
- **Steps Trend (Slope):** -0.0064 (Regressing)

## Problem Identification
The bot has converged on a local optimum of **"Suicide Eating"**.
- It collects high rewards from food (+10 per pellet).
- It dies very quickly (avg 57 steps) because the death penalty (-30 to -50) is smaller than the reward for eating just 3-5 pellets.
- The "Explorer" style currently active has `survival: 0.0`, meaning the bot has *zero* incentive to stay alive unless it's actively eating.

## Proposed Solution (Implemented)
To break this cycle and force the bot to learn survival skills:

1.  **Increase Death Penalties:** Making death significantly more painful (-200 for Wall, -100 for Snake) than a few food pellets.
2.  **Add Survival Reward:** Giving a small +0.05 reward per step simply for existing, incentivizing longer episodes.
3.  **Reduce Food Reward:** Lowering food reward from +10.0 to +5.0 to make survival relatively more important.
4.  **Accelerate Epsilon Decay:** The bot was still exploring 50% of the time after 16,000 episodes. We are speeding up the decay (300k -> 100k steps) to force it to exploit its knowledge sooner.
5.  **Increase Wall Penalty:** Added a proximity penalty to discourage hugging walls.

## Next Steps
- Restart training or continue from checkpoint. Ideally, the new rewards will cause a temporary drop in total reward (as it stops suicide-eating) followed by a rise as it learns to survive longer.
