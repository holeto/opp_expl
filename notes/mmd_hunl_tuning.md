# Architecture
MMD with moving magnet. No periodic swaps, 
instead polyak averaging at the same (similar) rate as the target network.

# Expected performance
Strong early game, weak on river. Good fit for
MAVS blueprint baseline.

# Training time
After 12 hours on h100 additional gains should be marginal

# Tuning tips
Small updates on the target network
and the magnet (around 5e-5).
Push batch size as far as the GPU allows.
The reported train was something like 8000 batch
of trajectories of size 40, where it was not padded, the environment was reset if the game ended within the 40 steps.
Use cosine decay on learning rate