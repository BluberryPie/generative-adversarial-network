# Vanilla GAN on MNIST

A from-scratch implementation of the original Goodfellow et al. (2014) GAN: MLP generator and discriminator, trained on MNIST. Built closely enough to the paper to reproduce its actual training pathologies, rather than the easier, more stable "modern standard" GAN recipe (BatchNorm, LeakyReLU-everywhere, Adam-by-default).

## Goal

Vanilla GANs are hard to train. The point of this project isn't sample quality, but to directly observe *why*:

- **Vanishing generator gradients** under the strict minimax loss from the paper's Algorithm 1.
- **Mode collapse**; the generator producing near-identical output regardless of its input noise.
- **Non-convergence / oscillation** between G and D, rather than settling at an equilibrium.

## Model

- **Generator**: MLP mixing ReLU (hidden layers) and Sigmoid (output layer). Noise z is injected only at the bottom (input) layer — never at intermediate layers, even though the paper notes its theoretical framework would technically permit that.
- **Discriminator**: MLP with maxout activations and dropout applied during discriminator training, per the paper. Final layer is a Sigmoid producing a scalar probability.
- **No BatchNorm**: It postdates this paper by about a year (Ioffe & Szegedy, 2015) and is explicitly out of scope for this sub-project.

## Loss: both formulations, as a toggle

- **Strict minimax** (Algorithm 1): G directly minimizes `log(1 - D(G(z)))`.
- **Non-saturating heuristic** (the paper's own recommended practical fix): G instead maximizes `log D(G(z))`.

Why this matters: `log(1-D(G(z)))`'s gradient into D's pre-activation is proportional to `-D(G(z))`, which vanishes as `D(G(z)) → 0`. This is exactly the regime where G's samples are obviously bad and D confidently rejects them. The non-saturating loss's gradient is instead proportional to `-(1-D(G(z)))`, which stays large in that same regime. Both share the same fixed point, but very different gradient behavior early in training.

## Diagnostics

Raw GAN loss curves are famously uninformative; there's no proper convergence objective the way there is in supervised learning. Three things were built to actually observe what's happening:

- **Loss curves** (`loss_D`, `loss_G`). Reference point: at the true equilibrium (`D(x)=0.5` everywhere), losses converge to `log(4)≈1.386` (D) and `±log(2)≈0.693` (G) — **not zero**. Zero is actually the signature of total discriminator victory, not healthy convergence.
- `D(real)`/`D(fake)` mean output, overlaid with G's gradient norm. More directly interpretable than the loss curves — watch whether the two probabilities hover near 0.5 (balanced) or get pinned near 1/0 (D winning), and whether G's gradient norm collapses toward 0 (vanishing gradient) or stays alive.
- **Fixed-noise sample grid, animated over training.** The same batch of `z`, sampled once before training and reused at every snapshot, so the evolution of specific latent codes' outputs can be tracked frame to frame. The only direct window into mode collapse.

## Findings

**Vanishing gradients (minimax, initial hyperparameters).** With the project's initial hyperparameter choices (`latent_dim=64`, a single shared learning rate `1e-3`, `D_hidden_dims=[64, 256]`, `D_hidden_dims=[256, 64]`, `maxout_k=3` — values we picked ourselves, since the paper's text doesn't specify layer widths, maxout piece count, or learning rates for MNIST), D dominates almost immediately under the strict minimax loss: `D(fake)` collapses toward 0, and G's gradient norm collapses in lockstep — the paper's predicted failure mode, observed directly.

![Loss curves — minimax](https://github.com/user-attachments/assets/7497ba3d-bf31-405a-a594-10e9e69a446b)

![D confidence & G gradient norm — minimax, gradient collapsing](https://github.com/user-attachments/assets/939ca149-46db-48aa-9644-dafe48a12466)

**Non-saturating comparison (same initial hyperparameters).** Under the identical setup, G's gradient norm stays large (and keeps growing) in the same D-dominant regime, directly confirming the theoretical fix. But D still wins overall across the run, and the dynamics oscillate — G claws back partial ground, D reasserts itself — rather than converging. This tracks with Algorithm 1's own convergence proof, which requires D to reach its true optimum before every G update; finite-capacity minibatch SGD never actually satisfies that.

![D confidence & G gradient norm — non-saturating, gradient staying alive](https://github.com/user-attachments/assets/e953cf1b-0c1d-4e26-817f-f1db87c97327)

**Rebalancing fixes D-domination — for both loss formulations.** A deliberate rebalancing experiment (separate G/D learning rates, `G_learning_rate=1e-2` / `D_learning_rate=1e-4`, and a much smaller discriminator, `D_hidden_dims=[64]`) pulled `D(real)`/`D(fake)` close to the ideal `0.5`/`0.5` balance under the non-saturating loss. Re-testing the *strict minimax* loss under this same rebalanced config was the more telling check: the near-instant collapse seen above disappeared entirely. Minimax now also oscillates, with `D(real)`/`D(fake)` reaching near `0.5` at points — behavior that, under the original hyperparameters, only the non-saturating loss's stronger gradient could produce. So the rebalancing targets the D-domination/vanishing-gradient axis specifically, and does so regardless of which G loss is used — it isn't a fix that happens to only work for one formulation.

![D confidence & G gradient norm — minimax, rebalanced config, oscillating instead of collapsing](https://github.com/user-attachments/assets/a4ded805-e305-499a-a322-40d305286c53)

**Mode collapse persists regardless.** The fixed-noise grid shows near-identical output across clearly distinct `z` vectors — and this holds across *all four* combinations tried: initial or rebalanced hyperparameters, minimax or non-saturating loss. Even the rebalanced-minimax run above, with its healthy-looking oscillating `D(real)`/`D(fake)` dynamics, still ends in the same collapsed grid by the final frame. The improved D/G balance also isn't stable on its own — it drifted back toward D-dominance over a longer run in one test. Taken together, this suggests vanishing gradients/D-domination and mode collapse are *separable* pathologies here, not one simply cascading from the other: fixing the balance problem measurably changes the training dynamics without fixing mode collapse. Pushed far enough in the other direction, G being trained too aggressively relative to D is plausibly the paper's own named **"Helvetica scenario"** (Section 6) — collapsing too many `z` values to the same `x` — a distinct, third explanation worth keeping in mind alongside plain undertrained diversity.

![Fixed-noise sample grid — mode collapse](https://github.com/user-attachments/assets/a40e97ed-3a06-4d16-8e9d-6a7b713bd753)

## Project structure

- `data.py` — MNIST loading (`[0,1]`-range, flattened, no normalization).
- `model.py` — `Generator`, `Discriminator`, and the custom `MaxOut` module.
- `config.py` — experiment hyperparameters, including the loss-mode toggle.
- `main.py` — training loop (Algorithm 1), diagnostics collection, fixed-noise snapshotting.
- `visualize.py` — loss curve, D-confidence/gradient-norm, and sample-grid-animation plotting.

## Running it

```bash
uv run vanilla_mnist/main.py
```
