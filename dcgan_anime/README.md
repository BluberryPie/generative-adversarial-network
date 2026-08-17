# DCGAN on Anime Faces

A from-scratch implementation of DCGAN(Radford, Metz & Chintala, 2015) trained on the anime face dataset(github.com/bchao1/Anime-Face-Dataset).

## Model

![DCGAN generator architecture, Figure 1 from Radford, Metz & Chintala (2015)](https://github.com/user-attachments/assets/4add3715-083d-43b0-901a-8966d20d264c)

Both networks follow Figure 1 of the DCGAN paper directly. 

The generator: a 100-dim latent `z` is projected into a `4×4×1024` feature map, then four fractionally-strided convolutions upsample it through `8×8×512 → 16×16×256 → 32×32×128 → 64×64×3`, with BatchNorm and ReLU after every layer except the last (which uses Tanh, no BatchNorm).

The discriminator mirrors this in reverse with strided `Conv2d` layers and LeakyReLU (slope `0.2`, one of the few hyperparameters the paper's text states explicitly), skipping BatchNorm on its first (input) layer, and collapsing the final `4×4×1024` feature map to a single scalar followed by Sigmoid.

Note: `kernel_size=4, stride=2, padding=1` (divisible, avoiding checkerboard artifacts) isn't stated explicitly in the paper's text. It comes from the authors' reference implementation. The paper itself only gives the five qualitative guidelines and Figure 1's resulting spatial sizes.

## Training

Optimizer: Adam, `lr=2e-4`, `beta1=0.5`; both stated explicitly in the paper's training-details paragraph, with `beta1` specifically lowered from Adam's default `0.9` because the authors found the default caused training oscillation. Both G and D share this same learning rate and beta.

Loss is the non-saturating BCE formulation only (no minimax-loss toggle).

Each iteration draws one fresh real minibatch and two independently-sampled noise batches (one for the D step, one for the G step) via a manual iterator with `StopIteration` handling, with a single D update and single G update per iteration.

Two concrete deviations from the paper's stated training details, noted honestly rather than glossed over: `batch_size=32` here versus the paper's `128`, and no explicit weight initialization scheme was implemented (the paper specifies drawing all weights from `N(0, 0.02)`; this project used PyTorch's default initialization instead).

## Findings

**Mode collapse, confirmed.** Every image in the fixed-noise grid looked identical early on. A controlled test confirmed real collapse, not undertraining: output diversity (std across a batch of distinct `z`) dropped from `≈0.31` at fresh initialization to `≈0.02` by iteration 25, and never recovered over 1000 iterations.

![Loss curves before mitigation](https://github.com/user-attachments/assets/9ab78004-809a-4cac-a587-5672c250ce59)

**One-sided label smoothing fixed it.** Softening D's real-image target from `1.0` to `0.9` (Salimans et al., 2016) capped D's confidence — the mechanism pinning `D(fake)` near 0. Afterward, `D(fake)` oscillated in a sustained band instead, and diversity returned.

![D(real)/D(fake) oscillating after label smoothing](https://github.com/user-attachments/assets/4b3a79c2-eb89-41bf-af13-d1aae1800843)

**Longer training improved quality, but along one axis.** At 2,000 iterations, diversity was real but just textured noise over a shared blurry silhouette. By 10,000 iterations (~5 epochs), recognizable face structure emerged — eyes, hair, bangs — with real hair-color diversity. Pose and composition stayed fixed across samples; whether that's a model limitation or just an accurate reflection of the dataset's own lack of pose variation is unverified.

![Fixed-noise sample grids along 10,000 iterations](https://github.com/user-attachments/assets/dc3f02a0-b14d-4f4f-bbde-cf6c741f727d)

## Project structure

- `data.py` — `AnimeFaceDataset` (lazy-loaded, RGB-forced JPEGs) and `load_anime`, with the `Resize(64,64) → ToTensor → Normalize(0.5, 0.5)` transform pipeline mapping images to `[-1, 1]`, matching G's Tanh output range.
- `model.py` — `Generator` and `Discriminator`, plus `build_g_conv_layers`/`build_d_conv_layers` helpers factoring out each network's repeating conv-BatchNorm-activation block.
- `config.py` — experiment hyperparameters.
- `main.py` — training loop, diagnostics collection, fixed-noise snapshotting via `sample_grid`.
- `visualize.py` — loss-curve, D-confidence, and sample-grid-animation plotting.

## Running it

```bash
uv run dcgan_anime/main.py
```

## Dataset citation

```bibtex
@online{chao2019/online,
  author       = {Brian Chao},
  title        = {Anime Face Dataset: a collection of high-quality anime faces.},
  date         = {2019-09-16},
  year         = {2019},
  url          = {https://github.com/bchao1/Anime-Face-Dataset}
}
```
