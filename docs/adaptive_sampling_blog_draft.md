# Uniform → Adaptive Sampling: Choosing Where TinyINR Learns

TinyINR learns from coordinate-value pairs: a pixel’s position and its color. Our next step is to explore **adaptive coordinate sampling**, using recent prediction errors to decide which points appear more often during training.

Uniform sampling gives every position the same chance of being selected. It is a useful baseline, but it does not respond to what the model is still struggling to learn. Can we make better use of a fixed training budget by spending more draws on difficult points?

## Why give every pixel the same chance?

Imagine an image with a smooth background and a small patch of detailed texture. Early in training, the model might struggle with both. Later, it might predict the background accurately while still missing the texture. Uniform sampling continues to select positions with equal probability, regardless of that progress.

Adaptive sampling lets the selection probabilities change as the model learns. A point with a large recent prediction error becomes more likely to appear in a future batch. As its error falls, its priority can fall too.

The detailed patch is just an example. We will use measured prediction errors to set priorities, so any region the model struggles with can receive more samples. Whether this improves learning enough to justify the extra work is what we want to test.

## Three places we sample

Uniform sampling currently appears in three parts of our training pipeline:

| Choice | What it does |
|---|---|
| **Training coordinates** | Chooses a subset of pixel positions and their matching RGB values. |
| **CICFM’s starting noise** | Samples each starting-value component uniformly between 0 and 1 in the random training path. |
| **CICFM’s training time** | Samples a time between 0 and 1, allowing the model to learn velocity prediction at different positions along the interpolation path. |

This milestone changes coordinate selection. Starting noise and training time will remain uniform in random CICFM training.

Our earlier article, [What is Flow Matching?](/post/what-is-flow-matching), introduces the idea using Gaussian starting noise. TinyINR’s current implementation uses a uniform source instead; the source distribution is a modeling choice.

There is also a distinction within coordinate sampling. The existing `randomSubsample` selects without replacement. Our new uniform baseline and adaptive sampler both draw **with replacement**, so a pixel can appear multiple times in one batch. Using the same rule for both makes their comparison more meaningful.

## Turning prediction errors into a feedback loop

The proposed loop starts with equal priorities for all points:

```text
Sample original pixel indices
          ↓
Gather matching positions and colors → embed coordinates → train
                                                           ↓
Update sampling probabilities ← update priorities ← per-point errors
          ↓
Sample the next batch
```

Each priority belongs to an original pixel within its image. Keeping those indices alongside the sampled data lets us connect a training error back to the right point, even when batches change order or contain duplicates.

After each training step, we will use the predictions from its existing forward pass to calculate a separate error for every sampled point. Those errors update the stored priorities, which determine the next draw distribution. A higher priority increases the chance of selection; it never guarantees inclusion.

## What counts as an error?

For ordinary MSE training, the per-point error is the mean squared difference between predicted and target RGB components.

For CICFM, the model predicts a velocity. Its target is the difference between the target value and the starting value, `z1 - z0`. The priority signal is therefore **velocity prediction error**, averaged over channels.

That error depends on the sampled starting value and time as well as the coordinate. One difficult observation does not establish that a pixel is consistently difficult, and a large velocity error is not itself a measurement of reconstructed image quality. This is one reason to smooth the observations before using them as priorities.

## Focusing while keeping every point eligible

Our initial plan mixes two distributions: 10% uniform sampling and 90% sampling based on priorities. These percentages describe the probability mixture, rather than an exact quota within each batch. Every point retains a nonzero chance of being selected, although a finite run may still miss some points.

For a small illustration, suppose four points have priorities `[1, 1, 1, 7]`. With this mixture, their selection probabilities are approximately:

| Point | Uniform | Adaptive |
|---|---:|---:|
| A | 25% | 11.5% |
| B | 25% | 11.5% |
| C | 25% | 11.5% |
| D | 25% | 65.5% |

Point D receives more draws on average, while the others remain eligible.

We will also smooth priority updates with an exponential moving average: 90% of the previous priority and 10% of the newly observed error. Unobserved points keep their previous priorities. When a point appears multiple times in a batch, we average its errors before updating its priority once.

<details>
<summary>The sampling rule</summary>

For an image with `N` points, let `s_i` be point `i`’s stored priority:

```text
q_i = (s_i + epsilon) / sum_j(s_j + epsilon)
p_i = eta/N + (1 - eta)*q_i

s_i ← beta*s_i + (1 - beta)*mean_observed_error_i
```

Here, `p_i` is the probability of selecting point `i` on one draw. Our initial settings are `eta = 0.10`, `beta = 0.90`, and `epsilon = 1e-6`. The small positive epsilon allows normalization even when all priorities are zero. These settings are configurable starting points, not experimentally established optima.

</details>

## Sampling changes what we optimize

Selecting difficult points more frequently gives them more influence on training. Our first version keeps the existing unweighted loss, so its expected coordinate objective becomes a probability-weighted average of errors instead of the uniform average over all pixels.

We call this **uncorrected priority sampling**. The change is deliberate, but it matters when interpreting results. High error is a practical signal to investigate; it does not guarantee that revisiting that point produces the most useful model update. Priorities can also become stale when points go unobserved.

Preserving the original uniform objective would require importance correction: weighting each sampled point’s loss and gradient by `1/(N*p_i)`, using its probability at draw time. That is a possible extension. The first experiment will evaluate the simpler uncorrected approach.

## How we will know whether it helps

We will compare uniform and adaptive sampling using the same image, initial model, batch size, learning rate, training-step count, and noise/time seed schedule.

The main measurements will be full-grid evaluation error against **training steps** and against **elapsed training time**. The second comparison includes the cost of selecting points and updating priorities. Learning in fewer steps is only part of the story if each step becomes more expensive.

For MSE, evaluation will measure RGB error across all original points. For CICFM, it will measure full-grid velocity error using the same fixed set of starting-value and time probes across runs. That evaluates velocity prediction, rather than generation quality.

We will also track point coverage, repeated draws, and sampling overhead. Raw minibatch loss alone would be misleading: an adaptive batch can become harder even while the model’s full-grid performance improves. Broader performance claims will need repeated runs with multiple seeds and consistent Release-build timings.

## Where we are now

The CPU probability sampler, indexed gathering, and uniform sampled training path are implemented. The next step is to connect prediction errors to persistent priority updates and run the comparison.

This milestone establishes a reproducible CPU reference. CUDA sampling, importance correction, and integration with attention remain later work. We have a testable hypothesis about where to spend training effort; the experiments will tell us whether it pays off.

## Further reading

- [Fast Neural Representations for Direct Volume Rendering](https://mediatum.ub.tum.de/doc/1693440/document.pdf), Section 5.4, offers a close spatial example: using prediction error to guide training-position sampling for a neural volume representation.
- [Not All Samples Are Created Equal: Deep Learning with Importance Sampling](https://proceedings.mlr.press/v80/katharopoulos18a.html) explores informative training-example selection and gradient variance. Its sampling method differs from our residual-based priorities.
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) provides a useful precedent for error-based priorities and sampling-bias correction in reinforcement learning.

## A cool related direction: adaptive steps during generation

Adaptivity can also enter after training. [CAT-Flow: Curvature-Adaptive sTeps for Flow Matching](https://arxiv.org/pdf/2609.01746) explores adjusting ODE step sizes while generating a sample with a trained flow-matching model.

The intuition is appealing: take smaller steps where the predicted velocity changes rapidly, and larger steps where the trajectory is easier to follow. Its methods use information from velocity evaluations to adapt those steps without additional training.

This addresses a different choice from our coordinate sampler. TinyINR is exploring **where to train**; CAT-Flow explores **how far to advance during generation**. Its results do not establish a benefit for our training method, but it opens another question for future work: once we can generate samples, could we also make better use of the computation spent following the learned flow?
