# The Ising Lab

A companion application to the book *The Discontinuous World: Artificial Intelligence and the
Physics of Phase Change*.

It reproduces, live in the browser and with no backend, the central result of:

> J. Carrasquilla and R. G. Melko, "Machine learning phases of matter",
> *Nature Physics* **13**, 431-434 (2017). <https://doi.org/10.1038/nphys4035>

A 2D Ising model is simulated by Monte Carlo. Configurations are harvested and labelled only as
"below T<sub>c</sub>" or "above T<sub>c</sub>". A small neural network is trained in the browser on
those labels alone. The temperature at which the network's output crosses one half is then read off
as its estimate of the critical temperature, and compared against the exact answer.

The exact answer is known. Onsager solved this model in 1944:

```
sinh(2J / kT_c) = 1     =>     kT_c / J = 2 / ln(1 + sqrt(2)) = 2.269185314213022
```

So every number the application produces is checkable.

---

## What you are looking at

Four panels:

| Panel | What it shows |
| --- | --- |
| **The lattice** | The live spin configuration, rendered with Three.js. Amber is +1, blue is -1. |
| **Thermodynamics** | Measured energy, order parameter, specific heat and susceptibility against temperature, with the exact T<sub>c</sub> drawn on every plot. |
| **The network** | The classifier: which experimental condition is running, the loss curve, and train/validation accuracy. |
| **The verdict** | The network's mean probability of "disordered" against temperature, and where it crosses one half. |

The **temperature slider** is the primary control. It also drives the live simulation, so you can
sit at T<sub>c</sub> and watch the domains grow to the size of the box.

---

## Physics conventions

These are fixed in `src/types.ts` and every module obeys them.

```
H = -J * sum_<ij> s_i s_j        J = 1, no external field
k_B = 1                          temperature is in units of J / k_B
s_i in {+1, -1}                  L x L square lattice, periodic boundaries
```

Energy is realised as `E = -sum_i s_i * (s_right(i) + s_down(i))`, so each site contributes its
right and its down bond. For `L >= 3` that counts every bond of the periodic lattice exactly once.

All extensive quantities are reported **per spin**, so different lattice sizes are directly
comparable.

```
C   = (N / T^2) * (<e^2> - <e>^2)        e = energy per spin
chi = (N / T)   * (<m^2> - <|m|>^2)
U   = 1 - <m^4> / (3 <m^2>^2)            Binder cumulant
```

`<m>` averages toward zero even deep in the ordered phase, because a finite lattice tunnels between
its two ordered states. `<|m|>` is used as the finite-size order parameter for that reason.

### Two update rules, and why the default is Wolff

Both Metropolis and Wolff sample the same Boltzmann distribution. They do not reach it at the same
speed, and that difference is not cosmetic here.

Single-spin Metropolis suffers critical slowing down: its correlation time grows roughly as
`L^2.17`, so a 128 x 128 lattice near T<sub>c</sub> needs of order 10<sup>4</sup> sweeps to forget
where it started. Coarsening out of a random start deep in the ordered phase is slower still.
Measured directly against this sampler, starting hot at `T = 2.0`:

| Run | `<\|m\|>` after 300 sweeps | Equilibrium value |
| --- | --- | --- |
| Metropolis, L = 32 | 0.123 | 0.911 |
| Metropolis, L = 128 | 0.049 | 0.911 |
| Wolff, L = 32 | 0.910 | 0.911 |
| Wolff, L = 128 | 0.910 | 0.911 |

Harvesting from configurations in that state hands the classifier ordered-phase samples that still
look disordered, which drags the estimated critical temperature upward. Wolff cluster updates
remove almost all of the penalty and land on the equilibrium value immediately.

**Wolff is therefore the default.** Metropolis remains selectable, gets twenty times the
equilibration when chosen, and the lattice panel warns you what it is costing. Watching critical
slowing down happen is worth doing once; it is not the setting to quote a number from.

---

## The machine learning

A multilayer perceptron written from scratch over typed arrays. No ML library, no WebGL compute, no
network access.

- One hidden layer of 64 ReLU units, two-unit softmax output over {ordered, disordered}.
- Cross-entropy loss, plain minibatch SGD, batch size 16, 40 epochs.
- Learning rate is `0.5 / sqrt(inputDim)`. The inputs are bare +/-1 spins, so an example's squared
  norm is exactly the input dimension; scaling this way keeps the update to a hidden unit's
  pre-activation independent of lattice size.
- Weights use He initialisation. Every PRNG is seeded, so a run is reproducible.

**Labels are the only supervision.** The network is never shown the temperature of a configuration,
and never shown T<sub>c</sub>.

### Held-out evaluation

The verdict curve is the headline claim, so it is not drawn on data the network trained on. One
configuration in every four at each temperature is withheld from training entirely and used only to
draw that curve. A network with enough capacity can memorise a training set, and under the
shuffled-label control it will try to, which would manufacture a crossing where there is no physics
at all.

### Three experimental conditions

| Condition | What it tests | What should happen |
| --- | --- | --- |
| **Raw spins** | The real experiment. The network sees the bare configuration and must construct an order parameter itself. | Accuracy well above chance, crossing near 2.2692. |
| **Magnetization only** | Control. The network is handed `\|m\|` and nothing else. | Succeeds, and therefore proves nothing about what a network can discover. |
| **Shuffled labels** | Falsification control. Labels are randomly permuted. | Validation accuracy pinned at chance. Any crossing is noise. |

---

## Measured results

From actual runs on one 2026 laptop, raw-spin condition, Wolff, default parameters.

Monte Carlo is stochastic, and so is SGD. A single run is not a result. Here are **four consecutive
runs at L = 32**, reported in the order they happened rather than selected:

| Run | Estimated T<sub>c</sub> | Error vs exact | Validation accuracy |
| --- | --- | --- | --- |
| 1 | 2.2529 | -0.0163 | 99.5 % |
| 2 | 2.2962 | +0.0270 | 98.9 % |
| 3 | 2.3257 | +0.0565 | 99.5 % |
| 4 | 2.2748 | +0.0056 | 97.8 % |

Mean 2.287, against an exact 2.269185. **Agreement at roughly the one percent level, with a
run-to-run spread of about ±0.04.** That spread is the honest precision of this measurement, and it
is set by the temperature grid spacing (0.1) and by how few held-out configurations sit at each
temperature near the boundary, where the network is genuinely uncertain.

Across lattice sizes, one run each:

| L | Configurations | Harvest | Train | Validation accuracy | Estimated T<sub>c</sub> | Error vs exact |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 1240 | 1.5 - 1.8 s | 5.0 s | 97.8 - 99.5 % | 2.287 (mean of 4) | +0.018 |
| 64 | 620 | 4.8 s | 9.8 s | 95.7 % | 2.3176 | +0.0484 |
| 128 | 248 | 7.7 s | 15.5 s | 97.3 % | 2.3372 | +0.0680 |

The error grows with L. That is not finite-size scaling working backwards, it is the harvest
shrinking as the input dimension grows, and it is the trade described under limitations below.

Controls at L = 32:

| Condition | Train accuracy | Validation accuracy | Crossing |
| --- | --- | --- | --- |
| Magnetization only | 98.8 % | 99.5 % | 2.2445 |
| Shuffled labels | 91.1 % | **50.0 %** | 2.1281, spurious |

The shuffled control is behaving exactly as it should: 50.0 % validation accuracy is pure chance,
and the gap against 91.1 % on the training set is the network memorising noise.

Equilibration is not a free parameter. Cutting the L = 32 harvest equilibration from 400 Wolff
sweeps to 60 to save about a second moved the mean estimate from +0.018 to +0.044 and widened the
spread. Under-equilibrated ordered-phase configurations still look disordered, and the estimate
drifts upward. The defaults here are set for accuracy, not speed.

---

## Known limitations

Stated plainly, because this is meant to be a scientific instrument.

1. **The crispest reproduction is at L = 32.** A 128 x 128 configuration is sixteen times longer
   than a 32 x 32 one, but the browser's time budget is fixed, so the harvest has to shrink as L
   grows. Larger lattices give better thermodynamics and a noisier classifier estimate. This is a
   limitation of a browser tab, not of the method.

2. **The shuffled control usually does produce a crossing.** The network panel's expected-outcome
   text says the curve "must never cross". In practice, a held-out curve that is pure noise centred
   on one half will wander across one half somewhere, so a crossing nearly always appears and the
   verdict panel correctly labels it spurious. The diagnostic that actually matters for this control
   is the validation accuracy sitting at chance, not the absence of a crossing.

3. **Metropolis at large L does not equilibrate** within any sweep count this application can spend.
   See above. The UI warns when it is selected.

4. **A crossing near 2.2692 is not understanding.** It shows that a classifier trained on binary
   labels alone locates a phase boundary it was never given. It does not show that the network
   discovered the magnetization, the correlation length, or anything a physicist would call insight.
   The two controls exist to keep that distinction visible.

---

## Running it

Requires Node 18 or newer.

```bash
npm install
npm run dev          # development server
npm run build        # typecheck + production build into dist/
npm run preview      # serve the production build
```

Then open the URL the dev server prints.

There are **no runtime network requests**. No external APIs, fonts, or images. Once the page has
loaded it works offline, and every number is computed in your browser during that session. Reload
and it is all recomputed from the same seeds.

## Running the tests

```bash
npm test             # vitest, single run
npm run test:watch   # vitest, watch mode
npx tsc -b           # typecheck only
```

The test suite is not decoration. Its centrepiece is an **exact-enumeration** check: for 2x2 and 3x3
lattices the partition function is summed over all 2<sup>N</sup> configurations, giving exact values
for `<E>/N`, `<|m|>`, `C`, `chi` and the Binder cumulant, which a long Monte Carlo run must
reproduce. That is what demonstrates the sampler obeys detailed balance rather than merely returning
plausible-looking numbers. The enumeration re-derives the energy from the Hamiltonian independently
of `ising.ts`, so it also checks the bond convention.

On the machine learning side the tests check that the analytic gradients match finite differences,
that the softmax is a probability distribution, that a fixed seed gives a bit-identical run, and
that the falsification control actually falsifies.

If a physics test fails, the physics is wrong. Do not weaken the assertion.

---

## Layout

```
src/
  types.ts                 shared interface contract; every module codes against it
  physics/ising.ts         lattice, Metropolis and Wolff sweeps, observable accumulator
  physics/ising.test.ts    exact enumeration and cross-validation of the two update rules
  workers/mc.worker.ts     owns the only lattice; speaks the WorkerRequest/WorkerResponse protocol
  ml/mlp.ts                the classifier, feature construction, verdict extraction
  ml/mlp.test.ts           gradient checks, determinism, falsification
  components/LatticeView.tsx    Three.js instanced renderer
  components/Chart.tsx          dependency-free SVG line chart
  components/Observables.tsx    thermodynamics panel
  components/NetworkPanel.tsx   training panel
  components/VerdictPanel.tsx   the panel that reproduces the published result
  App.tsx                  shell: worker lifecycle, controls, layout, training scheduler
  main.tsx                 entry point
  styles.css               application chrome
```

The simulation runs in a Web Worker, so the Monte Carlo never blocks the interface. Training runs on
the main thread, one epoch per animation frame, so the page keeps painting and the loss curve
advances in front of you rather than appearing all at once at the end.
