# The Ising Lab

## Companion application specification, for approval

**Repository:** github.com/alessoh/phase-change
**Deployment target:** Vercel
**Status:** DRAFT FOR APPROVAL. No code has been written.

---

## What it is

A browser application that reproduces, live and from scratch, the founding result of machine learning applied to phase transitions. Juan Carrasquilla and Roger Melko showed in 2017 that an ordinary feed-forward neural network trained on labelled Monte Carlo spin configurations learns to separate the ordered from the disordered phase of the two-dimensional Ising model, and that finite-size scaling of the network's own output recovers the critical temperature. The application does the entire thing in the browser, with no server and no precomputed results. It runs the Monte Carlo simulation, generates the training data, trains the network, and extracts the critical temperature while the reader watches.

This is the right choice for a companion to the book because it is the one result everything else in Part Two descends from, because the exact answer is known analytically from Lars Onsager's 1944 solution, and because a reader can therefore check the machine against ground truth rather than taking the demonstration on faith. The application is not an illustration of the book's argument. It is an instance of it.

---

## The four panels

**The lattice.** A Three.js rendering of an Ising lattice of selectable size, evolving under Metropolis or Wolff cluster dynamics at a temperature the reader controls with a single slider. Spins are drawn as an instanced mesh so that a 128 by 128 lattice stays at full frame rate. The visual point is that the reader drags the temperature down through the critical value and watches domains coarsen from noise into order, seeing the correlation length diverge rather than being told that it does. Wolff dynamics matter here rather than being an optional refinement, because Metropolis critical slowing down makes the lattice freeze visually near the transition, which is itself worth showing and worth being able to switch off.

**The observables.** Live magnetization, energy, specific heat and magnetic susceptibility plotted against temperature as the reader sweeps, with the exact Onsager critical temperature marked. The susceptibility peak sharpening and drifting toward the exact value as lattice size increases is the reader's first encounter with finite-size scaling, and it sets up the argument from Chapter 1 that a finite system has no true singularity.

**The network.** A small feed-forward classifier trained in the browser on configurations labelled only by whether they were sampled above or below the critical temperature. Training is visible, with a loss curve and a live output layer. The reader can choose to train on raw spins or on the absolute magnetization, which is the single most instructive control in the application, because it demonstrates directly what Chapter 7 argues, that a classifier may be succeeding for a reason that has nothing to do with the physics the experimenter intended.

**The verdict.** The network's output averaged over configurations at each temperature, crossing one half at the estimated critical temperature, plotted against the exact Onsager value. This is the panel that reproduces the published result, and it shows the estimate improving with lattice size.

---

## What makes it defensible rather than decorative

Every number in the application is checkable. The exact critical temperature follows from Onsager's solution as two over the natural logarithm of one plus the square root of two, approximately 2.269, and it is displayed as the target throughout. The application ships with a test suite that verifies the Monte Carlo sampler against known exact results, including the energy and magnetization of small lattices computed by exhaustive enumeration, so that the physics engine is validated rather than merely plausible.

The application also ships with its own falsification. A control lets the reader train the classifier on deliberately shuffled labels, and the resulting failure to find any transition is the honest counterpart to the successful run. A reader who has seen both understands the result in a way that a reader who has seen only the success does not.

---

## Technical shape

A single-page React application built with Vite and deployed to Vercel as a static site with no backend, since everything runs client-side. Three.js renders the lattice through instanced meshes. The Monte Carlo sweep runs in a Web Worker so that the interface stays responsive, and lattice updates are transferred without copying. The neural network is small enough to train in the browser in seconds rather than minutes, which is the constraint that sets the lattice sizes and sample counts. No external API keys, no external data sources, and no network requests after load, so the application cannot break because a third-party service changed.

---

## What I want you to decide

Whether the scope above is right. The four panels are the minimum that reproduces the result honestly, and I would rather build those four to an extremely high standard than add a fifth.

Whether the deliberate-failure controls stay. Training on shuffled labels and training on magnetization alone are the two features that make this a scientific instrument rather than a demonstration, and they are also the two features that make it look less impressive at a glance. My recommendation is that they stay and are given equal visual weight.

Whether you want the application to carry visible attribution to Carrasquilla and Melko on the interface itself. My recommendation is yes, with a direct link to the paper.
