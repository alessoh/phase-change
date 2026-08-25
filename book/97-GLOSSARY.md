# Glossary

This book is addressed to two readerships that do not share a vocabulary. A physicist may not know what a normalizing flow is, and a machine learning researcher may not know what an order parameter is. The glossary runs in both directions. Terms are defined as they are used in this book, and where a term carries a different meaning in another field, that is noted.

## The physics of phase transitions

**Binder cumulant.** The quantity $U = 1 - \langle m^4\rangle / (3\langle m^2\rangle^2)$, constructed so that its value at the critical point is independent of system size to leading order. Curves computed for several lattice sizes therefore cross at the critical temperature, which makes it one of the most reliable ways to locate a transition from finite simulations.

**Coarse-graining.** Replacing a group of microscopic degrees of freedom by a single effective variable, such as replacing a block of spins by their majority. Repeated coarse-graining is the operational content of the renormalization group, and its resemblance to representation learning is one of the recurring themes of this book.

**Continuous transition.** A transition at which the first derivatives of the free energy are continuous but higher derivatives diverge, producing power laws rather than jumps. Modern usage prefers this term to Ehrenfest's "second order", because real continuous transitions diverge as power laws rather than showing the finite jumps Ehrenfest assumed.

**Convex hull.** In materials science, the lower boundary of formation energy plotted against composition. A compound lying on the hull is thermodynamically stable against decomposition into any combination of other phases; one lying above it is not. The vertical distance above the hull, the energy above hull, is the standard machine learning target for stability prediction.

**Correlation length.** The distance over which fluctuations remain statistically related, written $\xi$. It diverges at a continuous transition, which is why the system becomes scale invariant there and why microscopic detail stops mattering.

**Critical exponent.** The power governing how a quantity diverges or vanishes near a continuous transition, such as $\beta$ for the order parameter or $\nu$ for the correlation length. Exponents are shared across systems in the same universality class.

**Critical slowing down.** The lengthening of the time a system takes to recover from a perturbation as it approaches a bifurcation, because the dominant eigenvalue of the linearized dynamics approaches zero. It produces rising variance and rising lag-one autocorrelation, which together form the basis of early warning signals.

**Ehrenfest classification.** The 1933 scheme that classifies a transition by the order of the lowest derivative of the thermodynamic potential that is discontinuous. Its first-order category survives in modern use; its higher-order categories do not.

**Finite-size scaling.** The systematic study of how a transition's apparent location and sharpness depend on system size. A finite system has no true singularity, only a rounded crossover, and finite-size scaling is how the infinite-system behaviour is extracted from finite data. Every machine learning result in this book depends on it.

**First-order transition.** A transition with a discontinuity in the first derivative of the free energy, and therefore with latent heat and a jump in entropy and volume. Boiling and freezing are the familiar cases. These transitions proceed by nucleation and can be delayed past their equilibrium point.

**Ginzburg criterion.** The self-consistency condition that determines when mean-field theory is reliable, by comparing the size of fluctuations to the size of the order parameter itself. It identifies the upper critical dimension, which for the Ising universality class is four.

**Kibble-Zurek mechanism.** The prediction that a system swept through a continuous transition at finite rate cannot stay adiabatic near the critical point, and freezes in a density of topological defects set by a power of the sweep rate. It links cosmology to condensed matter and is now tested directly in programmable quantum simulators.

**Kosterlitz-Thouless transition.** A transition in two dimensions driven by the unbinding of topological defects rather than by the appearance of conventional long-range order, which the Mermin-Wagner theorem forbids. It has no local order parameter in the Landau sense.

**Latent heat.** Energy absorbed or released at a first-order transition without any change in temperature. It is the direct experimental signature of a discontinuity in the first derivative of the free energy.

**Metastability.** A state that is locally stable but not the global free energy minimum, so that it persists until a sufficiently large fluctuation carries the system over the barrier. Supercooled water is the standard example.

**Nucleation barrier.** The free energy cost of forming a small region of a new phase, arising from competition between a favourable volume term and a costly interfacial term. Because the barrier is typically many times the thermal energy, nucleation is a rare event that ordinary molecular dynamics essentially never observes.

**Order parameter.** A quantity that is zero in the symmetric phase and nonzero in the broken-symmetry phase, introduced by Landau. Magnetization plays this role for a ferromagnet. Much of this book concerns systems where no such quantity is obvious, or where the machine is asked to find one.

**Ostwald's step rule.** The observation that the phase which forms first is often not the thermodynamically stable one but a metastable phase that is kinetically easier to nucleate.

**Pedestal.** The steep edge transport barrier characteristic of the high-confinement mode in a tokamak, whose formation marks the L to H transition.

**Replica symmetry breaking.** The structure Giorgio Parisi found in the mean-field spin glass, in which the order parameter is not a number but a function encoding a hierarchy of states. It is the canonical example of a system whose order could not have been guessed.

**Renormalization group.** The systematic study of how a system's effective description changes under coarse-graining. Critical points appear as fixed points of the resulting flow, and critical exponents as eigenvalues of the flow linearized about them, which explains universality.

**Spinodal decomposition.** Phase separation in a region where the free energy is locally unstable, so that no nucleation barrier exists and separation begins immediately and everywhere. It is the kinetic opposite of nucleation.

**Supercooling.** Cooling a liquid below its freezing point without crystallization, which is possible precisely because a first-order transition requires nucleation.

**Tearing mode.** A magnetic instability in a tokamak plasma that reconnects field lines and forms magnetic islands, degrading confinement and frequently initiating a disruption.

**Universality class.** The set of systems sharing the same critical exponents, determined only by dimensionality, symmetry and interaction range, and not by microscopic detail. It is why eight different fluids collapse onto one coexistence curve.

**Yang-Lee zeros.** The zeros of the partition function in the complex fugacity plane. A finite system has none on the positive real axis, so its free energy is analytic and no transition occurs; a transition appears in the thermodynamic limit exactly where the zeros pinch the real axis. This is the precise sense in which a singularity requires an infinite system.

## Machine learning

**Collective variable.** A low-dimensional function of atomic coordinates chosen to capture the slow progress of a transition, used to bias or analyse a simulation. Choosing a poor one does not announce itself; it yields a converged-looking free energy surface that is wrong, which is the problem that learned collective variables address.

**Committor.** The probability that a trajectory started from a given configuration reaches the product state before returning to the reactant state. It is the exact reaction coordinate in the sense that it captures everything about a configuration's position along the transition, and it can now be approximated by a neural network.

**Data leakage.** Any route by which information from the evaluation set influences training, producing performance estimates that do not survive genuine deployment. It is the single most common methodological failure in machine learning applied to science.

**Double descent.** The observation that test error, after rising to a peak at the interpolation threshold where model capacity just suffices to fit the training data, falls again as capacity grows further, contradicting the classical bias-variance picture.

**Equivariance.** The property that transforming a model's input produces a correspondingly transformed output, for instance that rotating a molecule rotates the predicted forces. Building it into the architecture rather than learning it from data dramatically improves data efficiency for interatomic potentials.

**Graph neural network.** A network operating on nodes and edges, passing messages between neighbours. It is the natural architecture for atoms and bonds, and underlies most modern machine-learned interatomic potentials and crystal property predictors.

**Grokking.** The phenomenon in which a network trained on a small algorithmic task memorizes the training set with chance-level validation accuracy for a long period, then abruptly generalizes. Mechanistic analysis has since found continuous structure beneath the apparent jump.

**Information bottleneck.** An objective that seeks a representation retaining as much information as possible about a target while discarding as much as possible about the input. Several methods for learning collective variables are built on it.

**Machine-learned interatomic potential.** A model trained to reproduce quantum-mechanical energies and forces at a small fraction of the cost, allowing simulations at the length and time scales that nucleation requires while retaining the accuracy needed to distinguish competing phases.

**Neural network quantum state.** A many-body wave function parameterized by a neural network and optimized variationally, which sidesteps the exponential growth of Hilbert space by never representing the full state vector.

**Normalizing flow.** An invertible network that transforms a simple distribution into a complex one with a tractable change of variables, allowing direct sampling. Applied to statistical mechanics, it can generate equilibrium configurations without simulating dynamics at all.

**Restricted Boltzmann machine.** A two-layer stochastic network with no connections within a layer, historically important both as a generative model and as the first ansatz used for neural network quantum states.

**Variational autoencoder.** A generative model that learns a probabilistic latent representation by encoding inputs to a distribution and decoding samples from it. Applied to spin configurations, its latent variable sometimes recovers the order parameter without supervision.

## Terms that mean different things in the two fields

**Relevant and irrelevant.** In the renormalization group these are technical terms describing whether a perturbation grows or shrinks under coarse-graining, and they classify directions in coupling space. They do not mean that a piece of data is informative or uninformative. Confusing the two senses is a genuine and easy error, and this book flags it where it arises.

**Emergence.** In physics, the appearance of behaviour at one scale that is not manifest in the description at the scale below, in the sense of Philip Anderson's essay. In machine learning it usually refers to a capability appearing abruptly with model scale, a usage that carries a contested empirical claim rather than an established one.

**Critical.** In statistical mechanics this refers specifically to a continuous transition, where the correlation length diverges. In casual usage, including some machine learning writing, it often means merely important or abrupt.
