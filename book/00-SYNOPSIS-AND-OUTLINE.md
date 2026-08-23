# The Discontinuous World

## Artificial Intelligence and the Physics of Phase Change

**Status:** DRAFT FOR APPROVAL. Nothing beyond this document has been written.
**Audience:** Graduate students and researchers in physics, chemistry, materials science, and machine learning.
**Length:** Twenty chapters at roughly 3,500 words each, approximately 70,000 words, plus a reference list with URLs and a glossary at the back of the book.

---

## Synopsis

Water heated past its boiling point does not gradually become steam. It boils. The density drops by a factor of more than a thousand across a boundary with no width, energy pours in while the temperature refuses to rise, and a quantity that was a smooth function of temperature a moment earlier acquires a discontinuity. Paul Ehrenfest proposed in 1933 that transitions be classified by which derivative of the thermodynamic potential goes singular, and the category he called first order, the one with latent heat and a jump in entropy and volume, remains the sharpest thing in physics. Nature is full of such jumps. Iron loses its magnetism, helium becomes a superfluid, a supercooled droplet crystallizes in microseconds after resisting for hours, a tokamak plasma that has been stable for seconds collapses in milliseconds, and an ocean circulation that has run for millennia may not.

Machine learning is, in its ordinary form, the opposite kind of object. A neural network is a smooth, differentiable function fitted by gradient descent to interpolate between examples. Pointing a smooth interpolator at a genuine singularity is not an obvious thing to do, and the tension between the two is the subject of this book.

It turns out to work, and the reasons it works are more interesting than the fact that it does. In 2017 Juan Carrasquilla and Roger Melko trained a plain feed-forward network on labelled Monte Carlo spin configurations and found that it located the critical temperature of the two-dimensional Ising model without being told what an order parameter was, then pushed the same architecture into Coulomb phases and the toric code, where no local order parameter exists at all. Evert van Nieuwenburg, Ye-Hua Liu and Sebastian Huber removed the labels entirely by deliberately lying to the network and reading the transition off the shape of its confusion. Lei Wang showed that principal component analysis, a technique older than most of the field, recovers the Ising order parameter from raw configurations. Within three years the question had inverted. The interesting problem was no longer whether a machine could find a phase boundary, but whether the feature it had latched onto was the physics or an artifact, and whether a machine could hand back something a physicist could read.

That question organizes the book. Kenneth Wilson's renormalization group, which won the 1982 Nobel Prize, is a procedure for throwing away detail that does not matter, and the reason universality exists at all is that almost everything about a system is irrelevant near a critical point. Pankaj Mehta and David Schwab proposed that stacked restricted Boltzmann machines implement something like variational renormalization, a claim that has been disputed ever since. Maciej Koch-Janusz and Zohar Ringel rejected that particular reading and built a different one, maximizing real-space mutual information to extract the degrees of freedom the renormalization group would keep, and recovered the Ising correlation-length exponent to within fifteen percent of its exact value. Whether any version of the claim that deep learning is renormalization survives is a live question rather than a settled analogy, and the book treats it as one. Elsewhere the machines are not finding physics but supplying speed. Machine-learned interatomic potentials trained on density functional theory, beginning with Jörg Behler and Michele Parrinello in 2007 and the Gaussian Approximation Potentials of Albert Bartók, Mike Payne, Risi Kondor and Gábor Csányi in 2010, dissolved a tradeoff that had blocked the field for decades: quantum-mechanical accuracy in the few meV per atom that separate competing phases, at the length and time scales nucleation actually requires. Full phase diagrams of water now come out of simulations that would have been unthinkable in 2005.

The applications reach further than condensed matter. Julian Kates-Harbeck, Alexey Svyatkovskiy and William Tang trained recurrent networks to forecast tokamak disruptions, and Jaemin Seo, Egemen Kolemen and colleagues later steered a real DIII-D plasma away from tearing-mode onset with a reinforcement learning agent. Thomas Bury, Chris Bauch and Madhur Anand trained a classifier on roughly half a million simulated bifurcations and applied it to sedimentary anoxia records and paleoclimate transitions. Autonomous beamlines choose their own next measurement, and electron microscopes decide where to look.

The book is equally concerned with what has not worked. A dedicated chapter treats the failures directly, because a field this young accumulates retractions and corrections faster than it accumulates textbooks. Anthony Cheetham and Ram Seshadri examined randomized samples of the GNoME stable-structure database and found little novelty or credibility in them. Alexander Leeman, Leslie Schoop, Robert Palgrave and co-workers found systematic Rietveld refinement errors in the A-Lab results and argued that most of its claimed new compounds were known disordered phases, and Nature published an author correction in 2026 that retitled the paper and withdrew its novelty framing. Sayash Kapoor and Arvind Narayanan documented data leakage across dozens of fields. The AMOC collapse forecasts drew specific, technical rebuttals, and an August 2025 author correction moved the central estimate by eight years after a coding error was found. None of this makes the field fraudulent. It makes it young, and a book that reported only the successes would be advertising rather than scholarship.

The final chapters turn the instrument on itself. When John Hopfield mapped associative memory onto a spin system in 1982, he made statistical mechanics literally applicable to a learning machine, and Daniel Amit, Hanoch Gutfreund and Haim Sompolinsky promptly showed that such a network undergoes a real first-order transition at a critical memory load. Elizabeth Gardner reframed the question as the geometry of the space of solutions. The 2024 Nobel Prize in Physics to Hopfield and Geoffrey Hinton formally recognized that lineage. Modern machine learning has revived every part of it, in the jamming transition that separates under- from over-parametrized networks, in double descent, and in the unresolved argument over whether the emergent abilities of large language models are genuine transitions or, as Rylan Schaeffer, Brando Miranda and Sanmi Koyejo argued, artifacts of discontinuous metrics. Physics lent machine learning its vocabulary for sudden change. The loan is now being called in.

---

## Structure

The book runs in six movements. Four chapters establish the physics, so that later claims about what a network discovered can be measured against what was already known. Four chapters cover learning to see a phase, moving from classification to interpretation. Four cover simulation, where the machine supplies speed rather than insight. Three cover quantum matter, laboratory instruments, and industrial materials discovery. Two cover forecasting an imminent catastrophic jump. Three close the book, one on failure and non-replication and two treating learning itself as a system with critical phenomena.

---

# Part One. The Physics of Discontinuity

### Chapter 1. The Sharpest Thing in Nature

The book opens on the boiling point, because everything difficult about the subject is already present there. Josiah Willard Gibbs supplied the framework of thermodynamic potentials and coexistence in 1878, Paul Ehrenfest supplied the classification in 1933, and Lev Landau supplied the order parameter and the language of symmetry breaking in 1937. The chapter establishes precisely what a first-order transition is, why latent heat implies a discontinuous first derivative, and why modern usage retains Ehrenfest's first-order category while replacing his second order with the word continuous, since real continuous transitions diverge as power laws rather than jumping by a finite amount. It then states the problem the rest of the book answers. A neural network is a smooth differentiable interpolator. A phase transition is a singularity in the thermodynamic limit. Finite systems have no true singularities at all, only rounded crossovers that sharpen with size, which means that every machine learning result in this field is inferring an idealization from data that does not contain it.

Built on Gibbs on heterogeneous equilibrium, Ehrenfest's 1933 Amsterdam classification with Tilman Sauer's 2016 translation and commentary, Landau's 1937 order parameter theory, and the finite-size rounding of transitions.

Leaves open: if no finite simulation contains a genuine singularity, what exactly is a network detecting when it announces a critical temperature?

### Chapter 2. Universality, and the Art of Throwing Things Away

The Ising model is the spine of this chapter and of the book, since it is also the system the companion application runs live. Wilhelm Lenz proposed the model, Ernst Ising solved it in one dimension in 1925 and wrongly generalized his negative conclusion, Rudolf Peierls showed in 1936 that two dimensions must order, and Lars Onsager solved it exactly in 1944, proving that critical exponents are not classical. Edward Guggenheim had already noticed in 1945 that eight different fluids collapse onto one coexistence curve. Benjamin Widom's scaling hypothesis, Leo Kadanoff's block-spin construction and Kenneth Wilson's renormalization group explained why. The chapter closes by making explicit the structural analogy that recurs throughout the book, which is that coarse-graining discards irrelevant detail, and so, in a different sense, does representation learning.

Built on Ising 1925 with the Lenz attribution, Peierls 1936, Onsager 1944 and the exact critical temperature, Yang's 1952 spontaneous magnetization, Guggenheim 1945, Widom 1965 with the correct Rushbrooke attribution, Kadanoff 1966, Wilson 1971, the Wilson and Fisher epsilon expansion of 1972, and the 1982 Nobel Prize awarded to Wilson alone.

Leaves open: if universality means that microscopic detail is irrelevant, what is a machine learning from microscopic configurations?

### Chapter 3. Disorder, Absolute Zero, and Emergence

Not every transition fits the Landau picture, and the exceptions are where machine learning has been most useful, because they are where human intuition about order parameters runs out. Norman Mermin and Herbert Wagner proved that continuous symmetries cannot break in two dimensions at finite temperature, and Vadim Berezinskii, John Kosterlitz and David Thouless found the topological transition that happens instead, recognized by the 2016 Nobel Prize. John Hertz and later Subir Sachdev developed transitions at absolute zero driven by a non-thermal parameter. Samuel Edwards and Philip Anderson, then David Sherrington and Scott Kirkpatrick, then Giorgio Parisi, found that spin glasses require not one order parameter but an infinite hierarchy, work recognized by the 2021 Nobel Prize and directly ancestral to Part Six. Anderson's 1972 essay supplies the framing.

Built on Mermin and Wagner 1966, Berezinskii 1971, Kosterlitz and Thouless 1973 with the 2016 Nobel Prize, Hertz 1976 and Sachdev's monograph, Broadbent and Hammersley on percolation, Edwards and Anderson 1975, Sherrington and Kirkpatrick 1975, Parisi's replica symmetry breaking and the 2021 Nobel Prize, and Anderson's "More Is Different."

Leaves open: Parisi needed an infinite-dimensional order parameter that no one could have guessed. What else are we failing to guess?

### Chapter 4. The Barrier

First-order transitions do not happen when thermodynamics says they should. They wait. This chapter covers the kinetics that the rest of the book depends on and that most treatments of machine learning in physics skip. Max Volmer and Alfred Weber, then Richard Becker and Werner Döring, then David Turnbull and John Fisher built classical nucleation theory around a free energy barrier arising from competition between a favorable volume term and a costly interfacial term. Ostwald's step rule predicts that the phase which forms first is often not the stable one. John Cahn and John Hilliard treated the opposite limit, spinodal decomposition, where no barrier exists. The chapter also introduces the glass and jamming transitions, where a system falls out of equilibrium entirely and whether a true thermodynamic transition underlies the kinetic arrest remains contested. That material is placed here deliberately, because Chapter 20 argues that neural networks jam, and a reader who has never met the physical jamming transition cannot judge whether the analogy is load-bearing. The chapter ends by quantifying the problem that motivates four later chapters, which is that barriers many times the thermal energy make the transition a rare event that brute-force molecular dynamics will essentially never observe.

Built on Volmer and Weber 1926, Becker and Döring, Turnbull and Fisher 1949, Ostwald's step rule, supercooling and metastability, Cahn and Hilliard on spinodal decomposition, and the glass and jamming transitions as the case where equilibrium thermodynamics stops applying.

Leaves open: if the event never happens on a timescale we can simulate, how can any amount of data teach a machine what it looks like?

---

# Part Two. Learning to See a Phase

### Chapter 5. The Machine That Found the Critical Temperature

The founding result, told properly. Juan Carrasquilla and Roger Melko fed labelled Ising configurations to an ordinary feed-forward network, which learned to separate ordered from disordered and, through finite-size scaling of its own output, returned the critical temperature. The genuinely surprising part came next, when the same architecture worked on a Coulomb phase and on the toric code, systems with no local order parameter for the network to have rediscovered. Tomoki Ohtsuki and Tomi Ohtsuki had independently applied deep learning to random two-dimensional electron systems in 2016. The chapter is careful about what these papers did and did not claim, since the secondary literature has garbled several of them.

Built on Carrasquilla and Melko in Nature Physics 2017, the toric code and Coulomb phase results, Ohtsuki and Ohtsuki in J. Phys. Soc. Jpn. 85, 123706, described accurately as comparing network output against previously published exponents rather than estimating exponents itself, and Broecker, Carrasquilla, Melko and Trebst on sign-problematic fermions, correctly described as classifying the sampled Green's function rather than auxiliary-field configurations.

Leaves open: the network needed labels, and labels require knowing the answer. What happens when nobody knows where the boundary is?

### Chapter 6. Learning Without Being Told

Removing the labels. Evert van Nieuwenburg, Ye-Hua Liu and Sebastian Huber built the confusion scheme, which deliberately mislabels data at a proposed boundary and reads the true transition off a characteristic W-shaped performance curve, an idea whose cleverness lies in extracting signal from induced failure. Lei Wang showed that principal component analysis alone recovers the Ising order parameter, and Sebastian Wetzel carried the comparison through autoencoders and variational autoencoders. The chapter also covers diffusion maps for topological order and anomaly detection for phases nobody thought to label.

Built on van Nieuwenburg, Liu and Huber in Nature Physics 2017, Wang in Phys. Rev. B 2016, Wetzel on principal component analysis and variational autoencoders, and Rodriguez-Nieva and Scheurer on unsupervised topological detection.

Leaves open: an unsupervised method that finds a boundary has still not said what changes across it.

### Chapter 7. What Is the Network Actually Looking At?

The skeptical chapter of Part Two, and the one that separates this book from promotional accounts. A classifier that achieves high accuracy may be reading magnetization, or it may be reading total energy, lattice artifacts, or the sampling temperature leaking through the data. Wenjian Hu, Rajiv Singh and Richard Scalettar's critical examination showed that principal component analysis fails outright on the charge correlations of the biquadratic spin-one model and on the vorticity of the XY model. Patrick Huembeli, Alexandre Dauphin, Peter Wittek and Christian Gogolin used an adversarial arrangement of two networks with conflicting objectives to let the machine choose its own features and identify an effective order parameter for the many-body localization transition. The chapter states plainly that in several celebrated cases the feature engineering, not the network, was doing the work.

Built on Hu, Singh and Scalettar in Phys. Rev. E 2017, Huembeli, Dauphin and Wittek on adversarial detection, Huembeli, Dauphin, Wittek and Gogolin on many-body localization, Zhang and Kim's quantum loop topography, and Greitemann, Liu and Pollet's tensorial kernel support vector machines, which emit explicit order-parameter polynomials.

Leaves open: if a support vector machine can hand back a polynomial a physicist can read, why settle for a network that cannot?

### Chapter 8. Machines That Hand Back Physics

The constructive answer to Chapter 7, and the intellectual peak of the methods half of the book. The chapter runs two lineages against each other. The first is the argument about whether deep learning is renormalization. Pankaj Mehta and David Schwab proposed that stacked restricted Boltzmann machines implement variational renormalization; Maciej Koch-Janusz and Zohar Ringel rejected that reading and built an alternative in which maximizing real-space mutual information between a coarse-grained variable and its environment extracts the relevant degrees of freedom, from which they recovered the Ising correlation-length exponent as one point zero plus or minus zero point one five against an exact value of one. Doruk Efe Gökmen, Zohar Ringel, Sebastian Huber and Maciej Koch-Janusz scaled this into RSMI-NE, which reconstructs phase diagrams and identifies symmetry-breaking patterns without prior knowledge, and Robert de Mello Koch and colleagues built observables that distinguish genuine renormalization flow from whatever a trained network is actually doing. The second lineage is symbolic regression, whose output is an equation rather than a prediction, from Michael Schmidt and Hod Lipson recovering conservation laws from motion-tracking data, through Silviu-Marian Udrescu and Max Tegmark's AI Feynman, through Miles Cranmer's PySR and the graph-network recipe of imposing a sparse bottleneck and symbolically regressing the learned messages, to Cristina Cornelio and colleagues at IBM scoring candidates on derivability from background axioms rather than fit alone, which is the difference between a fitted curve and a derived law. The chapter closes on the contrast case, physics-informed neural networks and neural operators applied to Allen-Cahn and Cahn-Hilliard dynamics, which enforce known physics rather than discovering new physics and whose documented failures on stiff, sharp-interface problems are optimization-landscape failures rather than expressivity failures.

Built on Mehta and Schwab on variational renormalization, Koch-Janusz and Ringel in Nature Physics 2018 with the extracted Ising exponent, Gökmen, Ringel, Huber and Koch-Janusz on RSMI-NE, de Mello Koch and colleagues on distinguishing renormalization flow from network behaviour, Schmidt and Lipson in Science 2009, Udrescu and Tegmark in Science Advances 2020 together with AI Feynman 2.0, Cranmer's PySR and the graph-network symbolic regression recipe with Lemos and colleagues recovering the inverse-square law and solar-system masses from thirty years of trajectories, Brunton, Proctor and Kutz on sparse identification of governing equations, Cornelio and colleagues in Nature Communications 2023, Raissi, Perdikaris and Karniadakis on physics-informed networks with Mattey and Ghosh on Allen-Cahn and Cahn-Hilliard and Krishnapriyan and colleagues on why they fail, and the Fourier Neural Operator.

Leaves open: every law recovered here was already known. Whether any version of the claim that deep learning is renormalization survives is still unsettled.

---

# Part Three. Simulating Matter That Jumps

### Chapter 9. Quantum Mechanics Without the Electrons

Why first-order transitions were, for decades, the hardest thing in atomistic simulation. Competing phases differ by a few meV per atom, which demands quantum-mechanical accuracy, while nucleation demands hundreds of thousands of atoms over hundreds of nanoseconds, which forbids it. Jörg Behler and Michele Parrinello decomposed the total energy into atomic contributions expressed through symmetry-invariant descriptors in 2007. Albert Bartók, Mike Payne, Risi Kondor and Gábor Csányi reframed the same problem as kernel regression in 2010. Deep Potential Molecular Dynamics made it scale, and equivariant graph networks made it dramatically more data-efficient.

Built on Behler and Parrinello 2007, Bartók, Payne, Kondor and Csányi 2010 with Kondor correctly placed at Caltech rather than Cambridge, Zhang, Han, Wang, Car and E on Deep Potential Molecular Dynamics, NequIP and Allegro from Batzner, Kozinsky and co-workers, MACE from Batatia, Csányi and co-workers, and MACE-MP-0 as a general-purpose foundation model cited to its published venue rather than the superseded preprint.

Leaves open: a potential fitted to equilibrium structures is being asked to describe a barrier it never saw during training.

### Chapter 10. Melting, Freezing, and the Phase Diagram of Water

The payoff chapter, and the strongest evidence in the book that these methods produce real physics rather than plausible pictures. Machine-learned potentials have produced full ab initio quality phase diagrams of water, homogeneous ice nucleation rates by seeding, the phase diagram and nucleation of gallium including its kinetic preference for the metastable beta phase, silicon crystallization from deep metadynamics, crystallization kinetics of the alloys used in phase-change memory, and evidence bearing on the disputed second critical point of supercooled water.

Built on the deep potential water phase diagram, ice nucleation by seeding, gallium and the Ostwald step rule made quantitative, Niu, Piaggi, Invernizzi and Parrinello on silica crystallization described accurately as to what the simulations did and did not resolve, phase-change memory alloys, and the second critical point controversy.

Leaves open: every one of these results depended on someone choosing the right coordinate to bias along.

### Chapter 11. Crossing the Barrier

Rare-event methods, presented as the machinery without which Chapter 10 is impossible. Alessandro Laio and Michele Parrinello introduced metadynamics in 2002, filling free energy minima with history-dependent bias until the system escapes. The well-tempered variant and the on-the-fly probability enhanced sampling framework followed, all implemented in the open-source PLUMED library. The chapter is careful with attribution here, since the standard secondary account misassigns the introduction of that framework. The alternative family avoids biasing altogether, in that Christoph Dellago, Peter Bolhuis, Félix Csajka and David Chandler's transition path sampling harvests reactive trajectories, needs no prior guess at the mechanism, and defines the reaction coordinate afterward through the committor. The chapter states the field's central failure mode plainly, which is that biasing along a wrong collective variable does not announce itself. It produces a converged-looking free energy surface that is wrong.

Built on Laio and Parrinello 2002, well-tempered metadynamics, Invernizzi and Parrinello's introduction of on-the-fly probability enhanced sampling in J. Phys. Chem. Lett. 2020 with the Invernizzi, Piaggi and Parrinello unification following in Phys. Rev. X 2020, PLUMED, Dellago, Bolhuis, Csajka and Chandler 1998, and the committor as the exact reaction coordinate.

Leaves open: the method fails silently when the coordinate is wrong, and nobody knew how to choose the coordinate.

### Chapter 12. Learning the Reaction Coordinate

The direct answer to Chapter 11, and the densest methods chapter in the book. Time-lagged independent component analysis and Markov state models gave a variational, data-driven definition of slow. VAMPnets collapsed the entire pipeline into a single neural network. Parallel lines learn variables from labelled metastable states, from biased trajectories reweighted back to equilibrium, from information-bottleneck objectives, and from spectral-gap maximization. A separate strand abandons dynamics altogether, training normalizing flows to draw one-shot equilibrium samples. Most recently networks learn the committor itself, closing a loop opened in 1998.

Built on Pérez-Hernández, Paul, Giorgino, De Fabritiis and Noé on time-lagged independent component analysis for molecular kinetics, Mardt, Pasquali, Wu and Noé on VAMPnets, Bonati, Rizzi and Parrinello on Deep-LDA, Bonati, Piccini and Parrinello on Deep-TICA with Piccini correctly placed at Istituto Eulero, Trizio and Parrinello on Deep-TDA, Tiwary's RAVE and SGOOP together with Wang and Tiwary's State Predictive Information Bottleneck, the mlcolvar library, Noé, Olsson, Köhler and Wu on Boltzmann generators, Li, Lin and Ren and later Kang, Trizio and Parrinello on learned committors, and Jung, Covino, Dellago, Bolhuis and Hummer.

Leaves open: every method in this chapter learns from trajectories that already crossed the barrier.

---

# Part Four. Quantum Matter, Instruments, and Industry

### Chapter 13. Neural Wave Functions

Giuseppe Carleo and Matthias Troyer parameterized a many-body wave function with a restricted Boltzmann machine in 2017 and optimized it variationally, sidestepping the exponential wall of Hilbert space. The field split immediately, with lattice spin and Hubbard physics moving through convolutional, autoregressive and transformer architectures on the NetKet substrate, while quantum chemistry applied deep networks directly to continuous-space electronic wave functions. The comparison baseline throughout is Steven White's density matrix renormalization group, still superior in one dimension. The chapter's centerpiece is the 2024 V-score benchmark, the field's first honest scorecard, which identifies where every method including neural quantum states remains far from converged, and it ends on programmable quantum simulators now measuring quantum phase transitions directly.

Built on Carleo and Troyer in Science 2017, Deng, Li and Das Sarma on volume-law entanglement, Nomura, Darmawan, Yamaji and Imada, Torlai and colleagues on tomography, NetKet and NetKet 3, Choo, Neupert and Carleo on the frustrated J1-J2 model, Schmitt and Heyl on dynamics, Hermann, Schätzle and Noé's PauliNet with DeepQMC correctly described as the package containing it rather than its successor, Pfau, Spencer, Matthews and Foulkes on FermiNet, White 1992, the Wu, Rossi, Vicentini and Carleo V-score paper in Science 2024 with its 33-author consortium, and Keesling and colleagues in the Lukin group, whose measured observable was the correlation length rather than defect density.

Leaves open: the benchmark says the hardest models are not solved. It does not say which method will solve them.

### Chapter 14. Instruments That Decide Where to Look

Between roughly 2017 and 2023 the job of deciding what phase a sample is in moved, in several experimental communities simultaneously, from a human reading a curve to a network reading raw detector output, and then from a human choosing the next measurement to an algorithm choosing it. Diffraction came first. Autonomy came second, with Gaussian-process surrogate models steering synchrotron and neutron instruments toward the regions of parameter space where a phase boundary actually lives. A parallel thread runs through electron microscopy, where atomically resolved imaging became a stream of machine-readable phase labels and the microscope itself began choosing its targets. Ultracold atoms complete the picture, with transitions extracted from single-shot momentum images.

Built on Park and Sohn on convolutional networks for powder diffraction, Oviedo and Buonassisi on physics-informed augmentation for thin films, Szymanski and Ceder on probabilistic ensembles for multi-phase mixtures, Maffettone and Olds on uncertainty-aware companion agents at a beamline, Noack with Yager, Fukuto and Sethian on Gaussian-process autonomous experimentation with the gpCAM attribution corrected, Kusne's CAMEO, Kalinin and Ziatdinov on scanning transmission electron microscopy, Rem, Köming, Tarnowski and Weitenberg in Nature Physics 15, 917 to 920, and Bohrdt and Demler on quantum gas microscope snapshots.

Leaves open: an instrument that chooses its own measurements is also choosing what will never be measured.

### Chapter 15. Phase Diagrams at Scale

The industrial chapter, and the one that must be written most carefully. High-throughput density functional theory databases turned crystal energetics into queryable data and made computed convex-hull phase diagrams routine. Surrogate models then replaced the density functional theory itself, and generative models began proposing crystals directly. The chapter's argument is that stability prediction is not discovery, a gap that Christopher Bartel and colleagues demonstrated in 2020 by showing that low formation-energy error does not imply correct stability classification, and which Matbench Discovery later formalized as a benchmark. Both headline 2023 results are presented together with their published criticism, in the same chapter rather than quarantined in a footnote.

Built on the Materials Project, the Open Quantum Materials Database and AFLOW, CALPHAD and its coupling to machine learning, Xie and Grossman's crystal graph convolutional networks, M3GNet and CHGNet, CDVAE and MatterGen, GNoME together with the Cheetham and Seshadri critique regarding novelty, cation ordering and space-group distribution, the A-Lab paper together with its Leeman, Schoop and Palgrave critique and the 2026 Nature author correction that retitled it to inorganic materials and withdrew the novelty framing, Bartel and colleagues 2020, and Matbench Discovery.

Leaves open: if the benchmark cannot tell a discovery from a rediscovery, what has been discovered?

---

# Part Five. Forecasting the Jump

### Chapter 16. Milliseconds: Disruption in Fusion Plasmas

Magnetically confined plasmas are the best-instrumented laboratory for discontinuous transitions that anyone has built. The chapter opens on the L-mode to H-mode transition, a genuine bifurcation in which edge turbulent transport collapses above a power threshold, theorized as a thermal confinement bifurcation and later reframed through self-regulating shear flow and zonal-flow predator-prey dynamics. It then turns to the disruption, an essentially irreversible collapse of plasma current and thermal energy that ends a discharge in milliseconds. Three levels of machine involvement follow, in prediction, in control, and in differentiable simulation. The chapter ends on the field's unsolved problem, which is that cross-machine generalization degrades sharply on unseen devices, precisely the regime ITER will occupy on its first day.

Built on Shaing, Crume and Houlberg 1990 and Hinton 1991 on bifurcation models of the L to H transition with the priority question stated accurately, Diamond 1994 and Kim and Diamond 2003, Schmitz 2012 on direct observation at DIII-D, Kates-Harbeck, Svyatkovskiy and Tang in Nature 2019, Rea, Montes and Granetz on the random-forest lineage, Vega, Murari, Dormido-Canto, Rattá and Gelfusa in Nature Physics 2022 characterized accurately as to article type, Degrave, Felici and Riedmiller in Nature 2022 on learned magnetic control at TCV, Seo and Kolemen in Nature 2024 on avoiding tearing onset at DIII-D, TORAX, DisruptionBench, and de Vries and colleagues on ITER mitigation triggering requirements.

Leaves open: every predictor in this chapter was trained on machines that are not ITER.

### Chapter 17. Centuries: Tipping Points and Early Warning

The same forecasting problem across fourteen orders of magnitude in time, and the most contested material in the book. As a system approaches a fold bifurcation its dominant eigenvalue approaches zero, recovery from perturbation slows, and fluctuations show rising variance and rising lag-one autocorrelation. Marten Scheffer, Stephen Carpenter, Vasilis Dakos and Timothy Lenton formalized this into a generic early-warning framework. Since 2021 the field has taken a machine learning turn, with classifiers trained on hundreds of thousands of simulated bifurcations. The chapter treats the Atlantic Meridional Overturning Circulation at length, and treats the rebuttals at equal length, because this is where the temptation to overclaim has been strongest and the corrections most instructive.

Built on Scheffer and colleagues in Nature 2009 and Dakos 2008, the standardized statistical toolbox, Bury, Bauch and Anand in PNAS 2021 trained on roughly 500,000 simulated series generated by continuation software, Dablander and Bury on preprocessing with the Lowess and Gaussian filters correctly distinguished, Boers 2021 and Ditlevsen and Ditlevsen 2023, Chen and Tung's Matters Arising correctly dated to 2024 on coverage artifacts and reversed proxy signs, Ben-Yami and colleagues on gap-filling and extrapolation, the August 2025 author correction that moved the tipping estimate by eight years after a Strang-splitting coding error, the Global Tipping Points Report, and the distinction between bifurcation-induced, noise-induced and rate-induced tipping.

Leaves open: a method calibrated on simulated bifurcations was applied to a system that may not be undergoing one.

---

# Part Six. The Machine as the System

### Chapter 18. What Did Not Replicate

A dedicated accounting, placed where it can be read against everything preceding it. Sayash Kapoor and Arvind Narayanan documented data leakage as a reproducibility crisis spanning dozens of fields and proposed reporting standards in response. Materials machine learning has its own version, in which benchmark performance rises while discovery does not. Superconductivity supplies the sharpest cautionary tales, in the Ranga Dias retractions and the 2023 LK-99 episode, both of which were claims about phase transitions and both of which collapsed under replication. The chapter's argument is not that the field is unsound. It is that a smooth interpolator asked to extrapolate into a phase it has never seen is being asked to do the one thing it is structurally worst at, and that most documented failures reduce to some version of that.

Built on Kapoor and Narayanan in Patterns 2023 together with the REFORMS checklist, the Dias retractions and the University of Rochester investigations, LK-99 and its resolution, benchmark overfitting and distribution shift, and the GNoME and A-Lab critiques revisited in a methodological rather than a domain frame.

Leaves open: Chapter 15 said stability is not discovery. This chapter suggests the problem is more general.

### Chapter 19. The Statistical Mechanics of Learning

Turning the instrument on itself, with a lineage most machine learning practitioners have never read. John Hopfield mapped associative memory onto a spin system in 1982, making statistical mechanics literally rather than metaphorically applicable to a learning machine, and Daniel Amit, Hanoch Gutfreund and Haim Sompolinsky used replica theory to show that such a network undergoes a genuine first-order transition at a critical memory load of roughly 0.138 patterns per neuron, above which retrieval collapses into a spin glass state. Elizabeth Gardner then moved the question from a fixed learning rule to the geometry of the space of solutions, computing the volume of weight configurations satisfying a set of constraints and, with Bernard Derrida, the optimal capacity of two patterns per weight, more than an order of magnitude above the Hebbian value. That calculation remains the template for nearly every capacity result since, and Gardner did the work before she was thirty and before her death in 1988. Through the late 1980s and 1990s the framework extended to generalization, where learning curves were shown to jump discontinuously to perfect generalization when weights are discrete. The chapter is careful with the word proof, since the replica method with a replica-symmetric ansatz is a non-rigorous calculation rather than a theorem, a distinction the secondary literature routinely loses.

Built on Hopfield 1982, Ackley, Hinton and Sejnowski on the Boltzmann machine and its learning rule from clamped and free-running correlations, Amit, Gutfreund and Sompolinsky on the capacity transition at roughly 0.138 patterns per neuron, Gardner on the space of interactions with Gardner and Derrida on optimal capacity, Seung, Sompolinsky and Tishby together with Györgyi on first-order transitions in learning curves, the 2024 Nobel Prize in Physics to Hopfield and Hinton, and Zdeborová and Krzakala on the thresholds separating impossible, hard and easy inference.

Leaves open: discontinuous jumps to perfect generalization were derived from theory decades before anyone gave the phenomenon a name.

### Chapter 20. Grokking, Emergence, and What the Word Transition Is Doing

The closing chapter, and the one that must resist its own thesis hardest. Modern deep learning has revived every element of Chapter 19, in the jamming transition separating under- from over-parametrized networks, in double descent, in grokking, where a network memorizes for thousands of steps and then abruptly generalizes, and in the claim that large language models exhibit emergent abilities appearing suddenly at scale. Twice over, the abruptness has turned out to be partly in the observer. Neel Nanda, Lawrence Chan and colleagues reverse-engineered grokking and found three continuous phases underneath an apparently discontinuous curve, and Rylan Schaeffer, Brando Miranda and Sanmi Koyejo showed that many reported emergent abilities dissolve into smooth curves when a discontinuous metric such as exact-match accuracy is replaced by a continuous one, which puts the discontinuity in the ruler rather than the system. The chapter takes both counterarguments as central rather than mentioning them in passing. The book ends by asking what work the word transition is actually doing in each of its uses across the preceding nineteen chapters, and where the analogy is load-bearing rather than decorative.

Built on Geiger, Spigler, Wyart and colleagues on jamming in deep networks read against the physical jamming transition introduced in Chapter 4, Belkin, Hsu, Ma and Mandal on double descent, Power, Burda and colleagues on grokking together with Nanda, Chan and colleagues on the three continuous phases beneath it, Wei and colleagues on emergent abilities, Schaeffer, Miranda and Koyejo at NeurIPS 2023, and the finite-size argument from Chapter 1 applied to model scale.

Leaves open: deliberately unresolved. The reader is left with the question the book cannot answer, which is what an order parameter for a transformer would even be, and therefore whether these systems have a phase diagram at all.

---

# Back Matter

### References

A single consolidated reference list at the back of the book, ordered by chapter and then alphabetically, with a resolvable URL for every entry. Entries are generated directly from the verified research corpus rather than assembled by hand at the end, so that no citation can drift from the version that passed fact-checking. Every entry carries a DOI link, an arXiv abstract link, or an official institutional page. Preprint and journal years are given separately wherever they differ.

### Glossary

A single glossary at the back of the book, covering the physics vocabulary a machine learning reader will not have and the machine learning vocabulary a physicist will not have. On the physics side it covers order parameter, latent heat, critical exponent, universality class, correlation length, renormalization group, coarse-graining, metastability, supercooling, nucleation barrier, spinodal decomposition, convex hull, replica symmetry breaking, tearing mode, pedestal, and critical slowing down. On the machine learning side it covers collective variable, committor, restricted Boltzmann machine, variational autoencoder, normalizing flow, equivariance, graph neural network, information bottleneck, double descent, grokking, and data leakage.

---

## Notes for Approval

Three points need your explicit decision.

The closing arc runs to two chapters rather than three. Chapters 19 and 20 cover the material you approved. If you want a third, I would split Chapter 20 into one chapter on jamming, double descent and algorithmic thresholds, and one on grokking and the emergence debate. That requires giving up a chapter elsewhere, and the cheapest source would be merging Chapters 16 and 17 into a single forecasting chapter. I do not recommend that, because both domains have enough verified material to stand alone, but the choice is yours.

Chapters 15 and 18 both handle the GNoME and A-Lab criticism. This is deliberate rather than duplicative, since Chapter 15 treats it as a materials-science question about whether the compounds are new and Chapter 18 treats it as a methodology question about what benchmarks measure. If you would rather it appeared once, Chapter 18 is the one to keep.

The proposed title is *The Discontinuous World: Artificial Intelligence and the Physics of Phase Change*. Alternatives worth considering are *The Sharpest Thing in Nature* and *Learning the Jump*.
