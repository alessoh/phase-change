# Verification corrections ledger

## [CORRECTED] collective-variables

**Claim as researched:** Deep-TICA, Luigi Bonati, GiovanniMaria Piccini, Michele Parrinello, 2021, PNAS 118(44), e2113533118, arXiv:2107.03943 — with affiliation 'GiovanniMaria Piccini: Pacific Northwest National Laboratory, Richland, and Istituto Eulero, Universita della Svizzera italiana, Lugano'

**Correction:** The citation itself (authors, order, year, venue, volume, issue, article number, arXiv ID, DOI) is correct, but the affiliation is wrong. Pacific Northwest National Laboratory does not appear anywhere in the paper. The affiliations as printed in PNAS 118(44), e2113533118 are: Luigi Bonati — Department of Physics, Eidgenoessische Technische Hochschule (ETH) Zuerich, 8092 Zuerich, Switzerland, and Atomistic Simulations, Italian Institute of Technology, 16163 Genova, Italy; GiovanniMaria Piccini — Istituto Eulero, Universita della Svizzera Italiana, 6900 Lugano, Switzerland (sole affiliation, no PNNL, no present-address footnote); Michele Parrinello — Atomistic Simulations, Italian Institute of Technology, 16163 Genova, Italy. Verified against the publisher-deposited record and the PMC full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC8612227/ ; article URL https://doi.org/10.1073/pnas.2113533118

---

## [CORRECTED] collective-variables

**Claim as researched:** SGOOP, Tiwary & Berne 2016, PNAS 113(11), 2839-2844: 'maximizing the spectral gap of the associated transfer operator ... using a maximum-caliber estimate of the dynamics built from short biased trajectories'

**Correction:** Citation is fully correct (Pratyush Tiwary and B. J. Berne, 'Spectral gap optimization of order parameters for sampling complex molecular systems', Proc. Natl. Acad. Sci. U.S.A. 113(11), 2839-2844, 2016). The method description is garbled: the Maximum Caliber (maximum path entropy) model is not built from biased trajectories. Accurate replacement text: 'SGOOP (Spectral Gap Optimization of Order Parameters), introduced by Pratyush Tiwary and B. J. Berne in 2016, selects a reaction coordinate by maximizing the spectral gap — the separation between slow and fast eigenvalues — of a transition matrix inferred through a Maximum Caliber (maximum path entropy) model. That model combines a stationary probability density along the trial coordinate, obtained from a biased simulation such as metadynamics, with dynamical constraints estimated from short unbiased trajectories.' URL: https://doi.org/10.1073/pnas.1600917113

---

## [CORRECTED] collective-variables

**Claim as researched:** Li, Lin & Ren 2019, neural committor from the variational (Dirichlet-form) backward Kolmogorov equation, J. Chem. Phys. 151(5), 054112: 'with adaptive sampling supplying training points in the transition region'

**Correction:** Citation, authors, order, year, venue and DOI are all correct (Qianxiao Li, Bo Lin, Weiqing Ren, 'Computing committor functions for the study of rare events using deep learning', J. Chem. Phys. 151(5), 054112, 2019). The variational/Dirichlet-form framing is correct, but the paper does not use 'adaptive sampling'; it uses two explicit data-generation schemes. Accurate replacement text: 'Qianxiao Li, Bo Lin and Weiqing Ren showed in 2019 that the committor function can be computed by training a deep neural network on the variational (Dirichlet-form) formulation of the backward Kolmogorov equation, with the boundary conditions built directly into the network's functional form rather than imposed by a penalty term, and with training points in the transition region generated either by sampling at an elevated temperature and correcting through a likelihood ratio, or by metadynamics that fills the metastable wells.' URL: https://doi.org/10.1063/1.5110439

---

## [CORRECTED] collective-variables

**Claim as researched:** Bonati & Parrinello 2018, Phys. Rev. Lett. 121, 265701: neural-network potential plus metadynamics simulating liquid silicon and its crystal nucleation, 'machine-learned potentials and machine-assisted enhanced sampling ... jointly reach a first-order phase transition'

**Correction:** Citation is fully correct (Luigi Bonati and Michele Parrinello, 'Silicon Liquid Structure and Crystal Nucleation from Ab Initio Deep Metadynamics', Phys. Rev. Lett. 121, 265701, 2018). Two overstatements: the collective variable was a hand-designed physical order parameter based on the Debye structure factor, not a machine-learned one, so 'machine-assisted enhanced sampling' misattributes the CV; and the work characterized the early stages of nucleation rather than simulating the full first-order transition. Accurate replacement text: 'Luigi Bonati and Michele Parrinello in 2018 trained a deep neural-network interatomic potential on DFT reference energies computed with the SCAN functional and combined it with metadynamics biased along a collective variable derived from the Debye structure factor. This recovered the free energy surface of silicon with DFT accuracy, gave thermodynamic properties near the melting point in good agreement with experiment, and revealed features of the nucleation mechanism in the early stages of crystallization — an early demonstration that a machine-learned potential can carry enhanced sampling to a first-order phase transition that ab initio molecular dynamics alone cannot reach.' URL: https://doi.org/10.1103/PhysRevLett.121.265701

---

## [CORRECTED] critique-limits

**Claim as researched:** Roberts et al., Nature Machine Intelligence 3, 199–217 (2021): screened 2,212 studies, included 61 after quality screening, concluded none of the models are of potential clinical use.

**Correction:** The published version says 62, not 61. Accurate text: 'Their search identified 2,212 studies, of which 415 were included after initial screening and, after quality screening, 62 studies were included in the systematic review; the review found that none of the models identified are of potential clinical use due to methodological flaws and/or underlying biases.' The figure 61 comes from the earlier arXiv preprint (arXiv:2008.06388), not the Nature Machine Intelligence paper. URL is correct: https://doi.org/10.1038/s42256-021-00307-0

---

## [CORRECTED] critique-limits

**Claim as researched:** Palgrave quotes on Rietveld refinement quality and 'two thirds' of predictions; Ceder's December 2023 response, in Chemistry World.

**Correction:** Substance verified verbatim, but tighten the first quote and add the byline/date. Accurate text: Chemistry World, 'New analysis raises doubts over autonomous lab's materials "discoveries"', 16 January 2024. Palgrave: '[The Rietveld refinement] was very bad, very beginner, completely novice human level – that led to them misidentifying things in some cases.' And: 'What they didn't realise is that two thirds of their entire predictions were just ordered versions of compounds that were already known to be disordered.' Ceder, writing on LinkedIn in December 2023: 'We have no doubt that a human can perform a higher-quality refinement on these samples. However, it was our objective to show what an autonomous laboratory can achieve, not to demonstrate what the best (or average) human can do externally to the A-Lab.' URL correct: https://www.chemistryworld.com/news/new-analysis-raises-doubts-over-autonomous-labs-materials-discoveries/4018791.article

---

## [CORRECTED] critique-limits

**Claim as researched:** GNoME: Merchant, Batzner, Schoenholz, Aykol, Cheon, Cubuk, Nature 624, 80–85 (2023), 2.2 million crystal structures, 381,000 newly discovered stable materials. DOI 10.1038/s41586-023-06018-3.

**Correction:** THE DOI IS WRONG AND POINTS TO AN UNRELATED PAPER. 10.1038/s41586-023-06018-3 resolves to Helson, Zwettler, Mivehvar, Colella, Roux, Konishi, Ritsch & Brantut, 'Density-wave ordering in a unitary Fermi gas with photon-mediated interactions', Nature 618, 716–720 (2023) — a cold-atoms paper with no connection to materials discovery. Correct citation: Amil Merchant, Simon Batzner, Samuel S. Schoenholz, Muratahan Aykol, Gowoon Cheon, Ekin Dogus Cubuk, 'Scaling deep learning for materials discovery', Nature 624, 80–85 (2023), https://doi.org/10.1038/s41586-023-06735-9. Second, on the numbers: the Nature abstract states 'the discovery of 2.2 million structures below the current convex hull' and does not contain the figure 381,000; that number comes from the paper body (structures on the final convex hull), and Google DeepMind's own communications round it to 380,000. Phrase it as '2.2 million structures below the convex hull, of which roughly 380,000 lie on the final convex hull' or attribute 381,000 to the article text explicitly.

---

## [CORRECTED] critique-limits

**Claim as researched:** van der Marel & Hirsch forensic reanalysis: susceptibility data 'pathological', not obtained by the described method nor by three alternatives, 'measured voltage' data not raw. International Journal of Modern Physics B 36, 2375001; arXiv:2201.07686.

**Correction:** The volume number is wrong. Accurate citation: D. van der Marel and J. E. Hirsch, 'Room-temperature superconductivity — or not? Comment on Nature 586, 373 (2020) by E. Snider et al.', International Journal of Modern Physics B 37(04), 2375001 (2023; published online September 2022), https://doi.org/10.1142/S0217979223750012. Volume 37, not 36. (An erratum to it exists at 10.1142/S0217979223920017, IJMPB 37(05).) The arXiv preprint URL given, https://arxiv.org/abs/2201.07686, is correct and is titled 'Extended Comment on Nature 586, 373 (2020) by E. Snider et al', submitted 19 January 2022; its abstract confirms 'the reported background-corrected susceptibility data are pathological' and that the 'measured voltage' data are not raw data.

---

## [CORRECTED] critique-limits

**Claim as researched:** Physicists interviewed after the Dias retractions: Boeri on reputational harm, Hamlin on damage to science and journals contacting coauthors, Ceperley on referees receiving only the accepted manuscript. Physics World, 2024.

**Correction:** The year is wrong and one quote is truncated in a way that changes its subject. Accurate citation: Michael Banks, 'Superconductivity "damaged" as researchers look to move on from retractions', Physics World, 20 October 2023 — not 2024. https://physicsworld.com/a/superconductivity-damaged-as-researchers-look-to-move-on-from-retractions/ Verified quotes: Boeri (Sapienza University of Rome) — 'Dias' inconsiderate behaviour has harmed the reputation of the field and it may take a few years to repair the damage.' Hamlin (University of Florida) — 'I do think the whole saga is damaging to science in general, and superconductivity research more so', and on coauthors, 'All authors are subject to potential reputational harm from a misconduct allegation, so all authors should be privy to the relevant communications from editors from the very beginning.' Ceperley (University of Illinois) — 'We were only provided with the accepted manuscript and not the data files or referee comments.' Because the piece is dated October 2023, it predates the 7 November 2023 lutetium hydride retraction; do not describe these physicists as speaking 'after the Dias retractions' plural without qualification — at the time only the 2022 Nature and 2023 PRL retractions had occurred.

---

## [CORRECTED] critique-limits

**Claim as researched:** Milad Abolhasani reported that roughly 95% of self-driving-laboratory papers omit operational lifetime data, chemical consumption metrics and experimental precision. Chemistry World, 2026.

**Correction:** The year is wrong by two years and the 95% figure is narrower than stated. Accurate citation: Julia Robinson, 'Are we rushing ahead with AI in the lab?', Chemistry World, 2 May 2024 (not 2026), https://www.chemistryworld.com/news/are-we-rushing-ahead-with-ai-in-the-lab/4019424.article Verified quote, Milad Abolhasani (North Carolina State University): 'We were surprised when we did the literature search that 95% of papers on self-driving labs did not report how long they could run the platform before it broke down.' The 95% figure attaches specifically to operational lifetime/durability. The additional items — chemical consumption metrics and experimental precision — are not covered by that percentage in the quoted passage; either drop them or source them separately (Abolhasani's group's own review in the primary literature) before attributing all three to one number.

---

## [CORRECTED] experiment-characterization

**Claim as researched:** The A-Lab produced 41 novel compounds from 58 targets over 17 days of continuous operation, with automated XRD phase analysis as the active-learning feedback signal (Szymanski et al., Nature 624(7990), 86-91, 2023)

**Correction:** Citation, venue, DOI and all 16 author names verified correct (https://doi.org/10.1038/s41586-023-06734-w). The problem is stating the novelty count as established fact. Recommended text: 'The A-Lab, an autonomous solid-state synthesis laboratory combining robotics, ab initio phase-stability data from the Materials Project, and machine-learned interpretation of X-ray diffraction, reported 41 successful syntheses from 58 targets over 17 days of continuous operation; automated XRD phase analysis was the feedback signal that let active learning revise failed synthesis recipes. The novelty of those compounds was subsequently disputed: Leeman, Liu, Stiles, Lee, Bhatt, Schoop and Palgrave argued that roughly two-thirds of the claimed successes are likely known, compositionally disordered variants of the predicted ordered compounds (PRX Energy 3(1), 011002, 2024, https://doi.org/10.1103/PRXEnergy.3.011002).' Any book text should attribute the 41-of-58 figure to the authors rather than assert it.

---

## [CORRECTED] experiment-characterization

**Claim as researched:** Extending GP autonomous experimentation with inhomogeneous measurement noise and anisotropic kernels made the method robust to real synchrotron signal-to-noise, and this machinery was released as the open-source gpCAM package (Noack, Doerk, Li, Streit, Vaia, Yager, Fukuto; Sci. Rep. 10, 17663, 2020)

**Correction:** Bibliographic data all verified correct (https://doi.org/10.1038/s41598-020-74394-1), and the first half of the claim matches the abstract. But gpCAM is never named anywhere in this paper — the software attribution is grafted on from elsewhere. Recommended text: 'Extending Gaussian process autonomous experimentation with inhomogeneous measurement noise and anisotropic kernels made the method robust to the real signal-to-noise structure of synchrotron data (Noack et al., Scientific Reports 10, 17663, 2020, https://doi.org/10.1038/s41598-020-74394-1). Noack later packaged this class of decision-making machinery as the open-source gpCAM library (https://github.com/lbl-camera/gpCAM), described in Noack et al., Nature Reviews Physics 3, 685-697 (2021), https://doi.org/10.1038/s42254-021-00345-y, so other facilities could run their own autonomous experiments.'

---

## [CORRECTED] experiment-characterization

**Claim as researched:** Implicit neural representations were trained to model the dynamical spin structure factor, allowing continuous reconstruction of dynamical correlations from sparse and noisy inelastic neutron scattering measurements instead of binned histograms (Chitturi et al., Nat. Commun. 14, 5852, 2023)

**Correction:** The citation itself is sound — Nature Communications 14, 5852 (2023), https://doi.org/10.1038/s41467-023-41378-4, all 19 authors correct and in order (arXiv:2304.03949). The CLAIM misdescribes the paper. It does not perform 'continuous reconstruction of dynamical correlations ... instead of binned histograms'; that contribution appears nowhere in the paper. What the paper actually does is train a neural implicit representation to mimic simulated S(Q,omega) from a model Hamiltonian, then use automatic differentiation through that network to recover unknown Hamiltonian parameters from experimental data. Recommended text: 'A neural implicit representation was trained to reproduce the dynamical structure factor S(Q, omega) computed from a model Hamiltonian, and automatic differentiation through the trained network was used to recover unknown exchange parameters directly from inelastic neutron scattering data on the square-lattice antiferromagnet La2NiO4 — matching conventional analytical fitting without manual peak-finding, and fast enough for near-real-time analysis of multidimensional scattering data.' The affiliation line should also add Howard University (Mardanya, Chowdhury) and Northeastern University (Bansil, Feiguin).

---

## [CORRECTED] foundations

**Claim as researched:** Ehrenfest 1933, 'Phasenumwandlungen im ueblichen und erweiterten Sinn...', Proc. Royal Acad. Amsterdam 36, 153-157, Commun. Kamerlingh Onnes Lab. Leiden Suppl. 75b; English translation and commentary by Tilman Sauer, EPJ Special Topics (2016), doi:10.1140/epjst/e2016-60344-y

**Correction:** The Ehrenfest original is exactly right (Proc. Royal Acad. Amsterdam 36, 153-157, 1933; Commun. Kamerlingh Onnes Lab. Leiden Suppl. 75b), and arXiv:1612.03062 resolves to Sauer's translation. But the Sauer translation year and volume are wrong. Correct text: 'An English translation with historical commentary by Tilman Sauer, "A look back at the Ehrenfest classification," European Physical Journal Special Topics 226, 539-549 (2017), doi:10.1140/epjst/e2016-60344-y (posted as arXiv:1612.03062 in December 2016).' Correct URL: https://doi.org/10.1140/epjst/e2016-60344-y (preprint: https://arxiv.org/abs/1612.03062). Separately, a caution on the editorial gloss: the dossier asserts books 'must note that Ehrenfest's higher-order categories are now obsolete.' Sauer's own commentary does not say this — it notes the He I / He II lambda-point logarithmic divergence forced modification but argues the framework 'remains important' and proved fruitful. Safer wording: Ehrenfest's higher-order categories do not describe real continuous transitions, which show power-law divergences rather than finite jumps, so modern usage is 'first-order' versus 'continuous.'

---

## [CORRECTED] foundations

**Claim as researched:** Widom 1965, 'Equation of State in the Neighborhood of the Critical Point,' Journal of Chemical Physics 43(11), 3898-3905; scaling relations including 'alpha + 2*beta + gamma = 2, later called the Rushbrooke relation' and gamma = beta*(delta-1)

**Correction:** The citation is exactly right (Widom, J. Chem. Phys. 43(11), 3898-3905, published 1 December 1965; doi 10.1063/1.1696618; Widom was at the Department of Chemistry, Cornell University). The word 'later' is chronologically wrong: Rushbrooke's inequality predates Widom's paper. Correct text: 'from which the scaling relations among critical exponents follow as equalities — including alpha + 2*beta + gamma = 2, which G. S. Rushbrooke had already established in 1963 as the inequality alpha + 2*beta + gamma >= 2 ("On the Thermodynamics of the Critical Region for the Ising Problem," Journal of Chemical Physics 39(3), 842-843, doi:10.1063/1.1734338), and gamma = beta*(delta - 1), the Widom relation.'

---

## [CORRECTED] foundations

**Claim as researched:** Berezinskii, Soviet Physics JETP 32(3), 493-500 (March 1971); 'Russian original ZhETF 59(3), 907 (1971)'; Berezinskii died in 1980 and was ineligible for the 2016 Nobel

**Correction:** The English translation, page range, title, and the 1980 death (23 June 1980, Moscow; born 15 July 1935, Kiev) are all correct, and the JETP URL resolves to the right article. The Russian original year is wrong: ZhETF volume 59 is a 1970 volume. Correct text: 'The paper is "Destruction of Long-range Order in One-dimensional and Two-dimensional Systems having a Continuous Symmetry Group. I. Classical Systems," Soviet Physics JETP 32(3), 493-500 (March 1971); Russian original Zh. Eksp. Teor. Fiz. 59(3), 907-920 (1970), received 31 March 1970.' Correct URL: http://jetp.ras.ru/cgi-bin/e/index/e/32/3/p493?a=list. Note for the author: the jetp.ras.ru database and INSPIRE both stamp the Russian line '1971' by propagating the translation date, so a book citing 1970 should not be 'corrected' back by a copyeditor consulting those two sources; Wikipedia and the paper's own submission date give 1970.

---

## [CORRECTED] foundations

**Claim as researched:** 2021 Nobel Prize in Physics awarded 'for groundbreaking contributions to our understanding of complex systems,' one half jointly to Manabe and Hasselmann, other half to Parisi (Sapienza University of Rome)

**Correction:** The division, laureates, Parisi's individual motivation, affiliation, and the 'around 1980' dating are all correct, but the overall prize citation is misquoted — the official wording contains 'physical.' Correct text: 'The Nobel Prize in Physics 2021 was awarded "for groundbreaking contributions to our understanding of complex physical systems," divided with one half jointly to Syukuro Manabe and Klaus Hasselmann "for the physical modelling of Earth's climate, quantifying variability and reliably predicting global warming," and the other half to Giorgio Parisi (Sapienza University of Rome) "for the discovery of the interplay of disorder and fluctuations in physical systems from atomic to planetary scales."' Correct URL for the full citation: https://www.nobelprize.org/prizes/physics/2021/summary/ (Parisi-specific page: https://www.nobelprize.org/prizes/physics/2021/parisi/facts/).

---

## [CORRECTED] foundations

**Claim as researched:** Turnbull and 'John Chipman Fisher' 1949, 'Rate of Nucleation in Condensed Systems,' Journal of Chemical Physics 17(1), 71-73

**Correction:** The author's name is fabricated. The co-author is John Crocker Fisher (born 19 December 1919, Ithaca NY; died 2 May 2018), Sc.D. MIT 1947, of General Electric Research Laboratory, who managed GE's physical metallurgy section. 'John Chipman' is an entirely different person — the MIT metallurgist and steelmaking chemist — and conflating the two would put a wrong name for a real man in print. Correct text: 'David Turnbull and John Crocker Fisher adapted classical nucleation theory to condensed systems... Published as "Rate of Nucleation in Condensed Systems," Journal of Chemical Physics 17(1), 71-73 (1949).' Everything else in the item checks out: the paper exists with that exact title, JCP volume 17, issue 1, pages 71-73, 1949, authors ordered Turnbull then Fisher, and the URL https://doi.org/10.1063/1.1747055 resolves to it. Note for the author: a stray erratum, JCP 17(4), 429 (1949), also exists, and some secondary databases circulate a wrong DOI (10.1063/1.1747279) and a wrong issue number (No. 4) for this paper — cite 10.1063/1.1747055 and issue 1. The supercooling figure (liquid metals to roughly 0.8 T_m) is accurate.

---

## [CORRECTED] fusion-plasma

**Claim as researched:** Item 4 — Tracey et al. catalogue the practical failure modes of RL tokamak magnetic control — 'control accuracy, steady-state error, sim-to-real transfer and training cost' — and report engineering fixes; Fusion Engineering and Design 200, 114161 (2024).

**Correction:** The citation (authors, venue, volume, article number, year, DOI, arXiv:2307.11546 of July 2023) is entirely correct, but the list of drawbacks overstates the paper's stated scope: sim-to-real transfer is not one of the drawbacks it sets out to fix. Accurate replacement text: "The same DeepMind–EPFL collaboration published a follow-up that is unusually candid about the gap between the Nature demonstration and a deployable controller: Brendan D. Tracey and colleagues address the key drawbacks of the RL method relative to traditional feedback control — control accuracy, steady-state error, and the time required to learn new tasks — reporting up to 65% improvement in shape accuracy, a substantial reduction in the long-term bias of the plasma current, and a threefold or greater reduction in training time, validated in new experiments on the TCV tokamak." Correct URL: https://doi.org/10.1016/j.fusengdes.2024.114161 (preprint https://arxiv.org/abs/2307.11546). Note the published byline also carries a corporate author, 'The TCV Team', after Riedmiller.

---

## [CORRECTED] fusion-plasma

**Claim as researched:** Item 16 — TORAX, open-source differentiable tokamak transport simulator in JAX; Citrin, Goodfellow, … Kohli, arXiv:2406.06718 (10 June 2024); JIT and autodiff make the simulator 'trainable-through', letting reinforcement learning, gradient-based trajectory optimization and neural surrogates operate inside a physics model.

**Correction:** The citation is exact — title, all fourteen authors in order, arXiv identifier and 10 June 2024 submission date all check out. But the paper never claims reinforcement learning or trajectory optimization as applications; neither phrase appears in the abstract or the arXiv v1 full text. Accurate replacement text: "Google DeepMind released TORAX, an open-source, differentiable tokamak core transport simulator implemented in Python using the JAX framework, whose just-in-time compilation gives fast runtimes and whose automatic differentiation enables gradient-based optimization workflows, Jacobian-based PDE solvers, and coupling to neural-network surrogates of physics models — the infrastructure step that lets optimization and learned models operate inside a physics simulation rather than beside it." Correct URL: https://arxiv.org/abs/2406.06718

---

## [CORRECTED] fusion-plasma

**Claim as researched:** Item 18 — Shaing, Crume Jr. & Houlberg 'gave the L–H transition its first explicit bifurcation model'; Physics of Fluids B 2(6), 1492–1498 (1990), doi:10.1063/1.859473.

**Correction:** The bibliographic record is exact (title, three authors, ORNL, volume, issue, pages, 1990, DOI), but the priority claim is wrong: the first explicit bifurcation model of the L–H transition was published a year earlier by two of the same authors. Accurate replacement text: "Ker-Chung Shaing and E. C. Crume Jr. gave the L–H transition its first explicit bifurcation model in 1989, showing that the poloidal momentum balance equation admits bifurcated solutions so that the poloidal flow and the radial electric field jump discontinuously, suppressing turbulent fluctuations — establishing the transition as a genuine bifurcation of the plasma's dynamical state rather than a smooth crossover. Shaing, Crume Jr. and W. A. Houlberg extended the model the following year to include the resulting suppression of turbulent fluctuations." Correct URLs: K. C. Shaing and E. C. Crume Jr., 'Bifurcation theory of poloidal rotation in tokamaks: A model for L-H transition', Physical Review Letters 63(21), 2369–2372 (1989), https://doi.org/10.1103/PhysRevLett.63.2369; and K. C. Shaing, E. C. Crume Jr. and W. A. Houlberg, 'Bifurcation of poloidal rotation and suppression of turbulent fluctuations: A model for the L–H transition in tokamaks', Physics of Fluids B 2(6), 1492–1498 (1990), https://doi.org/10.1063/1.859473

---

## [CORRECTED] fusion-plasma

**Claim as researched:** Summary prose — 'a state change theorized as a thermal-confinement bifurcation (Shaing 1990, Hinton 1991)'.

**Correction:** The label 'thermal-confinement bifurcation' belongs to Hinton 1991 only; Shaing, Crume and Houlberg's model is a bifurcation of poloidal rotation and the radial electric field, not of the heat-transport equation, and its priority date is 1989. Accurate replacement text: "...a state change theorized first as a bifurcation of poloidal rotation and the radial electric field (Shaing and Crume 1989; Shaing, Crume and Houlberg 1990) and then as a thermal-confinement bifurcation (Hinton 1991)..." Correct URLs: https://doi.org/10.1103/PhysRevLett.63.2369, https://doi.org/10.1063/1.859473, https://doi.org/10.1063/1.859866

---

## [CORRECTED] materials-discovery

**Claim as researched:** Cheetham and Seshadri found 'scant evidence' for novelty/credibility/utility; identified all 10 randomly selected Stable Structure entries (of 384,870) in the ICSD; Pnma is ~16% of ICSD structures but ranks 16th at 1.56% in GNoME; top four GNoME space groups all non-centrosymmetric, ~34% of entries, versus ~1% in ICSD; 18,138 compounds with Pm/Ac/Pa and 23,529 with Tc/Np/Pu. Chemistry of Materials 36(8), 3490-3495 (2024).

**Correction:** Citation, quotation, the 384,870 figure, the 10/10 ICSD identification, the 16th-place 1.56% Pnma ranking, the ~34% non-centrosymmetric share, and both radioactive-element counts (18,138 and 23,529) are all exactly right. TWO space-group statistics are garbled. (1) Pnma is NOT ~16% of ICSD structures. The paper says: 'the top two space groups in the ICSD are centrosymmetric Pnma and P2(1)/c, accounting for approximately 16% of all structures, with ~8% in each case.' The 16% is the two groups combined; Pnma alone is ~8%. (2) '~1% in ICSD' does not mean non-centrosymmetric groups are ~1% of the ICSD. The paper says: 'in the ICSD there is only one non-centrosymmetric space group in the top 24 and it accounts for only 1% of all structures.' Accurate replacement text: 'Pnma and P2(1)/c together account for roughly 16% of ICSD structures, about 8% each, yet Pnma ranks only 16th in GNoME's Stable Structure database at 1.56%; the four commonest GNoME space groups are all non-centrosymmetric and cover ~34% of entries, whereas the ICSD's top 24 space groups contain just one non-centrosymmetric group, accounting for 1% of structures.' Correct URL: https://doi.org/10.1021/acs.chemmater.4c00643 (open-access copy: https://escholarship.org/uc/item/9qx9t3kz). Published 8 April 2024.

---

## [CORRECTED] materials-discovery

**Claim as researched:** The A-Lab reported realizing 41 novel compounds from 58 targets over 17 days; Szymanski et al., Nature 624(7990), 86-91, 29 Nov 2023.

**Correction:** The 41/58/17-days figures are correct for the article AS ORIGINALLY PUBLISHED, but the article of record no longer reads that way and a reader following the DOI will find a mismatch. The corrected paper is titled 'An autonomous laboratory for the accelerated synthesis of inorganic materials' (the original was '...of novel materials') and its abstract now reads: 'Over 17 days of continuous operation, the A-Lab realized 36 compounds from a set of 57 targets including a variety of oxides and phosphates that were identified using large-scale ab initio phase-stability data from the Materials Project and Google DeepMind.' Publish as: 'As originally published, the A-Lab reported realizing 41 novel compounds from 58 targets over 17 days; after the 2026 author correction the paper reports 36 compounds from 57 targets and is retitled to drop the word novel.' URL is correct: https://doi.org/10.1038/s41586-023-06734-w

---

## [CORRECTED] materials-discovery

**Claim as researched:** The apparent resistivity drop in LK-99 was attributed to the first-order superionic phase transition of the Cu2S impurity near 385 K; Jain, J. Phys. Chem. C 127(37), 18253-18255 (2023).

**Correction:** The citation, authorship, venue, volume, issue, pages and DOI are all correct, but the temperature is wrong. Jain's abstract states: 'Copper(I) sulfide has a known phase transition at 104 degrees C from an ordered low-temperature phase to a high-temperature superionic phase' — that is ~377 K, not 385 K (385 K would be 112 C). The figure matters because the point of the paper is that 377 K coincides with LK-99's reported ~104.8 C resistivity drop. Accurate replacement text: 'The apparent resistivity drop in LK-99 was attributed to the superionic phase transition of the Cu2S impurity at 104 C (~377 K) — a mundane impurity artefact, not superconductivity.' Correct URL: https://doi.org/10.1021/acs.jpcc.3c05684 (title: 'Superionic Phase Transition of Copper(I) Sulfide and Its Implication for Purported Superconductivity of LK-99'; open preprint arXiv:2308.05222).

---

## [CORRECTED] materials-discovery

**Claim as researched:** OPEN QUESTION as written: 'DeepMind's blog claimed 736 had been independently created as of November 2023, but I found no peer-reviewed audit of that figure.'

**Correction:** The 736 figure is not blog-only. It appears verbatim in the peer-reviewed Nature abstract: 'Of the stable structures, 736 have already been independently experimentally realized.' The open question is still legitimate but must be reframed: the 736 claim IS peer-reviewed as part of the Merchant et al. paper; what is missing is an INDEPENDENT third-party audit of that figure. Rewrite as: 'The 736 figure appears in the peer-reviewed Nature paper as well as DeepMind's blog, but I found no independent audit of it, and Cheetham and Seshadri's sampling suggests the novelty rate is low.'

---

## [CORRECTED] ml-as-physics

**Claim as researched:** Ackley, Hinton, Sejnowski 1985, Cognitive Science 9(1):147-169, DOI 10.1207/s15516709cog0901_7 — Boltzmann machine; learning rule from clamped 'wake' phase minus free-running 'sleep' phase.

**Correction:** Bibliographic data fully confirmed via Crossref (title 'A Learning Algorithm for Boltzmann Machines'; authors, order, journal, volume, issue, pages, year all correct; DOI resolves). One factual anachronism: the 1985 paper uses the terms 'clamped' (phase+) and 'free-running'/'unclamped' (phase-) phases. The 'wake'/'sleep' vocabulary belongs to Hinton, Dayan, Frey and Neal's wake-sleep algorithm for the Helmholtz machine (Science, 1995), not to Ackley-Hinton-Sejnowski 1985. Replace with: '...and derived a learning rule based on the difference between unit correlations measured in a clamped phase and in a free-running phase.' URL unchanged: https://doi.org/10.1207/s15516709cog0901_7

---

## [CORRECTED] ml-as-physics

**Claim as researched:** Amit, Gutfreund, Sompolinsky 1985, PRL 55(14):1530-1533 — replica mean-field theory of the Hopfield model at p = alpha*N; 'the first analytic proof that associative memory survives as a distinct thermodynamic phase.'

**Correction:** Citation fully confirmed via Crossref: 'Storing Infinite Numbers of Patterns in a Spin-Glass Model of Neural Networks', Daniel J. Amit, Hanoch Gutfreund, H. Sompolinsky, Phys. Rev. Lett. 55(14), 1530-1533 (1985); DOI resolves. The word 'proof' overstates the work: the replica method with a replica-symmetric ansatz is a non-rigorous (and, in the spin-glass regime, replica-symmetry-breaking-corrected) calculation, not a proof; rigorous versions came later (e.g. Talagrand, Bovier-Gayrard). Replace 'the first analytic proof that' with 'the first analytic demonstration that' or 'the first replica-theoretic calculation showing that'. URL unchanged: https://doi.org/10.1103/PhysRevLett.55.1530

---

## [CORRECTED] ml-as-physics

**Claim as researched:** Gardner and Derrida 1988, J. Phys. A 21(1):271-284 — alpha_c(K) with alpha_c(0) = 2, 'twice the Hebbian-rule capacity'.

**Correction:** Citation fully confirmed via Crossref: 'Optimal storage properties of neural network models', E. Gardner and B. Derrida, J. Phys. A 21(1), 271-284 (1988); DOI 10.1088/0305-4470/21/1/031 resolves. The comparison is factually wrong: alpha_c = 2 is roughly fourteen times, not twice, the Hebbian capacity. The Hebbian (Hopfield) rule stores alpha_c ≈ 0.138 patterns per neuron (Amit-Gutfreund-Sompolinsky), so 2 / 0.138 ≈ 14.5. (alpha_c = 2 coincides with Cover's 1965 counting result that a perceptron with N inputs can realise 2N random dichotomies; that is the correct comparison to draw, not a factor of two over Hebb.) Replace with: '...obtaining a critical line alpha_c(K) with alpha_c(0) = 2 — the celebrated result that an optimally trained perceptron can store two patterns per weight, more than an order of magnitude above the roughly 0.138 patterns per neuron achieved by the Hebbian rule.' URL unchanged: https://doi.org/10.1088/0305-4470/21/1/031

---

## [CORRECTED] ml-as-physics

**Claim as researched:** Watkin, Rau, Biehl 1993, Rev. Mod. Phys. 65(2):499-556 — standard review 'The statistical mechanics of learning a rule'. Affiliation given as 'Department of Physics, University of Oxford'.

**Correction:** Bibliographic data fully confirmed via Crossref (title, three authors in the given order, RMP 65(2), 499-556, 1993; DOI 10.1103/RevModPhys.65.499 resolves). The affiliation field is wrong as a blanket statement: Watkin and Rau were at the Department of Physics, University of Oxford, but Michael Biehl was at the Institut für Theoretische Physik, Universität Würzburg. Replace the affiliation with: 'Department of Physics, University of Oxford (Watkin, Rau); Institut für Theoretische Physik, Universität Würzburg (Biehl)'. (Note: I could not fetch the APS article page directly — it returned HTTP 403 — so the Würzburg attribution rests on Biehl's documented affiliation history rather than on the printed byline; verify against the PDF before print.) URL unchanged: https://doi.org/10.1103/RevModPhys.65.499

---

## [CORRECTED] ml-as-physics

**Claim as researched:** Krotov and Hopfield 2016, NeurIPS 29, pp. 1172-1180, arXiv:1606.01164 — dense associative memory; capacity raised 'from linear in the number of neurons to superlinear (polynomial in the degree of the interaction)'; duality with one-hidden-layer feedforward networks; feature-matching vs prototype regimes controlled by the sharpness of the energy function.

**Correction:** Paper, authors, venue and page range confirmed (dblp lists NIPS 2016, pp. 1172-1180; arXiv:1606.01164 'Dense Associative Memory for Pattern Recognition', submitted 3 June 2016). The parenthetical is garbled: capacity is polynomial in N with an exponent set by the interaction degree, not 'polynomial in the degree of the interaction'. The paper gives K_max = alpha_n * N^(n-1) for a rectified-polynomial energy of power n. Replace with: '...which raises storage capacity from linear in the number of neurons N to polynomial, K_max ~ N^(n-1) for an interaction of power n.' The duality claim and the sharpness-controlled feature-to-prototype crossover are both confirmed against the paper text. URL unchanged: https://arxiv.org/abs/1606.01164

---

## [CORRECTED] ml-as-physics

**Claim as researched:** Koch-Janusz and Ringel 2018, Nature Physics 14(6):578-582 (arXiv:1704.06279) — challenged the assumption that ordinary RBM training performs RG; real-space mutual-information network; 'recovers the Ising critical exponent in one and two dimensions.'

**Correction:** Citation confirmed via Crossref and arXiv (title 'Mutual information, neural networks and the renormalization group'; Nature Physics 14, 578-582, 2018). The challenge to RBM-as-RG is confirmed in the paper text, which states that prior work assumes training by Kullback-Leibler divergence and shows the RSMI filters for the dimer model are orthogonal to the KL-derived ones. The last sub-clause misstates the result. The abstract reads: 'We apply the algorithm to classical statistical physics problems in one and two dimensions. We demonstrate RG flow and extract the Ising critical exponent.' The one-and-two-dimensions scope attaches to the demonstrations, not to the exponent; and it is a single exponent, not exponents. Replace with: '...the method demonstrates RG flow on classical statistical-physics problems in one and two dimensions and extracts the Ising critical exponent.' URL unchanged: https://doi.org/10.1038/s41567-018-0081-4

---

## [CORRECTED] ml-as-physics

**Claim as researched:** Nanda, Chan, Lieberum, Smith, Steinhardt 2023, ICLR 2023 'oral presentation' (arXiv:2301.05217) — reverse-engineering grokking in modular addition; three phases (memorization, circuit formation, cleanup).

**Correction:** Paper, author list and order, arXiv ID (2301.05217, submitted 12 January 2023), and every scientific claim confirmed against the paper: discrete Fourier transforms and trigonometric identities converting addition to rotation about a circle, and the three continuous phases memorization / circuit formation / cleanup. The 'oral presentation' designation could NOT be verified — OpenReview (both openreview.net and its API) returned a bot-verification challenge and iclr.cc returned 404, and ICLR 2023 labelled accepted papers 'notable-top-5%' / 'notable-top-25%' rather than 'oral' in its decision metadata. Drop the unverified descriptor: cite as 'The Eleventh International Conference on Learning Representations (ICLR 2023) (arXiv:2301.05217)'. URL unchanged: https://arxiv.org/abs/2301.05217

---

## [CORRECTED] ml-as-physics

**Claim as researched:** 'Jason Wei and sixteen coauthors' defined emergent abilities; TMLR 2022, arXiv:2206.07682, submitted 15 June 2022.

**Correction:** The author count is wrong. The arXiv record lists sixteen authors in total — Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, Ed H. Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy Liang, Jeff Dean, William Fedus — so Wei has fifteen coauthors, not sixteen. Replace with: 'Jason Wei and fifteen coauthors defined an emergent ability...'. Everything else checks out: title 'Emergent Abilities of Large Language Models', TMLR 2022, arXiv:2206.07682 v1 submitted 15 June 2022. URL unchanged: https://arxiv.org/abs/2206.07682

---

## [CORRECTED] ml-phase-detection

**Claim as researched:** Broecker, Carrasquilla, Melko & Trebst, Scientific Reports 7, 8823 (2017) — 'machine-learning classification of quantum Monte Carlo configurations remains effective for sign-problematic fermion models'

**Correction:** Bibliographic data are correct (Scientific Reports 7, 8823 (2017), DOI 10.1038/s41598-017-09098-0; arXiv:1608.07848 posted 28 August 2016), but the description of what was classified is wrong in a way the paper explicitly contradicts. Accurate text: 'Peter Broecker, Juan Carrasquilla, Roger G. Melko and Simon Trebst showed that a convolutional neural network fed the Green\'s function sampled by auxiliary-field quantum Monte Carlo — and specifically not the auxiliary-field configurations themselves, which they found do not carry sufficient information — can identify and locate quantum phase transitions in many-fermion systems, including systems with a severe sign problem where conventional estimators such as equal-time correlation functions fail.' URL: https://doi.org/10.1038/s41598-017-09098-0

---

## [CORRECTED] ml-phase-detection

**Claim as researched:** Hu, Singh & Scalettar, Phys. Rev. E 95, 062122 (2017), 'A critical examination'; arXiv:1704.00080 stated as posted 1 April 2017

**Correction:** Everything is correct except the preprint date: arXiv:1704.00080 v1 was submitted 31 March 2017 (announced in the April 2017 listing), not 1 April 2017. Correct text: 'arXiv preprint posted 31 March 2017 as arXiv:1704.00080'. URL: https://doi.org/10.1103/PhysRevE.95.062122 (preprint: https://arxiv.org/abs/1704.00080). Note also that the models list in the claim is exact, and the paper's negative results are worth keeping: PCA on raw configurations fails to capture the 'charge' correlations of the biquadratic spin-one model and the vorticity of the XY model.

---

## [CORRECTED] ml-phase-detection

**Claim as researched:** Huembeli, Dauphin & Wittek, 'Identifying Quantum Phase Transitions with Adversarial Neural Networks', Phys. Rev. B 97, 134109 (2018); arXiv:1710.08382 stated as posted 23 October 2017

**Correction:** Journal, volume, page, year, author names and order, DOI, and the three demonstration models (finite-temperature Ising, Bose-Hubbard, disordered SSH) are all correct; the preprint date is wrong. arXiv:1710.08382 v1 was submitted 11 October 2017 (v2 31 March 2018). Correct text: 'arXiv preprint posted 11 October 2017 as arXiv:1710.08382'. URL: https://doi.org/10.1103/PhysRevB.97.134109 (preprint: https://arxiv.org/abs/1710.08382)

---

## [CORRECTED] ml-phase-detection

**Claim as researched:** Huembeli, Dauphin, Wittek & Gogolin, Phys. Rev. B 99, 104106 (2019) — 'used interpretability tools on networks trained to detect the MBL transition to extract the characteristic features the network relies on'

**Correction:** Authors, venue (Phys. Rev. B 99, 104106 (2019)), DOI, and arXiv:1806.00419 posted 1 June 2018 are all correct, but the method is mischaracterized: this is not post-hoc interpretability applied to a trained classifier. Accurate text: 'Patrick Huembeli, Alexandre Dauphin, Peter Wittek and Christian Gogolin used an almost entirely unsupervised, adversarial (game-theoretic) setup between two neural networks with conflicting objectives to let the machine itself choose which features to predict from, thereby identifying a new effective order parameter for the disorder-driven many-body-localization transition and locating it from far fewer disorder realizations — a reduction in numerical effort of roughly two orders of magnitude — rather than only locating a transition already characterized by hand.' URL: https://doi.org/10.1103/PhysRevB.99.104106

---

## [CORRECTED] ml-phase-detection

**Claim as researched:** Rem, Käming, Tarnowski, Asteria, Fläschner, Becker, Sengstock & Weitenberg, cited as 'Nature Physics 15, 917' (2019)

**Correction:** Author names and order, year, DOI, arXiv:1809.05519 posted 14 September 2018, and the substance of the claim are all correct; only the page citation is incomplete. Crossref gives the full range. Correct text: 'Nature Physics 15, 917-920 (2019)'. URL: https://doi.org/10.1038/s41567-019-0554-0. (This resolves the dossier's own open question about 917-922 vs 917-920: the publisher record is 917-920.)

---

## [CORRECTED] ml-phase-detection

**Claim as researched:** Ohtsuki & Ohtsuki, J. Phys. Soc. Jpn. 85, 123706 (2016) — 'estimated the critical exponents and phase boundaries of the Anderson and quantum Hall transitions'

**Correction:** Authors, order, spelling, journal, volume, article number, year, DOI and arXiv v1 date (3 October 2016, arXiv:1610.00462) are all correct, but the claimed results are not in the paper. The paper reports no critical-exponent estimates — it takes previously published critical disorders and exponents from finite-size scaling of the localization length as external reference values against which the network output is compared — and it does not treat the quantum Hall plateau transition. Accurate text: 'Tomoki Ohtsuki (NTT DATA Mathematical Systems Inc.) and Tomi Ohtsuki (Sophia University) trained convolutional neural networks (a two-weight-layer network and a LeNet-like four-weight-layer network) on images of individual eigenfunctions of random two-dimensional electron systems, and showed the trained classifier locates the Anderson localization-delocalization transition of the 2D symplectic SU(2) model and the transition from a disordered Chern insulator to an Anderson insulator in the unitary class, with the learned classification transferring to the related Ando model — one of the earliest deep-learning treatments of a disorder-driven quantum phase transition.' URL: https://doi.org/10.7566/JPSJ.85.123706. The dossier's open question that 'Ohtsuki & Ohtsuki attempted exponents directly' rests on the same error and should be struck; exponent estimation appears in the later three-dimensional follow-up, J. Phys. Soc. Jpn. 86, 044708 (2017).

---

## [CORRECTED] ml-phase-detection

**Claim as researched:** Broecker, Carrasquilla, Melko & Trebst showed that machine-learning classification of quantum Monte Carlo CONFIGURATIONS remains effective for sign-problematic interacting fermion models. Scientific Reports 7, 8823 (2017); https://doi.org/10.1038/s41598-017-09098-0

**Correction:** Bibliographic data are all correct (authors, order, venue, article number, DOI; arXiv:1608.07848, posted 28 August 2016). The physics claim is misleading and would be actively wrong next to the Ch'ng et al. entry, which DOES use auxiliary-field configurations. Accurate text: 'Peter Broecker, Juan Carrasquilla, Roger G. Melko and Simon Trebst showed that a convolutional neural network fed the Green's function sampled by auxiliary-field quantum Monte Carlo — the auxiliary field itself, they found, does not hold sufficient information — can correctly identify and locate quantum phase transitions in many-fermion systems, and that this works even in systems with a severe fermion sign problem, where conventional equal-time correlation functions extracted from the same Green's function fail.' URL unchanged: https://doi.org/10.1038/s41598-017-09098-0

---

## [CORRECTED] ml-phase-detection

**Claim as researched:** Hu, Singh & Scalettar, critical examination of PCA and autoencoders on square/triangular Ising, Blume-Capel, biquadratic-exchange spin-one Ising, and 2D XY. Phys. Rev. E 95, 062122 (2017); arXiv:1704.00080 posted 1 April 2017; https://doi.org/10.1103/PhysRevE.95.062122

**Correction:** Everything is right except the preprint date. arXiv:1704.00080 v1 was submitted Friday, 31 March 2017 (22:51 UTC), not 1 April 2017. Corrected text: '...arXiv preprint posted 31 March 2017 as arXiv:1704.00080.' Venue, authors, article number, DOI and model list all confirmed against the abstract. URL correct: https://doi.org/10.1103/PhysRevE.95.062122

---

## [CORRECTED] ml-phase-detection

**Claim as researched:** Huembeli, Dauphin & Wittek, adversarial domain adaptation to derive phase diagrams from ground states; Ising at finite temperature, Bose-Hubbard, disordered SSH. Phys. Rev. B 97, 134109 (2018); arXiv preprint posted 23 October 2017 as arXiv:1710.08382; https://doi.org/10.1103/PhysRevB.97.134109

**Correction:** The arXiv posting date is wrong. arXiv:1710.08382 v1 was submitted Wednesday, 11 October 2017 (16:00:21 UTC); v2 on 31 March 2018. Corrected text: '...arXiv preprint posted 11 October 2017 as arXiv:1710.08382.' Authors, order, venue (Phys. Rev. B 97, 134109 (2018)), DOI and the substance of the claim are all confirmed. URL correct: https://doi.org/10.1103/PhysRevB.97.134109

---

## [CORRECTED] ml-phase-detection

**Claim as researched:** Huembeli, Dauphin, Wittek & Gogolin 'used interpretability tools on networks trained to detect the MBL transition in order to extract the characteristic features the network relies on'. Phys. Rev. B 99, 104106 (2019); arXiv:1806.00419 posted 1 June 2018; https://doi.org/10.1103/PhysRevB.99.104106

**Correction:** Authors, order, venue, article number, DOI and preprint date are all correct, but the method is mischaracterized. The paper is not post-hoc interpretability applied to a supervised MBL classifier; it is an adversarial, game-theoretic setup between two networks with conflicting objectives, described by the authors as 'almost entirely unsupervised'. Accurate text: 'Patrick Huembeli, Alexandre Dauphin, Peter Wittek and Christian Gogolin used an adversarial, almost entirely unsupervised game between two neural networks to discover automatically which features of the data mark the disorder-driven many-body-localization transition — in effect a machine-found order parameter — pinning down the transition from far fewer disorder realizations and cutting the numerical effort of mapping the phase diagram by roughly a factor of a hundred.' URL unchanged: https://doi.org/10.1103/PhysRevB.99.104106

---

## [CORRECTED] ml-phase-detection

**Claim as researched:** Tomoki Ohtsuki and Tomi Ohtsuki applied deep CNNs to wave-function images of random 2D electron systems and ESTIMATED THE CRITICAL EXPONENTS and phase boundaries of the Anderson and quantum Hall transitions. J. Phys. Soc. Jpn. 85, 123706 (2016); arXiv posted 3 October 2016; https://doi.org/10.7566/JPSJ.85.123706

**Correction:** This is the most serious error in the dossier and it inverts what the paper says. The bibliography is right — Tomoki Ohtsuki (NTT DATA Mathematical Systems Inc.) and Tomi Ohtsuki (Sophia University), in that order, J. Phys. Soc. Jpn. 85, 123706 (2016), arXiv:1610.00462 v1 3 October 2016, DOI 10.7566/JPSJ.85.123706 — but the paper does NOT estimate critical exponents. Its own text states, of the scaling law P(W,L)=f[(W-Wc)L^(1/nu)], that 'determining the exponent nu is beyond the scope of this Letter.' The transitions treated are the localization-delocalization (Anderson) transition and the disordered Chern insulator-to-Anderson insulator transition, i.e. the quantum ANOMALOUS Hall setting, not the integer quantum Hall plateau transition. Accurate text: 'Tomoki Ohtsuki and Tomi Ohtsuki applied a multilayered convolutional neural network to images of eigenfunctions of random two-dimensional electron systems, using it to judge from a single eigenfunction which phase it belongs to and thereby to locate the localization-delocalization transition and the disordered Chern insulator-to-Anderson insulator transition — one of the earliest deep-learning treatments of a disorder-driven quantum phase transition. The authors explicitly left determination of the critical exponent to future work.' URL unchanged: https://doi.org/10.7566/JPSJ.85.123706. If a critical-exponent claim is wanted, do not attach it to this paper; the authors' three-dimensional follow-up is Tomi Ohtsuki and Tomoki Ohtsuki (note the reversed order), J. Phys. Soc. Jpn. 86, 044708 (2017), arXiv:1612.04909, https://doi.org/10.7566/JPSJ.86.044708 — and its exponent content must be checked separately before being asserted.

---

## [CORRECTED] ml-potentials-nucleation

**Claim as researched:** Bartók, Payne, Kondor & Csányi (2010), PRL 104, 136403 — Gaussian Approximation Potentials via Gaussian process regression on QM energies and forces. Dossier lists affiliation as 'Cavendish Laboratory, University of Cambridge' for all four authors.

**Correction:** Citation, authors, order, year, venue, and DOI are all correct. The affiliation field is wrong for two of the four authors. Accurate text: 'Albert P. Bartók and Mike C. Payne (Cavendish Laboratory, University of Cambridge); Risi Kondor (Center for the Mathematics of Information, California Institute of Technology); Gábor Csányi (Engineering Laboratory, University of Cambridge).' Correct URL: https://doi.org/10.1103/PhysRevLett.104.136403

---

## [CORRECTED] ml-potentials-nucleation

**Claim as researched:** Invernizzi, Piaggi & Parrinello (2020), PRX 10, 041034 — 'presented a unified approach to enhanced sampling — the on-the-fly probability enhanced sampling (OPES) framework — that reformulates metadynamics-style and generalized-ensemble methods as targeted probability-distribution reweighting.'

**Correction:** The citation (authors, order, year, venue, article number, DOI) is correct, but the claim misattributes the introduction of OPES. The PRX paper's own abstract states that it adapts 'the recently developed on-the-fly probability enhanced sampling method [Invernizzi and Parrinello, J. Phys. Chem. Lett. 11.7 (2020)], which was originally introduced for metadynamics-like sampling.' Accurate replacement text: 'Michele Invernizzi, Pablo M. Piaggi and Michele Parrinello presented a unified perspective on enhanced sampling, focusing on the target probability distribution rather than the bias potential, and introduced a class of collective-variable-based bias potentials able to sample any expanded ensemble normally reached by replica exchange (OPES-expanded). Their practical implementation adapts the on-the-fly probability enhanced sampling (OPES) method, which had been introduced earlier that year by Michele Invernizzi and Michele Parrinello, "Rethinking Metadynamics: From Bias Potentials to Probability Distributions", Journal of Physical Chemistry Letters 11, 2731-2736 (2020).' URL for the PRX paper: https://doi.org/10.1103/PhysRevX.10.041034 ; URL for the paper that actually introduced OPES: https://doi.org/10.1021/acs.jpclett.0c00497

---

## [CORRECTED] ml-potentials-nucleation

**Claim as researched:** MACE-MP-0 is a general-purpose MACE model 'trained on roughly 150,000 inorganic crystal structures from the Materials Project'; cited only as arXiv:2401.00096.

**Correction:** Two defects. (1) The training-set figure is garbled: MACE-MP-0 was trained on the Materials Project Trajectory (MPtrj) dataset, which contains 1,580,395 DFT structures drawn from roughly 146,000 inorganic materials covering 89 elements. The ~150,000 figure is the number of MATERIALS, not structures; the number of training structures is ~1.6 million (an order of magnitude larger). The paper's own abstract says only 'a public dataset of moderate size' and gives no such count. (2) The work is no longer preprint-only. Accurate replacement text: 'Ilyes Batatia and a large collaboration released MACE-MP-0, a single general-purpose MACE model trained on the Materials Project Trajectory (MPtrj) dataset — about 1.6 million DFT structures drawn from roughly 146,000 inorganic materials — which runs stable molecular dynamics across solids, liquids, gases, chemical reactions and interfaces out of the box and can be fine-tuned on a handful of application-specific data points to reach ab initio accuracy.' Correct citation: Ilyes Batatia, Philipp Benner, Yuan Chiang, Alin M. Elena, Dávid P. Kovács et al., 'A foundation model for atomistic materials chemistry', Journal of Chemical Physics 163(18), 184110 (2025), https://doi.org/10.1063/5.0297006 — preprint arXiv:2401.00096 (v1 29 December 2023), https://arxiv.org/abs/2401.00096

---

## [CORRECTED] ml-potentials-nucleation

**Claim as researched:** Invernizzi, Piaggi & Parrinello (2020) 'presented a unified approach to enhanced sampling — the on-the-fly probability enhanced sampling (OPES) framework'. PRX 10, 041034.

**Correction:** The venue, authors, year and URL are correct, but the attribution overstates: the PRX paper did NOT introduce OPES. Its abstract explicitly states that it provides a practical implementation 'by properly adapting the iterative scheme of the recently developed on-the-fly probability enhanced sampling method [Invernizzi and Parrinello, J. Phys. Chem. Lett. 11.7 (2020)], which was originally introduced for metadynamics-like sampling.' Accurate replacement text: 'Michele Invernizzi, Pablo M. Piaggi and Michele Parrinello presented a unified approach to enhanced sampling, recasting collective-variable bias methods (umbrella sampling, metadynamics) and tempering/expanded-ensemble methods such as replica exchange within one perspective focused on the target probability distribution, and implemented it by adapting the iterative scheme of the on-the-fly probability enhanced sampling (OPES) method.' Cite as: Michele Invernizzi, Pablo M. Piaggi, Michele Parrinello, 'Unified Approach to Enhanced Sampling', Physical Review X 10, 041034 (2020), https://doi.org/10.1103/PhysRevX.10.041034. OPES itself must be credited to: Michele Invernizzi and Michele Parrinello, 'Rethinking Metadynamics: From Bias Potentials to Probability Distributions', Journal of Physical Chemistry Letters 11, 2731-2736 (2020), https://doi.org/10.1021/acs.jpclett.0c00497

---

## [CORRECTED] nnqs-quantum

**Claim as researched:** V-score paper: Wu, Rossi, Vicentini, Carleo 'with a 30-author consortium'; Science 386(6719), 296-301, published 18 October 2024, arXiv:2302.04919

**Correction:** The author count is wrong: the paper has 33 authors, not 30 (the dossier's own `people` field correctly lists all 33, matching arXiv exactly). Accurate text: 'Dian Wu, Riccardo Rossi, Filippo Vicentini and Giuseppe Carleo, with a 33-author consortium that included Steven R. White, Masatoshi Imada, Federico Becca, Juan Carrasquilla and Antoine Georges, defined the V-score...' Everything else in the item is correct: 'Variational benchmarks for quantum many-body problems', Science 386, 296-301 (18 October 2024), https://doi.org/10.1126/science.adg9774, preprint arXiv:2302.04919 (2023).

---

## [CORRECTED] nnqs-quantum

**Claim as researched:** Keesling et al. (Lukin group), Rydberg simulator verification of the quantum Kibble-Zurek mechanism, 'measuring the growth of spatial correlations and the universal scaling of defect density with ramp rate', Nature 568(7751), 207-211 (2019), arXiv:1809.05540

**Correction:** The citation is correct but the described measurement overstates/misstates the result. The paper measures the growth of spatial correlations across the transition and the universal power-law scaling of the correlation length with the ramp (sweep) rate — not defect density — and additionally extracts critical exponents for chiral clock models and observes corrections beyond the QKZM prediction. Accurate text: 'Alexander Keesling and co-workers in Mikhail D. Lukin's group used a programmable Rydberg-atom simulator to verify the quantum Kibble-Zurek mechanism across an Ising-type quantum phase transition, measuring the growth of spatial correlations and the universal scaling of the correlation length with the ramp rate, observing corrections beyond the QKZM prediction, and extracting critical exponents for chiral clock models.' Citation unchanged: Nature 568, 207-211 (2019), https://doi.org/10.1038/s41586-019-1070-1, preprint arXiv:1809.05540 (2018).

---

## [CORRECTED] symbolic-theory

**Claim as researched:** Lemos, Jeffrey, Cranmer, Ho, Battaglia (MLST 4:045002, 2023) rediscovered Newton's gravitation from 30 years of solar-system trajectories and 'recovered the masses of the bodies to within an order of magnitude'

**Correction:** The bibliographic data (Machine Learning: Science and Technology, vol. 4, no. 4, article 045002, 2023, DOI 10.1088/2632-2153/acfa63) is correct, but the accuracy figure is wrong and badly understates the result. Accurate text: 'A graph neural network trained on 30 years of observed trajectories of the Sun, planets and large moons, then distilled by symbolic regression, rediscovered Newton's law of gravitation and simultaneously inferred the masses of the bodies, with a mean percent error of about 9.1% from the learned simulator and about 1.6% after the masses were re-fitted using the discovered symbolic force law — without being given either the force law or the masses.' URL: https://doi.org/10.1088/2632-2153/acfa63 (preprint: https://arxiv.org/abs/2202.02306). 'Within an order of magnitude' appears nowhere in the paper and should not be printed.

---

## [CORRECTED] symbolic-theory

**Claim as researched:** AI-Descartes (Cornelio, Dash, Austel, Josephson, Goncalves, Clarkson, Megiddo, El Khadir, Horesh), Nature Communications 14:1777 (2023); rediscovered Kepler's third law, relativistic time dilation, Langmuir's adsorption equation; affiliation 'IBM Research'

**Correction:** Title, author list and order, venue (Nature Communications, vol. 14, article 1777, 2023), DOI 10.1038/s41467-023-37236-y and all three rediscovered laws are confirmed. Only the affiliation is wrong: the paper is not uniformly IBM Research. Accurate text: 'Cristina Cornelio, Sanjeeb Dash, Vernon Austel, Kenneth L. Clarkson, Nimrod Megiddo, Bachir El Khadir and Lior Horesh (IBM Research), with Tyler R. Josephson (University of Maryland, Baltimore County) and Joao Goncalves.' URL unchanged: https://doi.org/10.1038/s41467-023-37236-y

---

## [CORRECTED] symbolic-theory

**Claim as researched:** SINDy — Brunton, Proctor, Kutz, PNAS 113(15):3932-3937 (2016), sparse regression over a library of candidate nonlinear functions; affiliation 'University of Washington'

**Correction:** Title ('Discovering governing equations from data by sparse identification of nonlinear dynamical systems'), authors, order, venue, pages, year and DOI 10.1073/pnas.1517384113 are all confirmed. Affiliation is wrong for one author. Accurate text: 'Steven L. Brunton and J. Nathan Kutz (University of Washington); Joshua L. Proctor (Institute for Disease Modeling, Bellevue, Washington).' URL unchanged: https://doi.org/10.1073/pnas.1517384113

---

## [CORRECTED] symbolic-theory

**Claim as researched:** Companion RSMI-NE paper (Phys. Rev. E 104:064106, 2021) constructs phase diagrams and identifies symmetry-breaking patterns 'in systems including the anisotropic Ising and dimer models'

**Correction:** Authors, title ('Symmetries and phase diagrams with real-space mutual information neural estimation'), venue, volume, article number, year and DOI 10.1103/PhysRevE.104.064106 are confirmed. The model list is wrong: the anisotropic Ising model is not studied. Accurate text: '...identify the symmetry-breaking pattern of transitions in systems including the two-dimensional Ising model (ferromagnetic and antiferromagnetic), the interacting dimer model with its BKT transition, and a non-equilibrium one-dimensional chipping-and-aggregation model, making the learned representation itself the interpretable output.' URL unchanged: https://doi.org/10.1103/PhysRevE.104.064106

---

## [CORRECTED] symbolic-theory

**Claim as researched:** Carrasquilla & Melko, Nature Physics 13(5):431-434 (2017): a feedforward neural network classifies raw Ising configurations, and 'the same approach detects transitions in models with no conventional local order parameter, such as the Ising gauge theory'

**Correction:** Authors, title ('Machine learning phases of matter'), venue, volume, issue, pages, year and DOI 10.1038/nphys4035 are confirmed, but the claim overstates what the feedforward network did. The abstract states the network must be 'modified to a convolutional neural network' to handle topological phases with no conventional order parameter. Accurate text: 'A standard feed-forward neural network trained on raw Monte Carlo spin configurations of the two-dimensional Ising model detects the transition and encodes magnetization-like structure; the same network also handles non-trivial states such as Coulomb phases, but detecting topological phases with no conventional order parameter — the Ising lattice gauge theory — required modifying the architecture to a convolutional neural network.' URL unchanged: https://doi.org/10.1038/nphys4035

---

## [CORRECTED] symbolic-theory

**Claim as researched:** Wetzel, Melko, Scott, Panju, Ganesh, Phys. Rev. Research 2(3):033499 (2020): siamese networks yield conserved quantities and symmetry invariants, 'with the interpretation step performed by symbolic regression and automated reasoning tools'

**Correction:** Authors, order, title, venue, article number, year and DOI 10.1103/PhysRevResearch.2.033499 are confirmed, but the interpretation method is fabricated. The paper interprets the network's bottleneck using ridge regression on polynomial features of increasing degree; symbolic regression is named only as a future direction ('Future directions of this work include an upgrade of the polynomial regression to symbolic regression'), and no automated-reasoning or SAT/SMT solver is used. Accurate text: 'Siamese neural networks trained to map data points belonging to the same event, field configuration, or trajectory to the same output learn the relevant symmetry invariants and conserved quantities; the authors read these out in closed form by applying polynomial ridge regression to the network's low-dimensional bottleneck, demonstrated on events in special relativity, transformations of electromagnetic fields, and motion in a central potential.' URL unchanged: https://doi.org/10.1103/PhysRevResearch.2.033499

---

## [CORRECTED] symbolic-theory

**Claim as researched:** AI Poincaré (Liu & Tegmark), Phys. Rev. Lett. 126(18):180604 (2021): discovers conserved quantities via local intrinsic dimensionality of the trajectory manifold, 'correctly identifying the number of conservation laws in systems including the Kepler problem and a relativistic charged particle'

**Correction:** Authors, title ('Machine Learning Conservation Laws from Trajectories'), venue, volume, issue, article number, year and DOI 10.1103/PhysRevLett.126.180604 are confirmed, and the Kepler problem is genuinely one of the test systems (3 conserved quantities: energy, angular momentum, Runge-Lenz vector). The 'relativistic charged particle' is not among the systems tested and appears to be fabricated. The five Hamiltonian systems are the harmonic oscillator, the Kepler problem, the double pendulum, a magnetic mirror, and the gravitational three-body problem. Accurate text: 'The AI Poincaré method discovers conserved quantities from observed trajectories by measuring the local intrinsic dimensionality of the manifold the trajectory explores in phase space, correctly identifying the number of conservation laws in five Hamiltonian systems including the Kepler problem, a magnetic mirror and the gravitational three-body problem, without prior physical assumptions.' URL unchanged: https://doi.org/10.1103/PhysRevLett.126.180604

---

## [CORRECTED] symbolic-theory

**Claim as researched:** Farmer & Sidorowich, Phys. Rev. Lett. 59(8):845-848 (1987), local approximation forecasting of chaotic time series as 'an early demonstration that a system's equations of motion can be recovered empirically from observation alone'

**Correction:** Bibliographic data is fully correct: J. Doyne Farmer and John J. Sidorowich, 'Predicting chaotic time series', Physical Review Letters, vol. 59, no. 8, pp. 845-848 (1987), DOI 10.1103/PhysRevLett.59.845. The characterization overstates the result: the paper builds local linear and polynomial approximations to the dynamics on a delay-embedded attractor for the purpose of prediction; it does not recover equations of motion in closed or symbolic form. Accurate text: 'J. Doyne Farmer and John J. Sidorowich introduced local approximation methods for forecasting chaotic time series, fitting piecewise linear and polynomial models to the dynamics reconstructed from delay-embedded data — an early demonstration that a system's dynamics can be approximated empirically from observation alone, though not in symbolic or closed form.' URL unchanged: https://doi.org/10.1103/PhysRevLett.59.845. The dossier's own caveat is right: Farmer must be framed as prehistory, not as symbolic-regression or order-parameter work.

---

## [CORRECTED] tipping-points

**Claim as researched:** Dablander and Bury 2022 PNAS letter: because every training example was detrended with a Gaussian filter of a particular bandwidth, the network implicitly requires the same filter at application time.

**Correction:** The filters are reversed. Bury et al. (2021) detrended all training data with a LOWESS filter of span 0.2; the Gaussian filter is the alternative Dablander and Bury tested, on which the classifier failed. Accurate text: "In an unusual act of self-critique, Thomas M. Bury co-authored a PNAS letter with Fabian Dablander showing that the performance of the deep learning early-warning classifier is highly sensitive to preprocessing: because every training example in Bury et al. (2021) was detrended with a LOWESS filter of span 0.2, the network implicitly requires that same filter at application time — stationary AR(1) series left undetrended, or detrended instead with a Gaussian filter of bandwidth 0.2, are misclassified as approaching a bifurcation — so its apparent advantage over classical indicators shrinks when preprocessing choices are varied." Correct citation: Fabian Dablander, Thomas M. Bury, 'Deep learning for tipping points: Preprocessing matters', PNAS 119(37), e2207720119 (2022), https://doi.org/10.1073/pnas.2207720119

---

## [CORRECTED] tipping-points

**Claim as researched:** Chen and Tung Matters Arising, listed as year 2023, Nature Climate Change 14(1), 40-42, arguing coverage-driven variance artifacts and a sign error in two subpolar proxies.

**Correction:** Substance confirmed, but the citation year is wrong: the paper is formally 2024 (Nat. Clim. Chang. 14, 40–42, January 2024 issue; published online 30 November 2023). The dossier is also internally inconsistent, dating Boers's reply in the same issue as 2024. Accurate text: "Xianyao Chen and Ka-Kit Tung, 'Evidence lacking for a pending collapse of the Atlantic Meridional Overturning Circulation', Nature Climate Change 14(1), 40–42 (2024; published online 30 November 2023), https://doi.org/10.1038/s41558-023-01877-0. They argue that all eight proxies show artificial rises in variance as observational coverage expanded, that climate models with no genuine early-warning signal yield apparent warnings when sampled according to historical coverage, and that the signs of Boers's salinity proxy and one further subpolar AMOC proxy were reversed in error."

---

## [CORRECTED] tipping-points

**Claim as researched:** Hastings and Wysham 2010 demonstrated the class of systems exhibiting leading indicators is limited, 'constructing ecological models and identifying natural systems in which a regime change occurs with no forewarning at all'.

**Correction:** Overstates the paper: it identifies a class of MODELS, and only infers that a corresponding class of natural systems is LIKELY to exist. No natural system is identified. Accurate text: "Alan Hastings and Derin B. Wysham demonstrated that the class of ecological systems that will exhibit leading indicators of regime shifts is limited, constructing a set of ecological models — those whose nonlinearities combined with environmental variability yield descriptions without smooth potentials — for which there is no forewarning of a regime change, and arguing that there is therefore also likely to be a class of natural systems for which no forewarning exists." Citation: Alan Hastings, Derin B. Wysham, 'Regime shifts in ecological systems can occur with no warning', Ecology Letters 13(4), 464–472 (2010), https://doi.org/10.1111/j.1461-0248.2010.01439.x

---

## [CORRECTED] tipping-points

**Claim as researched:** A 2025 Nature Geoscience review led by Niklas Boers synthesized observational evidence for destabilization across Earth system tipping elements 'including the AMOC, the polar ice sheets, the Amazon and permafrost systems'.

**Correction:** The elements are wrong. The review assesses four tipping elements: the Greenland Ice Sheet, the AMOC, the Amazon rainforest and the South American monsoon system. It does not cover permafrost, and the ice-sheet analysis is Greenland specifically, not 'the polar ice sheets'. Accurate text: "A 2025 Nature Geoscience review led by Niklas Boers, with Timothy M. Lenton and Chris A. Boulton among the authors, presented observation-based evidence that four Earth system tipping elements — the Greenland Ice Sheet, the Atlantic Meridional Overturning Circulation, the Amazon rainforest and the South American monsoon system — have lost stability in recent decades, and assessed what critical-slowing-down indicators can and cannot establish for each, including how oceanic and atmospheric coupling between elements can generate spurious signals or mask genuine destabilization." Citation: Niklas Boers, Teng Liu, Sebastian Bathiany, Maya Ben-Yami, Lana L. Blaschke, Nils Bochow, Chris A. Boulton, Timothy M. Lenton, Andreas Morr, Da Nian, Martin Rypdal, Taylor Smith, 'Destabilization of Earth system tipping elements', Nature Geoscience 18(10), 949–960 (2025), https://doi.org/10.1038/s41561-025-01787-0

---

## [CORRECTED] tipping-points

**Claim as researched:** The Global Tipping Points Report, led by Timothy M. Lenton, 'with more than 160 authors from over 87 institutions across 23 countries', 2023 first edition and 2025 second edition; coral reefs crossing a tipping point at roughly 1.4 °C, ice sheets possibly past thresholds, Amazon and AMOC at risk below 2 °C.

**Correction:** All substantive findings check out, but the participation figures are inflated by 'more than'/'over'. The report's own figure is exactly 160 authors from 87 institutions in 23 countries. Accurate text: "The Global Tipping Points Report, led by Timothy M. Lenton at the University of Exeter's Global Systems Institute with 160 authors from 87 institutions across 23 countries, appeared in a first edition in 2023 and a second in 2025; the 2025 edition states that warm-water coral reefs are already crossing a tipping point at roughly 1.4 °C of warming, that parts of the polar ice sheets may have passed thresholds that would eventually commit the world to several metres of irreversible sea-level rise, and that the Amazon and the AMOC are both at risk below 2 °C." URL https://global-tipping-points.org/ resolves and is correct for the report series; for a stable citable record prefer the report's Zenodo deposits (e.g. https://zenodo.org/records/18611781 for Section 2, Earth System Tipping Points).

---

## [REFUTED] ml-phase-detection

**Claim as researched:** Open question in the dossier: 'Ohtsuki & Ohtsuki attempted exponents directly' (offered as evidence on whether ML can deliver reliable critical exponents).

**Correction:** False, and it propagates the item-28 error into the open-questions section. The 2016 Letter states the opposite: 'determining the exponent nu is beyond the scope of this Letter.' Corrected text: 'Whether machine-learned phase boundaries can deliver quantitatively reliable critical exponents and finite-size scaling — as opposed to approximate transition locations — is still unsettled; Ohtsuki and Ohtsuki explicitly deferred determination of the exponent nu, and the general reliability of the approach for critical phenomena is not established.'

---

## [REFUTED] symbolic-theory

**Claim as researched:** Open question: 'a definitive primary source in which a machine learning system autonomously produces accurate critical exponents was not confirmed'

**Correction:** This open question is answered by a paper already in the dossier. Koch-Janusz and Ringel (Nature Physics 14, 578-582, 2018) state in their abstract: 'We demonstrate RG flow and extract the Ising critical exponent.' The body reports the correlation-length exponent nu = 1.0 +/- 0.15 for the 2D Ising model (exact value 1), obtained by finite-size data collapse across successive RSMI-driven RG steps, with the critical point located to about 1% from the divergent RG flow. The dossier's framing is still partly right — the exponent is extracted by finite-size scaling applied to the machine-learned RG flow rather than emitted autonomously by the network — but the flat statement that no primary source was confirmed is wrong and should be replaced with that citation and that caveat. URL: https://doi.org/10.1038/s41567-018-0081-4

---

## [UNVERIFIABLE] critique-limits

**Claim as researched:** Confidential 124-page University of Rochester investigation report, commissioned at NSF's request, concluded 8 February 2024 after a ten-month inquiry, examined 16 allegations and found misconduct more likely than not in each; surfaced through court documents in a lawsuit Dias filed.

**Correction:** NOTE: partially verified only. CONFIRMED from the cited Nature article (Dan Garisto, 'Exclusive: official investigation reveals how superconductivity physicist faked blockbuster results', Nature, 2024, doi:10.1038/d41586-024-00976-y): the report is 'a confidential 124-page report from the University of Rochester', it was 'disclosed in a lawsuit', Nature's news team 'discovered the bombshell investigation report in court documents', and it found data fabrication, falsification and plagiarism. UNVERIFIED because the article body is paywalled and WebSearch was unavailable: the count of 16 allegations, the 'ten-month' duration, the 8 February 2024 conclusion date, and the 'more likely than not' standard applied to each of 16. The NSF-request element is corroborated only by Wikipedia ('at the request of the National Science Foundation'), which is not adequate sourcing for print. Obtain the Nature full text or the court filing before printing any of these four specifics.

---

## [UNVERIFIABLE] ml-phase-detection

**Claim as researched:** Affiliation fields given in the dossier (Perimeter/Waterloo; ETH Zurich; IOP CAS Beijing; Heidelberg; San José State; Cornell + KITP UCSB; UC Davis; ICFO; Universität Hamburg)

**Correction:** Only two were checked against a printed title page in this session: Zhang & Kim (arXiv:1611.01518v2 title page reads 'Department of Physics, Cornell University, Ithaca, New York 14853, USA and Kavli Institute for Theoretical Physics, University of California, Santa Barbara, California 93106, USA' — matches the dossier), and Ohtsuki & Ohtsuki (title page reads 'NTT DATA Mathematical Systems Inc, Shinjuku-ku, Tokyo' for Tomoki and 'Physics Division, Sophia University, Chiyoda-ku, Tokyo' for Tomi — the dossier left this field empty). The remaining affiliation strings could not be verified because APS and Nature title pages redirect to authentication endpoints and the session's web-search budget was exhausted; they are plausible but should be pulled from each published title page before print, since author affiliations frequently changed between preprint and publication in this cohort (Carrasquilla in particular moved to D-Wave Systems around the Nature Physics publication date).

---

## [UNVERIFIABLE] ml-phase-detection

**Claim as researched:** Open question in the dossier: a Nature Physics 2019 'Machine learning phases of matter' landscape article.

**Correction:** I could not confirm any such article, and the dossier's own scepticism is warranted. 'Machine learning phases of matter' is unambiguously the Carrasquilla-Melko title (Nature Physics 13, 431-434, 2017). Nature Physics volume 15 (2019) does contain the three research papers noted (Rodriguez-Nieva & Scheurer p. 790; Rem et al. p. 917-920; Bohrdt et al. p. 921-924), but no accompanying editorial, News & Views, or landscape/overview article under that or a similar title surfaced in searching, and nature.com blocks direct fetching (HTTP 403 / auth redirect). Recommendation: cut this citation from the manuscript unless someone verifies it against the physical issue or a library copy. Do not paraphrase it into existence.

---

