/**
 * Shared interface contract for The Ising Lab.
 *
 * Every module in this project codes against these types. Do not change a
 * signature here without updating every consumer.
 *
 * Physics conventions used throughout:
 *   Hamiltonian  H = -J * sum_<ij> s_i s_j   with J = 1 and no external field.
 *   Boltzmann constant k_B = 1, so temperature is measured in units of J/k_B.
 *   Spins take values +1 and -1. The lattice is L x L with periodic boundaries.
 */

/** Exact critical temperature of the 2D square-lattice Ising model.
 *  Onsager (1944): sinh(2J/kT_c) = 1, hence kT_c/J = 2 / ln(1 + sqrt(2)).
 *  Numerically 2.269185314213022. */
export const TC_EXACT = 2 / Math.log(1 + Math.SQRT2);

/** Monte Carlo update rule. Metropolis shows critical slowing down near T_c,
 *  which is physically instructive; Wolff cluster updates largely remove it. */
export type Algorithm = 'metropolis' | 'wolff';

/** What the classifier is fed. This is the single most important control in the
 *  application: 'spins' is the honest experiment, 'magnetization' demonstrates
 *  that a classifier can succeed on a hand-engineered feature rather than on the
 *  physics, and 'shuffled' is the deliberate falsification that must fail. */
export type FeatureMode = 'spins' | 'magnetization' | 'shuffled';

/** Thermodynamic observables accumulated over a set of samples at one temperature.
 *  All extensive quantities are reported per spin so that different lattice sizes
 *  are directly comparable. */
export interface Observables {
  temperature: number;
  /** <E>/N */
  energyPerSpin: number;
  /** <m> where m = (1/N) sum_i s_i, signed. Averages to zero on a finite lattice
   *  below T_c because the system tunnels between the two ordered states. */
  magnetizationPerSpin: number;
  /** <|m|>, the finite-size proxy for the spontaneous magnetization order parameter. */
  absMagnetizationPerSpin: number;
  /** C = (N / T^2) * (<e^2> - <e>^2), where e is energy per spin. */
  specificHeat: number;
  /** chi = (N / T) * (<m^2> - <|m|>^2). */
  susceptibility: number;
  /** Binder cumulant U = 1 - <m^4> / (3 <m^2>^2). Curves for different L cross at T_c. */
  binderCumulant: number;
  /** Number of measurements contributing to these averages. */
  samples: number;
}

/** A live snapshot pushed from the simulation worker to the renderer. */
export interface Frame {
  size: number;
  /** Length size*size, row-major, values +1 or -1. */
  spins: Int8Array;
  temperature: number;
  energyPerSpin: number;
  magnetizationPerSpin: number;
  /** Sweeps completed since the last reset. */
  sweep: number;
}

/** A labelled training set harvested across a temperature range.
 *  Labels are 0 for T below T_c and 1 for T above, which is the only supervision
 *  the network receives. It is never told the value of T_c. */
export interface Dataset {
  size: number;
  count: number;
  /** Length count * (size*size), values +1 or -1, one configuration per row. */
  configurations: Float32Array;
  /** Length count, 0 = ordered side, 1 = disordered side. */
  labels: Uint8Array;
  /** Length count, the temperature each configuration was sampled at. */
  temperatures: Float32Array;
}

/* ------------------------------------------------------------------ */
/* Worker protocol                                                     */
/* ------------------------------------------------------------------ */

export type WorkerRequest =
  | { type: 'init'; size: number; temperature: number; algorithm: Algorithm; seed: number }
  | { type: 'setTemperature'; temperature: number }
  | { type: 'setAlgorithm'; algorithm: Algorithm }
  | { type: 'setSize'; size: number }
  | { type: 'play' }
  | { type: 'pause' }
  | { type: 'randomize' }
  /** Sweep temperature from tMin to tMax in tSteps points, measuring observables. */
  | { type: 'measure'; tMin: number; tMax: number; tSteps: number; equilibrationSweeps: number; measurementSweeps: number }
  /** Harvest a labelled training set for the classifier. */
  | { type: 'harvest'; tMin: number; tMax: number; tSteps: number; perTemperature: number; equilibrationSweeps: number };

export type WorkerResponse =
  | { type: 'ready' }
  | { type: 'frame'; frame: Frame }
  | { type: 'measureProgress'; done: number; total: number; point: Observables }
  | { type: 'measureDone'; curve: Observables[] }
  | { type: 'harvestProgress'; done: number; total: number }
  | { type: 'harvestDone'; dataset: Dataset }
  | { type: 'error'; message: string };

/* ------------------------------------------------------------------ */
/* Classifier                                                          */
/* ------------------------------------------------------------------ */

export interface MLPConfig {
  inputDim: number;
  /** Hidden layer widths. A single hidden layer is enough to reproduce the result. */
  hidden: number[];
  learningRate: number;
  seed: number;
}

/** One epoch of training history. */
export interface TrainStep {
  epoch: number;
  loss: number;
  trainAccuracy: number;
  validationAccuracy: number;
}

/** The network's averaged output as a function of temperature, and the crossing
 *  point extracted from it. This is the panel that reproduces the published result. */
export interface VerdictCurve {
  temperatures: Float32Array;
  /** Mean probability assigned to the "disordered" class at each temperature. */
  pDisordered: Float32Array;
  /** Temperature at which pDisordered crosses 0.5, found by linear interpolation.
   *  Null when the curve never crosses, which is the expected outcome for shuffled labels. */
  estimatedTc: number | null;
  /** Signed difference between estimatedTc and TC_EXACT. */
  errorVsExact: number | null;
}
