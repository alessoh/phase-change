/**
 * The Ising Lab classifier.
 *
 * A small multilayer perceptron written from scratch over typed arrays: no ML
 * library, no WebGL, no network access. Architecture follows Carrasquilla and
 * Melko, "Machine learning phases of matter", Nature Physics 13, 431-434 (2017):
 * a fully connected network with ReLU hidden units and a two-unit softmax output
 * over {ordered, disordered}, trained with cross-entropy and plain minibatch
 * stochastic gradient descent on configurations labelled only by which side of
 * T_c they were sampled from.
 *
 * Conventions
 *   Output unit 0 is the "ordered" class   (Dataset label 0, T < T_c).
 *   Output unit 1 is the "disordered" class (Dataset label 1, T > T_c).
 *   Parameters live in Float32Array. Activations and gradients are accumulated
 *   in double precision, which costs nothing in JavaScript (all arithmetic is
 *   double anyway) and keeps the finite-difference gradient check meaningful.
 */

import { TC_EXACT } from '../types';
import type { Dataset, FeatureMode, MLPConfig, TrainStep, VerdictCurve } from '../types';

/* ------------------------------------------------------------------ */
/* Seeded pseudo-random numbers                                        */
/* ------------------------------------------------------------------ */

/** mulberry32: a small, fast, well-distributed 32-bit generator.
 *  Deterministic for a given seed, which is what reproducibility requires. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function next(): number {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard normal deviate by the Box-Muller transform. */
function gaussian(rand: () => number): number {
  let u = rand();
  if (u < 1e-12) u = 1e-12;
  const v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** In-place Fisher-Yates shuffle of the first `n` entries of `idx`. */
function shuffleInPlace(idx: Int32Array, n: number, rand: () => number): void {
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    const tmp = idx[i];
    idx[i] = idx[j];
    idx[j] = tmp;
  }
}

/* ------------------------------------------------------------------ */
/* Public option and result shapes                                     */
/* ------------------------------------------------------------------ */

/** Options for one epoch of stochastic gradient descent. */
export interface TrainOptions {
  /** Number of examples per gradient update. Must be at least 1. */
  batchSize: number;
  /** Fraction of the examples held out for validation, in [0, 1).
   *  The held-out set is chosen once, deterministically from the network seed,
   *  and stays fixed for as long as `n` and this fraction stay the same. */
  validationSplit: number;
}

/** Loss and accuracy of the current parameters on a set of examples. */
export interface Evaluation {
  /** Mean cross-entropy in nats. */
  loss: number;
  /** Fraction of examples whose argmax matches the label. */
  accuracy: number;
}

/** Analytic gradient of the mean cross-entropy, one array per layer.
 *  The arrays are copies, safe to keep across further calls. */
export interface Gradients {
  loss: number;
  /** gradWeights[l][o * dims[l] + i] = dLoss / dW_l[o][i]. */
  gradWeights: Float32Array[];
  /** gradBiases[l][o] = dLoss / db_l[o]. */
  gradBiases: Float32Array[];
}

/** The network input built from a `Dataset` by `buildFeatures`. */
export interface FeatureSet {
  /** Length count * inputDim, row-major, one example per row. */
  x: Float32Array;
  /** Length count, 0 = ordered, 1 = disordered. Permuted when mode is 'shuffled'. */
  y: Uint8Array;
  /** Length count, the sampling temperature of each example. Never permuted. */
  temperatures: Float32Array;
  inputDim: number;
  count: number;
  mode: FeatureMode;
}

/* ------------------------------------------------------------------ */
/* The network                                                         */
/* ------------------------------------------------------------------ */

const EPS_LOG = 1e-12;

export class MLP {
  readonly config: MLPConfig;
  /** Layer widths, from input to output: [inputDim, ...hidden, 2]. */
  readonly dims: number[];

  private readonly weights: Float32Array[];
  private readonly biases: Float32Array[];
  private readonly gradW: Float32Array[];
  private readonly gradB: Float32Array[];
  /** Pre-activations, one array per weight layer. */
  private readonly zs: Float64Array[];
  /** Activations; acts[0] is the input, acts[dims.length - 1] the softmax output. */
  private readonly acts: Float64Array[];
  /** Backpropagated error at each layer's pre-activation. */
  private readonly deltas: Float64Array[];

  private readonly rand: () => number;
  private epochsRun = 0;

  /** Cached, deterministic train/validation partition. */
  private splitKey = '';
  private trainIdx: Int32Array = new Int32Array(0);
  private valIdx: Int32Array = new Int32Array(0);

  constructor(config: MLPConfig) {
    if (!Number.isInteger(config.inputDim) || config.inputDim < 1) {
      throw new Error('MLP: inputDim must be a positive integer');
    }
    for (const h of config.hidden) {
      if (!Number.isInteger(h) || h < 1) throw new Error('MLP: hidden widths must be positive integers');
    }
    if (!(config.learningRate > 0)) throw new Error('MLP: learningRate must be positive');

    this.config = config;
    this.dims = [config.inputDim, ...config.hidden, 2];
    this.rand = mulberry32(config.seed);

    const layers = this.dims.length - 1;
    this.weights = new Array<Float32Array>(layers);
    this.biases = new Array<Float32Array>(layers);
    this.gradW = new Array<Float32Array>(layers);
    this.gradB = new Array<Float32Array>(layers);
    this.zs = new Array<Float64Array>(layers);
    this.deltas = new Array<Float64Array>(layers);
    this.acts = new Array<Float64Array>(layers + 1);

    this.acts[0] = new Float64Array(this.dims[0]);
    for (let l = 0; l < layers; l++) {
      const nIn = this.dims[l];
      const nOut = this.dims[l + 1];
      const w = new Float32Array(nOut * nIn);
      // He initialisation: variance 2 / fanIn, the right scale for ReLU units.
      const std = Math.sqrt(2 / nIn);
      for (let k = 0; k < w.length; k++) w[k] = gaussian(this.rand) * std;
      this.weights[l] = w;
      this.biases[l] = new Float32Array(nOut);
      this.gradW[l] = new Float32Array(nOut * nIn);
      this.gradB[l] = new Float32Array(nOut);
      this.zs[l] = new Float64Array(nOut);
      this.deltas[l] = new Float64Array(nOut);
      this.acts[l + 1] = new Float64Array(nOut);
    }
  }

  /** Number of epochs completed so far. */
  get epoch(): number {
    return this.epochsRun;
  }

  /** Live references to the weight matrices, row-major, shape dims[l+1] x dims[l].
   *  Exposed so that a gradient check can perturb a single parameter. */
  getWeights(): Float32Array[] {
    return this.weights;
  }

  /** Live references to the bias vectors. */
  getBiases(): Float32Array[] {
    return this.biases;
  }

  /* -------------------------------------------------------------- */
  /* Forward and backward                                            */
  /* -------------------------------------------------------------- */

  /** Forward pass for the example stored at x[offset .. offset + inputDim).
   *  Leaves the softmax output in this.acts[last]. */
  private forward(x: Float32Array, offset: number): void {
    const dims = this.dims;
    const layers = dims.length - 1;

    const input = this.acts[0];
    for (let i = 0; i < dims[0]; i++) input[i] = x[offset + i];

    for (let l = 0; l < layers; l++) {
      const nIn = dims[l];
      const nOut = dims[l + 1];
      const w = this.weights[l];
      const b = this.biases[l];
      const aIn = this.acts[l];
      const z = this.zs[l];
      for (let o = 0; o < nOut; o++) {
        const base = o * nIn;
        let s = b[o];
        for (let i = 0; i < nIn; i++) s += w[base + i] * aIn[i];
        z[o] = s;
      }
      const aOut = this.acts[l + 1];
      if (l < layers - 1) {
        for (let o = 0; o < nOut; o++) aOut[o] = z[o] > 0 ? z[o] : 0;
      } else {
        // Numerically stable softmax over the two class logits.
        let max = z[0];
        for (let o = 1; o < nOut; o++) if (z[o] > max) max = z[o];
        let sum = 0;
        for (let o = 0; o < nOut; o++) {
          const e = Math.exp(z[o] - max);
          aOut[o] = e;
          sum += e;
        }
        for (let o = 0; o < nOut; o++) aOut[o] /= sum;
      }
    }
  }

  /** Backward pass for one example, accumulating into gradW / gradB.
   *  Assumes `forward` has just been called for that example. */
  private backward(label: number): void {
    const dims = this.dims;
    const layers = dims.length - 1;
    const out = this.acts[layers];

    // d(cross-entropy)/d(logit) for a softmax output is p - onehot(label).
    const dLast = this.deltas[layers - 1];
    for (let o = 0; o < dims[layers]; o++) dLast[o] = out[o] - (o === label ? 1 : 0);

    for (let l = layers - 1; l >= 0; l--) {
      const nIn = dims[l];
      const nOut = dims[l + 1];
      const d = this.deltas[l];
      const aIn = this.acts[l];
      const gW = this.gradW[l];
      const gB = this.gradB[l];

      for (let o = 0; o < nOut; o++) {
        const dv = d[o];
        if (dv === 0) continue;
        gB[o] += dv;
        const base = o * nIn;
        for (let i = 0; i < nIn; i++) gW[base + i] += dv * aIn[i];
      }

      if (l > 0) {
        const w = this.weights[l];
        const dPrev = this.deltas[l - 1];
        dPrev.fill(0);
        for (let o = 0; o < nOut; o++) {
          const dv = d[o];
          if (dv === 0) continue;
          const base = o * nIn;
          for (let i = 0; i < nIn; i++) dPrev[i] += w[base + i] * dv;
        }
        // ReLU derivative of the previous layer.
        const zPrev = this.zs[l - 1];
        for (let i = 0; i < nIn; i++) if (!(zPrev[i] > 0)) dPrev[i] = 0;
      }
    }
  }

  private zeroGradients(): void {
    for (let l = 0; l < this.gradW.length; l++) {
      this.gradW[l].fill(0);
      this.gradB[l].fill(0);
    }
  }

  /** Apply one SGD step from the accumulated gradients, scaled by 1 / count. */
  private applyGradients(count: number): void {
    const step = this.config.learningRate / count;
    for (let l = 0; l < this.weights.length; l++) {
      const w = this.weights[l];
      const gW = this.gradW[l];
      for (let k = 0; k < w.length; k++) w[k] -= step * gW[k];
      const b = this.biases[l];
      const gB = this.gradB[l];
      for (let o = 0; o < b.length; o++) b[o] -= step * gB[o];
    }
  }

  private checkShapes(x: Float32Array, y: Uint8Array, n: number): void {
    if (!Number.isInteger(n) || n < 0) throw new Error('MLP: n must be a non-negative integer');
    if (x.length < n * this.dims[0]) {
      throw new Error(`MLP: x holds ${x.length} values, expected at least ${n * this.dims[0]}`);
    }
    if (y.length < n) throw new Error(`MLP: y holds ${y.length} labels, expected at least ${n}`);
  }

  /* -------------------------------------------------------------- */
  /* Train / evaluate / predict                                      */
  /* -------------------------------------------------------------- */

  /**
   * Partition indices into train and validation once, deterministically.
   * Derived from the network seed rather than from the training PRNG stream so
   * that the held-out set does not depend on how many epochs have been run.
   */
  private ensureSplit(n: number, validationSplit: number): void {
    const key = `${n}:${validationSplit}`;
    if (key === this.splitKey) return;
    if (!(validationSplit >= 0 && validationSplit < 1)) {
      throw new Error('MLP: validationSplit must be in [0, 1)');
    }
    const order = new Int32Array(n);
    for (let i = 0; i < n; i++) order[i] = i;
    shuffleInPlace(order, n, mulberry32((this.config.seed ^ 0x5bf03635) >>> 0));
    const nVal = Math.min(n, Math.floor(n * validationSplit));
    this.valIdx = order.slice(0, nVal);
    this.trainIdx = order.slice(nVal);
    this.splitKey = key;
  }

  /**
   * Run one epoch of minibatch stochastic gradient descent.
   *
   * `x` holds n examples of length inputDim, row-major; `y` holds n labels in
   * {0, 1}. Training indices are reshuffled every epoch with the seeded PRNG,
   * so two networks built from the same config and fed the same data produce
   * bit-identical histories.
   *
   * The reported loss and training accuracy are running averages over the
   * epoch, measured on each minibatch just before its update. Validation
   * accuracy is measured after the epoch on the held-out examples; when
   * `validationSplit` is 0 there are none, and the training accuracy is
   * reported in its place.
   */
  trainEpoch(x: Float32Array, y: Uint8Array, n: number, opts: TrainOptions): TrainStep {
    this.checkShapes(x, y, n);
    if (!Number.isInteger(opts.batchSize) || opts.batchSize < 1) {
      throw new Error('MLP: batchSize must be a positive integer');
    }
    this.ensureSplit(n, opts.validationSplit);

    const idx = this.trainIdx;
    const nTrain = idx.length;
    this.epochsRun += 1;

    if (nTrain === 0) {
      const valEmpty = this.evaluateIndices(x, y, this.valIdx);
      return {
        epoch: this.epochsRun,
        loss: valEmpty.loss,
        trainAccuracy: 0,
        validationAccuracy: valEmpty.accuracy,
      };
    }

    shuffleInPlace(idx, nTrain, this.rand);

    const inputDim = this.dims[0];
    const outIdx = this.dims.length - 1;
    let lossSum = 0;
    let correct = 0;

    for (let start = 0; start < nTrain; start += opts.batchSize) {
      const end = Math.min(start + opts.batchSize, nTrain);
      this.zeroGradients();
      for (let k = start; k < end; k++) {
        const sample = idx[k];
        const label = y[sample];
        this.forward(x, sample * inputDim);
        const out = this.acts[outIdx];
        const p = out[label];
        lossSum -= Math.log(p > EPS_LOG ? p : EPS_LOG);
        if ((out[1] > out[0] ? 1 : 0) === label) correct++;
        this.backward(label);
      }
      this.applyGradients(end - start);
    }

    const trainAccuracy = correct / nTrain;
    const val = this.valIdx.length > 0 ? this.evaluateIndices(x, y, this.valIdx) : null;

    return {
      epoch: this.epochsRun,
      loss: lossSum / nTrain,
      trainAccuracy,
      validationAccuracy: val === null ? trainAccuracy : val.accuracy,
    };
  }

  /** Mean cross-entropy and accuracy of the current parameters on n examples. */
  evaluate(x: Float32Array, y: Uint8Array, n: number): Evaluation {
    this.checkShapes(x, y, n);
    if (n === 0) return { loss: 0, accuracy: 0 };
    const inputDim = this.dims[0];
    const outIdx = this.dims.length - 1;
    let lossSum = 0;
    let correct = 0;
    for (let s = 0; s < n; s++) {
      const label = y[s];
      this.forward(x, s * inputDim);
      const out = this.acts[outIdx];
      const p = out[label];
      lossSum -= Math.log(p > EPS_LOG ? p : EPS_LOG);
      if ((out[1] > out[0] ? 1 : 0) === label) correct++;
    }
    return { loss: lossSum / n, accuracy: correct / n };
  }

  /** Same as `evaluate` but restricted to an explicit index list. */
  private evaluateIndices(x: Float32Array, y: Uint8Array, list: Int32Array): Evaluation {
    if (list.length === 0) return { loss: 0, accuracy: 0 };
    const inputDim = this.dims[0];
    const outIdx = this.dims.length - 1;
    let lossSum = 0;
    let correct = 0;
    for (let k = 0; k < list.length; k++) {
      const sample = list[k];
      const label = y[sample];
      this.forward(x, sample * inputDim);
      const out = this.acts[outIdx];
      const p = out[label];
      lossSum -= Math.log(p > EPS_LOG ? p : EPS_LOG);
      if ((out[1] > out[0] ? 1 : 0) === label) correct++;
    }
    return { loss: lossSum / list.length, accuracy: correct / list.length };
  }

  /** Full softmax output: length 2n, [pOrdered, pDisordered] for each example. */
  predict(x: Float32Array, n: number): Float32Array {
    if (!Number.isInteger(n) || n < 0) throw new Error('MLP: n must be a non-negative integer');
    if (x.length < n * this.dims[0]) {
      throw new Error(`MLP: x holds ${x.length} values, expected at least ${n * this.dims[0]}`);
    }
    const inputDim = this.dims[0];
    const outIdx = this.dims.length - 1;
    const result = new Float32Array(2 * n);
    for (let s = 0; s < n; s++) {
      this.forward(x, s * inputDim);
      const out = this.acts[outIdx];
      result[2 * s] = out[0];
      result[2 * s + 1] = out[1];
    }
    return result;
  }

  /** Probability assigned to the disordered class, one value per example. */
  predictProbabilities(x: Float32Array, n: number): Float32Array {
    if (!Number.isInteger(n) || n < 0) throw new Error('MLP: n must be a non-negative integer');
    if (x.length < n * this.dims[0]) {
      throw new Error(`MLP: x holds ${x.length} values, expected at least ${n * this.dims[0]}`);
    }
    const inputDim = this.dims[0];
    const outIdx = this.dims.length - 1;
    const result = new Float32Array(n);
    for (let s = 0; s < n; s++) {
      this.forward(x, s * inputDim);
      result[s] = this.acts[outIdx][1];
    }
    return result;
  }

  /**
   * Mean cross-entropy over n examples together with its exact analytic
   * gradient with respect to every parameter. Used internally by `trainEpoch`
   * and exposed so that the gradients can be checked against finite differences.
   * The returned arrays are copies and do not alias the network's buffers.
   */
  computeLossAndGradients(x: Float32Array, y: Uint8Array, n: number): Gradients {
    this.checkShapes(x, y, n);
    this.zeroGradients();
    const inputDim = this.dims[0];
    const outIdx = this.dims.length - 1;
    let lossSum = 0;
    for (let s = 0; s < n; s++) {
      const label = y[s];
      this.forward(x, s * inputDim);
      const p = this.acts[outIdx][label];
      lossSum -= Math.log(p > EPS_LOG ? p : EPS_LOG);
      this.backward(label);
    }
    const scale = n > 0 ? 1 / n : 0;
    const gradWeights = this.gradW.map((g) => {
      const c = new Float32Array(g.length);
      for (let k = 0; k < g.length; k++) c[k] = g[k] * scale;
      return c;
    });
    const gradBiases = this.gradB.map((g) => {
      const c = new Float32Array(g.length);
      for (let k = 0; k < g.length; k++) c[k] = g[k] * scale;
      return c;
    });
    return { loss: lossSum * scale, gradWeights, gradBiases };
  }
}

/* ------------------------------------------------------------------ */
/* Feature construction                                                */
/* ------------------------------------------------------------------ */

/** Default seed for the label permutation used by the 'shuffled' control. */
const SHUFFLE_SEED = 0x15173a1b;

/**
 * Turn a harvested `Dataset` into network input.
 *
 *   'spins'         the raw configuration, unchanged; inputDim = size * size.
 *                   This is the honest experiment.
 *   'magnetization' one feature per configuration, |m| = |sum_i s_i| / N;
 *                   inputDim = 1. Succeeding here says nothing about whether a
 *                   network can find the order parameter, only that we handed
 *                   it one.
 *   'shuffled'      the raw configuration with the LABELS randomly permuted.
 *                   The deliberate falsification control: temperature and label
 *                   are now independent, so a working pipeline must fail to
 *                   learn and must produce no crossing in `extractVerdict`.
 *
 * Temperatures are never permuted, so the verdict curve is still indexed by the
 * true sampling temperature of each configuration.
 *
 * For 'spins' and 'shuffled' the returned `x` aliases `dataset.configurations`
 * rather than copying it; a harvest of thousands of L x L configurations is
 * large and is only ever read. Do not mutate it.
 */
export function buildFeatures(dataset: Dataset, mode: FeatureMode, seed: number = SHUFFLE_SEED): FeatureSet {
  const count = dataset.count;
  const spinsPerConfig = dataset.size * dataset.size;
  if (dataset.configurations.length < count * spinsPerConfig) {
    throw new Error('buildFeatures: dataset.configurations is shorter than count * size * size');
  }
  if (dataset.labels.length < count) throw new Error('buildFeatures: dataset.labels is shorter than count');
  if (dataset.temperatures.length < count) {
    throw new Error('buildFeatures: dataset.temperatures is shorter than count');
  }

  const temperatures =
    dataset.temperatures.length === count ? dataset.temperatures : dataset.temperatures.subarray(0, count);

  if (mode === 'magnetization') {
    const x = new Float32Array(count);
    for (let k = 0; k < count; k++) {
      let sum = 0;
      const base = k * spinsPerConfig;
      for (let i = 0; i < spinsPerConfig; i++) sum += dataset.configurations[base + i];
      x[k] = Math.abs(sum) / spinsPerConfig;
    }
    return { x, y: dataset.labels.slice(0, count), temperatures, inputDim: 1, count, mode };
  }

  const x =
    dataset.configurations.length === count * spinsPerConfig
      ? dataset.configurations
      : dataset.configurations.subarray(0, count * spinsPerConfig);

  const y = dataset.labels.slice(0, count);
  if (mode === 'shuffled') {
    const rand = mulberry32(seed);
    for (let i = count - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      const tmp = y[i];
      y[i] = y[j];
      y[j] = tmp;
    }
  }

  return { x, y, temperatures, inputDim: spinsPerConfig, count, mode };
}

/* ------------------------------------------------------------------ */
/* Reading the critical temperature off the network's output           */
/* ------------------------------------------------------------------ */

/**
 * Average the network's disordered-class probability over every configuration
 * sampled at each temperature, then locate where that averaged curve crosses
 * one half.
 *
 * The crossing is the point at which the network is maximally undecided, and in
 * Carrasquilla and Melko it lands on the Onsager value for T_c. It is found by
 * linear interpolation between the two bracketing temperatures:
 *
 *     T_c = T_i + (0.5 - p_i) * (T_{i+1} - T_i) / (p_{i+1} - p_i)
 *
 * Temperatures are scanned in ascending order and the first crossing is
 * returned. `estimatedTc` and `errorVsExact` are null when the curve never
 * reaches one half, which is the expected outcome for shuffled labels.
 */
export function extractVerdict(temperatures: Float32Array, pDisordered: Float32Array): VerdictCurve {
  const n = Math.min(temperatures.length, pDisordered.length);

  // Group configurations by their exact sampling temperature.
  const sums = new Map<number, number>();
  const counts = new Map<number, number>();
  for (let k = 0; k < n; k++) {
    const t = temperatures[k];
    sums.set(t, (sums.get(t) ?? 0) + pDisordered[k]);
    counts.set(t, (counts.get(t) ?? 0) + 1);
  }

  const uniqueT = Array.from(sums.keys()).sort((a, b) => a - b);
  const tOut = new Float32Array(uniqueT.length);
  const pOut = new Float32Array(uniqueT.length);
  for (let i = 0; i < uniqueT.length; i++) {
    const t = uniqueT[i];
    tOut[i] = t;
    pOut[i] = sums.get(t)! / counts.get(t)!;
  }

  let estimatedTc: number | null = null;
  for (let i = 0; i < pOut.length; i++) {
    if (pOut[i] === 0.5) {
      estimatedTc = tOut[i];
      break;
    }
    if (i + 1 >= pOut.length) break;
    const a = pOut[i] - 0.5;
    const b = pOut[i + 1] - 0.5;
    if (a * b < 0) {
      estimatedTc = tOut[i] + (a * (tOut[i + 1] - tOut[i])) / (pOut[i] - pOut[i + 1]);
      break;
    }
  }

  return {
    temperatures: tOut,
    pDisordered: pOut,
    estimatedTc,
    errorVsExact: estimatedTc === null ? null : estimatedTc - TC_EXACT,
  };
}
