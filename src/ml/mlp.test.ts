/**
 * Tests for the in-browser classifier.
 *
 * These are not decoration. The Ising Lab claims to recover a number that
 * Onsager derived in 1944, so the machinery that produces that number has to be
 * checkable in isolation: the gradients must be the real gradients, the softmax
 * must be a probability distribution, the same seed must give the same run, and
 * the falsification control must actually falsify.
 */

import { describe, expect, it } from 'vitest';
import { TC_EXACT } from '../types';
import type { Dataset } from '../types';
import { MLP, buildFeatures, extractVerdict, mulberry32 } from './mlp';
import type { TrainOptions } from './mlp';

/* ------------------------------------------------------------------ */
/* Fixtures                                                            */
/* ------------------------------------------------------------------ */

/** A linearly separable two-dimensional problem with a clear margin. */
function separableProblem(n: number, seed: number): { x: Float32Array; y: Uint8Array } {
  const rand = mulberry32(seed);
  const x = new Float32Array(2 * n);
  const y = new Uint8Array(n);
  let k = 0;
  while (k < n) {
    const a = rand() * 2 - 1;
    const b = rand() * 2 - 1;
    const score = 0.7 * a - 0.4 * b + 0.1;
    if (Math.abs(score) < 0.05) continue; // keep a margin so the target is unambiguous
    x[2 * k] = a;
    x[2 * k + 1] = b;
    y[k] = score > 0 ? 1 : 0;
    k++;
  }
  return { x, y };
}

/**
 * A stand-in for a harvested Dataset: 6 x 6 lattices, half of them nearly
 * fully aligned and labelled ordered, half of them random and labelled
 * disordered. Trivially learnable from the raw spins, which is what makes it a
 * fair test of whether shuffling the labels destroys the signal.
 */
function syntheticDataset(count: number, seed: number): Dataset {
  const size = 6;
  const dim = size * size;
  const rand = mulberry32(seed);
  const configurations = new Float32Array(count * dim);
  const labels = new Uint8Array(count);
  const temperatures = new Float32Array(count);
  for (let k = 0; k < count; k++) {
    const label = k % 2; // exactly balanced
    labels[k] = label;
    temperatures[k] = label === 0 ? 1.5 : 3.5;
    const base = k * dim;
    for (let i = 0; i < dim; i++) {
      if (label === 0) {
        configurations[base + i] = rand() < 0.05 ? -1 : 1; // ordered, a few flipped spins
      } else {
        configurations[base + i] = rand() < 0.5 ? -1 : 1; // disordered
      }
    }
  }
  return { size, count, configurations, labels, temperatures };
}

const OPTS: TrainOptions = { batchSize: 16, validationSplit: 0.2 };

/* ------------------------------------------------------------------ */

describe('MLP training', () => {
  it('learns a linearly separable problem to better than 95 percent', () => {
    const n = 600;
    const { x, y } = separableProblem(n, 20240517);
    const net = new MLP({ inputDim: 2, hidden: [8], learningRate: 0.1, seed: 7 });

    let last = net.trainEpoch(x, y, n, OPTS);
    for (let e = 1; e < 60; e++) last = net.trainEpoch(x, y, n, OPTS);

    expect(last.epoch).toBe(60);
    expect(last.trainAccuracy).toBeGreaterThan(0.95);
    expect(last.validationAccuracy).toBeGreaterThan(0.95);
    expect(net.evaluate(x, y, n).accuracy).toBeGreaterThan(0.95);
    expect(last.loss).toBeLessThan(0.3);
  });

  it('drives the loss down monotonically enough to show the descent is working', () => {
    const n = 400;
    const { x, y } = separableProblem(n, 99);
    const net = new MLP({ inputDim: 2, hidden: [8], learningRate: 0.1, seed: 3 });
    const first = net.trainEpoch(x, y, n, OPTS);
    let last = first;
    for (let e = 1; e < 40; e++) last = net.trainEpoch(x, y, n, OPTS);
    expect(last.loss).toBeLessThan(first.loss);
  });
});

describe('softmax output', () => {
  it('is a probability distribution: strictly positive and summing to one', () => {
    const n = 50;
    const { x, y } = separableProblem(n, 5150);
    const net = new MLP({ inputDim: 2, hidden: [6, 4], learningRate: 0.1, seed: 11 });
    net.trainEpoch(x, y, n, OPTS);

    const both = net.predict(x, n);
    expect(both.length).toBe(2 * n);
    for (let s = 0; s < n; s++) {
      const p0 = both[2 * s];
      const p1 = both[2 * s + 1];
      expect(p0).toBeGreaterThan(0);
      expect(p1).toBeGreaterThan(0);
      expect(p0 + p1).toBeCloseTo(1, 5);
    }

    const pDis = net.predictProbabilities(x, n);
    expect(pDis.length).toBe(n);
    for (let s = 0; s < n; s++) {
      expect(pDis[s]).toBeGreaterThan(0);
      expect(pDis[s]).toBeLessThan(1);
      // predictProbabilities must report the disordered class, unit 1.
      expect(pDis[s]).toBeCloseTo(both[2 * s + 1], 6);
    }
  });
});

describe('determinism', () => {
  it('reproduces an identical run from an identical seed', () => {
    const n = 300;
    const { x, y } = separableProblem(n, 424242);
    const a = new MLP({ inputDim: 2, hidden: [8, 5], learningRate: 0.08, seed: 1234 });
    const b = new MLP({ inputDim: 2, hidden: [8, 5], learningRate: 0.08, seed: 1234 });

    for (let e = 0; e < 12; e++) {
      const sa = a.trainEpoch(x, y, n, OPTS);
      const sb = b.trainEpoch(x, y, n, OPTS);
      expect(sa).toEqual(sb);
    }

    const pa = a.predictProbabilities(x, n);
    const pb = b.predictProbabilities(x, n);
    for (let s = 0; s < n; s++) expect(pa[s]).toBe(pb[s]);
  });

  it('gives a different run from a different seed', () => {
    const n = 300;
    const { x } = separableProblem(n, 424242);
    const a = new MLP({ inputDim: 2, hidden: [8], learningRate: 0.08, seed: 1 });
    const b = new MLP({ inputDim: 2, hidden: [8], learningRate: 0.08, seed: 2 });
    const pa = a.predictProbabilities(x, n);
    const pb = b.predictProbabilities(x, n);
    let differs = 0;
    for (let s = 0; s < n; s++) if (pa[s] !== pb[s]) differs++;
    expect(differs).toBeGreaterThan(0);
  });
});

describe('extractVerdict', () => {
  it('finds the crossing exactly on a hand-constructed monotonic curve', () => {
    // Crossing sits between T = 2.0 (p = 0.3) and T = 3.0 (p = 0.7),
    // so linear interpolation must land on T = 2.5.
    const t = new Float32Array([1, 2, 3, 4]);
    const p = new Float32Array([0.1, 0.3, 0.7, 0.9]);
    const v = extractVerdict(t, p);

    expect(v.estimatedTc).not.toBeNull();
    expect(v.estimatedTc as number).toBeCloseTo(2.5, 6);
    expect(v.errorVsExact as number).toBeCloseTo(2.5 - TC_EXACT, 6);
    expect(Array.from(v.temperatures)).toEqual([1, 2, 3, 4]);
  });

  it('averages over every configuration sharing a temperature', () => {
    // Two configurations per temperature; the means are 0.2, 0.4, 0.6.
    const t = new Float32Array([1, 1, 2, 2, 3, 3]);
    const p = new Float32Array([0.1, 0.3, 0.35, 0.45, 0.5, 0.7]);
    const v = extractVerdict(t, p);

    expect(v.temperatures.length).toBe(3);
    expect(Array.from(v.temperatures)).toEqual([1, 2, 3]);
    expect(v.pDisordered[0]).toBeCloseTo(0.2, 6);
    expect(v.pDisordered[1]).toBeCloseTo(0.4, 6);
    expect(v.pDisordered[2]).toBeCloseTo(0.6, 6);
    // Crossing between T = 2 (0.4) and T = 3 (0.6) is at T = 2.5.
    expect(v.estimatedTc as number).toBeCloseTo(2.5, 5);
  });

  it('sorts unsorted input by temperature before looking for a crossing', () => {
    const t = new Float32Array([4, 1, 3, 2]);
    const p = new Float32Array([0.9, 0.1, 0.7, 0.3]);
    const v = extractVerdict(t, p);
    expect(Array.from(v.temperatures)).toEqual([1, 2, 3, 4]);
    expect(v.estimatedTc as number).toBeCloseTo(2.5, 6);
  });

  it('returns the temperature itself when a point sits exactly on one half', () => {
    const t = new Float32Array([1, 2, 3]);
    const p = new Float32Array([0.2, 0.5, 0.8]);
    const v = extractVerdict(t, p);
    expect(v.estimatedTc).toBe(2);
  });

  it('returns null when the curve never crosses one half', () => {
    const below = extractVerdict(new Float32Array([1, 2, 3, 4]), new Float32Array([0.1, 0.2, 0.3, 0.4]));
    expect(below.estimatedTc).toBeNull();
    expect(below.errorVsExact).toBeNull();

    const above = extractVerdict(new Float32Array([1, 2, 3, 4]), new Float32Array([0.6, 0.7, 0.8, 0.9]));
    expect(above.estimatedTc).toBeNull();
    expect(above.errorVsExact).toBeNull();

    // A flat, undecided curve pinned just under one half: no crossing either.
    const flat = extractVerdict(new Float32Array([1, 2, 3]), new Float32Array([0.499, 0.499, 0.499]));
    expect(flat.estimatedTc).toBeNull();
  });

  it('recovers the Onsager value from a curve centred on it', () => {
    const t = new Float32Array([TC_EXACT - 0.5, TC_EXACT + 0.5]);
    const p = new Float32Array([0.25, 0.75]);
    const v = extractVerdict(t, p);
    expect(v.estimatedTc as number).toBeCloseTo(TC_EXACT, 4);
    expect(Math.abs(v.errorVsExact as number)).toBeLessThan(1e-4);
  });
});

describe('buildFeatures', () => {
  it("passes raw spins through unchanged for 'spins'", () => {
    const ds = syntheticDataset(20, 8);
    const f = buildFeatures(ds, 'spins');
    expect(f.inputDim).toBe(ds.size * ds.size);
    expect(f.count).toBe(ds.count);
    expect(Array.from(f.x)).toEqual(Array.from(ds.configurations));
    expect(Array.from(f.y)).toEqual(Array.from(ds.labels));
  });

  it("reduces each configuration to its absolute magnetization for 'magnetization'", () => {
    const ds = syntheticDataset(20, 9);
    const f = buildFeatures(ds, 'magnetization');
    const dim = ds.size * ds.size;
    expect(f.inputDim).toBe(1);
    expect(f.x.length).toBe(ds.count);
    for (let k = 0; k < ds.count; k++) {
      let sum = 0;
      for (let i = 0; i < dim; i++) sum += ds.configurations[k * dim + i];
      expect(f.x[k]).toBeCloseTo(Math.abs(sum) / dim, 6);
      expect(f.x[k]).toBeGreaterThanOrEqual(0);
      expect(f.x[k]).toBeLessThanOrEqual(1);
    }
  });

  it("permutes the labels for 'shuffled' while leaving spins and temperatures alone", () => {
    const ds = syntheticDataset(400, 10);
    const f = buildFeatures(ds, 'shuffled');

    // The spins are untouched.
    expect(f.x.length).toBe(ds.count * ds.size * ds.size);
    let spinMismatches = 0;
    for (let i = 0; i < f.x.length; i++) if (f.x[i] !== ds.configurations[i]) spinMismatches++;
    expect(spinMismatches).toBe(0);

    // The temperatures are untouched: the verdict curve still uses the true T.
    let tempMismatches = 0;
    for (let k = 0; k < ds.count; k++) if (f.temperatures[k] !== ds.temperatures[k]) tempMismatches++;
    expect(tempMismatches).toBe(0);

    // The labels are a permutation: same multiset, different arrangement.
    const originalOnes = ds.labels.reduce((a, b) => a + b, 0);
    const shuffledOnes = f.y.reduce((a: number, b: number) => a + b, 0);
    expect(shuffledOnes).toBe(originalOnes);

    let moved = 0;
    for (let k = 0; k < ds.count; k++) if (f.y[k] !== ds.labels[k]) moved++;
    expect(moved).toBeGreaterThan(ds.count * 0.2);

    // And the original dataset was not mutated in the process.
    for (let k = 0; k < ds.count; k++) expect(ds.labels[k]).toBe(k % 2);
  });
});

describe('the shuffled-label falsification control', () => {
  it('learns the honest task but stays near chance once the labels are permuted', () => {
    const ds = syntheticDataset(400, 2718);
    const opts: TrainOptions = { batchSize: 20, validationSplit: 0.25 };

    const honest = buildFeatures(ds, 'spins');
    const honestNet = new MLP({ inputDim: honest.inputDim, hidden: [8], learningRate: 0.1, seed: 31 });
    let honestStep = honestNet.trainEpoch(honest.x, honest.y, honest.count, opts);
    for (let e = 1; e < 40; e++) {
      honestStep = honestNet.trainEpoch(honest.x, honest.y, honest.count, opts);
    }
    expect(honestStep.validationAccuracy).toBeGreaterThan(0.95);

    const control = buildFeatures(ds, 'shuffled');
    const controlNet = new MLP({ inputDim: control.inputDim, hidden: [8], learningRate: 0.1, seed: 31 });
    const tail: number[] = [];
    for (let e = 0; e < 40; e++) {
      const step = controlNet.trainEpoch(control.x, control.y, control.count, opts);
      if (e >= 35) tail.push(step.validationAccuracy);
    }
    const meanTail = tail.reduce((a, b) => a + b, 0) / tail.length;

    // Chance is 0.5 on a balanced set. Anything much above it would mean the
    // control is leaking signal and the whole experiment is worthless.
    expect(meanTail).toBeGreaterThan(0.3);
    expect(meanTail).toBeLessThan(0.65);
  });

  it('produces no crossing, hence no estimated Tc, on shuffled labels', () => {
    // Held at a constant, undecided output: the honest signature of failure.
    const temps = new Float32Array([2.0, 2.2, 2.4, 2.6]);
    const p = new Float32Array([0.51, 0.52, 0.51, 0.52]);
    expect(extractVerdict(temps, p).estimatedTc).toBeNull();
  });
});

describe('gradient check', () => {
  it('matches finite differences on a tiny network', () => {
    const inputDim = 3;
    const n = 5;
    const rand = mulberry32(777);
    const x = new Float32Array(inputDim * n);
    for (let k = 0; k < x.length; k++) x[k] = rand() * 2 - 1;
    const y = new Uint8Array(n);
    for (let k = 0; k < n; k++) y[k] = k % 2;

    const net = new MLP({ inputDim, hidden: [4, 3], learningRate: 0.05, seed: 4242 });
    const analytic = net.computeLossAndGradients(x, y, n);
    const weights = net.getWeights();
    const biases = net.getBiases();

    const eps = 1e-4;
    /** Central difference against the parameter values actually stored in
     *  Float32Array, so that float32 rounding of the perturbation cancels. */
    const numeric = (store: Float32Array, index: number): number => {
      const original = store[index];
      store[index] = original + eps;
      const plusValue = store[index];
      const lossPlus = net.computeLossAndGradients(x, y, n).loss;
      store[index] = original - eps;
      const minusValue = store[index];
      const lossMinus = net.computeLossAndGradients(x, y, n).loss;
      store[index] = original;
      return (lossPlus - lossMinus) / (plusValue - minusValue);
    };

    let checked = 0;
    for (let l = 0; l < weights.length; l++) {
      const g = analytic.gradWeights[l];
      for (let k = 0; k < g.length && checked < 24; k++) {
        if (Math.abs(g[k]) < 1e-6) continue; // a dead ReLU unit checks nothing
        const num = numeric(weights[l], k);
        expect(Math.abs(num - g[k])).toBeLessThan(1e-3 * Math.max(1, Math.abs(g[k])));
        checked++;
      }
    }
    expect(checked).toBeGreaterThanOrEqual(6);

    let biasChecked = 0;
    for (let l = 0; l < biases.length; l++) {
      const g = analytic.gradBiases[l];
      for (let o = 0; o < g.length; o++) {
        if (Math.abs(g[o]) < 1e-6) continue;
        const num = numeric(biases[l], o);
        expect(Math.abs(num - g[o])).toBeLessThan(1e-3 * Math.max(1, Math.abs(g[o])));
        biasChecked++;
      }
    }
    expect(biasChecked).toBeGreaterThanOrEqual(3);

    // Calling computeLossAndGradients must leave the parameters untouched.
    const again = net.computeLossAndGradients(x, y, n);
    expect(again.loss).toBeCloseTo(analytic.loss, 12);
  });
});
