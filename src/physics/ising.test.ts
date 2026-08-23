/**
 * Physics validation for the Ising Monte Carlo core.
 *
 * The centrepiece is the exact-enumeration test: for 2x2 and 3x3 lattices the
 * partition function can be summed over all 2^(L*L) configurations, giving the
 * exact <E>/N, <|m|>, C, chi and Binder cumulant. A long Monte Carlo run must
 * reproduce those numbers. That is what proves the sampler obeys detailed
 * balance rather than merely returning plausible-looking numbers.
 *
 * The enumeration below deliberately re-derives the energy from the
 * Hamiltonian definition (every site contributes its right and its down bond
 * under periodic boundaries) instead of calling into ising.ts, so it is an
 * independent check of the convention documented there.
 *
 * Every run is seeded, so these tests are deterministic: a tolerance that
 * passes once passes always.
 */

import { describe, it, expect } from 'vitest';
import { TC_EXACT, type Algorithm, type Observables } from '../types';
import { IsingLattice, ObservableAccumulator, mulberry32 } from './ising';

/* ------------------------------------------------------------------ */
/* Exact enumeration                                                   */
/* ------------------------------------------------------------------ */

/** Brute-force thermodynamics: sum over all 2^N spin configurations. */
function exactObservables(L: number, T: number): Observables {
  const N = L * L;
  const s = new Int8Array(N);
  const total = 1 << N;

  let Z = 0;
  let sE = 0;
  let sE2 = 0;
  let sM = 0;
  let sAbsM = 0;
  let sM2 = 0;
  let sM4 = 0;

  for (let c = 0; c < total; c++) {
    for (let i = 0; i < N; i++) s[i] = ((c >> i) & 1) === 1 ? 1 : -1;

    let energy = 0;
    let mag = 0;
    for (let y = 0; y < L; y++) {
      for (let x = 0; x < L; x++) {
        const i = y * L + x;
        const rightNeighbour = y * L + ((x + 1) % L);
        const downNeighbour = ((y + 1) % L) * L + x;
        energy -= s[i] * (s[rightNeighbour] + s[downNeighbour]);
        mag += s[i];
      }
    }

    const w = Math.exp(-energy / T);
    const e = energy / N;
    const m = mag / N;
    const m2 = m * m;
    Z += w;
    sE += e * w;
    sE2 += e * e * w;
    sM += m * w;
    sAbsM += Math.abs(m) * w;
    sM2 += m2 * w;
    sM4 += m2 * m2 * w;
  }

  const meanE = sE / Z;
  const meanE2 = sE2 / Z;
  const meanAbsM = sAbsM / Z;
  const meanM2 = sM2 / Z;
  const meanM4 = sM4 / Z;

  return {
    temperature: T,
    energyPerSpin: meanE,
    magnetizationPerSpin: sM / Z,
    absMagnetizationPerSpin: meanAbsM,
    specificHeat: (N / (T * T)) * (meanE2 - meanE * meanE),
    susceptibility: (N / T) * (meanM2 - meanAbsM * meanAbsM),
    binderCumulant: 1 - meanM4 / (3 * meanM2 * meanM2),
    samples: total,
  };
}

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

function measure(
  L: number,
  T: number,
  algorithm: Algorithm,
  seed: number,
  equilibration: number,
  measurement: number,
): Observables {
  const lattice = new IsingLattice(L, seed);
  const acc = new ObservableAccumulator(T, L * L);
  for (let i = 0; i < equilibration; i++) lattice.sweep(T, algorithm);
  for (let i = 0; i < measurement; i++) {
    lattice.sweep(T, algorithm);
    acc.sample(lattice);
  }
  return acc.result();
}

const EQUILIBRATION = 2000;
const MEASUREMENT = 120000;

/* ------------------------------------------------------------------ */
/* Tests                                                               */
/* ------------------------------------------------------------------ */

describe('Onsager critical temperature', () => {
  it('TC_EXACT is 2 / ln(1 + sqrt(2))', () => {
    expect(TC_EXACT).toBeCloseTo(2.269185314213022, 12);
  });

  it('satisfies the Onsager condition sinh(2J/kTc) = 1', () => {
    expect(Math.sinh(2 / TC_EXACT)).toBeCloseTo(1, 12);
  });
});

describe('seeded PRNG', () => {
  it('produces uniforms in [0, 1) and repeats exactly for the same seed', () => {
    const a = mulberry32(20250822);
    const b = mulberry32(20250822);
    let sum = 0;
    for (let i = 0; i < 10000; i++) {
      const x = a();
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThan(1);
      expect(x).toBe(b());
      sum += x;
    }
    // Mean of 10^4 uniforms: 0.5 +/- 0.003. A generator stuck in one corner of
    // the interval would fail this.
    expect(sum / 10000).toBeCloseTo(0.5, 2);
  });
});

describe('exact enumeration vs Monte Carlo', () => {
  // Two lattice sizes, three temperatures spanning below / at / above T_c.
  const temperatures = [1.8, TC_EXACT, 3.2];
  const algorithms: Algorithm[] = ['metropolis', 'wolff'];

  for (const L of [2, 3]) {
    for (const T of temperatures) {
      for (const algorithm of algorithms) {
        it(`L=${L} T=${T.toFixed(3)} ${algorithm} reproduces the exact averages`, () => {
          const exact = exactObservables(L, T);
          const mc = measure(L, T, algorithm, 0xc0ffee + L, EQUILIBRATION, MEASUREMENT);

          expect(mc.samples).toBe(MEASUREMENT);
          // Absolute tolerances. The statistical error at 1.2e5 samples on
          // these tiny lattices is a few times 1e-3, so 0.015 is a comfortable
          // margin that a genuine detailed-balance violation would blow past.
          expect(Math.abs(mc.energyPerSpin - exact.energyPerSpin)).toBeLessThan(0.015);
          expect(
            Math.abs(mc.absMagnetizationPerSpin - exact.absMagnetizationPerSpin),
          ).toBeLessThan(0.015);
          // <m> must vanish by up/down symmetry. The tolerance is looser than
          // for <|m|> because below T_c the sign of m decorrelates slowly even
          // on a lattice this small, so <m> is the noisiest statistic here. A
          // sampler that failed to visit both ordered states would return
          // something near +/-1 and fail this by a wide margin.
          expect(Math.abs(mc.magnetizationPerSpin)).toBeLessThan(0.15);
          // The Binder cumulant is a ratio of moments and converges quickly.
          expect(Math.abs(mc.binderCumulant - exact.binderCumulant)).toBeLessThan(0.02);
          // Fluctuation observables are variances, which converge more slowly,
          // so these get a relative tolerance.
          expect(Math.abs(mc.specificHeat - exact.specificHeat)).toBeLessThan(
            0.06 * exact.specificHeat + 0.02,
          );
          expect(Math.abs(mc.susceptibility - exact.susceptibility)).toBeLessThan(
            0.06 * exact.susceptibility + 0.02,
          );
        });
      }
    }
  }
});

describe('cross-validation of the two update rules on a larger lattice', () => {
  // L = 8 is far beyond what can be enumerated, so instead the two algorithms
  // must agree with each other. Metropolis and Wolff share no code path apart
  // from the energy bookkeeping, so agreement is strong evidence that both
  // sample the Boltzmann distribution.
  for (const T of [2.0, 3.0]) {
    it(`Metropolis and Wolff agree at L=8, T=${T}`, () => {
      const met = measure(8, T, 'metropolis', 4242, 2000, 60000);
      const wol = measure(8, T, 'wolff', 9977, 2000, 60000);
      expect(Math.abs(met.energyPerSpin - wol.energyPerSpin)).toBeLessThan(0.02);
      expect(
        Math.abs(met.absMagnetizationPerSpin - wol.absMagnetizationPerSpin),
      ).toBeLessThan(0.02);
    });
  }
});

describe('incremental bookkeeping', () => {
  it('Metropolis keeps energy and magnetization exact over many sweeps', () => {
    const lattice = new IsingLattice(16, 777);
    for (let i = 0; i < 400; i++) {
      lattice.metropolisSweep(2.3);
      if (i % 50 === 0) {
        expect(lattice.energy).toBe(lattice.computeEnergy());
        expect(lattice.magnetization).toBe(lattice.computeMagnetization());
      }
    }
    expect(lattice.energy).toBe(lattice.computeEnergy());
    expect(lattice.magnetization).toBe(lattice.computeMagnetization());
  });

  it('Wolff keeps energy and magnetization exact over many sweeps', () => {
    for (const L of [2, 3, 16]) {
      const lattice = new IsingLattice(L, 31337 + L);
      for (let i = 0; i < 400; i++) {
        lattice.wolffSweep(2.1);
        expect(lattice.energy).toBe(lattice.computeEnergy());
        expect(lattice.magnetization).toBe(lattice.computeMagnetization());
      }
    }
  });

  it('stays exact when the algorithms are interleaved and the lattice is resized', () => {
    const lattice = new IsingLattice(12, 8080);
    for (let i = 0; i < 200; i++) {
      lattice.metropolisSweep(2.4);
      lattice.wolffSweep(2.4);
    }
    expect(lattice.energy).toBe(lattice.computeEnergy());
    expect(lattice.magnetization).toBe(lattice.computeMagnetization());

    lattice.setSize(9);
    expect(lattice.size).toBe(9);
    expect(lattice.spins.length).toBe(81);
    expect(lattice.energy).toBe(lattice.computeEnergy());
    expect(lattice.magnetization).toBe(lattice.computeMagnetization());
    for (let i = 0; i < 200; i++) lattice.sweep(1.9, 'wolff');
    expect(lattice.energy).toBe(lattice.computeEnergy());
    expect(lattice.magnetization).toBe(lattice.computeMagnetization());
  });

  it('setAllUp gives the ground state energy -2N and magnetization N', () => {
    const lattice = new IsingLattice(10, 5);
    lattice.setAllUp();
    expect(lattice.energy).toBe(-200);
    expect(lattice.magnetization).toBe(100);
    expect(lattice.energyPerSpin).toBe(-2);
    expect(lattice.magnetizationPerSpin).toBe(1);
  });
});

describe('limiting regimes', () => {
  it('orders as T -> 0: |m| -> 1 and e -> -2', () => {
    const L = 16;
    const T = 0.5;
    const lattice = new IsingLattice(L, 606);
    const acc = new ObservableAccumulator(T, L * L);
    lattice.setAllUp();
    for (let i = 0; i < 500; i++) lattice.metropolisSweep(T);
    for (let i = 0; i < 2000; i++) {
      lattice.metropolisSweep(T);
      acc.sample(lattice);
    }
    const o = acc.result();
    expect(o.absMagnetizationPerSpin).toBeGreaterThan(0.95);
    expect(o.energyPerSpin).toBeLessThan(-1.9);
    expect(o.energyPerSpin).toBeGreaterThanOrEqual(-2);
  });

  it('disorders as T -> infinity: |m| -> 0 and e -> 0', () => {
    const L = 16;
    const T = 1000;
    const lattice = new IsingLattice(L, 909);
    const acc = new ObservableAccumulator(T, L * L);
    for (let i = 0; i < 200; i++) lattice.metropolisSweep(T);
    for (let i = 0; i < 2000; i++) {
      lattice.metropolisSweep(T);
      acc.sample(lattice);
    }
    const o = acc.result();
    // On a finite lattice <|m|> ~ sqrt(2 / (pi N)) = 0.05 at N = 256, not 0.
    expect(o.absMagnetizationPerSpin).toBeLessThan(0.12);
    expect(Math.abs(o.energyPerSpin)).toBeLessThan(0.05);
    // For a Gaussian order parameter the Binder cumulant tends to 1 - 3/3 = 0.
    expect(Math.abs(o.binderCumulant)).toBeLessThan(0.1);
  });

  it('orders below T_c and disorders above it', () => {
    const below = measure(16, 1.8, 'wolff', 1234, 1000, 4000);
    const above = measure(16, 2.8, 'wolff', 1234, 1000, 4000);
    expect(below.absMagnetizationPerSpin).toBeGreaterThan(0.85);
    expect(above.absMagnetizationPerSpin).toBeLessThan(0.35);
    expect(below.energyPerSpin).toBeLessThan(above.energyPerSpin);
  });
});

describe('determinism', () => {
  it('two lattices with the same seed follow identical trajectories', () => {
    const a = new IsingLattice(12, 424242);
    const b = new IsingLattice(12, 424242);
    expect(Array.from(a.spins)).toEqual(Array.from(b.spins));

    for (let i = 0; i < 100; i++) {
      a.metropolisSweep(2.3);
      b.metropolisSweep(2.3);
      a.wolffSweep(2.3);
      b.wolffSweep(2.3);
      expect(a.energy).toBe(b.energy);
      expect(a.magnetization).toBe(b.magnetization);
    }
    expect(Array.from(a.spins)).toEqual(Array.from(b.spins));
  });

  it('different seeds diverge', () => {
    const a = new IsingLattice(12, 1);
    const b = new IsingLattice(12, 2);
    for (let i = 0; i < 50; i++) {
      a.metropolisSweep(2.3);
      b.metropolisSweep(2.3);
    }
    expect(Array.from(a.spins)).not.toEqual(Array.from(b.spins));
  });

  it('snapshot returns an independent copy', () => {
    const lattice = new IsingLattice(8, 11);
    const before = lattice.snapshot();
    for (let i = 0; i < 50; i++) lattice.metropolisSweep(2.3);
    expect(before).not.toBe(lattice.spins);
    expect(Array.from(before)).not.toEqual(Array.from(lattice.spins));
  });
});

describe('ObservableAccumulator', () => {
  it('reports zeros with no samples and applies the documented formulas', () => {
    const acc = new ObservableAccumulator(2.5, 64);
    const empty = acc.result();
    expect(empty.samples).toBe(0);
    expect(empty.energyPerSpin).toBe(0);
    expect(empty.specificHeat).toBe(0);
    expect(Number.isNaN(empty.binderCumulant)).toBe(false);

    acc.add(-1.5, 0.5);
    acc.add(-1.3, -0.5);
    expect(acc.samples).toBe(2);
    const r = acc.result();
    expect(r.temperature).toBe(2.5);
    expect(r.energyPerSpin).toBeCloseTo(-1.4, 12);
    expect(r.magnetizationPerSpin).toBeCloseTo(0, 12);
    expect(r.absMagnetizationPerSpin).toBeCloseTo(0.5, 12);
    // <e^2> - <e>^2 = 0.01, so C = 64 * 0.01 / 2.5^2.
    expect(r.specificHeat).toBeCloseTo((64 / 6.25) * 0.01, 10);
    // <m^2> = 0.25 = <|m|>^2, so chi vanishes for these two samples.
    expect(r.susceptibility).toBeCloseTo(0, 12);
    // <m^4> = 0.0625 and <m^2>^2 = 0.0625, so U = 1 - 1/3.
    expect(r.binderCumulant).toBeCloseTo(2 / 3, 12);

    acc.reset();
    expect(acc.result().samples).toBe(0);
  });
});
