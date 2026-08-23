/**
 * The Ising Lab: 2D square-lattice Ising Monte Carlo core.
 *
 * Hamiltonian
 *
 *     H = -J * sum_<ij> s_i s_j ,   J = 1,   no external field,
 *
 * on an L x L square lattice with periodic boundary conditions and spins
 * s_i in {+1, -1}. Temperature is in units of J / k_B (k_B = 1).
 *
 * Bond convention. The bond sum is realised as
 *
 *     E = - sum_i s_i * ( s_{right(i)} + s_{down(i)} )
 *
 * i.e. every site contributes its right and its down bond, for 2N terms in
 * total. For L >= 3 those 2N terms are 2N distinct bonds. For L = 2 the right
 * and left neighbours of a site are the same site, so each geometric bond is
 * counted twice and the lattice behaves as if it had coupling 2J on four
 * bonds. That is the standard consequence of periodic boundaries on a 2x2
 * lattice; it is internally consistent here (the Metropolis energy difference,
 * the Wolff bond probability and computeEnergy() all use the same convention),
 * which is what lets the exact-enumeration tests be exact.
 *
 * Randomness is supplied by a seeded mulberry32 generator, never Math.random,
 * so that every trajectory in this file is reproducible.
 */

import type { Algorithm, Observables } from '../types';

/**
 * Temperatures below this are clamped. The Ising model has no dynamics at
 * T = 0 under single-spin flips and 1/T diverges, so a floor keeps the
 * interactive controls well behaved without changing any physics above it.
 */
export const MIN_TEMPERATURE = 1e-3;

/**
 * mulberry32: a 32-bit seeded PRNG. Small, fast, and good enough for Monte
 * Carlo of this size. Returns a function producing uniform doubles in [0, 1).
 */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function next(): number {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * The only possible energy changes for a single spin flip are
 * dE in {-8, -4, 0, +4, +8} (dE = 2 s_i * sum of four neighbours).
 * Index the returned table with (dE + 8) >> 2, giving 0..4.
 * Entries are min(1, exp(-dE / T)).
 */
export function metropolisAcceptanceTable(temperature: number): Float64Array {
  const T = temperature > MIN_TEMPERATURE ? temperature : MIN_TEMPERATURE;
  const table = new Float64Array(5);
  for (let k = 0; k < 5; k++) {
    const dE = -8 + 4 * k;
    table[k] = dE <= 0 ? 1 : Math.exp(-dE / T);
  }
  return table;
}

/** Wolff bond activation probability p_add = 1 - exp(-2J/T) with J = 1. */
export function wolffAddProbability(temperature: number): number {
  const T = temperature > MIN_TEMPERATURE ? temperature : MIN_TEMPERATURE;
  return 1 - Math.exp(-2 / T);
}

export class IsingLattice {
  private _L: number;
  private _n: number;
  private _spins!: Int8Array;

  /** Precomputed periodic neighbour indices. */
  private _right!: Int32Array;
  private _left!: Int32Array;
  private _up!: Int32Array;
  private _down!: Int32Array;

  private _rng: () => number;

  /** Incrementally maintained totals (both integer valued). */
  private _energy = 0;
  private _mag = 0;

  /** Cached Metropolis acceptance table and the temperature it was built for. */
  private _table: Float64Array = metropolisAcceptanceTable(1);
  private _tableT = 1;

  /** Wolff scratch: an explicit stack, the member list, and a generation stamp
   *  that marks cluster membership without an O(N) clear per cluster. */
  private _stack!: Int32Array;
  private _members!: Int32Array;
  private _stamp!: Int32Array;
  private _generation = 0;

  /** Running cluster-size statistics for the current temperature. See
   *  wolffSweep() for why the number of cluster flips in a sweep has to be
   *  fixed from these before the sweep starts. */
  private _clusterSizeSum = 0;
  private _clusterFlips = 0;
  private _statsT = Number.NaN;

  constructor(L: number, seed: number) {
    this._rng = mulberry32(seed);
    this._L = L;
    this._n = L * L;
    this.setSize(L);
  }

  /** Linear lattice dimension L. */
  get size(): number {
    return this._L;
  }

  /** Number of sites N = L * L. */
  get siteCount(): number {
    return this._n;
  }

  /** Live view of the spin array, length N, row-major, values +1 / -1.
   *  Treat as read-only; use snapshot() for a copy that is safe to transfer. */
  get spins(): Int8Array {
    return this._spins;
  }

  /** Total energy, maintained incrementally. */
  get energy(): number {
    return this._energy;
  }

  /** Total magnetization sum_i s_i, maintained incrementally. */
  get magnetization(): number {
    return this._mag;
  }

  /** <E>/N for the current configuration. */
  get energyPerSpin(): number {
    return this._energy / this._n;
  }

  /** m = (1/N) sum_i s_i for the current configuration, signed. */
  get magnetizationPerSpin(): number {
    return this._mag / this._n;
  }

  /** Independent copy of the spin array. */
  snapshot(): Int8Array {
    return this._spins.slice();
  }

  /**
   * Resize to an L x L lattice. Reallocates the spin array and the neighbour
   * tables and installs a fresh random configuration. The PRNG stream is
   * continued, not reset.
   */
  setSize(L: number): void {
    if (!Number.isInteger(L) || L < 2) {
      throw new RangeError(`lattice size must be an integer >= 2, received ${L}`);
    }
    const n = L * L;
    this._L = L;
    this._n = n;
    this._spins = new Int8Array(n);
    this._right = new Int32Array(n);
    this._left = new Int32Array(n);
    this._up = new Int32Array(n);
    this._down = new Int32Array(n);
    for (let y = 0; y < L; y++) {
      for (let x = 0; x < L; x++) {
        const i = y * L + x;
        this._right[i] = y * L + ((x + 1) % L);
        this._left[i] = y * L + ((x + L - 1) % L);
        this._down[i] = ((y + 1) % L) * L + x;
        this._up[i] = ((y + L - 1) % L) * L + x;
      }
    }
    this._stack = new Int32Array(n);
    this._members = new Int32Array(n);
    this._stamp = new Int32Array(n);
    this._generation = 0;
    this._clusterSizeSum = 0;
    this._clusterFlips = 0;
    this._statsT = Number.NaN;
    this.randomize();
  }

  /** Infinite-temperature start: every spin independently +1 or -1. */
  randomize(): void {
    const s = this._spins;
    const rng = this._rng;
    for (let i = 0; i < this._n; i++) {
      s[i] = rng() < 0.5 ? -1 : 1;
    }
    this.resetTotals();
  }

  /** Zero-temperature start: one of the two ground states. */
  setAllUp(): void {
    this._spins.fill(1);
    this.resetTotals();
  }

  /** Recompute the totals from scratch and adopt them. */
  private resetTotals(): void {
    this._energy = this.computeEnergy();
    this._mag = this.computeMagnetization();
  }

  /**
   * Total energy recomputed from scratch, E = -sum_i s_i (s_right + s_down).
   * Used by the tests to prove the incremental bookkeeping is exact; the
   * simulation itself never needs to call it.
   */
  computeEnergy(): number {
    const s = this._spins;
    const right = this._right;
    const down = this._down;
    let e = 0;
    for (let i = 0; i < this._n; i++) {
      e -= s[i] * (s[right[i]] + s[down[i]]);
    }
    return e;
  }

  /** Total magnetization recomputed from scratch. */
  computeMagnetization(): number {
    const s = this._spins;
    let m = 0;
    for (let i = 0; i < this._n; i++) {
      m += s[i];
    }
    return m;
  }

  /** Dispatch one sweep of the requested algorithm. */
  sweep(temperature: number, algorithm: Algorithm): void {
    if (algorithm === 'wolff') {
      this.wolffSweep(temperature);
    } else {
      this.metropolisSweep(temperature);
    }
  }

  /**
   * One Metropolis sweep: N attempted single-spin flips at uniformly random
   * sites, each accepted with probability min(1, exp(-dE/T)) read from the
   * five-entry lookup table (no Math.exp inside the loop).
   */
  metropolisSweep(temperature: number): void {
    const T = temperature > MIN_TEMPERATURE ? temperature : MIN_TEMPERATURE;
    if (T !== this._tableT) {
      this._table = metropolisAcceptanceTable(T);
      this._tableT = T;
    }
    const s = this._spins;
    const right = this._right;
    const left = this._left;
    const up = this._up;
    const down = this._down;
    const table = this._table;
    const rng = this._rng;
    const n = this._n;

    let energy = this._energy;
    let mag = this._mag;

    for (let k = 0; k < n; k++) {
      const i = (rng() * n) | 0;
      const si = s[i];
      const neighbours = s[up[i]] + s[down[i]] + s[left[i]] + s[right[i]];
      const dE = 2 * si * neighbours;
      if (dE <= 0 || rng() < table[(dE + 8) >> 2]) {
        s[i] = -si;
        energy += dE;
        mag -= 2 * si;
      }
    }

    this._energy = energy;
    this._mag = mag;
  }

  /**
   * One Wolff sweep: enough single-cluster flips that roughly N spins are
   * touched, so the work is comparable to one Metropolis sweep. Aligned
   * neighbours join the cluster with probability p_add = 1 - exp(-2J/T) and
   * the whole cluster is flipped unconditionally.
   *
   * The number of cluster flips is fixed BEFORE the sweep runs, from the mean
   * cluster size observed so far at this temperature. It must be: the obvious
   * alternative, "keep flipping until a running total of N spins have been
   * touched", is a stopping rule that depends on the sizes of the clusters it
   * just flipped, so a sweep terminates preferentially on a large cluster.
   * Large clusters are exactly what ordered configurations produce, so
   * measuring once per such sweep oversamples the ordered states and biases
   * every average. The effect is not subtle: on a 2x2 lattice at T = 3.2 that
   * rule reports <|m|> = 0.935 where the exact answer is 0.777. Averaging the
   * cluster size over the whole run instead makes the flip count effectively a
   * constant, and a constant number of applications of the Wolff kernel leaves
   * its stationary distribution - the Boltzmann distribution - intact.
   *
   * The statistics are reset when the temperature changes; the first sweep at
   * a new temperature flips a single cluster, which is enough to bootstrap the
   * estimate in either direction.
   */
  wolffSweep(temperature: number): void {
    const T = temperature > MIN_TEMPERATURE ? temperature : MIN_TEMPERATURE;
    if (T !== this._statsT) {
      this._statsT = T;
      this._clusterSizeSum = 0;
      this._clusterFlips = 0;
    }
    const pAdd = wolffAddProbability(T);
    const n = this._n;
    // Cluster sizes lie in [1, N], so meanSize in [1, N] and flips in [1, N].
    const meanSize = this._clusterFlips > 0 ? this._clusterSizeSum / this._clusterFlips : n;
    const flips = Math.round(n / meanSize);
    for (let k = 0; k < flips; k++) {
      this._clusterSizeSum += this.flipCluster(pAdd);
      this._clusterFlips++;
    }
  }

  /**
   * Build one Wolff cluster from a random seed site and flip it.
   * Returns the number of spins in the cluster.
   *
   * Growth uses an explicit stack. Membership is marked with a generation
   * stamp so that a cluster costs O(cluster size) rather than O(N).
   * A neighbour is only stamped when the bond activation succeeds, so on a
   * 2x2 lattice, where the same neighbour is reached through two directions,
   * it gets two independent chances to join - which is exactly right for the
   * doubled bond of that geometry.
   */
  private flipCluster(pAdd: number): number {
    const s = this._spins;
    const right = this._right;
    const left = this._left;
    const up = this._up;
    const down = this._down;
    const stack = this._stack;
    const members = this._members;
    const stamp = this._stamp;
    const rng = this._rng;
    const n = this._n;

    // Generation stamps are monotone; recycle long before Int32 overflow.
    if (this._generation >= 0x7ffffffe) {
      stamp.fill(0);
      this._generation = 0;
    }
    const gen = ++this._generation;

    const seedSite = (rng() * n) | 0;
    const spinValue = s[seedSite];
    stamp[seedSite] = gen;
    stack[0] = seedSite;
    members[0] = seedSite;
    let top = 1;
    let count = 1;

    while (top > 0) {
      const i = stack[--top];
      for (let d = 0; d < 4; d++) {
        const j = d === 0 ? right[i] : d === 1 ? left[i] : d === 2 ? up[i] : down[i];
        if (stamp[j] === gen || s[j] !== spinValue) continue;
        if (rng() < pAdd) {
          stamp[j] = gen;
          members[count] = j;
          stack[top++] = j;
          count++;
        }
      }
    }

    // Only bonds crossing the cluster boundary change energy; internal bonds
    // are untouched because both endpoints flip. Each such bond contributes
    // -s_i s_j before the flip and +s_i s_j after, hence dE = 2 s_i s_j.
    // Iterating all four directions per member reproduces the same bond
    // multiplicity that computeEnergy() uses.
    let dE = 0;
    for (let k = 0; k < count; k++) {
      const i = members[k];
      if (stamp[right[i]] !== gen) dE += 2 * spinValue * s[right[i]];
      if (stamp[left[i]] !== gen) dE += 2 * spinValue * s[left[i]];
      if (stamp[up[i]] !== gen) dE += 2 * spinValue * s[up[i]];
      if (stamp[down[i]] !== gen) dE += 2 * spinValue * s[down[i]];
    }

    for (let k = 0; k < count; k++) {
      s[members[k]] = -spinValue;
    }

    this._energy += dE;
    this._mag -= 2 * spinValue * count;
    return count;
  }
}

/**
 * Streaming accumulator for thermodynamic averages at one temperature.
 * Formulas follow the documentation on Observables in types.ts exactly:
 *
 *   C   = (N / T^2) * (<e^2> - <e>^2),  e = energy per spin
 *   chi = (N / T)   * (<m^2> - <|m|>^2)
 *   U   = 1 - <m^4> / (3 <m^2>^2)
 */
export class ObservableAccumulator {
  private readonly temperature: number;
  private readonly n: number;
  private count = 0;
  private sumE = 0;
  private sumE2 = 0;
  private sumM = 0;
  private sumAbsM = 0;
  private sumM2 = 0;
  private sumM4 = 0;

  /** @param temperature simulation temperature. @param siteCount N = L*L. */
  constructor(temperature: number, siteCount: number) {
    this.temperature = temperature;
    this.n = siteCount;
  }

  /** Discard all samples, keeping the temperature and site count. */
  reset(): void {
    this.count = 0;
    this.sumE = 0;
    this.sumE2 = 0;
    this.sumM = 0;
    this.sumAbsM = 0;
    this.sumM2 = 0;
    this.sumM4 = 0;
  }

  get samples(): number {
    return this.count;
  }

  /** Add one measurement, given per-spin energy and signed per-spin magnetization. */
  add(energyPerSpin: number, magnetizationPerSpin: number): void {
    const e = energyPerSpin;
    const m = magnetizationPerSpin;
    const m2 = m * m;
    this.count++;
    this.sumE += e;
    this.sumE2 += e * e;
    this.sumM += m;
    this.sumAbsM += Math.abs(m);
    this.sumM2 += m2;
    this.sumM4 += m2 * m2;
  }

  /** Add one measurement taken from the lattice's incremental totals. */
  sample(lattice: IsingLattice): void {
    this.add(lattice.energyPerSpin, lattice.magnetizationPerSpin);
  }

  /** Averages so far. With no samples every field is zero. */
  result(): Observables {
    const k = this.count;
    if (k === 0) {
      return {
        temperature: this.temperature,
        energyPerSpin: 0,
        magnetizationPerSpin: 0,
        absMagnetizationPerSpin: 0,
        specificHeat: 0,
        susceptibility: 0,
        binderCumulant: 0,
        samples: 0,
      };
    }
    const T = this.temperature > MIN_TEMPERATURE ? this.temperature : MIN_TEMPERATURE;
    const N = this.n;
    const meanE = this.sumE / k;
    const meanE2 = this.sumE2 / k;
    const meanM = this.sumM / k;
    const meanAbsM = this.sumAbsM / k;
    const meanM2 = this.sumM2 / k;
    const meanM4 = this.sumM4 / k;
    return {
      temperature: this.temperature,
      energyPerSpin: meanE,
      magnetizationPerSpin: meanM,
      absMagnetizationPerSpin: meanAbsM,
      specificHeat: (N / (T * T)) * (meanE2 - meanE * meanE),
      susceptibility: (N / T) * (meanM2 - meanAbsM * meanAbsM),
      binderCumulant: meanM2 > 0 ? 1 - meanM4 / (3 * meanM2 * meanM2) : 0,
      samples: k,
    };
  }
}
