/**
 * Monte Carlo simulation worker for The Ising Lab.
 *
 * Owns the only IsingLattice instance in the application and speaks exactly the
 * WorkerRequest / WorkerResponse protocol declared in src/types.ts.
 *
 * Physics conventions (see src/types.ts):
 *   H = -J * sum_<ij> s_i s_j, J = 1, k_B = 1, spins +-1, periodic boundaries.
 *
 * Design notes that matter physically:
 *   - Temperature sweeps run from the HIGH temperature end downwards, reusing the
 *     previous configuration. A hot start is trivially equilibrated (infinite
 *     temperature is the disordered fixed point), and each subsequent point begins
 *     from a configuration already close to the new equilibrium, which suppresses
 *     the long transients a cold start suffers near T_c.
 *   - Energy and magnetization are computed here, directly from the spin array, so
 *     that the normalisation convention (per spin vs extensive) is unambiguous.
 *     Every extensive quantity reported to the main thread is per spin.
 */

import { IsingLattice } from '../physics/ising';
import { TC_EXACT } from '../types';
import type {
  Algorithm,
  Dataset,
  Frame,
  Observables,
  WorkerRequest,
  WorkerResponse,
} from '../types';

/* ------------------------------------------------------------------ */
/* Worker global                                                       */
/* ------------------------------------------------------------------ */

/** Minimal structural view of the dedicated worker scope. Declared locally so the
 *  file does not depend on which of the DOM / WebWorker lib declarations wins. */
interface WorkerScope {
  postMessage(message: WorkerResponse, transfer?: ArrayBufferLike[]): void;
  addEventListener(type: 'message', listener: (event: { data: WorkerRequest }) => void): void;
}

const ctx: WorkerScope = self as unknown as WorkerScope;

function post(message: WorkerResponse, transfer?: ArrayBufferLike[]): void {
  if (transfer !== undefined && transfer.length > 0) ctx.postMessage(message, transfer);
  else ctx.postMessage(message);
}

/* ------------------------------------------------------------------ */
/* Tunables                                                            */
/* ------------------------------------------------------------------ */

/** ~30 frames per second. */
const TARGET_FRAME_MS = 1000 / 30;
/** Work budget per animation frame; the rest of the frame is left to the event loop. */
const SWEEP_BUDGET_MS = 16;
/** Upper bound on batched sweeps, so a tiny lattice cannot monopolise the worker. */
const MAX_SWEEPS_PER_FRAME = 256;
/** How often long-running measure / harvest loops return to the event loop. */
const YIELD_INTERVAL_MS = 40;

/** Sweeps discarded between successive harvested configurations, to cut the
 *  autocorrelation between training samples. Wolff cluster updates decorrelate far
 *  faster than single-spin Metropolis, especially near T_c. */
function decorrelationSweeps(algorithm: Algorithm): number {
  return algorithm === 'wolff' ? 2 : 8;
}

/* ------------------------------------------------------------------ */
/* State                                                               */
/* ------------------------------------------------------------------ */

let lattice: IsingLattice | null = null;
let temperature = TC_EXACT;
let algorithm: Algorithm = 'metropolis';
let sweepCount = 0;

let playing = false;
let frameTimer: ReturnType<typeof setTimeout> | null = null;
let sweepsPerFrame = 1;

/** Which long-running job, if any, currently owns the lattice. */
let busy: 'measure' | 'harvest' | null = null;
/** Set by 'pause' to abort the running job at the next yield point. */
let cancelRequested = false;
let lastYieldAt = 0;

/** Monotonic clock. performance.now() exists in dedicated workers and has
 *  sub-millisecond resolution, which Date.now() does not on Windows. */
function now(): number {
  return performance.now();
}

function requireLattice(): IsingLattice {
  if (lattice === null) throw new Error('Simulation received a command before "init".');
  return lattice;
}

function sweepOnce(lat: IsingLattice, t: number, algo: Algorithm): void {
  if (algo === 'wolff') lat.wolffSweep(t);
  else lat.metropolisSweep(t);
}

/* ------------------------------------------------------------------ */
/* Observables computed from the raw spin array                        */
/* ------------------------------------------------------------------ */

/** Total energy E = -J * sum over distinct bonds. Each site contributes its right
 *  and down bonds, which counts every bond of the periodic lattice exactly once. */
function totalEnergy(spins: Int8Array, size: number): number {
  let bonds = 0;
  for (let y = 0; y < size; y++) {
    const row = y * size;
    const rowDown = (y + 1 === size ? 0 : y + 1) * size;
    for (let x = 0; x < size; x++) {
      const s = spins[row + x];
      const right = spins[row + (x + 1 === size ? 0 : x + 1)];
      const down = spins[rowDown + x];
      bonds += s * (right + down);
    }
  }
  return -bonds;
}

/** Total magnetization M = sum_i s_i (signed). */
function totalMagnetization(spins: Int8Array): number {
  let sum = 0;
  for (let i = 0; i < spins.length; i++) sum += spins[i];
  return sum;
}

interface Accumulator {
  sumE: number;
  sumE2: number;
  sumM: number;
  sumAbsM: number;
  sumM2: number;
  sumM4: number;
  samples: number;
}

function newAccumulator(): Accumulator {
  return { sumE: 0, sumE2: 0, sumM: 0, sumAbsM: 0, sumM2: 0, sumM4: 0, samples: 0 };
}

/** Record one measurement. `e` and `m` are per spin. */
function accumulate(acc: Accumulator, e: number, m: number): void {
  const absM = Math.abs(m);
  const m2 = m * m;
  acc.sumE += e;
  acc.sumE2 += e * e;
  acc.sumM += m;
  acc.sumAbsM += absM;
  acc.sumM2 += m2;
  acc.sumM4 += m2 * m2;
  acc.samples += 1;
}

function finalize(acc: Accumulator, t: number, spinCount: number): Observables {
  const n = acc.samples > 0 ? acc.samples : 1;
  const e = acc.sumE / n;
  const e2 = acc.sumE2 / n;
  const m = acc.sumM / n;
  const absM = acc.sumAbsM / n;
  const m2 = acc.sumM2 / n;
  const m4 = acc.sumM4 / n;
  // Clamp against round-off producing a tiny negative variance.
  const varE = Math.max(0, e2 - e * e);
  const varM = Math.max(0, m2 - absM * absM);
  return {
    temperature: t,
    energyPerSpin: e,
    magnetizationPerSpin: m,
    absMagnetizationPerSpin: absM,
    specificHeat: t > 0 ? (spinCount * varE) / (t * t) : 0,
    susceptibility: t > 0 ? (spinCount * varM) / t : 0,
    binderCumulant: m2 > 0 ? 1 - m4 / (3 * m2 * m2) : 0,
    samples: acc.samples,
  };
}

/* ------------------------------------------------------------------ */
/* Frames                                                              */
/* ------------------------------------------------------------------ */

function postFrame(): void {
  const lat = requireLattice();
  const size = lat.size;
  const spinCount = size * size;
  // Copy: the lattice keeps mutating its own array, and the buffer is transferred.
  const spins = new Int8Array(lat.spins);
  const frame: Frame = {
    size,
    spins,
    temperature,
    energyPerSpin: totalEnergy(spins, size) / spinCount,
    magnetizationPerSpin: totalMagnetization(spins) / spinCount,
    sweep: sweepCount,
  };
  post({ type: 'frame', frame }, [spins.buffer]);
}

/* ------------------------------------------------------------------ */
/* Animation loop                                                      */
/* ------------------------------------------------------------------ */

function stopLoop(): void {
  playing = false;
  if (frameTimer !== null) {
    clearTimeout(frameTimer);
    frameTimer = null;
  }
}

function startLoop(): void {
  requireLattice();
  if (playing) return;
  playing = true;
  frameTimer = setTimeout(tick, 0);
}

function tick(): void {
  frameTimer = null;
  if (!playing || lattice === null) return;
  try {
    const started = now();
    const lat = lattice;
    const batch = sweepsPerFrame;
    for (let i = 0; i < batch; i++) sweepOnce(lat, temperature, algorithm);
    sweepCount += batch;
    const workMs = now() - started;

    // Keep the per-frame work near SWEEP_BUDGET_MS so small lattices animate fast
    // and large ones still hand the event loop back promptly.
    if (workMs > SWEEP_BUDGET_MS && batch > 1) {
      sweepsPerFrame = Math.max(1, batch >> 1);
    } else if (workMs * 2 < SWEEP_BUDGET_MS && batch < MAX_SWEEPS_PER_FRAME) {
      sweepsPerFrame = Math.min(MAX_SWEEPS_PER_FRAME, batch * 2);
    }

    postFrame();
    frameTimer = setTimeout(tick, Math.max(0, TARGET_FRAME_MS - (now() - started)));
  } catch (err) {
    stopLoop();
    post({ type: 'error', message: describe(err) });
  }
}

/* ------------------------------------------------------------------ */
/* Cooperative scheduling                                              */
/* ------------------------------------------------------------------ */

function yieldToEventLoop(): Promise<void> {
  return new Promise<void>((resolve) => {
    setTimeout(resolve, 0);
  });
}

/** Return to the event loop if we have been hogging it, so that a 'pause' message
 *  queued by the main thread is actually delivered mid-run. */
async function pump(): Promise<void> {
  const t = now();
  if (t - lastYieldAt >= YIELD_INTERVAL_MS) {
    lastYieldAt = t;
    await yieldToEventLoop();
  }
}

/** Run `count` sweeps, yielding periodically. Returns false if cancelled. */
async function sweepBlock(lat: IsingLattice, t: number, algo: Algorithm, count: number): Promise<boolean> {
  for (let i = 0; i < count; i++) {
    if (cancelRequested) return false;
    sweepOnce(lat, t, algo);
    await pump();
  }
  return !cancelRequested;
}

/** Ascending ladder of tSteps temperatures spanning [tMin, tMax]. The endpoints
 *  are sorted here, so the callers' "walk the ladder downwards" is a hot start
 *  even if the request arrives with the bounds the wrong way round. */
function temperatureLadder(tMin: number, tMax: number, tSteps: number): number[] {
  const lo = Math.min(tMin, tMax);
  const hi = Math.max(tMin, tMax);
  const steps = Math.max(1, Math.floor(tSteps));
  if (steps === 1) return [hi];
  const out = new Array<number>(steps);
  const dT = (hi - lo) / (steps - 1);
  for (let i = 0; i < steps; i++) out[i] = lo + i * dT;
  return out;
}

/* ------------------------------------------------------------------ */
/* Measurement sweep                                                   */
/* ------------------------------------------------------------------ */

async function runMeasure(
  tMin: number,
  tMax: number,
  tSteps: number,
  equilibrationSweeps: number,
  measurementSweeps: number,
): Promise<void> {
  const lat = requireLattice();
  const size = lat.size;
  const spinCount = size * size;
  const algo = algorithm;
  const temps = temperatureLadder(tMin, tMax, tSteps);
  const total = temps.length;
  const equilibration = Math.max(0, Math.floor(equilibrationSweeps));
  const measurements = Math.max(1, Math.floor(measurementSweeps));

  // Start from a fully disordered configuration: that IS the equilibrium state at
  // the high temperature end, so the descending ladder begins already equilibrated.
  lat.randomize();
  sweepCount = 0;
  lastYieldAt = now();

  const descending: Observables[] = [];
  for (let i = total - 1; i >= 0; i--) {
    const t = temps[i];
    if (!(await sweepBlock(lat, t, algo, equilibration))) break;

    const acc = newAccumulator();
    let aborted = false;
    for (let s = 0; s < measurements; s++) {
      if (cancelRequested) {
        aborted = true;
        break;
      }
      sweepOnce(lat, t, algo);
      sweepCount += 1;
      const spins = lat.spins;
      accumulate(acc, totalEnergy(spins, size) / spinCount, totalMagnetization(spins) / spinCount);
      await pump();
    }
    if (aborted) break;

    const point = finalize(acc, t, spinCount);
    descending.push(point);
    post({ type: 'measureProgress', done: total - i, total, point });
    await yieldToEventLoop();
  }

  // Reported ascending in temperature, the natural order for plotting.
  descending.reverse();
  post({ type: 'measureDone', curve: descending });
}

/* ------------------------------------------------------------------ */
/* Dataset harvest                                                     */
/* ------------------------------------------------------------------ */

async function runHarvest(
  tMin: number,
  tMax: number,
  tSteps: number,
  perTemperature: number,
  equilibrationSweeps: number,
): Promise<void> {
  const lat = requireLattice();
  const size = lat.size;
  const spinCount = size * size;
  const algo = algorithm;
  const spacing = decorrelationSweeps(algo);
  const temps = temperatureLadder(tMin, tMax, tSteps);
  const total = temps.length;
  const per = Math.max(1, Math.floor(perTemperature));
  const equilibration = Math.max(0, Math.floor(equilibrationSweeps));

  const rows = total * per;
  const configurations = new Float32Array(rows * spinCount);
  const labels = new Uint8Array(rows);
  const temperatures = new Float32Array(rows);

  lat.randomize();
  sweepCount = 0;
  lastYieldAt = now();

  // Rows are laid out in ascending temperature order; the loop fills them from the
  // hot end downwards, so a cancelled run leaves a contiguous filled tail.
  let lowestFilledBlock = total;
  for (let i = total - 1; i >= 0; i--) {
    const t = temps[i];
    if (!(await sweepBlock(lat, t, algo, equilibration))) break;

    const label = t < TC_EXACT ? 0 : 1;
    let aborted = false;
    for (let k = 0; k < per; k++) {
      if (!(await sweepBlock(lat, t, algo, spacing))) {
        aborted = true;
        break;
      }
      sweepCount += spacing;
      const row = i * per + k;
      const offset = row * spinCount;
      const spins = lat.spins;
      for (let j = 0; j < spinCount; j++) configurations[offset + j] = spins[j];
      labels[row] = label;
      temperatures[row] = t;
    }
    if (aborted) break;

    lowestFilledBlock = i;
    post({ type: 'harvestProgress', done: total - i, total });
    await yieldToEventLoop();
  }

  const startRow = lowestFilledBlock * per;
  const count = rows - startRow;
  const dataset: Dataset =
    startRow === 0
      ? { size, count: rows, configurations, labels, temperatures }
      : {
          size,
          count,
          configurations: configurations.slice(startRow * spinCount),
          labels: labels.slice(startRow),
          temperatures: temperatures.slice(startRow),
        };
  post({ type: 'harvestDone', dataset }, [
    dataset.configurations.buffer,
    dataset.labels.buffer,
    dataset.temperatures.buffer,
  ]);
}

/* ------------------------------------------------------------------ */
/* Message handling                                                    */
/* ------------------------------------------------------------------ */

function describe(err: unknown): string {
  if (err instanceof Error) return err.message;
  return String(err);
}

/** Requests accepted while a measure / harvest job owns the lattice. Everything
 *  else would corrupt the run in progress, so it is refused rather than silently
 *  producing wrong physics. */
function handleWhileBusy(request: WorkerRequest): void {
  switch (request.type) {
    case 'pause':
      cancelRequested = true;
      stopLoop();
      return;
    case 'setTemperature':
      temperature = request.temperature;
      return;
    case 'setAlgorithm':
      algorithm = request.algorithm;
      return;
    default:
      post({
        type: 'error',
        message: `Cannot handle "${request.type}" while a ${busy ?? 'background'} run is in progress. Pause first.`,
      });
      return;
  }
}

async function handleMessage(request: WorkerRequest): Promise<void> {
  try {
    if (busy !== null) {
      handleWhileBusy(request);
      return;
    }

    switch (request.type) {
      case 'init': {
        stopLoop();
        lattice = new IsingLattice(request.size, request.seed);
        lattice.randomize();
        temperature = request.temperature;
        algorithm = request.algorithm;
        sweepCount = 0;
        sweepsPerFrame = 1;
        post({ type: 'ready' });
        // Give the renderer something to draw before the first 'play', and keep
        // 'init' consistent with 'setSize' / 'randomize', which also post a frame.
        postFrame();
        return;
      }
      case 'setTemperature': {
        temperature = request.temperature;
        return;
      }
      case 'setAlgorithm': {
        algorithm = request.algorithm;
        return;
      }
      case 'setSize': {
        const lat = requireLattice();
        const wasPlaying = playing;
        stopLoop();
        lat.setSize(request.size);
        lat.randomize();
        sweepCount = 0;
        sweepsPerFrame = 1;
        postFrame();
        if (wasPlaying) startLoop();
        return;
      }
      case 'play': {
        startLoop();
        return;
      }
      case 'pause': {
        stopLoop();
        return;
      }
      case 'randomize': {
        const lat = requireLattice();
        lat.randomize();
        sweepCount = 0;
        postFrame();
        return;
      }
      case 'measure': {
        requireLattice();
        stopLoop();
        busy = 'measure';
        cancelRequested = false;
        try {
          await runMeasure(
            request.tMin,
            request.tMax,
            request.tSteps,
            request.equilibrationSweeps,
            request.measurementSweeps,
          );
        } finally {
          busy = null;
          cancelRequested = false;
        }
        postFrame();
        return;
      }
      case 'harvest': {
        requireLattice();
        stopLoop();
        busy = 'harvest';
        cancelRequested = false;
        try {
          await runHarvest(
            request.tMin,
            request.tMax,
            request.tSteps,
            request.perTemperature,
            request.equilibrationSweeps,
          );
        } finally {
          busy = null;
          cancelRequested = false;
        }
        postFrame();
        return;
      }
      default: {
        const unreachable: never = request;
        post({ type: 'error', message: `Unknown request: ${JSON.stringify(unreachable)}` });
        return;
      }
    }
  } catch (err) {
    stopLoop();
    busy = null;
    cancelRequested = false;
    post({ type: 'error', message: describe(err) });
  }
}

ctx.addEventListener('message', (event) => {
  void handleMessage(event.data);
});
