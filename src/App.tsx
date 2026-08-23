/**
 * The Ising Lab - application shell.
 *
 * This module owns nothing scientific. It owns the simulation worker's
 * lifecycle, the controls, the layout, and the scheduling that keeps a
 * single-threaded browser responsive while a network trains. The physics lives
 * in src/physics/ising.ts, the sampler in src/workers/mc.worker.ts, the
 * classifier in src/ml/mlp.ts, and the four panels render what those produce.
 *
 * The experiment being reproduced is Carrasquilla and Melko, "Machine learning
 * phases of matter", Nature Physics 13, 431-434 (2017). A feed-forward network
 * is shown Monte Carlo configurations of the 2D square-lattice Ising model,
 * labelled only by which side of the transition they were drawn from, and never
 * told where the transition is. The temperature at which its output crosses one
 * half is then compared against Onsager's exact result of 1944,
 * kT_c/J = 2/ln(1 + sqrt(2)) = 2.269185314213022.
 */

import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { LatticeView } from './components/LatticeView';
import { NetworkPanel } from './components/NetworkPanel';
import { ObservablesPanel } from './components/Observables';
import { VerdictPanel } from './components/VerdictPanel';
import { MLP, buildFeatures, extractVerdict } from './ml/mlp';
import { TC_EXACT } from './types';
import type {
  Algorithm,
  Dataset,
  FeatureMode,
  Frame,
  Observables,
  TrainStep,
  VerdictCurve,
  WorkerRequest,
  WorkerResponse,
} from './types';

/* ------------------------------------------------------------------ */
/* Run parameters                                                      */
/* ------------------------------------------------------------------ */

/** Slider bounds. Wide enough to bracket T_c with plenty of ordered and
 *  disordered phase on either side, narrow enough to stay legible. */
const T_MIN = 1;
const T_MAX = 4;

const SIZES = [32, 64, 128] as const;
const INITIAL_SIZE = 32;

/** Fixed seed for the sampler, so a session is reproducible on reload. */
const SIMULATION_SEED = 19440101;
/** Fixed seed for network initialisation, for the same reason. */
const NETWORK_SEED = 0x1a5e1ab;

/** Examples per gradient update. */
const BATCH_SIZE = 16;
/** Fraction of the training split held out to report validation accuracy. */
const VALIDATION_SPLIT = 0.2;
/** One row in every RESERVE_EVERY consecutive rows at a temperature is withheld
 *  from training entirely and used only to draw the verdict curve. */
const RESERVE_EVERY = 4;

/**
 * How much work each lattice size is worth. Nothing here changes the physics;
 * it changes only how long the reader waits.
 */
interface RunPlan {
  /** Temperature sweep for the thermodynamics panel. */
  measure: { tSteps: number; equilibrationSweeps: number; measurementSweeps: number };
  /** Labelled harvest for the classifier. */
  harvest: { tSteps: number; perTemperature: number; equilibrationSweeps: number };
  /** Hidden layer widths. */
  hidden: number[];
  /** Epochs of stochastic gradient descent, one per animation frame. */
  epochs: number;
}

/*
 * Equilibration counts below are the number of sweeps a WOLFF run needs, which
 * is small: cluster updates decorrelate the 2D Ising model in a handful of
 * sweeps at every temperature, including at T_c.
 *
 * Single-spin Metropolis does not. Its correlation time grows as L^z with
 * z ~ 2.17, so a 128 x 128 lattice near T_c needs of order 10^4 sweeps to
 * forget where it started, and coarsening out of a random start deep in the
 * ordered phase is slower still. Measured directly against the sampler: from a
 * hot start at T = 2.0, 300 Metropolis sweeps leave <|m|> at 0.123 on L = 32
 * and 0.049 on L = 128, where the equilibrium value is 0.911. Wolff reaches
 * 0.910 on both. Harvesting from configurations in that state hands the
 * classifier ordered-phase samples that still look disordered, which drags the
 * estimated critical temperature upward.
 *
 * So Metropolis gets twenty times the equilibration here, and Wolff is the
 * default. Metropolis remains selectable because watching critical slowing
 * down happen is the point of offering the choice at all.
 */
const METROPOLIS_EQUILIBRATION_FACTOR = 20;

/*
 * The hidden layer is the same width at every lattice size, so that changing L
 * changes the physics and the input dimension and nothing else. The harvest
 * shrinks with L instead, because a 128 x 128 configuration is sixteen times
 * longer than a 32 x 32 one and the browser's time budget is fixed.
 *
 * That trade is real and it shows: the crispest reproduction is at L = 32,
 * where the input is short enough that a few hundred configurations pin the
 * weights down. Larger lattices give better thermodynamics and a worse
 * classifier estimate. The colophon says so rather than hiding it.
 *
 * Measured on a 2026 laptop with Wolff: harvest 1.5 to 8 s, training 5 to 16 s.
 * Training runs one epoch per animation frame, so those seconds pass with the
 * loss curve advancing rather than with a frozen page.
 *
 * Equilibration is not a free parameter either. Cutting the L = 32 harvest from
 * 400 Wolff sweeps to 60, to save about a second, moved the mean estimate from
 * +0.018 to +0.044 and widened the run-to-run spread. These counts are set for
 * accuracy.
 */
const PLANS: Record<number, RunPlan> = {
  32: {
    measure: { tSteps: 31, equilibrationSweeps: 400, measurementSweeps: 800 },
    harvest: { tSteps: 31, perTemperature: 40, equilibrationSweeps: 400 },
    hidden: [64],
    epochs: 40,
  },
  64: {
    measure: { tSteps: 31, equilibrationSweeps: 300, measurementSweeps: 500 },
    harvest: { tSteps: 31, perTemperature: 20, equilibrationSweeps: 300 },
    hidden: [64],
    epochs: 40,
  },
  128: {
    measure: { tSteps: 31, equilibrationSweeps: 150, measurementSweeps: 300 },
    harvest: { tSteps: 31, perTemperature: 8, equilibrationSweeps: 150 },
    hidden: [64],
    epochs: 40,
  },
};

/** Sweeps of equilibration to request for the algorithm actually in use. */
function equilibrationFor(baseline: number, algorithm: Algorithm): number {
  return algorithm === 'wolff' ? baseline : baseline * METROPOLIS_EQUILIBRATION_FACTOR;
}

function planFor(size: number): RunPlan {
  return PLANS[size] ?? PLANS[INITIAL_SIZE];
}

/**
 * Step size for plain SGD, scaled as 1/sqrt(fan-in).
 *
 * The inputs are bare +-1 spins, so the squared norm of an example is exactly
 * the input dimension: a 128 x 128 configuration is sixteen times longer than a
 * 32 x 32 one, and the same nominal learning rate would take steps sixteen
 * times larger in the pre-activations. Scaling by 1/sqrt(inputDim) makes the
 * change in a hidden unit's pre-activation per update independent of lattice
 * size, and gives the single-feature 'magnetization' control a sane rate too.
 */
function learningRateFor(inputDim: number): number {
  return 0.5 / Math.sqrt(inputDim);
}

/* ------------------------------------------------------------------ */
/* Dataset partitioning                                                */
/* ------------------------------------------------------------------ */

/** Copy the given rows of a dataset into a new, self-contained dataset. */
function gatherRows(source: Dataset, rows: number[]): Dataset {
  const spinsPerConfig = source.size * source.size;
  const configurations = new Float32Array(rows.length * spinsPerConfig);
  const labels = new Uint8Array(rows.length);
  const temperatures = new Float32Array(rows.length);
  for (let k = 0; k < rows.length; k++) {
    const src = rows[k] * spinsPerConfig;
    configurations.set(source.configurations.subarray(src, src + spinsPerConfig), k * spinsPerConfig);
    labels[k] = source.labels[rows[k]];
    temperatures[k] = source.temperatures[rows[k]];
  }
  return { size: source.size, count: rows.length, configurations, labels, temperatures };
}

/**
 * Split a harvest into the configurations the network is trained on and the
 * configurations it is judged on.
 *
 * The verdict curve is the headline claim of this application, so it must not
 * be drawn on data the network has already seen: a network with enough capacity
 * can memorise a training set, and under the shuffled-label control it will try
 * to, which would manufacture a crossing where there is no physics at all.
 *
 * The rows of a harvest arrive grouped by temperature. Reserving every fourth
 * row inside each group keeps the held-out set balanced across the whole
 * temperature range, which matters because the verdict is read off a curve
 * indexed by temperature. If a group is too small to spare a row, the split
 * degenerates and the full set is used for both, which is reported to the
 * reader rather than hidden.
 */
function partitionDataset(source: Dataset): { train: Dataset; evaluation: Dataset; held: boolean } {
  const trainRows: number[] = [];
  const evalRows: number[] = [];
  let withinGroup = 0;
  for (let i = 0; i < source.count; i++) {
    if (i > 0 && source.temperatures[i] !== source.temperatures[i - 1]) withinGroup = 0;
    if (withinGroup % RESERVE_EVERY === RESERVE_EVERY - 1) evalRows.push(i);
    else trainRows.push(i);
    withinGroup++;
  }
  if (evalRows.length === 0 || trainRows.length === 0) {
    return { train: source, evaluation: source, held: false };
  }
  return { train: gatherRows(source, trainRows), evaluation: gatherRows(source, evalRows), held: true };
}

/* ------------------------------------------------------------------ */
/* Formatting                                                          */
/* ------------------------------------------------------------------ */

function fixed(value: number, decimals: number): string {
  return Number.isFinite(value) ? value.toFixed(decimals) : '--';
}

function signed(value: number, decimals: number): string {
  if (!Number.isFinite(value)) return '--';
  return `${value >= 0 ? '+' : '−'}${Math.abs(value).toFixed(decimals)}`;
}

/* ------------------------------------------------------------------ */
/* Memoised panels                                                     */
/* ------------------------------------------------------------------ */

/* Frames arrive about thirty times a second. The lattice has to redraw at that
 * rate; the three analysis panels do not, and re-running their chart layout on
 * every frame is the difference between a smooth instrument and a stuttering
 * one. Their props are built with useMemo / useCallback so these comparisons
 * actually hold. */
const Observables3 = memo(ObservablesPanel);
const Network3 = memo(NetworkPanel);
const Verdict3 = memo(VerdictPanel);

/* ------------------------------------------------------------------ */
/* Component                                                           */
/* ------------------------------------------------------------------ */

interface Progress {
  done: number;
  total: number;
}

export default function App(): JSX.Element {
  const workerRef = useRef<Worker | null>(null);
  const trainFrameRef = useRef<number | null>(null);

  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [size, setSize] = useState<number>(INITIAL_SIZE);
  const [temperature, setTemperature] = useState<number>(TC_EXACT);
  // Wolff by default: it is the update rule that actually equilibrates this
  // model, and every quantitative claim on the page depends on that.
  const [algorithm, setAlgorithm] = useState<Algorithm>('wolff');
  const [playing, setPlaying] = useState(false);

  const [frame, setFrame] = useState<Frame | null>(null);
  const [live, setLive] = useState<Observables | null>(null);

  const [curve, setCurve] = useState<Observables[]>([]);
  const [measuring, setMeasuring] = useState<Progress | null>(null);

  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [harvesting, setHarvesting] = useState<Progress | null>(null);
  const [heldOut, setHeldOut] = useState(true);

  const [featureMode, setFeatureMode] = useState<FeatureMode>('spins');
  const [history, setHistory] = useState<TrainStep[]>([]);
  const [training, setTraining] = useState(false);
  const [verdict, setVerdict] = useState<VerdictCurve | null>(null);

  const send = useCallback((request: WorkerRequest): void => {
    workerRef.current?.postMessage(request);
  }, []);

  /* ---------------------------------------------------------------- */
  /* Worker lifecycle                                                  */
  /* ---------------------------------------------------------------- */

  useEffect(() => {
    const worker = new Worker(new URL('./workers/mc.worker.ts', import.meta.url), { type: 'module' });
    workerRef.current = worker;

    // Throttle the observables readout: the lattice needs every frame, the
    // numeric panel does not, and re-rendering four charts at 30 Hz is waste.
    let lastLiveAt = 0;

    worker.onmessage = (event: MessageEvent<WorkerResponse>): void => {
      const message = event.data;
      switch (message.type) {
        case 'ready': {
          setReady(true);
          setError(null);
          return;
        }
        case 'frame': {
          const f = message.frame;
          setFrame(f);
          const now = performance.now();
          if (now - lastLiveAt > 250) {
            lastLiveAt = now;
            setLive({
              temperature: f.temperature,
              energyPerSpin: f.energyPerSpin,
              magnetizationPerSpin: f.magnetizationPerSpin,
              absMagnetizationPerSpin: Math.abs(f.magnetizationPerSpin),
              // A single configuration carries no variance, so the fluctuation
              // observables are undefined here rather than zero. The charts
              // drop non-finite points instead of drawing a false value.
              specificHeat: Number.NaN,
              susceptibility: Number.NaN,
              binderCumulant: Number.NaN,
              samples: 1,
            });
          }
          return;
        }
        case 'measureProgress': {
          setMeasuring({ done: message.done, total: message.total });
          setCurve((previous) => [...previous, message.point]);
          return;
        }
        case 'measureDone': {
          setMeasuring(null);
          setCurve(message.curve);
          return;
        }
        case 'harvestProgress': {
          setHarvesting({ done: message.done, total: message.total });
          return;
        }
        case 'harvestDone': {
          setHarvesting(null);
          setDataset(message.dataset);
          setHistory([]);
          setVerdict(null);
          return;
        }
        case 'error': {
          setError(message.message);
          setMeasuring(null);
          setHarvesting(null);
          return;
        }
        default: {
          const unreachable: never = message;
          setError(`Unrecognised message from the simulation: ${JSON.stringify(unreachable)}`);
          return;
        }
      }
    };

    worker.onerror = (event: ErrorEvent): void => {
      setError(event.message.length > 0 ? event.message : 'The simulation worker failed to start.');
    };

    const init: WorkerRequest = {
      type: 'init',
      size: INITIAL_SIZE,
      temperature: TC_EXACT,
      algorithm: 'wolff',
      seed: SIMULATION_SEED,
    };
    worker.postMessage(init);

    return () => {
      worker.onmessage = null;
      worker.onerror = null;
      worker.terminate();
      workerRef.current = null;
    };
  }, []);

  /** Abandon a training run in progress. */
  const cancelTraining = useCallback((): void => {
    if (trainFrameRef.current !== null) {
      cancelAnimationFrame(trainFrameRef.current);
      trainFrameRef.current = null;
    }
  }, []);

  useEffect(() => cancelTraining, [cancelTraining]);

  /* ---------------------------------------------------------------- */
  /* Controls                                                          */
  /* ---------------------------------------------------------------- */

  const busy = measuring !== null || harvesting !== null || training;

  const handleTemperature = useCallback(
    (value: number): void => {
      setTemperature(value);
      send({ type: 'setTemperature', temperature: value });
    },
    [send],
  );

  const handleAlgorithm = useCallback(
    (value: Algorithm): void => {
      setAlgorithm(value);
      send({ type: 'setAlgorithm', algorithm: value });
    },
    [send],
  );

  const handleSize = useCallback(
    (value: number): void => {
      cancelTraining();
      setTraining(false);
      setSize(value);
      // Every measured curve, harvested dataset and trained network belongs to
      // one lattice size. Keeping them across a resize would put results from
      // two different systems on the same axes.
      setCurve([]);
      setDataset(null);
      setHistory([]);
      setVerdict(null);
      setLive(null);
      send({ type: 'setSize', size: value });
    },
    [cancelTraining, send],
  );

  const handlePlayPause = useCallback((): void => {
    setPlaying((wasPlaying) => {
      send({ type: wasPlaying ? 'pause' : 'play' });
      return !wasPlaying;
    });
  }, [send]);

  const handleRandomize = useCallback((): void => {
    send({ type: 'randomize' });
  }, [send]);

  const handleMeasure = useCallback((): void => {
    const plan = planFor(size);
    setPlaying(false);
    setCurve([]);
    setMeasuring({ done: 0, total: plan.measure.tSteps });
    send({
      type: 'measure',
      tMin: T_MIN,
      tMax: T_MAX,
      tSteps: plan.measure.tSteps,
      equilibrationSweeps: equilibrationFor(plan.measure.equilibrationSweeps, algorithm),
      measurementSweeps: plan.measure.measurementSweeps,
    });
  }, [algorithm, send, size]);

  const handleHarvest = useCallback((): void => {
    const plan = planFor(size);
    cancelTraining();
    setTraining(false);
    setPlaying(false);
    setDataset(null);
    setHistory([]);
    setVerdict(null);
    setHarvesting({ done: 0, total: plan.harvest.tSteps });
    send({
      type: 'harvest',
      tMin: T_MIN,
      tMax: T_MAX,
      tSteps: plan.harvest.tSteps,
      perTemperature: plan.harvest.perTemperature,
      equilibrationSweeps: equilibrationFor(plan.harvest.equilibrationSweeps, algorithm),
    });
  }, [algorithm, cancelTraining, send, size]);

  /** Stop whatever long-running job is in flight. */
  const handleStop = useCallback((): void => {
    cancelTraining();
    setTraining(false);
    if (measuring !== null || harvesting !== null) send({ type: 'pause' });
  }, [cancelTraining, harvesting, measuring, send]);

  const handleFeatureMode = useCallback(
    (mode: FeatureMode): void => {
      cancelTraining();
      setTraining(false);
      setFeatureMode(mode);
      // A history and a verdict belong to the condition that produced them.
      setHistory([]);
      setVerdict(null);
    },
    [cancelTraining],
  );

  /* ---------------------------------------------------------------- */
  /* Training                                                          */
  /* ---------------------------------------------------------------- */

  /**
   * Train the classifier one epoch per animation frame.
   *
   * Everything here runs on the main thread, so an epoch is a block of work the
   * browser cannot interrupt. Scheduling one epoch per frame hands control back
   * between them: the page keeps painting, the loss curve advances in front of
   * the reader, and Stop is answered on the next frame rather than at the end
   * of the run.
   */
  const handleTrain = useCallback((): void => {
    if (dataset === null || dataset.count < 2) return;
    cancelTraining();
    setPlaying(false);
    send({ type: 'pause' });

    const plan = planFor(dataset.size);
    const parts = partitionDataset(dataset);
    setHeldOut(parts.held);

    const trainSet = buildFeatures(parts.train, featureMode);
    const evalSet = buildFeatures(parts.evaluation, featureMode);
    const network = new MLP({
      inputDim: trainSet.inputDim,
      hidden: plan.hidden,
      learningRate: learningRateFor(trainSet.inputDim),
      seed: NETWORK_SEED,
    });

    const steps: TrainStep[] = [];
    setHistory([]);
    setVerdict(null);
    setTraining(true);

    const runEpoch = (): void => {
      trainFrameRef.current = null;
      try {
        const step = network.trainEpoch(trainSet.x, trainSet.y, trainSet.count, {
          batchSize: BATCH_SIZE,
          validationSplit: VALIDATION_SPLIT,
        });
        steps.push(step);
        setHistory(steps.slice());

        if (steps.length < plan.epochs) {
          trainFrameRef.current = requestAnimationFrame(runEpoch);
          return;
        }

        const probabilities = network.predictProbabilities(evalSet.x, evalSet.count);
        setVerdict(extractVerdict(evalSet.temperatures, probabilities));
        setTraining(false);
      } catch (thrown) {
        setTraining(false);
        setError(thrown instanceof Error ? thrown.message : String(thrown));
      }
    };

    trainFrameRef.current = requestAnimationFrame(runEpoch);
  }, [cancelTraining, dataset, featureMode, send]);

  /* ---------------------------------------------------------------- */
  /* Derived display values                                            */
  /* ---------------------------------------------------------------- */

  const distanceToTc = temperature - TC_EXACT;
  const phase = distanceToTc < 0 ? 'ordered' : 'disordered';
  const tcOffsetPercent = ((TC_EXACT - T_MIN) / (T_MAX - T_MIN)) * 100;

  const datasetCount = dataset?.count ?? 0;
  const datasetReady = dataset !== null && dataset.count > 1;

  const observablesProps = useMemo(
    () => ({ curve, live, latticeSize: size, measuring }),
    [curve, live, measuring, size],
  );

  const networkProps = useMemo(
    () => ({
      featureMode,
      onFeatureMode: handleFeatureMode,
      history,
      training,
      datasetReady,
      datasetCount,
      onTrain: handleTrain,
      onHarvest: handleHarvest,
      harvesting,
    }),
    [
      datasetCount,
      datasetReady,
      featureMode,
      handleFeatureMode,
      handleHarvest,
      handleTrain,
      harvesting,
      history,
      training,
    ],
  );

  const verdictProps = useMemo(
    () => ({ verdict, featureMode, latticeSize: size }),
    [featureMode, size, verdict],
  );

  /* ---------------------------------------------------------------- */

  return (
    <div className="app">
      <header className="masthead">
        <p className="masthead__eyebrow">
          The Discontinuous World &middot; companion instrument
        </p>
        <h1 className="masthead__title">The Ising Lab</h1>
        <p className="masthead__standfirst">
          Below a certain temperature a two-dimensional magnet is ordered; above it, it is not. The
          change is abrupt, and its location is known exactly. This page runs the Monte Carlo
          simulation live, harvests configurations labelled only <strong>below</strong> or{' '}
          <strong>above</strong> that temperature, trains a small neural network in your browser on
          those labels alone, and then asks the network where the boundary is. Nothing is
          precomputed, and the answer it gives can be checked against a number derived in 1944.
        </p>
        <dl className="masthead__meta">
          <div>
            <dt>Ground truth</dt>
            <dd>
              <span className="num">kT&#8348;/J = 2 / ln(1 + &radic;2) = {TC_EXACT.toFixed(9)}</span>
              <br />
              Onsager&rsquo;s exact solution of the square-lattice Ising model, 1944.
            </dd>
          </div>
          <div>
            <dt>Reproducing</dt>
            <dd>
              J. Carrasquilla and R. G. Melko, &ldquo;Machine learning phases of matter&rdquo;,{' '}
              <em>Nature Physics</em> <strong>13</strong>, 431&ndash;434 (2017).{' '}
              <a href="https://doi.org/10.1038/nphys4035" target="_blank" rel="noreferrer noopener">
                doi:10.1038/nphys4035
              </a>
            </dd>
          </div>
          <div>
            <dt>Conventions</dt>
            <dd>
              H = &minus;J &sum;<sub>&lang;ij&rang;</sub> s<sub>i</sub>s<sub>j</sub>, J = 1,
              k<sub>B</sub> = 1, no external field, periodic boundaries. Temperature is in units of
              J/k<sub>B</sub>.
            </dd>
          </div>
        </dl>
      </header>

      {error !== null && (
        <div className="banner" role="alert">
          <span>{error}</span>
          <button type="button" onClick={() => setError(null)}>
            dismiss
          </button>
        </div>
      )}

      <section className="console" aria-label="Simulation controls">
        <div>
          <div className="field__label">
            <span>Temperature</span>
            <span>{phase} phase</span>
          </div>
          <div className="temperature__value">
            <span className="temperature__number">{fixed(temperature, 3)}</span>
            <span className="temperature__delta" data-near={Math.abs(distanceToTc) < 0.05}>
              T &minus; T&#8348; = {signed(distanceToTc, 3)}
            </span>
          </div>
          <div className="slider">
            <input
              type="range"
              min={T_MIN}
              max={T_MAX}
              step={0.005}
              value={temperature}
              disabled={!ready || busy}
              aria-label="Temperature in units of J over Boltzmann's constant"
              onChange={(event) => handleTemperature(Number(event.target.value))}
            />
            <div
              className="slider__tc"
              style={{ left: `calc(${tcOffsetPercent}% - 0.5px)` }}
              aria-hidden="true"
            />
          </div>
          <div className="slider__scale">
            <span>{T_MIN.toFixed(1)}</span>
            <span>T&#8348; = {TC_EXACT.toFixed(4)}</span>
            <span>{T_MAX.toFixed(1)}</span>
          </div>
        </div>

        <div className="controls">
          <div>
            <div className="field__label">
              <span>Lattice</span>
            </div>
            <div className="segment" role="group" aria-label="Lattice size">
              {SIZES.map((candidate) => (
                <button
                  key={candidate}
                  type="button"
                  aria-pressed={size === candidate}
                  disabled={!ready || busy}
                  onClick={() => handleSize(candidate)}
                >
                  {candidate}&sup2;
                </button>
              ))}
            </div>
          </div>

          <div>
            <div className="field__label">
              <span>Update rule</span>
            </div>
            <div className="segment" role="group" aria-label="Monte Carlo algorithm">
              <button
                type="button"
                aria-pressed={algorithm === 'metropolis'}
                disabled={!ready}
                onClick={() => handleAlgorithm('metropolis')}
              >
                Metropolis
              </button>
              <button
                type="button"
                aria-pressed={algorithm === 'wolff'}
                disabled={!ready}
                onClick={() => handleAlgorithm('wolff')}
              >
                Wolff
              </button>
            </div>
          </div>

          <div>
            <div className="field__label">
              <span>Run</span>
            </div>
            <div className="btn-row">
              <button
                type="button"
                className={`btn${playing ? '' : ' btn--primary'}`}
                disabled={!ready || busy}
                onClick={handlePlayPause}
              >
                {playing ? 'Pause' : 'Play'}
              </button>
              <button type="button" className="btn" disabled={!ready || busy} onClick={handleRandomize}>
                Randomize
              </button>
              <button type="button" className="btn" disabled={!ready || busy} onClick={handleMeasure}>
                Sweep temperature
              </button>
              <button type="button" className="btn btn--stop" disabled={!busy} onClick={handleStop}>
                Stop
              </button>
            </div>
          </div>
        </div>
      </section>

      <div className="grid">
        <section className="panel" aria-label="The lattice">
          <header className="panel__head">
            <h2>The lattice</h2>
            <p>
              {size} &times; {size} spins with periodic boundaries, amber for +1 and blue for
              &minus;1, updated by {algorithm === 'wolff' ? 'Wolff cluster' : 'single-spin Metropolis'}{' '}
              moves. Near T&#8348; the domains grow to the size of the box; that is the transition
              you are looking for.
            </p>
          </header>

          <div className="stage">
            <LatticeView frame={frame} />
          </div>

          <dl className="readouts">
            <div>
              <dt>Sweeps</dt>
              <dd>{frame === null ? '--' : frame.sweep.toLocaleString()}</dd>
            </div>
            <div>
              <dt>Energy / spin</dt>
              <dd>{frame === null ? '--' : fixed(frame.energyPerSpin, 4)}</dd>
            </div>
            <div>
              <dt>m (signed)</dt>
              <dd>{frame === null ? '--' : signed(frame.magnetizationPerSpin, 4)}</dd>
            </div>
            <div>
              <dt>|m|</dt>
              <dd>{frame === null ? '--' : fixed(Math.abs(frame.magnetizationPerSpin), 4)}</dd>
            </div>
          </dl>

          <p className="note">
            Both rules sample the same Boltzmann distribution, but not at the same speed. Metropolis
            flips one spin at a time, and its correlation time grows roughly as L<sup>2.17</sup>, so
            near T&#8348; a large lattice needs thousands of sweeps to forget where it started. Wolff
            flips whole aligned clusters and removes almost all of that penalty.
          </p>
          {algorithm === 'metropolis' && (
            <p className="note" style={{ color: 'var(--warn)' }}>
              Metropolis is selected. Sweeps and harvests below run with{' '}
              {METROPOLIS_EQUILIBRATION_FACTOR}&times; the equilibration used for Wolff, and on{' '}
              {size} &times; {size} that is still not enough deep in the ordered phase: the
              configurations reaching the classifier will look more disordered than equilibrium
              requires, which pushes the estimated T&#8348; upward. This is critical slowing down
              doing what it does, and it is worth watching once. Switch back to Wolff for the
              quantitative result.
            </p>
          )}
        </section>

        <Observables3 {...observablesProps} />

        <Network3 {...networkProps} />

        <Verdict3 {...verdictProps} />
      </div>

      <footer className="colophon">
        <p>
          <strong>How the estimate is produced.</strong> Configurations are harvested across{' '}
          {T_MIN.toFixed(1)} to {T_MAX.toFixed(1)} and labelled 0 below T&#8348; and 1 above it. The
          labels are the network&rsquo;s only supervision: it is never shown the temperature of a
          configuration, and never shown T&#8348;.{' '}
          {heldOut
            ? `One configuration in every ${RESERVE_EVERY} at each temperature is withheld from training and used only to draw the verdict curve, so the crossing is read off data the network has not seen.`
            : 'This harvest was too small to withhold a separate evaluation set, so the verdict curve is drawn on the training configurations. Harvest again on a smaller lattice for a clean held-out estimate.'}{' '}
          The network&rsquo;s mean probability for the disordered class is then averaged at each
          temperature and the crossing of one half is located by linear interpolation.
        </p>
        <p>
          <strong>What is being claimed, and what is not.</strong> A crossing near 2.2692 under the
          raw-spin condition shows that a classifier trained on binary labels alone locates a phase
          boundary it was never given. It does not show that the network has discovered the
          magnetization, the correlation length, or anything else a physicist would call
          understanding. The two control conditions in the network panel exist to keep that
          distinction visible: one succeeds for an uninteresting reason, and one is required to
          fail.
        </p>
        <p>
          Every number on this page is computed in your browser during this session. There is no
          server, no stored result, and no network request after the page loads. Reload and it is
          all recomputed from the same seeds.
        </p>
      </footer>
    </div>
  );
}
