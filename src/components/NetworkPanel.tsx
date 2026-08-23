/**
 * The network panel.
 *
 * Reproduces the training half of Carrasquilla & Melko, "Machine learning phases
 * of matter", Nature Physics 13, 431-434 (2017). A small feed-forward classifier
 * is trained on Monte Carlo configurations labelled only by whether they were
 * drawn below or above the critical temperature. It is never told T_c.
 *
 * The three feature modes are presented as three equally legitimate experimental
 * conditions, not as one headline result with two footnotes. Two of them are
 * controls, and one of those controls is required to fail.
 */

import { useMemo } from 'react';
import type { CSSProperties } from 'react';
import type { FeatureMode, TrainStep } from '../types';
import { Chart } from './Chart';

export interface NetworkPanelProps {
  featureMode: FeatureMode;
  onFeatureMode: (m: FeatureMode) => void;
  history: TrainStep[];
  training: boolean;
  datasetReady: boolean;
  datasetCount: number;
  onTrain: () => void;
  onHarvest: () => void;
  harvesting: { done: number; total: number } | null;
}

interface FeatureModeSpec {
  mode: FeatureMode;
  name: string;
  /** One line: what this condition tests. */
  tests: string;
  /** What a reader should expect to see if the pipeline is honest. */
  expect: string;
  accent: string;
}

/** Colours are literal so the panel is legible with or without an app theme. */
const C = {
  fg: 'var(--fg, #e9ecf1)',
  muted: 'var(--muted, #949cab)',
  panel: 'var(--panel, #12151c)',
  inset: 'var(--inset, #0c0f15)',
  border: 'var(--border, #262c38)',
  accent: 'var(--accent, #6ea8fe)',
  ok: 'var(--ok, #4ade80)',
  warn: 'var(--warn, #fbbf24)',
  danger: 'var(--danger, #f87171)',
} as const;

const FEATURE_MODES: readonly FeatureModeSpec[] = [
  {
    mode: 'spins',
    name: 'Raw spins',
    tests: 'The real experiment: the network sees only the bare ±1 configuration and must construct an order parameter for itself.',
    expect: 'Validation accuracy should climb well above chance, and the verdict curve should cross one half near 2.2692.',
    accent: C.accent,
  },
  {
    mode: 'magnetization',
    name: 'Magnetization only',
    tests: 'Control: the network is handed the hand-engineered feature |m| and nothing else, testing whether the physicist, not the network, is doing the work.',
    expect: 'Accuracy should be at least as good as raw spins. Success here is therefore not evidence that a network can learn a phase.',
    accent: C.warn,
  },
  {
    mode: 'shuffled',
    name: 'Shuffled labels',
    tests: 'Falsification control: the labels are randomly permuted, destroying any relationship between configuration and phase.',
    expect: 'Validation accuracy must sit at chance while training accuracy climbs, which is the signature of a network memorising noise. The verdict curve becomes meaningless, and any crossing it happens to show is an artifact of that noise rather than a measurement. Success here would indict the pipeline, not confirm it.',
    accent: C.danger,
  },
];

const panelStyle: CSSProperties = {
  background: C.panel,
  border: `1px solid ${C.border}`,
  borderRadius: 8,
  padding: '16px 18px 18px',
  color: C.fg,
  font: '400 14px/1.5 system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif',
  display: 'flex',
  flexDirection: 'column',
  gap: 14,
};

const headingStyle: CSSProperties = {
  margin: 0,
  font: '600 15px/1.3 inherit',
  letterSpacing: '0.01em',
};

const subheadStyle: CSSProperties = {
  margin: 0,
  color: C.muted,
  fontSize: 12.5,
};

const legendStyle: CSSProperties = {
  border: 0,
  padding: 0,
  margin: 0,
  minInlineSize: 0,
};

const modeGridStyle: CSSProperties = {
  display: 'grid',
  gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))',
  gap: 10,
  marginTop: 8,
};

const buttonBase: CSSProperties = {
  font: 'inherit',
  fontWeight: 600,
  fontSize: 13,
  padding: '8px 14px',
  borderRadius: 6,
  cursor: 'pointer',
  border: `1px solid ${C.border}`,
  background: C.inset,
  color: C.fg,
};

function buttonStyle(enabled: boolean, primary: boolean): CSSProperties {
  return {
    ...buttonBase,
    cursor: enabled ? 'pointer' : 'not-allowed',
    opacity: enabled ? 1 : 0.45,
    borderColor: primary && enabled ? C.accent : C.border,
    color: primary && enabled ? C.accent : C.fg,
  };
}

function modeCardStyle(selected: boolean, accent: string, disabled: boolean): CSSProperties {
  return {
    display: 'block',
    background: selected ? C.inset : 'transparent',
    border: `1px solid ${selected ? accent : C.border}`,
    borderRadius: 6,
    padding: '10px 12px',
    cursor: disabled ? 'not-allowed' : 'pointer',
    opacity: disabled && !selected ? 0.55 : 1,
  };
}

function pct(x: number): string {
  return `${(100 * x).toFixed(1)}%`;
}

export function NetworkPanel(props: NetworkPanelProps): JSX.Element {
  const {
    featureMode,
    onFeatureMode,
    history,
    training,
    datasetReady,
    datasetCount,
    onTrain,
    onHarvest,
    harvesting,
  } = props;

  const epochs = useMemo<number[]>(() => history.map((s) => s.epoch), [history]);
  const losses = useMemo<number[]>(() => history.map((s) => s.loss), [history]);
  const trainAcc = useMemo<number[]>(() => history.map((s) => s.trainAccuracy), [history]);
  const valAcc = useMemo<number[]>(() => history.map((s) => s.validationAccuracy), [history]);

  const last: TrainStep | null = history.length > 0 ? history[history.length - 1] : null;
  const busy = training || harvesting !== null;
  const canHarvest = !busy;
  const canTrain = !busy && datasetReady;
  const selected = FEATURE_MODES.find((f) => f.mode === featureMode) ?? FEATURE_MODES[0];

  return (
    <section style={panelStyle} aria-label="The network">
      <header style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        <h2 style={headingStyle}>The network</h2>
        <p style={subheadStyle}>
          A feed-forward classifier trained on configurations labelled only &ldquo;below T
          <sub>c</sub>&rdquo; or &ldquo;above T<sub>c</sub>&rdquo;. The value of T<sub>c</sub> is
          never supplied.
        </p>
      </header>

      <fieldset style={legendStyle} disabled={busy}>
        <legend style={{ ...subheadStyle, color: C.fg, fontWeight: 600, fontSize: 13, padding: 0 }}>
          Experimental condition
        </legend>
        <p style={{ ...subheadStyle, margin: '4px 0 0' }}>
          Three conditions, run the same way. Two of them are controls, and the third is required
          to fail.
        </p>
        <div style={modeGridStyle} role="radiogroup" aria-label="Experimental condition">
          {FEATURE_MODES.map((f) => {
            const isSelected = f.mode === featureMode;
            return (
              <label key={f.mode} style={modeCardStyle(isSelected, f.accent, busy)}>
                <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <input
                    type="radio"
                    name="ising-lab-feature-mode"
                    value={f.mode}
                    checked={isSelected}
                    onChange={() => onFeatureMode(f.mode)}
                    style={{ accentColor: f.accent, margin: 0 }}
                  />
                  <span style={{ fontWeight: 600, fontSize: 13, color: isSelected ? f.accent : C.fg }}>
                    {f.name}
                  </span>
                </span>
                <span
                  style={{
                    display: 'block',
                    marginTop: 6,
                    fontSize: 12,
                    lineHeight: 1.45,
                    color: C.muted,
                  }}
                >
                  {f.tests}
                </span>
              </label>
            );
          })}
        </div>
        <p
          style={{
            margin: '10px 0 0',
            fontSize: 12,
            lineHeight: 1.5,
            color: C.muted,
            borderLeft: `2px solid ${selected.accent}`,
            paddingLeft: 10,
          }}
        >
          <strong style={{ color: selected.accent, fontWeight: 600 }}>Expected outcome. </strong>
          {selected.expect}
        </p>
      </fieldset>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center' }}>
        <button type="button" style={buttonStyle(canHarvest, !datasetReady)} disabled={!canHarvest} onClick={onHarvest}>
          {harvesting !== null ? 'Harvesting…' : datasetReady ? 'Harvest again' : 'Harvest dataset'}
        </button>
        <button type="button" style={buttonStyle(canTrain, datasetReady)} disabled={!canTrain} onClick={onTrain}>
          {training ? 'Training…' : history.length > 0 ? 'Train again' : 'Train network'}
        </button>
        <span style={{ fontSize: 12, color: C.muted }} aria-live="polite">
          {harvesting !== null
            ? `${harvesting.done} / ${harvesting.total} configurations sampled`
            : datasetReady
              ? `${datasetCount.toLocaleString()} labelled configurations ready`
              : 'No dataset yet — harvest one first.'}
        </span>
      </div>

      {harvesting !== null && (
        <div
          role="progressbar"
          aria-valuemin={0}
          aria-valuemax={harvesting.total}
          aria-valuenow={harvesting.done}
          style={{ height: 4, background: C.inset, borderRadius: 2, overflow: 'hidden' }}
        >
          <div
            style={{
              height: '100%',
              width: `${harvesting.total > 0 ? (100 * harvesting.done) / harvesting.total : 0}%`,
              background: C.accent,
              transition: 'width 120ms linear',
            }}
          />
        </div>
      )}

      {last === null ? (
        <p
          style={{
            margin: 0,
            padding: '22px 12px',
            textAlign: 'center',
            fontSize: 12.5,
            color: C.muted,
            border: `1px dashed ${C.border}`,
            borderRadius: 6,
          }}
        >
          {datasetReady
            ? 'Dataset ready. Train the network to see the loss and accuracy curves.'
            : 'Harvest a dataset, then train. Nothing here is precomputed.'}
        </p>
      ) : (
        <>
          <dl
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(110px, 1fr))',
              gap: 10,
              margin: 0,
            }}
          >
            <Readout label="Epoch" value={String(last.epoch)} />
            <Readout label="Loss" value={last.loss.toFixed(4)} />
            <Readout label="Train accuracy" value={pct(last.trainAccuracy)} />
            <Readout
              label="Validation accuracy"
              value={pct(last.validationAccuracy)}
              color={selected.accent}
            />
          </dl>

          <Chart
            series={[{ label: 'cross-entropy loss', color: C.accent, x: epochs, y: losses }]}
            xLabel="epoch"
            yLabel="loss"
            height={150}
          />

          <Chart
            series={[
              { label: 'train', color: C.accent, x: epochs, y: trainAcc },
              { label: 'validation', color: C.ok, x: epochs, y: valAcc },
            ]}
            markers={[{ axis: 'y', value: 0.5, label: 'chance', color: C.danger }]}
            xLabel="epoch"
            yLabel="accuracy"
            yMin={0}
            yMax={1}
            height={150}
          />
        </>
      )}
    </section>
  );
}

function Readout(props: { label: string; value: string; color?: string }): JSX.Element {
  return (
    <div>
      <dt style={{ fontSize: 11, color: C.muted, letterSpacing: '0.03em', textTransform: 'uppercase' }}>
        {props.label}
      </dt>
      <dd
        style={{
          margin: '2px 0 0',
          fontSize: 16,
          fontWeight: 600,
          fontVariantNumeric: 'tabular-nums',
          color: props.color ?? C.fg,
        }}
      >
        {props.value}
      </dd>
    </div>
  );
}
