/**
 * The thermodynamics panel.
 *
 * Plots the measured observables of the 2D square-lattice Ising model against
 * temperature: the order parameter |m|, the energy per spin, the specific heat
 * and the magnetic susceptibility. The exact Onsager critical temperature,
 * kT_c/J = 2/ln(1+sqrt(2)) = 2.269185..., is drawn on every plot as a fixed
 * reference, so the reader can see for themselves how far the finite lattice
 * misses it.
 *
 * The scientific point of this panel is finite-size scaling. On an infinite
 * lattice the susceptibility diverges at T_c and the specific heat diverges
 * logarithmically. On an L x L lattice nothing diverges: the correlation length
 * cannot exceed L, so both quantities show rounded peaks of finite height,
 * displaced from T_c, that sharpen and move toward T_c as L grows. The lattice
 * size is therefore labelled on every plot, and the measured peak positions are
 * reported next to the exact value.
 */

import { useMemo } from 'react';
import type { CSSProperties } from 'react';
import { TC_EXACT } from '../types';
import type { Observables } from '../types';
import { Chart } from './Chart';
import type { Series } from './Chart';

export interface ObservablesPanelProps {
  curve: Observables[];
  live: Observables | null;
  latticeSize: number;
  measuring: { done: number; total: number } | null;
}

/** Literal fallbacks keep the panel legible with or without an app theme. */
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
  violet: '#c084fc',
} as const;

const panelStyle: CSSProperties = {
  background: C.panel,
  border: `1px solid ${C.border}`,
  borderRadius: 10,
  padding: 16,
  color: C.fg,
  fontFamily: 'ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
};

const gridStyle: CSSProperties = {
  display: 'grid',
  gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
  gap: 14,
};

const plotBoxStyle: CSSProperties = {
  background: C.inset,
  border: `1px solid ${C.border}`,
  borderRadius: 8,
  padding: '10px 10px 4px',
};

const monoStyle: CSSProperties = {
  fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace',
};

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

/** Extract one field of the curve as a plain array, in temperature order. */
function column(rows: Observables[], pick: (o: Observables) => number): number[] {
  const out = new Array<number>(rows.length);
  for (let i = 0; i < rows.length; i++) out[i] = pick(rows[i]);
  return out;
}

/**
 * Temperature of the largest measured value, refined by fitting a parabola
 * through the peak and its two neighbours. This is the standard way to read a
 * pseudo-critical temperature T_c(L) off a coarsely sampled finite-size peak.
 * Returns null when there are too few points to locate a peak.
 */
function peakTemperature(t: number[], v: number[]): number | null {
  const n = Math.min(t.length, v.length);
  if (n < 3) return null;
  let best = -1;
  let bestV = -Infinity;
  for (let i = 0; i < n; i++) {
    if (Number.isFinite(v[i]) && v[i] > bestV) {
      bestV = v[i];
      best = i;
    }
  }
  if (best < 0) return null;
  if (best === 0 || best === n - 1) return t[best];

  const x0 = t[best - 1];
  const x1 = t[best];
  const x2 = t[best + 1];
  const y0 = v[best - 1];
  const y1 = v[best];
  const y2 = v[best + 1];
  if (![x0, x1, x2, y0, y1, y2].every(Number.isFinite)) return t[best];

  // Vertex of the parabola through three points (unequal spacing allowed).
  const d1 = (y1 - y0) / (x1 - x0);
  const d2 = (y2 - y1) / (x2 - x1);
  const a = (d2 - d1) / (x2 - x0);
  if (!Number.isFinite(a) || a >= 0) return t[best];
  const b = d1 - a * (x0 + x1);
  const vertex = -b / (2 * a);
  // Refuse a vertex outside the bracketing interval; the sampling is too coarse.
  return vertex > x0 && vertex < x2 ? vertex : t[best];
}

function fmt(v: number, decimals: number): string {
  return Number.isFinite(v) ? v.toFixed(decimals) : '--';
}

function signed(v: number, decimals: number): string {
  if (!Number.isFinite(v)) return '--';
  return (v >= 0 ? '+' : '') + v.toFixed(decimals);
}

/* ------------------------------------------------------------------ */
/* Component                                                           */
/* ------------------------------------------------------------------ */

export function ObservablesPanel(props: ObservablesPanelProps): JSX.Element {
  const { curve, live, latticeSize, measuring } = props;

  const data = useMemo(() => {
    const rows = curve.slice().sort((a, b) => a.temperature - b.temperature);
    const t = column(rows, (o) => o.temperature);
    return {
      rows,
      t,
      absM: column(rows, (o) => o.absMagnetizationPerSpin),
      m: column(rows, (o) => o.magnetizationPerSpin),
      energy: column(rows, (o) => o.energyPerSpin),
      heat: column(rows, (o) => o.specificHeat),
      chi: column(rows, (o) => o.susceptibility),
    };
  }, [curve]);

  const chiPeak = useMemo(() => peakTemperature(data.t, data.chi), [data.t, data.chi]);
  const heatPeak = useMemo(() => peakTemperature(data.t, data.heat), [data.t, data.heat]);

  const spins = latticeSize * latticeSize;
  const tcLabel = `T_c = ${TC_EXACT.toFixed(4)}`;

  /** A single measured point for the temperature the simulation is sitting at. */
  const livePoint = (pick: (o: Observables) => number): Series[] => {
    if (!live) return [];
    const y = pick(live);
    if (!Number.isFinite(y) || !Number.isFinite(live.temperature)) return [];
    return [{ label: 'live', color: C.violet, x: [live.temperature], y: [y] }];
  };

  return (
    <section style={panelStyle}>
      <header
        style={{
          display: 'flex',
          alignItems: 'baseline',
          justifyContent: 'space-between',
          gap: 12,
          flexWrap: 'wrap',
          marginBottom: 4,
        }}
      >
        <h2 style={{ margin: 0, fontSize: 16, letterSpacing: 0.2 }}>Thermodynamics</h2>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, flexWrap: 'wrap' }}>
          <span
            style={{
              ...monoStyle,
              fontSize: 15,
              fontWeight: 700,
              color: C.accent,
              border: `1px solid ${C.border}`,
              borderRadius: 6,
              padding: '2px 8px',
              background: C.inset,
            }}
          >
            L = {latticeSize} x {latticeSize}
          </span>
          <span style={{ ...monoStyle, fontSize: 12, color: C.muted }}>
            N = {spins} spins
          </span>
          <span style={{ ...monoStyle, fontSize: 12, color: C.warn }}>{tcLabel}</span>
        </div>
      </header>

      <p style={{ margin: '6px 0 12px', fontSize: 12.5, lineHeight: 1.5, color: C.muted }}>
        Energies and magnetizations are per spin, temperature is in units of J/k<sub>B</sub>, and the
        dashed line is Onsager&rsquo;s exact critical temperature for the infinite lattice. Nothing
        here diverges: on {latticeSize} x {latticeSize} sites the correlation length is capped at L,
        so the susceptibility and specific heat show rounded peaks of finite height. Increase L and
        the peaks grow taller, narrower, and closer to the dashed line.
      </p>

      {measuring && (
        <div style={{ margin: '0 0 12px' }}>
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              fontSize: 12,
              color: C.muted,
              marginBottom: 4,
            }}
          >
            <span>Measuring temperature sweep on L = {latticeSize}</span>
            <span style={monoStyle}>
              {measuring.done} / {measuring.total} temperatures
            </span>
          </div>
          <div
            style={{
              height: 6,
              background: C.inset,
              border: `1px solid ${C.border}`,
              borderRadius: 999,
              overflow: 'hidden',
            }}
            role="progressbar"
            aria-valuemin={0}
            aria-valuemax={measuring.total}
            aria-valuenow={measuring.done}
          >
            <div
              style={{
                height: '100%',
                width: `${measuring.total > 0 ? (100 * measuring.done) / measuring.total : 0}%`,
                background: C.accent,
                transition: 'width 120ms linear',
              }}
            />
          </div>
        </div>
      )}

      {!measuring && data.rows.length === 0 && (
        <p style={{ margin: '0 0 12px', fontSize: 12.5, color: C.muted }}>
          No sweep measured yet. Run a temperature sweep to accumulate the curves below; the live
          point marks the temperature the simulation is currently equilibrated at.
        </p>
      )}

      <div style={gridStyle}>
        <PlotBox
          title="Order parameter"
          note={`<|m|> per spin, L = ${latticeSize}`}
          yLabel="|m|"
          xLabel="temperature  T"
          yDomain={[-1, 1]}
          series={[
            { label: '<|m|>', color: C.accent, x: data.t, y: data.absM },
            { label: '<m>  (signed)', color: C.muted, x: data.t, y: data.m, dashed: true },
            ...livePoint((o) => o.absMagnetizationPerSpin),
          ]}
        />

        <PlotBox
          title="Energy"
          note={`<E>/N, L = ${latticeSize}`}
          yLabel="e = E/N"
          xLabel="temperature  T"
          series={[
            { label: '<E>/N', color: C.ok, x: data.t, y: data.energy },
            ...livePoint((o) => o.energyPerSpin),
          ]}
        />

        <PlotBox
          title="Specific heat"
          note={`C = (N/T^2)(<e^2> - <e>^2), L = ${latticeSize}`}
          yLabel="C"
          xLabel="temperature  T"
          series={[
            { label: 'C', color: C.warn, x: data.t, y: data.heat },
            ...livePoint((o) => o.specificHeat),
          ]}
        />

        <PlotBox
          title="Susceptibility"
          note={`chi = (N/T)(<m^2> - <|m|>^2), L = ${latticeSize}`}
          yLabel="chi"
          xLabel="temperature  T"
          series={[
            { label: 'chi', color: C.danger, x: data.t, y: data.chi },
            ...livePoint((o) => o.susceptibility),
          ]}
        />
      </div>

      <div
        style={{
          marginTop: 14,
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: 10,
        }}
      >
        <Readout
          label={`chi peak, L = ${latticeSize}`}
          value={chiPeak === null ? '--' : `T = ${fmt(chiPeak, 4)}`}
          detail={chiPeak === null ? 'needs three or more points' : `${signed(chiPeak - TC_EXACT, 4)} from exact`}
          color={C.danger}
        />
        <Readout
          label={`C peak, L = ${latticeSize}`}
          value={heatPeak === null ? '--' : `T = ${fmt(heatPeak, 4)}`}
          detail={heatPeak === null ? 'needs three or more points' : `${signed(heatPeak - TC_EXACT, 4)} from exact`}
          color={C.warn}
        />
        <Readout
          label="Onsager exact"
          value={`T_c = ${TC_EXACT.toFixed(6)}`}
          detail="2 / ln(1 + sqrt 2), infinite lattice"
          color={C.fg}
        />
        <Readout
          label="Sweep coverage"
          value={`${data.rows.length} temperatures`}
          detail={
            data.rows.length > 0
              ? `${fmt(data.t[0], 2)} to ${fmt(data.t[data.rows.length - 1], 2)}, ${data.rows[0].samples} samples each`
              : 'no measurements yet'
          }
          color={C.muted}
        />
      </div>

      <p style={{ margin: '12px 0 0', fontSize: 12, lineHeight: 1.5, color: C.muted }}>
        The signed magnetization &lt;m&gt; averages toward zero even deep in the ordered phase,
        because a finite lattice tunnels between its two ordered states over a long enough run. That
        is why &lt;|m|&gt; is used as the order parameter here, and it is also why the peak positions
        above are pseudo-critical temperatures T<sub>c</sub>(L), not the critical temperature itself.
      </p>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* Sub-components                                                      */
/* ------------------------------------------------------------------ */

interface PlotBoxProps {
  title: string;
  note: string;
  xLabel: string;
  yLabel: string;
  series: Series[];
  yDomain?: [number, number];
}

function PlotBox(props: PlotBoxProps): JSX.Element {
  return (
    <div style={plotBoxStyle}>
      <div
        style={{
          display: 'flex',
          alignItems: 'baseline',
          justifyContent: 'space-between',
          gap: 8,
          marginBottom: 2,
        }}
      >
        <span style={{ fontSize: 13, fontWeight: 600, color: C.fg }}>{props.title}</span>
        <span style={{ ...monoStyle, fontSize: 11, color: C.muted }}>{props.note}</span>
      </div>
      <Chart
        series={props.series}
        xLabel={props.xLabel}
        yLabel={props.yLabel}
        height={210}
        yDomain={props.yDomain}
        markerX={{ value: TC_EXACT, label: 'T_c exact', color: C.warn }}
      />
    </div>
  );
}

function Readout(props: { label: string; value: string; detail: string; color: string }): JSX.Element {
  return (
    <div
      style={{
        background: C.inset,
        border: `1px solid ${C.border}`,
        borderRadius: 8,
        padding: '8px 10px',
      }}
    >
      <div style={{ fontSize: 11, color: C.muted, marginBottom: 3 }}>{props.label}</div>
      <div style={{ ...monoStyle, fontSize: 14, color: props.color }}>{props.value}</div>
      <div style={{ fontSize: 11, color: C.muted, marginTop: 2 }}>{props.detail}</div>
    </div>
  );
}
