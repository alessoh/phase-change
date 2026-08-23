/**
 * The verdict panel.
 *
 * This is the panel that reproduces the published result of Carrasquilla & Melko,
 * "Machine learning phases of matter", Nature Physics 13, 431-434 (2017). The
 * trained classifier is evaluated on fresh configurations at each temperature and
 * its mean probability for the "disordered" class is plotted against temperature.
 * Where that curve crosses one half is the network's own estimate of the critical
 * temperature. The exact answer for the 2D square-lattice Ising model is known
 * from Onsager (1944), kT_c/J = 2 / ln(1 + sqrt(2)) = 2.269185..., so the estimate
 * is checkable to as many digits as the reader cares to count.
 *
 * A null estimate is reported as a null estimate. For shuffled labels a null
 * estimate is the correct result and is labelled as such.
 */

import { useMemo } from 'react';
import type { CSSProperties } from 'react';
import { TC_EXACT } from '../types';
import type { FeatureMode, VerdictCurve } from '../types';
import { Chart } from './Chart';

export interface VerdictPanelProps {
  verdict: VerdictCurve | null;
  featureMode: FeatureMode;
  latticeSize: number;
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

const emptyStyle: CSSProperties = {
  margin: 0,
  padding: '22px 12px',
  textAlign: 'center',
  fontSize: 12.5,
  color: C.muted,
  border: `1px dashed ${C.border}`,
  borderRadius: 6,
};

function noteStyle(accent: string): CSSProperties {
  return {
    margin: 0,
    fontSize: 12.5,
    lineHeight: 1.55,
    color: C.muted,
    borderLeft: `2px solid ${accent}`,
    paddingLeft: 10,
  };
}

/** Four decimal places everywhere, as promised in the caption. */
function fixed4(x: number): string {
  return x.toFixed(4);
}

/** Signed, so that a low estimate reads as negative rather than as a bare magnitude. */
function signed4(x: number): string {
  return `${x >= 0 ? '+' : '−'}${Math.abs(x).toFixed(4)}`;
}

interface Reading {
  accent: string;
  headline: string;
  body: string;
}

/** What the reader should conclude, given the condition that was actually run. */
function interpret(
  featureMode: FeatureMode,
  estimatedTc: number | null,
  error: number | null,
  latticeSize: number,
): Reading {
  if (featureMode === 'shuffled') {
    if (estimatedTc === null) {
      return {
        accent: C.ok,
        headline: 'No transition found, which is the correct result.',
        body:
          'The labels were randomly permuted before training, so no configuration carried any information about which side of the transition it came from. A network that reported a critical temperature here would be reporting an artefact of the fitting procedure. The absence of a crossing is the falsification control passing, not the application failing.',
      };
    }
    return {
      accent: C.danger,
      headline: 'A crossing appeared under shuffled labels. Treat it as spurious.',
      body:
        'With permuted labels there is no signal to find, so any apparent critical temperature comes from noise in the mean output rather than from the physics. Enlarge the evaluation sample at each temperature, or harvest a larger training set, and the crossing should dissolve. Do not report this number.',
    };
  }

  if (estimatedTc === null || error === null) {
    return {
      accent: C.warn,
      headline: 'The network found no transition.',
      body:
        'The mean probability of the disordered class never crosses one half anywhere in the temperature range that was scanned, so no critical temperature can be extracted. The usual causes are a network that has not been trained yet, a training set that does not straddle the transition, or a scanned range that stops short of it.',
    };
  }

  if (featureMode === 'magnetization') {
    return {
      accent: C.warn,
      headline: 'Recovered from a hand-engineered feature, not from the raw physics.',
      body:
        `The classifier was fed |m| alone. Magnetization is the order parameter of this model, so a sharp crossing near the exact value is expected and demonstrates only that a threshold can be fitted to a quantity a physicist already chose. Compare against raw spins on an L = ${latticeSize} lattice before crediting the network with anything.`,
    };
  }

  return {
    accent: C.ok,
    headline: 'Recovered from raw spin configurations alone.',
    body:
      `The network was shown only bare ±1 configurations labelled below or above the transition, never the value of T_c and never the magnetization. The residual offset from the Onsager value is dominated by finite-size effects: on an L = ${latticeSize} lattice the crossing is displaced from the thermodynamic-limit value by an amount that shrinks as L grows, so repeating this at several lattice sizes is the honest way to test the result.`,
  };
}

export function VerdictPanel(props: VerdictPanelProps): JSX.Element {
  const { verdict, featureMode, latticeSize } = props;

  const temperatures = useMemo<number[]>(
    () => (verdict === null ? [] : Array.from(verdict.temperatures)),
    [verdict],
  );
  const pDisordered = useMemo<number[]>(
    () => (verdict === null ? [] : Array.from(verdict.pDisordered)),
    [verdict],
  );

  if (verdict === null) {
    return (
      <section style={panelStyle} aria-label="The verdict">
        <header style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <h2 style={headingStyle}>The verdict</h2>
          <p style={subheadStyle}>
            Mean probability of the disordered class against temperature. The crossing of one half
            is the network&rsquo;s estimate of T<sub>c</sub>.
          </p>
        </header>
        <p style={emptyStyle}>
          No verdict yet. Harvest a dataset and train the network, and its output will be evaluated
          across the temperature range here.
        </p>
      </section>
    );
  }

  const { estimatedTc, errorVsExact } = verdict;
  const reading = interpret(featureMode, estimatedTc, errorVsExact, latticeSize);

  const markers = [
    { axis: 'y' as const, value: 0.5, label: 'p = 0.5', color: C.muted },
    { axis: 'x' as const, value: TC_EXACT, label: `Onsager Tc = ${fixed4(TC_EXACT)}`, color: C.warn },
    ...(estimatedTc === null
      ? []
      : [
          {
            axis: 'x' as const,
            value: estimatedTc,
            label: `estimate = ${fixed4(estimatedTc)}`,
            color: reading.accent,
          },
        ]),
  ];

  return (
    <section style={panelStyle} aria-label="The verdict">
      <header style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        <h2 style={headingStyle}>The verdict</h2>
        <p style={subheadStyle}>
          Mean probability of the disordered class against temperature, evaluated on an L ={' '}
          {latticeSize} lattice. The crossing of one half is the network&rsquo;s estimate of T
          <sub>c</sub>.
        </p>
      </header>

      <Chart
        series={[
          {
            label: 'P(disordered)',
            color: reading.accent,
            x: temperatures,
            y: pDisordered,
          },
        ]}
        markers={markers}
        xLabel="temperature (J / k_B)"
        yLabel="P(disordered)"
        yMin={0}
        yMax={1}
        height={220}
      />

      <dl
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
          gap: 12,
          margin: 0,
        }}
      >
        <Readout
          label="Estimated Tc"
          value={estimatedTc === null ? 'none' : fixed4(estimatedTc)}
          color={estimatedTc === null ? C.muted : reading.accent}
          note={estimatedTc === null ? 'no crossing of 0.5' : 'network output, interpolated'}
        />
        <Readout label="Exact Tc (Onsager 1944)" value={fixed4(TC_EXACT)} note="2 / ln(1 + √2)" />
        <Readout
          label="Signed error"
          value={errorVsExact === null ? 'undefined' : signed4(errorVsExact)}
          color={errorVsExact === null ? C.muted : reading.accent}
          note={errorVsExact === null ? 'no estimate to compare' : 'estimate minus exact'}
        />
      </dl>

      <p style={noteStyle(reading.accent)}>
        <strong style={{ color: reading.accent, fontWeight: 600 }}>{reading.headline} </strong>
        {reading.body}
      </p>
    </section>
  );
}

function Readout(props: { label: string; value: string; note: string; color?: string }): JSX.Element {
  return (
    <div>
      <dt
        style={{
          fontSize: 11,
          color: C.muted,
          letterSpacing: '0.03em',
          textTransform: 'uppercase',
        }}
      >
        {props.label}
      </dt>
      <dd
        style={{
          margin: '2px 0 0',
          fontSize: 20,
          fontWeight: 600,
          fontVariantNumeric: 'tabular-nums',
          color: props.color ?? C.fg,
        }}
      >
        {props.value}
      </dd>
      <dd style={{ margin: '2px 0 0', fontSize: 11, color: C.muted }}>{props.note}</dd>
    </div>
  );
}
