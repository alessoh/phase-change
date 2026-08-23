/**
 * A dependency-free SVG line chart.
 *
 * The Ising Lab ships no charting library, so this module is the only plotting
 * primitive in the application. It is deliberately small and deliberately
 * scientific: rounded axis ticks, honest handling of empty and single-point
 * series, and explicit reference lines so that a plot can be annotated with a
 * known exact value (the Onsager critical temperature, or the probability one
 * half at which a binary classifier is undecided).
 *
 * The chart is drawn into a fixed viewBox and stretched by CSS, so it scales
 * with its container without any resize observer or layout measurement.
 */

import { useId, useMemo } from 'react';

export interface Series {
  label: string;
  color: string;
  x: ArrayLike<number>;
  y: ArrayLike<number>;
  dashed?: boolean;
}

/** A reference line on either axis. */
export interface AxisMarker {
  axis: 'x' | 'y';
  value: number;
  label?: string;
  color?: string;
}

export interface ChartProps {
  series: Series[];
  xLabel: string;
  yLabel: string;
  height?: number;
  yDomain?: [number, number];
  /** Vertical reference line, e.g. the exact critical temperature. */
  markerX?: { value: number; label: string; color?: string };
  /** Horizontal reference line, e.g. probability one half. */
  markerY?: { value: number; label?: string; color?: string };
  /** Additional reference lines on either axis. Equivalent to markerX/markerY. */
  markers?: AxisMarker[];
  /** Convenience alternative to yDomain; both bounds must be given to take effect. */
  yMin?: number;
  yMax?: number;
}

/* ------------------------------------------------------------------ */
/* Constants                                                           */
/* ------------------------------------------------------------------ */

/** Internal drawing width. The viewBox maps this onto the container width. */
const VIEW_W = 640;

const MARGIN_L = 58;
const MARGIN_R = 16;
const MARGIN_B = 42;
const MARGIN_T_BASE = 10;
const LEGEND_ROW_H = 15;

const FONT_SANS =
  'ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif';
const FONT_MONO = 'ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace';

/** Literal fallbacks keep the chart legible on a dark background with no theme. */
const C = {
  fg: 'var(--fg, #e9ecf1)',
  muted: 'var(--muted, #949cab)',
  inset: 'var(--inset, #0c0f15)',
  border: 'var(--border, #262c38)',
  grid: 'rgba(233, 236, 241, 0.09)',
  marker: 'var(--warn, #fbbf24)',
} as const;

/* ------------------------------------------------------------------ */
/* Axis arithmetic                                                     */
/* ------------------------------------------------------------------ */

/** Round a range to a "nice" number: 1, 2, 5 or 10 times a power of ten. */
function niceNum(range: number, round: boolean): number {
  if (!(range > 0) || !Number.isFinite(range)) return 1;
  const exp = Math.floor(Math.log10(range));
  const f = range / Math.pow(10, exp);
  let nf: number;
  if (round) {
    nf = f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10;
  } else {
    nf = f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10;
  }
  return nf * Math.pow(10, exp);
}

interface AxisScale {
  min: number;
  max: number;
  step: number;
  ticks: number[];
}

/**
 * Build an axis. When `fixed` is true the supplied bounds are honoured exactly
 * and only the tick positions are rounded; otherwise the bounds themselves are
 * pushed out to the nearest round tick.
 */
function makeAxis(lo: number, hi: number, target: number, fixed: boolean): AxisScale {
  let a = Number.isFinite(lo) ? lo : 0;
  let b = Number.isFinite(hi) ? hi : 1;
  if (b < a) {
    const t = a;
    a = b;
    b = t;
  }
  if (!(b > a)) {
    // Degenerate: a single distinct value. Open a symmetric window around it.
    const pad = Math.abs(a) > 1e-12 ? Math.abs(a) * 0.1 : 1;
    a -= pad;
    b += pad;
  }

  const step = niceNum((b - a) / Math.max(1, target - 1), true);
  const min = fixed ? a : Math.floor(a / step + 1e-9) * step;
  const max = fixed ? b : Math.ceil(b / step - 1e-9) * step;

  const ticks: number[] = [];
  const first = Math.ceil(min / step - 1e-9) * step;
  for (let k = 0; k < 200; k++) {
    const v = first + k * step;
    if (v > max + step * 1e-6) break;
    // Snap away accumulated floating point drift.
    ticks.push(Math.round(v / step) * step);
  }
  return { min, max: max > min ? max : min + step, step, ticks };
}

/** Tick text with a decimal count implied by the tick spacing. */
function formatTick(v: number, step: number): string {
  const av = Math.abs(v);
  if (av < step * 1e-6) return '0';
  if (av >= 1e5 || av < 1e-4) return v.toExponential(1);
  const decimals = Math.min(6, Math.max(0, -Math.floor(Math.log10(step) + 1e-9)));
  return v.toFixed(decimals);
}

/** Crude but stable text width estimate, in viewBox units, at 11px. */
function textWidth(s: string, fontSize: number): number {
  return s.length * fontSize * 0.55;
}

/* ------------------------------------------------------------------ */
/* Geometry                                                            */
/* ------------------------------------------------------------------ */

interface Extent {
  xLo: number;
  xHi: number;
  yLo: number;
  yHi: number;
  count: number;
}

function dataExtent(series: Series[]): Extent {
  let xLo = Infinity;
  let xHi = -Infinity;
  let yLo = Infinity;
  let yHi = -Infinity;
  let count = 0;
  for (const s of series) {
    const n = Math.min(s.x.length, s.y.length);
    for (let i = 0; i < n; i++) {
      const x = s.x[i];
      const y = s.y[i];
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      if (x < xLo) xLo = x;
      if (x > xHi) xHi = x;
      if (y < yLo) yLo = y;
      if (y > yHi) yHi = y;
      count++;
    }
  }
  return { xLo, xHi, yLo, yHi, count };
}

interface Pt {
  px: number;
  py: number;
}

/** Path data plus the projected points, splitting the line at any gap. */
function projectSeries(
  s: Series,
  sx: (v: number) => number,
  sy: (v: number) => number,
): { d: string; pts: Pt[] } {
  const n = Math.min(s.x.length, s.y.length);
  const pts: Pt[] = [];
  let d = '';
  let pen = false;
  for (let i = 0; i < n; i++) {
    const x = s.x[i];
    const y = s.y[i];
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      pen = false;
      continue;
    }
    const px = sx(x);
    const py = sy(y);
    pts.push({ px, py });
    d += `${pen ? 'L' : 'M'}${px.toFixed(2)} ${py.toFixed(2)} `;
    pen = true;
  }
  return { d: d.trim(), pts };
}

/* ------------------------------------------------------------------ */
/* Component                                                           */
/* ------------------------------------------------------------------ */

export function Chart(props: ChartProps): JSX.Element {
  const { series, xLabel, yLabel } = props;
  const height = props.height ?? 220;
  const rawId = useId();
  const clipId = `isinglab-clip-${rawId.replace(/[^a-zA-Z0-9]/g, '')}`;

  // Reference lines: the named markerX/markerY plus any extras.
  const markers = useMemo<AxisMarker[]>(() => {
    const out: AxisMarker[] = [];
    if (props.markerX) {
      out.push({ axis: 'x', value: props.markerX.value, label: props.markerX.label, color: props.markerX.color });
    }
    if (props.markerY) {
      out.push({ axis: 'y', value: props.markerY.value, label: props.markerY.label, color: props.markerY.color });
    }
    if (props.markers) {
      for (const m of props.markers) out.push(m);
    }
    return out.filter((m) => Number.isFinite(m.value));
  }, [props.markerX, props.markerY, props.markers]);

  const fixedDomain: [number, number] | null = props.yDomain
    ? props.yDomain
    : props.yMin !== undefined && props.yMax !== undefined
      ? [props.yMin, props.yMax]
      : null;

  // Legend layout comes first because it sets the top margin.
  const legend = useMemo(() => {
    const entries = series.filter((s) => s.label.length > 0);
    const rows: { s: Series; x: number; row: number }[] = [];
    let x = MARGIN_L;
    let row = 0;
    const limit = VIEW_W - MARGIN_R;
    for (const s of entries) {
      const w = 20 + textWidth(s.label, 11) + 14;
      if (x > MARGIN_L && x + w > limit) {
        row++;
        x = MARGIN_L;
      }
      rows.push({ s, x, row });
      x += w;
    }
    return { rows, rowCount: entries.length > 0 ? row + 1 : 0 };
  }, [series]);

  const marginT = MARGIN_T_BASE + legend.rowCount * LEGEND_ROW_H;
  const plotL = MARGIN_L;
  const plotT = marginT;
  const plotW = VIEW_W - MARGIN_L - MARGIN_R;
  const plotH = Math.max(40, height - marginT - MARGIN_B);
  const plotR = plotL + plotW;
  const plotB = plotT + plotH;

  const ext = dataExtent(series);
  const hasData = ext.count > 0;

  // x domain: data plus any vertical reference line, so the line is never off-plot.
  let xLo = ext.xLo;
  let xHi = ext.xHi;
  for (const m of markers) {
    if (m.axis !== 'x') continue;
    xLo = Math.min(xLo, m.value);
    xHi = Math.max(xHi, m.value);
  }

  // y domain: 6% headroom when automatic; exact when the caller fixes it.
  let yLo = ext.yLo;
  let yHi = ext.yHi;
  if (!fixedDomain) {
    for (const m of markers) {
      if (m.axis !== 'y') continue;
      yLo = Math.min(yLo, m.value);
      yHi = Math.max(yHi, m.value);
    }
    if (Number.isFinite(yLo) && Number.isFinite(yHi) && yHi > yLo) {
      const pad = (yHi - yLo) * 0.06;
      const loWasNonNegative = yLo >= 0;
      const hiWasNonPositive = yHi <= 0;
      yLo -= pad;
      yHi += pad;
      // Never invent a sign the data does not have: a susceptibility or a
      // probability that is nowhere negative should not get a negative axis.
      if (loWasNonNegative && yLo < 0) yLo = 0;
      if (hiWasNonPositive && yHi > 0) yHi = 0;
    }
  }

  const xAxis = makeAxis(xLo, xHi, 6, false);
  const yAxis = fixedDomain
    ? makeAxis(fixedDomain[0], fixedDomain[1], 5, true)
    : makeAxis(yLo, yHi, 5, false);

  const sx = (v: number): number => plotL + ((v - xAxis.min) / (xAxis.max - xAxis.min)) * plotW;
  const sy = (v: number): number => plotB - ((v - yAxis.min) / (yAxis.max - yAxis.min)) * plotH;

  // Vertical reference lines that sit close together would otherwise draw their
  // labels on top of each other, which happens whenever the estimated critical
  // temperature lands near the exact one. Assign each label the first vertical
  // lane in which it does not overlap a label already placed.
  const markerLane = new Map<number, number>();
  {
    const lanes: { x0: number; x1: number }[][] = [];
    const ordered = markers
      .map((m, i) => ({ m, i }))
      .filter((e) => e.m.axis === 'x' && !!e.m.label)
      .sort((a, b) => sx(a.m.value) - sx(b.m.value));
    for (const { m, i } of ordered) {
      const px = sx(m.value);
      const w = textWidth(m.label ?? '', 11) + 8;
      const flip = px > plotR - 70;
      const x0 = flip ? px - 5 - w : px + 5;
      const x1 = x0 + w;
      let lane = 0;
      while (lanes[lane] && lanes[lane].some((s) => x0 < s.x1 && x1 > s.x0)) lane++;
      if (!lanes[lane]) lanes[lane] = [];
      lanes[lane].push({ x0, x1 });
      markerLane.set(i, lane);
    }
  }

  const drawn = series.map((s) => ({ s, ...projectSeries(s, sx, sy) }));

  return (
    <svg
      viewBox={`0 0 ${VIEW_W} ${height}`}
      preserveAspectRatio="xMidYMid meet"
      role="img"
      aria-label={`${yLabel} against ${xLabel}`}
      style={{ display: 'block', width: '100%', height: 'auto', overflow: 'visible' }}
    >
      <defs>
        <clipPath id={clipId}>
          <rect x={plotL} y={plotT} width={plotW} height={plotH} />
        </clipPath>
      </defs>

      {/* Plot frame */}
      <rect x={plotL} y={plotT} width={plotW} height={plotH} fill={C.inset} stroke={C.border} strokeWidth={1} />

      {hasData && (
        <g>
          {/* Grid */}
          {yAxis.ticks.map((t) => (
            <line key={`gy${t}`} x1={plotL} x2={plotR} y1={sy(t)} y2={sy(t)} stroke={C.grid} strokeWidth={1} />
          ))}
          {xAxis.ticks.map((t) => (
            <line key={`gx${t}`} y1={plotT} y2={plotB} x1={sx(t)} x2={sx(t)} stroke={C.grid} strokeWidth={1} />
          ))}

          {/* Tick labels */}
          {yAxis.ticks.map((t) => (
            <g key={`ty${t}`}>
              <line x1={plotL - 4} x2={plotL} y1={sy(t)} y2={sy(t)} stroke={C.muted} strokeWidth={1} />
              <text
                x={plotL - 7}
                y={sy(t) + 3.6}
                textAnchor="end"
                fill={C.muted}
                fontSize={11}
                fontFamily={FONT_MONO}
              >
                {formatTick(t, yAxis.step)}
              </text>
            </g>
          ))}
          {xAxis.ticks.map((t) => (
            <g key={`tx${t}`}>
              <line x1={sx(t)} x2={sx(t)} y1={plotB} y2={plotB + 4} stroke={C.muted} strokeWidth={1} />
              <text
                x={sx(t)}
                y={plotB + 16}
                textAnchor="middle"
                fill={C.muted}
                fontSize={11}
                fontFamily={FONT_MONO}
              >
                {formatTick(t, xAxis.step)}
              </text>
            </g>
          ))}

          {/* Reference lines */}
          <g clipPath={`url(#${clipId})`}>
            {markers.map((m, i) => {
              const color = m.color ?? C.marker;
              if (m.axis === 'x') {
                const px = sx(m.value);
                const flip = px > plotR - 70;
                return (
                  <g key={`m${i}`}>
                    <line
                      x1={px}
                      x2={px}
                      y1={plotT}
                      y2={plotB}
                      stroke={color}
                      strokeWidth={1.25}
                      strokeDasharray="5 4"
                    />
                    {m.label && (
                      <text
                        x={flip ? px - 5 : px + 5}
                        y={plotT + 12 + (markerLane.get(i) ?? 0) * 13}
                        textAnchor={flip ? 'end' : 'start'}
                        fill={color}
                        fontSize={11}
                        fontFamily={FONT_SANS}
                      >
                        {m.label}
                      </text>
                    )}
                  </g>
                );
              }
              const py = sy(m.value);
              return (
                <g key={`m${i}`}>
                  <line
                    x1={plotL}
                    x2={plotR}
                    y1={py}
                    y2={py}
                    stroke={color}
                    strokeWidth={1.25}
                    strokeDasharray="5 4"
                  />
                  {m.label && (
                    <text
                      x={plotR - 5}
                      y={py - 5}
                      textAnchor="end"
                      fill={color}
                      fontSize={11}
                      fontFamily={FONT_SANS}
                    >
                      {m.label}
                    </text>
                  )}
                </g>
              );
            })}
          </g>

          {/* Data */}
          <g clipPath={`url(#${clipId})`}>
            {drawn.map((item, i) => (
              <g key={`s${i}`}>
                {item.d.length > 0 && (
                  <path
                    d={item.d}
                    fill="none"
                    stroke={item.s.color}
                    strokeWidth={1.8}
                    strokeLinejoin="round"
                    strokeLinecap="round"
                    strokeDasharray={item.s.dashed ? '5 4' : undefined}
                  />
                )}
                {/* Individual measurements are worth seeing when they are few,
                    and a one-point series would otherwise be invisible. */}
                {item.pts.length <= 60 &&
                  item.pts.map((p, j) => (
                    <circle
                      key={`p${j}`}
                      cx={p.px}
                      cy={p.py}
                      r={item.pts.length === 1 ? 3 : 2}
                      fill={item.s.color}
                    />
                  ))}
              </g>
            ))}
          </g>
        </g>
      )}

      {!hasData && (
        <text
          x={plotL + plotW / 2}
          y={plotT + plotH / 2}
          textAnchor="middle"
          fill={C.muted}
          fontSize={12}
          fontFamily={FONT_SANS}
        >
          no data yet
        </text>
      )}

      {/* Axis labels */}
      <text
        x={plotL + plotW / 2}
        y={height - 8}
        textAnchor="middle"
        fill={C.fg}
        fontSize={12}
        fontFamily={FONT_SANS}
      >
        {xLabel}
      </text>
      <text
        transform={`translate(14 ${plotT + plotH / 2}) rotate(-90)`}
        textAnchor="middle"
        fill={C.fg}
        fontSize={12}
        fontFamily={FONT_SANS}
      >
        {yLabel}
      </text>

      {/* Legend */}
      {legend.rows.map((entry, i) => {
        const y = MARGIN_T_BASE + entry.row * LEGEND_ROW_H - 2;
        return (
          <g key={`l${i}`}>
            <line
              x1={entry.x}
              x2={entry.x + 16}
              y1={y}
              y2={y}
              stroke={entry.s.color}
              strokeWidth={2.4}
              strokeDasharray={entry.s.dashed ? '4 3' : undefined}
            />
            <text
              x={entry.x + 20}
              y={y + 3.8}
              fill={C.muted}
              fontSize={11}
              fontFamily={FONT_SANS}
            >
              {entry.s.label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
