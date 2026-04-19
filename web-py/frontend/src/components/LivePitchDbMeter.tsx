import type { AudioAnalysis } from '../lib/audio';

/**
 * Live dBFS + pitch meters with a highlighted "sweet spot" band on each axis.
 *
 * Drop-in, read-only visualiser: pass the latest `AudioAnalysis` snapshot and
 * the desired targets, and the component renders a horizontal loudness bar
 * (with sweet-spot shading + edge tick lines) on top of a vertical pitch
 * column (with the same sweet-spot shading). Purely theme-token styled so it
 * tracks light/dark with the rest of the UI.
 */

export interface LivePitchDbMeterProps {
  /** Latest audio analysis frame, or null when idle. */
  analysis: AudioAnalysis | null;
  /** Centre of the dBFS sweet spot (e.g. -18). */
  targetDbfs?: number;
  /** Half-width of the dBFS sweet spot, in dB. Band = target ± this. */
  dbfsTolerance?: number;
  /** Sweet-spot pitch range in Hz. Defaults to 110..180 (typical adult speech). */
  pitchSweetSpot?: [number, number];
  /** Whether the stream is live. When false, the meters render inert. */
  active?: boolean;
  /** Optional heading shown above the meters. */
  title?: string;
}

const DBFS_MIN = -60;
const DBFS_MAX = 0;
const PITCH_MIN = 80;
const PITCH_MAX = 400;

const clamp01 = (x: number) => Math.max(0, Math.min(1, x));
const dbToFrac = (db: number) =>
  isFinite(db) ? clamp01((db - DBFS_MIN) / (DBFS_MAX - DBFS_MIN)) : 0;
const hzToFrac = (hz: number) =>
  clamp01((hz - PITCH_MIN) / (PITCH_MAX - PITCH_MIN));

export default function LivePitchDbMeter({
  analysis,
  targetDbfs = -18,
  dbfsTolerance = 8,
  pitchSweetSpot = [95, 240],
  active = true,
  title = 'Live signal',
}: LivePitchDbMeterProps) {
  const dbfs = analysis?.dbfs ?? -Infinity;
  const pitchHz = analysis?.pitchHz ?? null;
  const speaking = (analysis?.speaking ?? false) && active;

  const loudFrac = dbToFrac(dbfs);
  // Sweet-spot band is biased a few dB quieter than the target: missing the
  // band on the loud side is a bigger problem than on the quiet side, so we
  // want the visual "green zone" to sit a little to the left of the target.
  const SWEET_SPOT_OFFSET_DB = -8;
  const dbLoFrac = dbToFrac(targetDbfs - dbfsTolerance + SWEET_SPOT_OFFSET_DB);
  const dbHiFrac = dbToFrac(targetDbfs + dbfsTolerance + SWEET_SPOT_OFFSET_DB);
  const dbCenterFrac = dbToFrac(targetDbfs);

  const inDbSweetSpot =
    active && isFinite(dbfs) && Math.abs(dbfs - targetDbfs) <= dbfsTolerance;

  const [pitchLo, pitchHi] = pitchSweetSpot;
  const pitchLoFrac = hzToFrac(pitchLo);
  const pitchHiFrac = hzToFrac(pitchHi);
  const pitchValueFrac =
    pitchHz != null
      ? hzToFrac(Math.max(PITCH_MIN, Math.min(PITCH_MAX, pitchHz)))
      : null;
  const inPitchSweetSpot =
    active && pitchHz != null && pitchHz >= pitchLo && pitchHz <= pitchHi;

  const loudBarColor = !active
    ? 'var(--border-strong)'
    : loudFrac > 0.95
    ? 'var(--danger)'
    : inDbSweetSpot
    ? 'var(--accent)'
    : 'var(--accent-strong)';

  return (
    <div className="card p-4">
      <div className="mb-3 flex items-center justify-between text-[10px] uppercase tracking-wider text-[var(--text-faint)]">
        <span>{title}</span>
        <span className="flex items-center gap-1.5">
          <span
            className={
              'inline-block h-2 w-2 rounded-full transition ' +
              (speaking
                ? 'bg-[var(--accent)] shadow-[0_0_6px_2px_var(--accent-ring)]'
                : active
                ? 'bg-[var(--text-faint)]'
                : 'bg-[var(--border-strong)]')
            }
          />
          {speaking ? 'voice' : active ? 'silence' : 'idle'}
        </span>
      </div>

      <div className="grid grid-cols-[1fr_auto] gap-5">
        {/* ---- Loudness row ---------------------------------------------- */}
        <div>
          <div className="flex items-center justify-between text-[10px] uppercase tracking-wider text-[var(--text-faint)]">
            <span>Loudness</span>
            <span className="font-mono normal-case tracking-normal text-[var(--text-muted)]">
              {active && isFinite(dbfs) ? `${dbfs.toFixed(1)} dBFS` : '—'}
              <span className="ml-2 text-[var(--text-faint)]">
                sweet {targetDbfs - dbfsTolerance}…{targetDbfs + dbfsTolerance}
              </span>
            </span>
          </div>
          <div className="relative mt-2 h-3 w-full overflow-hidden rounded-full bg-[var(--surface-inset)]">
            {/* Live bar. */}
            <div
              className="absolute inset-y-0 left-0 rounded-full transition-[width] duration-75"
              style={{
                width: `${(loudFrac * 100).toFixed(2)}%`,
                background: loudBarColor,
              }}
            />
            {/* Sweet-spot band — rendered on top of the bar at partial opacity
             * so it tints rather than obscures the live fill. */}
            <div
              className="pointer-events-none absolute inset-y-0 bg-[var(--accent)]/35"
              style={{
                left: `${(dbLoFrac * 100).toFixed(2)}%`,
                width: `${((dbHiFrac - dbLoFrac) * 100).toFixed(2)}%`,
              }}
            />
            {/* Sweet-spot edge ticks. */}
            <SweetSpotTick frac={dbLoFrac} />
            <SweetSpotTick frac={dbHiFrac} />
            {/* Target centre marker (slightly stronger). */}
            <div
              className="pointer-events-none absolute top-[-2px] h-[calc(100%+4px)] w-0.5 bg-[var(--text)]/60"
              style={{ left: `calc(${(dbCenterFrac * 100).toFixed(2)}% - 1px)` }}
              title={`target ${targetDbfs} dBFS`}
            />
          </div>
          {/* Axis labels below the bar. */}
          <div className="mt-1 flex justify-between font-mono text-[9px] text-[var(--text-faint)]">
            <span>{DBFS_MIN}</span>
            <span>{targetDbfs}</span>
            <span>{DBFS_MAX}</span>
          </div>
        </div>

        {/* ---- Pitch column ---------------------------------------------- */}
        <div className="flex w-24 flex-col items-stretch">
          <div className="flex items-center justify-between text-[10px] uppercase tracking-wider text-[var(--text-faint)]">
            <span>Pitch</span>
          </div>
          <div className="relative mt-2 h-24 w-full overflow-hidden rounded-md border border-[var(--border)] bg-[var(--surface-inset)]">
            {/* Sweet-spot band (bottom = PITCH_MIN, top = PITCH_MAX). */}
            <div
              className="pointer-events-none absolute inset-x-0 bg-[var(--accent)]/35"
              style={{
                bottom: `${(pitchLoFrac * 100).toFixed(2)}%`,
                height: `${((pitchHiFrac - pitchLoFrac) * 100).toFixed(2)}%`,
              }}
            />
            {/* Sweet-spot edge lines. */}
            <PitchSweetLine frac={pitchLoFrac} />
            <PitchSweetLine frac={pitchHiFrac} />
            {/* Live dot. */}
            {pitchValueFrac != null && active && (
              <div
                className={
                  'absolute left-1/2 h-3 w-3 -translate-x-1/2 translate-y-1/2 rounded-full transition-[bottom] duration-100 ' +
                  (inPitchSweetSpot
                    ? 'bg-[var(--accent-fg)] ring-2 ring-[var(--accent-strong)] shadow-[0_0_6px_2px_var(--accent-ring)]'
                    : 'bg-[var(--info)] shadow-[0_0_6px_2px_var(--info-soft)]')
                }
                style={{ bottom: `${(pitchValueFrac * 100).toFixed(2)}%` }}
              />
            )}
          </div>
          <div className="mt-1 text-center font-mono text-[11px] text-[var(--text-muted)]">
            {active && pitchHz != null ? `${Math.round(pitchHz)} Hz` : '—'}
          </div>
          <div className="text-center font-mono text-[9px] text-[var(--text-faint)]">
            sweet {pitchLo}…{pitchHi} Hz
          </div>
        </div>
      </div>
    </div>
  );
}

function SweetSpotTick({ frac }: { frac: number }) {
  return (
    <div
      className="pointer-events-none absolute top-[-2px] h-[calc(100%+4px)] w-px bg-[var(--accent-strong)]"
      style={{ left: `${(frac * 100).toFixed(2)}%` }}
    />
  );
}

function PitchSweetLine({ frac }: { frac: number }) {
  return (
    <div
      className="pointer-events-none absolute inset-x-0 h-px bg-[var(--accent-strong)]"
      style={{ bottom: `${(frac * 100).toFixed(2)}%` }}
    />
  );
}
