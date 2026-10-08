"use client";

import { useEffect, useMemo, useState } from "react";
import Shell from "../../components/Shell";

// =============================================================================
// Types — mirror benchmark.json produced by compute_metrics.py
// =============================================================================

type MetricEntry = {
  image: string;
  psnr: number;
  ssim: number;
  lpips: number | null;
} | null;

type SceneEntry = {
  id: string;
  slug: string;
  label: string;
  lr_image: string;
  gt_image: string;
  lr_size: [number, number];
  gt_size: [number, number];
  results: {
    esrgan: Record<string, MetricEntry>;
    diffusion: Record<string, MetricEntry>;
  };
};

type ModelKey = "esrgan" | "diffusion";

type AverageEntry = {
  psnr: number;
  ssim: number;
  lpips: number | null;
  n: number;
} | null;

type BenchmarkData = {
  generated_at: string;
  scale: number;
  models: Record<ModelKey, { label: string; strategies: string[] }>;
  scenes: SceneEntry[];
  averages: Record<ModelKey, Record<string, AverageEntry>>;
};

const ALL_STRATEGIES = ["zero_shot", "head_only", "partial", "lora", "full"];

function formatStrategy(strategy: string) {
  return strategy.replace(/_/g, "-");
}

function fmtPsnr(value: number | null | undefined) {
  return value == null ? "—" : `${value.toFixed(2)} dB`;
}

function fmtSsim(value: number | null | undefined) {
  return value == null ? "—" : value.toFixed(3);
}

function fmtLpips(value: number | null | undefined) {
  return value == null ? "—" : value.toFixed(3);
}

// =============================================================================
// Component
// =============================================================================

export default function BenchmarkPage() {
  const [data, setData] = useState<BenchmarkData | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [sceneSlug, setSceneSlug] = useState<string | null>(null);
  const [slider, setSlider] = useState(50);
  const [model, setModel] = useState<"All" | ModelKey>("All");
  const [strategy, setStrategy] = useState<string>("All");

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const response = await fetch("/benchmark/benchmark.json", {
          cache: "no-store",
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }

        const payload: BenchmarkData = await response.json();

        if (cancelled) return;

        setData(payload);
        setLoadError(null);
        setSceneSlug((current) => current ?? payload.scenes[0]?.slug ?? null);
      } catch (error) {
        if (cancelled) return;

        console.error("Failed to load benchmark data:", error);

        setLoadError(
          "Could not load /benchmark/benchmark.json. Make sure the " +
            "benchmark JSON + images are in the public/benchmark folder."
        );
      }
    }

    load();

    return () => {
      cancelled = true;
    };
  }, []);

  const scene = useMemo(
    () => data?.scenes.find((item) => item.slug === sceneSlug) ?? null,
    [data, sceneSlug]
  );

  // Strategies actually usable in the dropdown: union of what each model
  // supports, in a fixed sensible order.
  const availableStrategies = useMemo(() => {
    if (!data) return ALL_STRATEGIES;

    const set = new Set<string>();
    (Object.keys(data.models) as ModelKey[]).forEach((key) => {
      data.models[key].strategies.forEach((s) => set.add(s));
    });

    return ALL_STRATEGIES.filter((s) => set.has(s));
  }, [data]);

  // The concrete strategy used to pick images for the compare view — "All"
  // isn't a real checkpoint, so it falls back to zero-shot.
  const compareStrategy = strategy === "All" ? "zero_shot" : strategy;

  const esrganEntry: MetricEntry =
    scene?.results.esrgan[compareStrategy] ?? null;

  const diffusionEntryRaw: MetricEntry =
    scene?.results.diffusion[compareStrategy] ?? null;

  const diffusionFallback = !diffusionEntryRaw && compareStrategy === "lora";
  const diffusionEntry: MetricEntry =
    diffusionEntryRaw ?? scene?.results.diffusion["zero_shot"] ?? null;

  // Top metric cards: average over the current scene, filtered by the
  // selected model(s) and strategy(ies).
  const sliceAverage = useMemo(() => {
    if (!scene) return null;

    const models: ModelKey[] =
      model === "All" ? ["esrgan", "diffusion"] : [model];

    const strategies =
      strategy === "All" ? availableStrategies : [strategy];

    const entries: MetricEntry[] = [];

    for (const m of models) {
      for (const s of strategies) {
        const entry = scene.results[m][s];
        if (entry) entries.push(entry);
      }
    }

    if (entries.length === 0) return null;

    const psnrAvg =
      entries.reduce((sum, e) => sum + (e as NonNullable<MetricEntry>).psnr, 0) /
      entries.length;

    const ssimAvg =
      entries.reduce((sum, e) => sum + (e as NonNullable<MetricEntry>).ssim, 0) /
      entries.length;

    const lpipsEntries = entries.filter(
      (e) => (e as NonNullable<MetricEntry>).lpips != null
    );

    const lpipsAvg =
      lpipsEntries.length === 0
        ? null
        : lpipsEntries.reduce(
            (sum, e) => sum + ((e as NonNullable<MetricEntry>).lpips as number),
            0
          ) / lpipsEntries.length;

    return { psnr: psnrAvg, ssim: ssimAvg, lpips: lpipsAvg, n: entries.length };
  }, [scene, model, strategy, availableStrategies]);

  const loading = !data && !loadError;

  return (
    <Shell active="/benchmark">
      <div className="page-intro">
        <div className="eyebrow">03 / benchmark explorer</div>
        <h1 className="hero-title">Benchmark Dashboard</h1>
        <p className="hero-copy">
          Explore quantitative results, scene-specific behavior, side-by-side
          outputs and the model × strategy matrix, computed from a
          precomputed offline evaluation run
          {data ? ` across ${data.scenes.length} scenes` : ""}.
        </p>
      </div>

      {loadError && (
        <section className="panel">
          <div className="panel-body">
            <strong>Benchmark data warning</strong>
            <p style={{ marginTop: 8 }}>{loadError}</p>
          </div>
        </section>
      )}

      {loading && (
        <section className="panel">
          <div className="panel-body">Loading benchmark data…</div>
        </section>
      )}

      {data && scene && (
        <>
          <div className="dashboard-grid">
            <div className="metric-card">
              <div className="metric-label">Average PSNR</div>
              <div className="metric-value">
                {fmtPsnr(sliceAverage?.psnr)}
              </div>
              <div className="metric-note">
                {scene.label} · selected benchmark slice
              </div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Average SSIM</div>
              <div className="metric-value">
                {fmtSsim(sliceAverage?.ssim)}
              </div>
              <div className="metric-note">
                {scene.label} · selected benchmark slice
              </div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Average LPIPS</div>
              <div className="metric-value">
                {fmtLpips(sliceAverage?.lpips)}
              </div>
              <div className="metric-note">
                {scene.label} · lower is better
              </div>
            </div>
          </div>

          <div className="dashboard-layout">
            <section className="panel">
              <div className="panel-header">
                <div className="panel-title">Scene explorer</div>
                <span className="status-chip">{scene.label}</span>
              </div>
              <div className="panel-body">
                <div className="scene-tabs">
                  {data.scenes.map((item) => (
                    <button
                      key={item.slug}
                      className={`scene-tab ${
                        item.slug === scene.slug ? "scene-tab-active" : ""
                      }`}
                      onClick={() => setSceneSlug(item.slug)}
                    >
                      {item.label}
                    </button>
                  ))}
                </div>

                <div
                  className="compare-view"
                  style={{
                    position: "relative",
                    aspectRatio: "1 / 1",
                    height: "auto",
                    overflow: "hidden",
                  }}
                >
                  <div className="compare-side compare-left">
                    {esrganEntry ? (
                      <img
                        src={esrganEntry.image}
                        alt={`Real-ESRGAN ${formatStrategy(compareStrategy)} output`}
                        style={{
                          width: "100%",
                          height: "100%",
                          objectFit: "cover",
                          display: "block",
                        }}
                      />
                    ) : (
                      "no checkpoint for this strategy"
                    )}
                  </div>

                  <div
                    className="compare-side compare-right"
                    style={{ clipPath: `inset(0 0 0 ${slider}%)` }}
                  >
                    {diffusionEntry ? (
                      <img
                        src={diffusionEntry.image}
                        alt={`Diffusion ${formatStrategy(compareStrategy)} output`}
                        style={{
                          width: "100%",
                          height: "100%",
                          objectFit: "cover",
                          display: "block",
                        }}
                      />
                    ) : (
                      "no checkpoint for this strategy"
                    )}
                  </div>

                  <div
                    className="compare-divider"
                    style={{ left: `${slider}%` }}
                  />

                  <span
                    className="status-chip"
                    style={{ position: "absolute", top: 10, left: 10 }}
                  >
                    ESRGAN / {formatStrategy(compareStrategy)}
                    {esrganEntry ? ` · ${fmtPsnr(esrganEntry.psnr)}` : ""}
                  </span>

                  <span
                    className="status-chip"
                    style={{ position: "absolute", top: 10, right: 10 }}
                  >
                    Diffusion /{" "}
                    {formatStrategy(
                      diffusionFallback ? "zero_shot" : compareStrategy
                    )}
                    {diffusionEntry ? ` · ${fmtPsnr(diffusionEntry.psnr)}` : ""}
                  </span>
                </div>

                {diffusionFallback && (
                  <div
                    style={{ marginTop: 10, fontSize: 12, opacity: 0.68 }}
                  >
                    Diffusion has no LoRA checkpoint in this run — showing
                    zero-shot on the right instead.
                  </div>
                )}

                <input
                  className="compare-control"
                  type="range"
                  min="0"
                  max="100"
                  value={slider}
                  onChange={(event) => setSlider(Number(event.target.value))}
                  aria-label="Comparison slider"
                />
              </div>
            </section>

            <section className="panel">
              <div className="panel-header">
                <div className="panel-title">Filters</div>
                <button
                  type="button"
                  className="secondary-button"
                  onClick={() => {
                    setModel("All");
                    setStrategy("All");
                  }}
                  disabled={model === "All" && strategy === "All"}
                  style={{ fontSize: 12, padding: "6px 12px" }}
                >
                  Reset
                </button>
              </div>
              <div className="panel-body">
                <label className="group-label">Model</label>
                <div
                  style={{
                    display: "flex",
                    gap: 6,
                    marginTop: 8,
                    marginBottom: 18,
                  }}
                >
                  {(["All", "esrgan", "diffusion"] as const).map((value) => {
                    const active = model === value;
                    const label =
                      value === "All" ? "All" : data.models[value].label;

                    return (
                      <button
                        key={value}
                        type="button"
                        onClick={() => {
                          setModel(value);
                          // Diffusion has no LoRA checkpoint in this run —
                          // don't leave the strategy filter stuck on a
                          // combo that can never match anything.
                          if (value === "diffusion" && strategy === "lora") {
                            setStrategy("All");
                          }
                        }}
                        className={`scene-tab ${
                          active ? "scene-tab-active" : ""
                        }`}
                        style={{ flex: 1, textAlign: "center" }}
                      >
                        {label}
                      </button>
                    );
                  })}
                </div>

                <label className="group-label">Strategy</label>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
                    gap: 6,
                    marginTop: 8,
                  }}
                >
                  {["All", ...availableStrategies].map((item) => {
                    const active = strategy === item;
                    const disabled =
                      item !== "All" &&
                      model !== "All" &&
                      !data.models[model].strategies.includes(item);

                    return (
                      <button
                        key={item}
                        type="button"
                        disabled={disabled}
                        onClick={() => setStrategy(item)}
                        className={`scene-tab ${
                          active ? "scene-tab-active" : ""
                        }`}
                        style={{
                          opacity: disabled ? 0.35 : 1,
                          cursor: disabled ? "not-allowed" : "pointer",
                        }}
                        title={
                          disabled && model !== "All"
                            ? `${data.models[model].label} has no ${formatStrategy(
                                item
                              )} checkpoint`
                            : undefined
                        }
                      >
                        {item === "All" ? "All" : formatStrategy(item)}
                      </button>
                    );
                  })}
                </div>

                <div
                  style={{
                    fontSize: 11,
                    opacity: 0.55,
                    marginTop: 10,
                    marginBottom: 18,
                  }}
                >
                  Model narrows the averages above — the comparison view
                  always shows Real-ESRGAN vs Diffusion side by side.
                </div>

                <div className="metric-card">
                  <div className="metric-label">
                    Head-to-head · {scene.label} ·{" "}
                    {formatStrategy(compareStrategy)}
                  </div>
                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns: "1fr 1fr",
                      gap: 12,
                      marginTop: 10,
                    }}
                  >
                    <div>
                      <div style={{ fontSize: 12, opacity: 0.7 }}>
                        Real-ESRGAN
                      </div>
                      <div
                        style={{
                          fontWeight: 700,
                          fontSize: 18,
                          marginTop: 4,
                        }}
                      >
                        {fmtPsnr(esrganEntry?.psnr)}
                      </div>
                      <div
                        style={{ fontSize: 12, opacity: 0.75, marginTop: 2 }}
                      >
                        SSIM {fmtSsim(esrganEntry?.ssim)} · LPIPS{" "}
                        {fmtLpips(esrganEntry?.lpips)}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: 12, opacity: 0.7 }}>
                        Diffusion{diffusionFallback ? " (zero-shot)" : ""}
                      </div>
                      <div
                        style={{
                          fontWeight: 700,
                          fontSize: 18,
                          marginTop: 4,
                        }}
                      >
                        {fmtPsnr(diffusionEntry?.psnr)}
                      </div>
                      <div
                        style={{ fontSize: 12, opacity: 0.75, marginTop: 2 }}
                      >
                        SSIM {fmtSsim(diffusionEntry?.ssim)} · LPIPS{" "}
                        {fmtLpips(diffusionEntry?.lpips)}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </section>
          </div>

          <section className="panel" style={{ marginTop: 18 }}>
            <div className="panel-header">
              <div className="panel-title">Strategy matrix</div>
              <span className="status-chip">
                avg PSNR across all {data.scenes.length} scenes
              </span>
            </div>
            <div className="panel-body">
              <table className="matrix">
                <thead>
                  <tr>
                    <th>Model</th>
                    {ALL_STRATEGIES.map((item) => (
                      <th key={item}>{formatStrategy(item)}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(Object.keys(data.models) as ModelKey[]).map((m) => (
                    <tr key={m}>
                      <td>{data.models[m].label}</td>
                      {ALL_STRATEGIES.map((s) => {
                        const avg = data.averages[m]?.[s];

                        return (
                          <td
                            key={s}
                            title={
                              avg
                                ? `PSNR ${fmtPsnr(avg.psnr)} · SSIM ${fmtSsim(
                                    avg.ssim
                                  )} · LPIPS ${fmtLpips(avg.lpips)}`
                                : "no checkpoint"
                            }
                          >
                            <div
                              className={`matrix-dot ${
                                avg ? "" : "matrix-dot-muted"
                              }`}
                            />
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </>
      )}
    </Shell>
  );
}