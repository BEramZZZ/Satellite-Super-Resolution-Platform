"use client";

import {
  ChangeEvent,
  DragEvent,
  useEffect,
  useMemo,
  useState,
} from "react";

import Shell from "../../components/Shell";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

type Strategy =
  | "zero-shot"
  | "head-only"
  | "partial"
  | "lora"
  | "full";

type BackendModel = {
  id?: string;
  name?: string;
  model?: string;
  family?: string;
  strategy?: string;
  strategy_id?: string;
  available?: boolean;
  exists?: boolean;
};

type SRResult = {
  id: string;
  model: string;
  strategy: string;
  imageUrl: string;
  psnr: string | null;
  ssim: string | null;
  lpips: string | null;
  metricsAvailable: boolean;
  metricsError: string | null;
  scale: string;
  device: string;
};

const ESRGAN_STRATEGIES: Strategy[] = [
  "zero-shot",
  "head-only",
  "partial",
  "lora",
  "full",
];

const DIFFUSION_STRATEGIES: Strategy[] = [
  "zero-shot",
  "head-only",
  "partial",
  "full",
];

const INITIAL_SELECTION = [
  "Real-ESRGAN / zero-shot",
  "Real-ESRGAN / head-only",
];

function normalizeStrategy(
  value: string | undefined
): Strategy | null {
  if (!value) return null;

  const normalized = value
    .toLowerCase()
    .trim()
    .replace(/_/g, "-");

  switch (normalized) {
    case "zero":
    case "zero-shot":
      return "zero-shot";

    case "head":
    case "head-only":
      return "head-only";

    case "partial":
      return "partial";

    case "lora":
      return "lora";

    case "full":
      return "full";

    default:
      return null;
  }
}

/*
 * The ESRGAN backend accepts both hyphenated and underscored strategy
 * names (see _normalize_esrgan_strategy on the server). The diffusion
 * backend does not have that alias layer -- it only does
 * strategy.strip().lower() -- so "head-only" must be converted to
 * "head_only" before it reaches /api/inference/diffusion.
 */
function toBackendStrategy(
  family: "Real-ESRGAN" | "Diffusion",
  strategy: Strategy
): string {
  if (family === "Diffusion") {
    return strategy.replace(/-/g, "_");
  }

  return strategy;
}

function formatMetric(value: string | null) {
  if (!value) return "—";

  const parsed = Number(value);

  if (!Number.isFinite(parsed)) {
    return value;
  }

  return parsed.toFixed(4);
}

function modelFamily(model: BackendModel) {
  return (
    model.family ||
    model.model ||
    model.name ||
    model.id ||
    ""
  ).toLowerCase();
}

function isESRGAN(model: BackendModel) {
  const family = modelFamily(model);

  return (
    family.includes("esrgan") ||
    family.includes("real-esrgan") ||
    family.includes("real_esrgan")
  );
}

function strategyAvailable(
  models: BackendModel[],
  strategy: Strategy
) {
  /*
   * Keep the UI usable if /api/models has not responded
   * or if the registry uses a slightly different shape.
   */
  if (models.length === 0) {
    return true;
  }

  const matches = models.filter((model) => {
    if (!isESRGAN(model)) {
      return false;
    }

    const backendStrategy = normalizeStrategy(
      model.strategy ||
        model.strategy_id ||
        model.id ||
        model.name
    );

    return backendStrategy === strategy;
  });

  if (matches.length === 0) {
    return true;
  }

  return matches.some(
    (model) =>
      model.available !== false &&
      model.exists !== false
  );
}

export default function LabPage() {
  const [lrFile, setLrFile] =
    useState<File | null>(null);

  const [lrPreview, setLrPreview] =
    useState<string | null>(null);

  const [lrDimensions, setLrDimensions] =
    useState<string | null>(null);

  const [hrFile, setHrFile] =
    useState<File | null>(null);

  const [hrPreview, setHrPreview] =
    useState<string | null>(null);

  const [hrDimensions, setHrDimensions] =
    useState<string | null>(null);

  const [dragActive, setDragActive] =
    useState(false);

  const [selected, setSelected] =
    useState<string[]>(INITIAL_SELECTION);

  const [backendModels, setBackendModels] =
    useState<BackendModel[]>([]);

  const [backendError, setBackendError] =
    useState<string | null>(null);

  const [runError, setRunError] =
    useState<string | null>(null);

  const [running, setRunning] =
    useState(false);

  const [progress, setProgress] =
    useState(0);

  const [currentPipeline, setCurrentPipeline] =
    useState<string | null>(null);

  const [completedPipelines, setCompletedPipelines] =
    useState<Set<string>>(new Set());

  const [results, setResults] =
    useState<SRResult[]>([]);

  const selectedESRGAN = useMemo(
    () =>
      selected.filter((item) =>
        item.startsWith("Real-ESRGAN / ")
      ),
    [selected]
  );

  const selectedDiffusion = useMemo(
    () =>
      selected.filter((item) =>
        item.startsWith("Diffusion / ")
      ),
    [selected]
  );

  /*
   * Load available models.
   */
  useEffect(() => {
    let cancelled = false;

    async function loadModels() {
      try {
        setBackendError(null);

        const response = await fetch(
          `${API_BASE_URL}/api/models`,
          {
            method: "GET",
            cache: "no-store",
          }
        );

        if (!response.ok) {
          throw new Error(
            `HTTP ${response.status}`
          );
        }

        const payload = await response.json();

        const models: BackendModel[] =
          Array.isArray(payload)
            ? payload
            : Array.isArray(payload?.models)
            ? payload.models
            : [];

        if (!cancelled) {
          setBackendModels(models);
        }
      } catch (error) {
        if (cancelled) return;

        console.error(
          "Failed to load models:",
          error
        );

        /*
         * Do not disable the entire lab because
         * the registry request failed.
         */
        setBackendModels([]);

        setBackendError(
          `Could not connect to ${API_BASE_URL}`
        );
      }
    }

    loadModels();

    return () => {
      cancelled = true;
    };
  }, []);

  /*
   * Clean up preview URLs.
   */
  useEffect(() => {
    return () => {
      if (lrPreview) {
        URL.revokeObjectURL(lrPreview);
      }
    };
  }, [lrPreview]);

  useEffect(() => {
    return () => {
      if (hrPreview) {
        URL.revokeObjectURL(hrPreview);
      }
    };
  }, [hrPreview]);

  /*
   * Clean up result image URLs.
   */
  useEffect(() => {
    return () => {
      results.forEach((result) => {
        URL.revokeObjectURL(result.imageUrl);
      });
    };
  }, [results]);

  function readImageDimensions(
    file: File,
    callback: (
      width: number,
      height: number
    ) => void
  ) {
    const objectUrl =
      URL.createObjectURL(file);

    const image = new Image();

    image.onload = () => {
      callback(
        image.naturalWidth,
        image.naturalHeight
      );

      URL.revokeObjectURL(objectUrl);
    };

    image.onerror = () => {
      URL.revokeObjectURL(objectUrl);
    };

    image.src = objectUrl;
  }

  function clearResults() {
    setResults((oldResults) => {
      oldResults.forEach((result) => {
        URL.revokeObjectURL(result.imageUrl);
      });

      return [];
    });

    setProgress(0);
    setCurrentPipeline(null);
    setCompletedPipelines(new Set());
  }

  function acceptLRFile(file?: File) {
    if (!file) return;

    const allowedTypes = [
      "image/png",
      "image/jpeg",
      "image/webp",
    ];

    if (!allowedTypes.includes(file.type)) {
      setRunError(
        "Please upload a PNG, JPEG, or WebP image."
      );
      return;
    }

    setRunError(null);

    clearResults();

    setLrFile(file);

    setLrDimensions(null);

    setLrPreview((oldPreview) => {
      if (oldPreview) {
        URL.revokeObjectURL(oldPreview);
      }

      return URL.createObjectURL(file);
    });

    readImageDimensions(
      file,
      (width, height) => {
        setLrDimensions(
          `${width} × ${height}`
        );
      }
    );
  }

  function acceptHRFile(file?: File) {
    if (!file) return;

    const allowedTypes = [
      "image/png",
      "image/jpeg",
      "image/webp",
    ];

    if (!allowedTypes.includes(file.type)) {
      setRunError(
        "Please upload a PNG, JPEG, or WebP HR reference."
      );
      return;
    }

    setRunError(null);

    setHrFile(file);
    setHrDimensions(null);

    setHrPreview((oldPreview) => {
      if (oldPreview) {
        URL.revokeObjectURL(oldPreview);
      }

      return URL.createObjectURL(file);
    });

    readImageDimensions(
      file,
      (width, height) => {
        setHrDimensions(
          `${width} × ${height}`
        );
      }
    );
  }

  function handleDrop(
    event: DragEvent<HTMLLabelElement>
  ) {
    event.preventDefault();
    setDragActive(false);

    acceptLRFile(
      event.dataTransfer.files?.[0]
    );
  }

  function handleInput(
    event: ChangeEvent<HTMLInputElement>
  ) {
    acceptLRFile(
      event.target.files?.[0]
    );
  }

  function handleHRInput(
    event: ChangeEvent<HTMLInputElement>
  ) {
    acceptHRFile(
      event.target.files?.[0]
    );
  }

  function togglePipeline(
    family: "Real-ESRGAN" | "Diffusion",
    strategy: Strategy
  ) {
    if (running) return;

    const value =
      `${family} / ${strategy}`;

    setSelected((current) =>
      current.includes(value)
        ? current.filter(
            (item) => item !== value
          )
        : [...current, value]
    );
  }

  function selectAllAvailable() {
    if (running) return;

    const esrgan = ESRGAN_STRATEGIES
      .filter((strategy) =>
        strategyAvailable(
          backendModels,
          strategy
        )
      )
      .map(
        (strategy) =>
          `Real-ESRGAN / ${strategy}`
      );

    const diffusion =
      DIFFUSION_STRATEGIES.map(
        (strategy) =>
          `Diffusion / ${strategy}`
      );

    setSelected([
      ...esrgan,
      ...diffusion,
    ]);
  }

  async function runESRGAN(
    strategy: Strategy,
    index: number,
    total: number
  ) {
    if (!lrFile) {
      throw new Error(
        "Please select an LR image first."
      );
    }

    const pipelineName =
      `Real-ESRGAN / ${strategy}`;

    setCurrentPipeline(pipelineName);

    const formData = new FormData();

    formData.append(
      "file",
      lrFile
    );

    formData.append(
      "strategy",
      toBackendStrategy("Real-ESRGAN", strategy)
    );

    if (hrFile) {
      formData.append(
        "reference_file",
        hrFile
      );
    }

    const response = await fetch(
      `${API_BASE_URL}/api/inference/esrgan`,
      {
        method: "POST",
        body: formData,
      }
    );

    if (!response.ok) {
      let detail =
        `HTTP ${response.status}`;

      try {
        const payload =
          await response.json();

        if (payload?.detail) {
          detail =
            typeof payload.detail ===
            "string"
              ? payload.detail
              : JSON.stringify(
                  payload.detail
                );
        }
      } catch {
        // Keep HTTP status.
      }

      throw new Error(
        `${pipelineName} failed: ${detail}`
      );
    }

    const blob =
      await response.blob();

    if (
      !blob.type.startsWith(
        "image/"
      )
    ) {
      throw new Error(
        "The backend did not return an image."
      );
    }

    const imageUrl =
      URL.createObjectURL(blob);

    const metricsAvailable =
      response.headers.get(
        "X-SR-Metrics-Available"
      ) === "true";

    const result: SRResult = {
      id:
        `${strategy}-${Date.now()}-${index}`,

      model:
        response.headers.get(
          "X-SR-Model"
        ) || "Real-ESRGAN",

      strategy:
        response.headers.get(
          "X-SR-Strategy"
        ) || strategy,

      imageUrl,

      psnr:
        response.headers.get(
          "X-SR-PSNR"
        ),

      ssim:
        response.headers.get(
          "X-SR-SSIM"
        ),

      lpips:
        response.headers.get(
          "X-SR-LPIPS"
        ),

      metricsAvailable,

      metricsError:
        response.headers.get(
          "X-SR-Metrics-Error"
        ),

      scale:
        response.headers.get(
          "X-SR-Scale"
        ) || "4×",

      device:
        response.headers.get(
          "X-SR-Device"
        ) || "unknown",
    };

    setResults((current) => [
      ...current,
      result,
    ]);

    setCompletedPipelines((prev) =>
      new Set(prev).add(pipelineName)
    );

    setProgress(
      Math.round(
        ((index + 1) / total) * 100
      )
    );
  }

  async function runDiffusion(
    strategy: Strategy,
    index: number,
    total: number
  ) {
    if (!lrFile) {
      throw new Error(
        "Please select an LR image first."
      );
    }

    const pipelineName =
      `Diffusion / ${strategy}`;

    setCurrentPipeline(pipelineName);

    const formData = new FormData();

    formData.append(
      "file",
      lrFile
    );

    formData.append(
      "strategy",
      toBackendStrategy("Diffusion", strategy)
    );

    // Note: /api/inference/diffusion does not accept a reference_file
    // yet, so PSNR/SSIM/LPIPS will show "—" for diffusion results
    // even when an HR reference has been uploaded.

    const response = await fetch(
      `${API_BASE_URL}/api/inference/diffusion`,
      {
        method: "POST",
        body: formData,
      }
    );

    if (!response.ok) {
      let detail =
        `HTTP ${response.status}`;

      try {
        const payload =
          await response.json();

        if (payload?.detail) {
          detail =
            typeof payload.detail ===
            "string"
              ? payload.detail
              : JSON.stringify(
                  payload.detail
                );
        }
      } catch {
        // Keep HTTP status.
      }

      throw new Error(
        `${pipelineName} failed: ${detail}`
      );
    }

    const blob =
      await response.blob();

    if (
      !blob.type.startsWith(
        "image/"
      )
    ) {
      throw new Error(
        "The backend did not return an image."
      );
    }

    const imageUrl =
      URL.createObjectURL(blob);

    const result: SRResult = {
      id:
        `diffusion-${strategy}-${Date.now()}-${index}`,

      model:
        response.headers.get(
          "X-SR-Model"
        ) || "Diffusion",

      strategy:
        response.headers.get(
          "X-SR-Strategy"
        ) || strategy,

      imageUrl,

      // Diffusion backend doesn't compute metrics yet.
      psnr: null,
      ssim: null,
      lpips: null,
      metricsAvailable: false,
      metricsError: null,

      scale:
        response.headers.get(
          "X-SR-Scale"
        ) || "4×",

      device:
        response.headers.get(
          "X-SR-Device"
        ) || "unknown",
    };

    setResults((current) => [
      ...current,
      result,
    ]);

    setCompletedPipelines((prev) =>
      new Set(prev).add(pipelineName)
    );

    setProgress(
      Math.round(
        ((index + 1) / total) * 100
      )
    );
  }

  async function runSelected() {
    if (!lrFile) {
      setRunError(
        "Please upload an LR image first."
      );
      return;
    }

    if (selected.length === 0) {
      setRunError(
        "Please select at least one pipeline."
      );
      return;
    }

    if (running) return;

    setRunError(null);
    setRunning(true);
    setProgress(0);
    setCurrentPipeline(null);

    clearResults();

    const totalPipelines =
      selectedESRGAN.length +
      selectedDiffusion.length;

    let runIndex = 0;

    try {
      for (const item of selectedESRGAN) {
        const strategy =
          normalizeStrategy(
            item.replace(
              "Real-ESRGAN / ",
              ""
            )
          );

        if (!strategy) {
          continue;
        }

        await runESRGAN(
          strategy,
          runIndex,
          totalPipelines
        );

        runIndex++;
      }

      for (const item of selectedDiffusion) {
        const strategy =
          normalizeStrategy(
            item.replace(
              "Diffusion / ",
              ""
            )
          );

        if (!strategy) {
          continue;
        }

        await runDiffusion(
          strategy,
          runIndex,
          totalPipelines
        );

        runIndex++;
      }

      setProgress(100);
      setCurrentPipeline(null);
    } catch (error) {
      console.error(
        "Inference error:",
        error
      );

      setRunError(
        error instanceof Error
          ? error.message
          : "Inference failed."
      );

      setCurrentPipeline(null);
    } finally {
      setRunning(false);
    }
  }

  function downloadResult(
    result: SRResult
  ) {
    const link =
      document.createElement("a");

    link.href = result.imageUrl;

    link.download =
      `sr_${result.model
        .toLowerCase()
        .replace(/\s+/g, "_")}_${result.strategy
        .toLowerCase()
        .replace(/\s+/g, "_")}.png`;

    document.body.appendChild(link);

    link.click();

    link.remove();
  }

  return (
    <Shell active="/lab">
      <div className="eyebrow">
        01 / interactive laboratory
      </div>

      <h1 className="hero-title">
        Satelite imagery <br></br>
        <span>Super-resolution</span>
      </h1>

      <p className="hero-copy">
        Upload an LR image, choose the exact
        model/strategy checkpoints you want
        to run, and compare the generated HR
        outputs in one workspace.
      </p>

      {backendError && (
        <section className="panel">
          <div className="panel-body">
            <strong>
              Backend connection warning
            </strong>

            <p style={{ marginTop: 8 }}>
              {backendError}
            </p>
          </div>
        </section>
      )}

      {runError && (
        <section className="panel">
          <div className="panel-body">
            <strong>
              Inference message
            </strong>

            <p style={{ marginTop: 8 }}>
              {runError}
            </p>
          </div>
        </section>
      )}

      {/* =========================================================
          TOP WORKSPACE
          Old upload preview + old pipeline selector
          ========================================================= */}

      <div className="panel-grid">
        {/* =======================================================
            UPLOAD
            ======================================================= */}

        <section className="panel">
          <div className="panel-header">
            <div className="panel-title">
              1. Upload low-resolution image
            </div>

            <span className="status-chip">
              4× SR
            </span>
          </div>

          <div className="panel-body">
            <label
              className={`dropzone ${
                dragActive
                  ? "dropzone-active"
                  : ""
              }`}
              onDragOver={(event) => {
                event.preventDefault();
                setDragActive(true);
              }}
              onDragLeave={() => {
                setDragActive(false);
              }}
              onDrop={handleDrop}
            >
              {lrPreview ? (
                <div
                  style={{
                    width: "100%",
                  }}
                >
                  <div
                    style={{
                      width: "100%",
                      height: 280,
                      borderRadius: 8,
                      overflow: "hidden",
                      marginBottom: 14,
                      background:
                        "rgba(0,0,0,0.18)",
                      border:
                        "1px solid rgba(255,255,255,0.08)",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                    }}
                  >
                    <img
                      src={lrPreview}
                      alt="Uploaded LR image preview"
                      style={{
                        width: "100%",
                        height: "100%",
                        objectFit: "contain",
                        display: "block",
                      }}
                    />
                  </div>

                  <div className="dropzone-title">
                    {lrFile?.name}
                  </div>

                  <div
                    className="dropzone-copy"
                    style={{
                      marginTop: 6,
                    }}
                  >
                    {lrDimensions
                      ? lrDimensions
                      : "Reading image dimensions…"}
                    {lrFile
                      ? ` · ${(
                          lrFile.size /
                          1024 /
                          1024
                        ).toFixed(2)} MB`
                      : ""}
                  </div>

                  <div
                    style={{
                      display: "flex",
                      justifyContent:
                        "center",
                      gap: 8,
                      marginTop: 14,
                    }}
                  >
                    <span className="secondary-button">
                      Choose another image
                    </span>
                  </div>
                </div>
              ) : (
                <div>
                  <div className="dropzone-icon">
                    IMG
                  </div>

                  <div className="dropzone-title">
                    Drop your LR image here
                  </div>

                  <div className="dropzone-copy">
                    PNG, JPEG or WebP
                  </div>

                  <input
                    className="file-input"
                    type="file"
                    accept="image/png,image/jpeg,image/webp"
                    onChange={handleInput}
                  />

                  <span className="secondary-button">
                    Choose file
                  </span>
                </div>
              )}
            </label>

            {/* =================================================
                OPTIONAL HR REFERENCE
                ================================================= */}

            <div
              style={{
                marginTop: 14,
                padding: 14,
                border:
                  "1px solid rgba(255,255,255,0.08)",
                borderRadius: 8,
              }}
            >
              <div className="metric-label">
                OPTIONAL HR REFERENCE
              </div>

              <div
                style={{
                  fontSize: 12,
                  opacity: 0.68,
                  marginTop: 4,
                  marginBottom: 12,
                }}
              >
                Add the matching HR image to
                calculate PSNR, SSIM and LPIPS.
                (Currently computed for
                Real-ESRGAN results only.)
              </div>

              {hrPreview ? (
                <div>
                  <div
                    style={{
                      width: "100%",
                      height: 160,
                      borderRadius: 8,
                      overflow: "hidden",
                      background:
                        "rgba(0,0,0,0.15)",
                      marginBottom: 10,
                    }}
                  >
                    <img
                      src={hrPreview}
                      alt="Uploaded HR reference preview"
                      style={{
                        width: "100%",
                        height: "100%",
                        objectFit: "contain",
                      }}
                    />
                  </div>

                  <div
                    style={{
                      fontSize: 12,
                    }}
                  >
                    {hrFile?.name}
                  </div>

                  <div
                    style={{
                      fontSize: 11,
                      opacity: 0.6,
                      marginTop: 4,
                    }}
                  >
                    {hrDimensions ||
                      "Reading dimensions…"}
                  </div>

                  <label
                    className="secondary-button"
                    style={{
                      display:
                        "inline-flex",
                      marginTop: 10,
                      cursor:
                        running
                          ? "not-allowed"
                          : "pointer",
                      opacity:
                        running ? 0.5 : 1,
                    }}
                  >
                    Replace HR reference

                    <input
                      className="file-input"
                      type="file"
                      accept="image/png,image/jpeg,image/webp"
                      onChange={
                        handleHRInput
                      }
                      disabled={running}
                    />
                  </label>
                </div>
              ) : (
                <label
                  className="secondary-button"
                  style={{
                    display:
                      "inline-flex",
                    cursor:
                      running
                        ? "not-allowed"
                        : "pointer",
                    opacity:
                      running ? 0.5 : 1,
                  }}
                >
                  Choose HR reference

                  <input
                    className="file-input"
                    type="file"
                    accept="image/png,image/jpeg,image/webp"
                    onChange={
                      handleHRInput
                    }
                    disabled={running}
                  />
                </label>
              )}
            </div>

            <div className="button-row">
              <button
                className="primary-button full-width"
                onClick={runSelected}
                disabled={
                  !lrFile ||
                  selected.length === 0 ||
                  running
                }
              >
                {running
                  ? "Running pipelines…"
                  : "Run selected pipelines"}
              </button>
            </div>

            {/* =================================================
                PROGRESS
                ================================================= */}

            {(running ||
              progress > 0) && (
              <div className="progress-panel">
                <div className="progress-row">
                  <span>
                    {running
                      ? currentPipeline ||
                        "processing"
                      : "completed"}
                  </span>

                  <span>
                    {progress}%
                  </span>
                </div>

                <div className="progress-track">
                  <div
                    className="progress-fill"
                    style={{
                      width: `${progress}%`,
                    }}
                  />
                </div>

                <div className="pipeline-status-list">
                  {selected.map(
                    (item, index) => {
                      const completed =
                        completedPipelines.has(
                          item
                        );

                      const current =
                        currentPipeline ===
                        item;

                      return (
                        <div
                          className="pipeline-status"
                          key={`${item}-${index}`}
                        >
                          <span
                            className={`dot ${
                              completed
                                ? "dot-done"
                                : current
                                ? "dot-running"
                                : ""
                            }`}
                          />

                          <span>
                            {item}
                          </span>

                          <span className="status-chip">
                            {completed
                              ? "done"
                              : current
                              ? "running"
                              : "queued"}
                          </span>
                        </div>
                      );
                    }
                  )}
                </div>
              </div>
            )}
          </div>
        </section>

        {/* =======================================================
            OLD PIPELINE SELECTION DESIGN
            Large selectable pipeline cards / toggles
            ======================================================= */}

        <section className="panel">
          <div className="panel-header">
            <div className="panel-title">
              2. Pipeline selection
            </div>

            <button
              className="secondary-button"
              onClick={
                selectAllAvailable
              }
              disabled={running}
            >
              Select all available
            </button>
          </div>

          <div className="panel-body">
            {/* =========================
                REAL-ESRGAN
               ========================= */}

            <div
              style={{
                marginBottom: 24,
              }}
            >
              <div
                className="group-label"
                style={{
                  marginBottom: 12,
                }}
              >
                REAL-ESRGAN
              </div>

              <div
                style={{
                  display: "grid",
                  gridTemplateColumns:
                    "repeat(2, minmax(0, 1fr))",
                  gap: 10,
                }}
              >
                {ESRGAN_STRATEGIES.map(
                  (strategy) => {
                    const value =
                      `Real-ESRGAN / ${strategy}`;

                    const active =
                      selected.includes(
                        value
                      );

                    const available =
                      strategyAvailable(
                        backendModels,
                        strategy
                      );

                    return (
                      <button
                        key={value}
                        type="button"
                        disabled={
                          running ||
                          !available
                        }
                        onClick={() =>
                          togglePipeline(
                            "Real-ESRGAN",
                            strategy
                          )
                        }
                        style={{
                          appearance:
                            "none",
                          border:
                            active
                              ? "1px solid rgba(45,212,191,0.65)"
                              : "1px solid rgba(255,255,255,0.10)",
                          background:
                            active
                              ? "rgba(45,212,191,0.10)"
                              : "rgba(255,255,255,0.025)",
                          color:
                            active
                              ? "inherit"
                              : "inherit",
                          borderRadius: 8,
                          padding:
                            "15px 14px",
                          textAlign: "left",
                          cursor:
                            running ||
                            !available
                              ? "not-allowed"
                              : "pointer",
                          opacity:
                            !available
                              ? 0.4
                              : 1,
                          transition:
                            "border-color 0.15s ease, background 0.15s ease",
                        }}
                      >
                        <div
                          style={{
                            display:
                              "flex",
                            alignItems:
                              "center",
                            justifyContent:
                              "space-between",
                            gap: 10,
                          }}
                        >
                          <span
                            style={{
                              fontSize: 14,
                              fontWeight:
                                600,
                            }}
                          >
                            {strategy}
                          </span>

                          <span
                            style={{
                              width: 9,
                              height: 9,
                              borderRadius:
                                "50%",
                              background:
                                active
                                  ? "currentColor"
                                  : "rgba(255,255,255,0.25)",
                              flexShrink: 0,
                            }}
                          />
                        </div>

                        <div
                          style={{
                            marginTop: 7,
                            fontSize: 11,
                            opacity: 0.58,
                            textTransform:
                              "uppercase",
                            letterSpacing:
                              "0.06em",
                          }}
                        >
                          {available
                            ? active
                              ? "selected"
                              : "ready"
                            : "not available"}
                        </div>
                      </button>
                    );
                  }
                )}
              </div>
            </div>

            {/* =========================
                DIFFUSION
               ========================= */}

            <div>
              <div
                className="group-label"
                style={{
                  marginBottom: 12,
                }}
              >
                DIFFUSION
              </div>

              <div
                style={{
                  display: "grid",
                  gridTemplateColumns:
                    "repeat(2, minmax(0, 1fr))",
                  gap: 10,
                }}
              >
                {DIFFUSION_STRATEGIES.map(
                  (strategy) => {
                    const value =
                      `Diffusion / ${strategy}`;

                    const active =
                      selected.includes(
                        value
                      );

                    return (
                      <button
                        key={value}
                        type="button"
                        disabled={running}
                        onClick={() =>
                          togglePipeline(
                            "Diffusion",
                            strategy
                          )
                        }
                        style={{
                          appearance:
                            "none",
                          border:
                            active
                              ? "1px solid rgba(45,212,191,0.65)"
                              : "1px solid rgba(255,255,255,0.10)",
                          background:
                            active
                              ? "rgba(45,212,191,0.10)"
                              : "rgba(255,255,255,0.025)",
                          borderRadius: 8,
                          padding:
                            "15px 14px",
                          textAlign:
                            "left",
                          cursor:
                            running
                              ? "not-allowed"
                              : "pointer",
                          opacity: 1,
                          transition:
                            "border-color 0.15s ease, background 0.15s ease",
                        }}
                      >
                        <div
                          style={{
                            display:
                              "flex",
                            alignItems:
                              "center",
                            justifyContent:
                              "space-between",
                            gap: 10,
                          }}
                        >
                          <span
                            style={{
                              fontSize: 14,
                              fontWeight:
                                600,
                            }}
                          >
                            {strategy}
                          </span>

                          <span
                            style={{
                              width: 9,
                              height: 9,
                              borderRadius:
                                "50%",
                              background:
                                active
                                  ? "currentColor"
                                  : "rgba(255,255,255,0.25)",
                            }}
                          />
                        </div>

                        <div
                          style={{
                            marginTop: 7,
                            fontSize: 11,
                            opacity: 0.58,
                            textTransform:
                              "uppercase",
                            letterSpacing:
                              "0.06em",
                          }}
                        >
                          {active
                            ? "selected"
                            : "ready"}
                        </div>
                      </button>
                    );
                  }
                )}
              </div>
            </div>

            <div
              className="metric-card"
              style={{
                marginTop: 18,
              }}
            >
              <div className="metric-label">
                Selected pipelines
              </div>

              <div className="metric-value">
                {selected.length}
              </div>

              <div className="metric-note">
                Pipelines run sequentially
                (ESRGAN first, then diffusion)
                to keep GPU memory usage
                controlled.
              </div>
            </div>
          </div>
        </section>
      </div>

      {/* =========================================================
          RESULT WORKSPACE — KEEPING THE DESIGN YOU LIKED
          ========================================================= */}

      <section className="results-section">
        <div className="section-header">
          <div className="section-kicker">
            3 / results workspace
          </div>

          <h2 className="section-title">
            Generated HR outputs
          </h2>

          <p className="section-copy">
            Compare the generated outputs side
            by side. When an HR reference is
            supplied, PSNR, SSIM and LPIPS are
            displayed when available.
          </p>
        </div>

        {results.length > 0 ? (
          <div className="result-grid">
            {results.map((result) => (
              <article
                className="result-card"
                key={result.id}
              >
                <div
                  className="result-media"
                  style={{
                    background:
                      "transparent",
                    padding: 0,
                  }}
                >
                  <img
                    src={result.imageUrl}
                    alt={`${result.model} ${result.strategy} output`}
                    style={{
                      width: "100%",
                      height: "100%",
                      objectFit:
                        "contain",
                      display:
                        "block",
                    }}
                  />
                </div>

                <div className="result-card-body">
                  <div className="result-name">
                    {result.model} /{" "}
                    {result.strategy}
                  </div>

                  <div className="result-meta">
                    Scale:{" "}
                    {result.scale} ·
                    Device:{" "}
                    {result.device}
                  </div>

                  <div
                    style={{
                      display:
                        "flex",
                      gap: 8,
                      flexWrap:
                        "wrap",
                      marginTop: 12,
                    }}
                  >
                    <span className="status-chip">
                      PSNR:{" "}
                      {result.metricsAvailable
                        ? formatMetric(
                            result.psnr
                          )
                        : "—"}
                    </span>

                    <span className="status-chip">
                      SSIM:{" "}
                      {result.metricsAvailable
                        ? formatMetric(
                            result.ssim
                          )
                        : "—"}
                    </span>

                    <span className="status-chip">
                      LPIPS:{" "}
                      {result.metricsAvailable
                        ? formatMetric(
                            result.lpips
                          )
                        : "—"}
                    </span>
                  </div>

                  {!result.metricsAvailable && (
                    <div
                      style={{
                        marginTop: 10,
                        fontSize: 12,
                        opacity: 0.68,
                      }}
                    >
                      Metrics unavailable
                      {result.metricsError
                        ? `: ${result.metricsError}`
                        : result.model.toLowerCase().includes(
                            "diffusion"
                          )
                        ? " — not computed for diffusion yet."
                        : !hrFile
                        ? " — no HR reference supplied."
                        : ""}
                    </div>
                  )}

                  <div
                    className="button-row"
                    style={{
                      marginTop: 12,
                    }}
                  >
                    <button
                      className="secondary-button"
                      onClick={() =>
                        downloadResult(
                          result
                        )
                      }
                    >
                      Download image
                    </button>
                  </div>
                </div>
              </article>
            ))}
          </div>
        ) : (
          <div className="result-grid">
            {selected.length >
            0 ? (
              selected.map(
                (item) => (
                  <article
                    className="result-card"
                    key={item}
                  >
                    <div className="result-media">
                      HR OUTPUT PREVIEW
                    </div>

                    <div className="result-card-body">
                      <div className="result-name">
                        {item}
                      </div>

                      <div className="result-meta">
                        Waiting for
                        inference
                      </div>
                    </div>
                  </article>
                )
              )
            ) : (
              <div className="panel">
                <div className="panel-body">
                  Select a pipeline to
                  begin.
                </div>
              </div>
            )}
          </div>
        )}
      </section>
    </Shell>
  );
}