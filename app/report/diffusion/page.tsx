import Shell from "../../../components/Shell";

export default function DiffusionReportPage() {
  return (
    <Shell active="/report">
      <div className="report-page">
        <div className="report-topbar">
          <div>
            <div className="eyebrow">
              REPORT / COMPANION DEEP DIVE
            </div>

            <h1 className="hero-title">
              Diffusion Architecture
            </h1>

            <p className="hero-copy">
              End-to-end explanation of the Stable Diffusion x4 Upscaler
              pipeline and its fine-tuning behavior.
            </p>
          </div>

          <a
            className="secondary-button"
            href="/report/diffusion_architecture.html"
            target="_blank"
            rel="noreferrer"
          >
            Open full document ↗
          </a>
        </div>

        <section className="report-viewer">
          <div className="report-viewer-header">
            <div>
              <div className="report-viewer-label">
                DIFFUSION DEEP DIVE
              </div>

              <div className="report-viewer-title">
                Stable Diffusion x4 Upscaler
              </div>
            </div>

            <div className="report-viewer-status">
              <span className="status-indicator" />
              HTML document loaded
            </div>
          </div>

          <div className="report-viewer-body">
            <iframe
              className="report-frame"
              title="Diffusion architecture deep dive"
              src="/report/diffusion_architecture.html"
            />
          </div>
        </section>
      </div>
    </Shell>
  );
}