import Shell from "../../components/Shell";

export default function ReportPage() {
  return (
    <Shell active="/report">
      <div className="report-page">
        <div className="report-topbar">
          <div>
            <div className="eyebrow">02 / FORMAL RESEARCH RECORD</div>
            <h1 className="hero-title">Research Report</h1>
            <p className="hero-copy">
              Diffusion-Based Super-Resolution of Low-Cost Satellite Imagery
            </p>
          </div>
          <a
            className="secondary-button"
            href="/report/main-report.html"
            target="_blank"
            rel="noreferrer">
            Open full report ↗
          </a>
        </div>
        <div className="report-links">
          <a
            className="report-link-card report-link-active"
            href="/report/main-report.html"
          >
            <div className="report-link-number">01</div>

            <div>
              <div className="report-link-title">Main research report</div>
              <div className="report-link-copy">
                Complete project report, methodology, experiments,
                quantitative results and discussion.
              </div>
            </div>

            <div className="report-link-arrow">→</div>
          </a>

          <a
            className="report-link-card"
            href="/report/esrgan"
          >
            <div className="report-link-number">02</div>

            <div>
              <div className="report-link-title">
                Real-ESRGAN architecture
              </div>
              <div className="report-link-copy">
                RRDBNet architecture, SRGAN/ESRGAN derivation and
                hallucination analysis.
              </div>
            </div>

            <div className="report-link-arrow">→</div>
          </a>

          <a
            className="report-link-card"
            href="/report/diffusion"
          >
            <div className="report-link-number">03</div>

            <div>
              <div className="report-link-title">
                Diffusion architecture
              </div>
              <div className="report-link-copy">
                Stable Diffusion x4 Upscaler pipeline, denoising process,
                training objective and fine-tuning behavior.
              </div>
            </div>

            <div className="report-link-arrow">→</div>
          </a>
        </div>

        <section className="report-viewer">
          <div className="report-viewer-header">
            <div>
              <div className="report-viewer-label">DOCUMENT VIEWER</div>
              <div className="report-viewer-title">
                Diffusion-Based Super-Resolution of Low-Cost Satellite Imagery
              </div>
            </div>

            <div className="report-viewer-status">
              <span className="status-indicator" />
              HTML report loaded
            </div>
          </div>

          <div className="report-viewer-body">
            <iframe
              className="report-frame"
              title="Diffusion-Based Super-Resolution research report"
              src="/report/main-report.html"
            />
          </div>
        </section>
      </div>
    </Shell>
  );
}