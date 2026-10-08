import Shell from "../../../components/Shell";

export default function EsrganReportPage() {
  return (
    <Shell active="/report">
      <div className="report-page">
        <div className="report-topbar">
          <div>
            <div className="eyebrow">
              REPORT / COMPANION DEEP DIVE
            </div>

            <h1 className="hero-title">
              Real-ESRGAN Architecture
            </h1>

            <p className="hero-copy">
              Detailed architecture and hallucination analysis.
            </p>
          </div>

          <a
            className="secondary-button"
            href="/report/real_esrgan_architecture_and_hallucination.html"
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
                ARCHITECTURE DEEP DIVE
              </div>

              <div className="report-viewer-title">
                Real-ESRGAN / RRDBNet
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
              title="Real-ESRGAN architecture deep dive"
              src="/report/real_esrgan_architecture_and_hallucination.html"
            />
          </div>
        </section>
      </div>
    </Shell>
  );
}