import Shell from "../../components/Shell";

const concepts = [
  {
    n: "01",
    title: "Architecture",
    copy:
      "Real-ESRGAN is a deterministic single-forward-pass SR regressor. The diffusion pipeline repeatedly predicts a denoising update for a latent state and lets the scheduler produce the next state.",
    code:
      "LR → RRDBNet → HR       vs       LR condition + z_t → UNet → εθ → scheduler → z_(t-1)"
  },
  {
    n: "02",
    title: "Fine-tuning strategies",
    copy:
      "The project compares zero-shot, head-only, partial and full adaptation, with LoRA as a parameter-efficient variant where available. The important variable is not only the percentage of parameters but which part of the computation is changed.",
    code:
      "zero-shot: 0%   |   head-only: output region   |   partial: late blocks   |   full: entire trainable model"
  },
  {
    n: "03",
    title: "Loss functions",
    copy:
      "The current Real-ESRGAN fine-tuning uses direct image-space L1. The original diffusion formulation supervises noise prediction with MSE. These losses live on different targets and should not be numerically compared across model families.",
    code:
      "L_ESRGAN = |I_HR - I_SR|_1       vs       L_diff = ||ε - εθ(z_t,t)||²"
  },
  {
    n: "04",
    title: "Why diffusion can hallucinate",
    copy:
      "A diffusion model can choose one plausible high-frequency completion where the LR observation does not uniquely determine the unseen detail. Plausibility is therefore not the same as factual fidelity.",
    code:
      "missing information → multiple plausible HR completions → model samples one completion"
  },
  {
    n: "05",
    title: "Why head-only behaves differently",
    copy:
      "In ESRGAN, the final layer directly converts the frozen feature representation into RGB, so its influence is terminal. In diffusion, the output block predicts the denoising residual and participates repeatedly in the iterative trajectory.",
    code:
      "ESRGAN: frozen features → trainable conv_last → RGB\nDiffusion: frozen features → trainable output → εθ → next latent → repeat"
  },
  {
    n: "06",
    title: "Why partial can become unstable",
    copy:
      "The project observed severe degradation for one tested diffusion partial configuration. The iterative denoising chain provides a mechanism by which a modified prediction can shift the next latent state and affect later steps, but this is an experimental observation for a specific setup, not a universal law.",
    code:
      "Δε_t → Δz_(t-1) → different input at next step → propagated trajectory difference"
  }
];

export default function ConceptsPage() {
  return (
    <Shell active="/concepts">
      <div className="page-intro">
        <div className="eyebrow">04 / concepts & methodology</div>
        <h1 className="hero-title">The ideas behind the results.</h1>
        <p className="hero-copy">
          This section deliberately complements the report rather than
          duplicating it. It explains the mechanisms that matter most when
          interpreting your experiments.
        </p>
      </div>

      <div className="concept-grid">
        {concepts.map((concept) => (
          <article className="concept-card" key={concept.n}>
            <div className="concept-number">{concept.n}</div>
            <h2 className="concept-title">{concept.title}</h2>
            <p className="concept-copy">{concept.copy}</p>
            <pre className="code-strip">{concept.code}</pre>
          </article>
        ))}
      </div>
    </Shell>
  );
}
