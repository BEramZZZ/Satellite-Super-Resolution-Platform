# Satellite Super-Resolution Lab — Step 1

Frontend shell for the research/demo website.

## Current scope

- Next.js + React + TypeScript
- Four main sections:
  - Super-Resolution Lab
  - Research Report
  - Benchmark
  - Concepts
- Responsive dark research-lab visual system
- Functional drag/drop image selection
- Pipeline selection UI
- "Run selected" demo state (frontend-only for now)
- Result cards and progress UI skeleton
- Benchmark dashboard skeleton
- Scene explorer + comparison slider UI
- Strategy matrix
- Concepts page covering the key technical topics

## Not connected yet

The ML backend is intentionally not connected in Step 1.

The next integration phase will add:

- FastAPI backend
- model registry
- asynchronous inference jobs
- SSE progress events
- ESRGAN checkpoints
- Diffusion checkpoints
- real benchmark JSON
- real report HTML files

## Run locally

```bash
npm install
npm run dev
```

Then open:

http://localhost:3000

## Recommended Node version

Node 20+.
