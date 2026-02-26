# Seeing Bias — Concept Attention Explorer

Seeing Bias is an interactive web tool for exploring how a diffusion model's internal attention distributes across user-defined concept tokens — and how that distribution shifts when a face is swapped into an image.

Built on the [ConceptAttention](https://arxiv.org/abs/2502.04320) interpretability method for Flux DiT (Black Forest Labs), extended with a face-swap pipeline and a collective "Brain of AI" visualization.

---

## Demo

[![Watch Demo](https://drive.google.com/file/d/11InzsXwZHNEq-7Gx2LWfSWHgKJAISnPz&sz=w1280)](https://drive.google.com/file/d/11InzsXwZHNEq-7Gx2LWfSWHgKJAISnPz/view?usp=sharing)

> Click the image to play the video on Google Drive.

---

## What It Does

The model encodes an image through the Flux Schnell transformer and measures, for each spatial patch, how strongly the model attends to each concept word you provide. This produces per-concept spatial heatmaps — no training required, fully zero-shot.

Three pages let you explore this from different angles:

### Concept Attention Explorer (`/index.html`)
Upload an image, write a prompt, and list concept tokens (e.g. `face, skin, hair, background`). The pipeline:
1. Encodes your image with the Flux transformer
2. Runs image-to-image generation (controllable strength)
3. Computes attention heatmaps for every concept token at every diffusion step
4. Streams per-step heatmaps to the browser in real time via WebSocket

A time scrubber lets you replay how attention evolves across diffusion steps. Original and generated images are shown side by side for each concept.

### Face-Swap Attention Explorer (`/faceswap.html`)
Upload two images. The pipeline:
1. Swaps the dominant face from the Face Source into the Original image (InsightFace `inswapper_128`)
2. Runs Concept Attention on all three images: Original, Face Source, Swapped
3. Returns a comparison grid — one row per image, one column per concept

This reveals how the model's spatial attention changes purely due to a face change, holding the scene constant.

### Brain of AI (`/brain.html`)
After any analysis, click **Submit to Brain of AI**. Results accumulate in a 3D force-directed graph:
- Each concept token is a node
- Thumbnail images orbit each node, masked so only the high-attention region is visible (top 30% luminance threshold)
- Link thickness encodes how often two concepts co-occur across submissions
- Click a thumbnail → detail panel with heatmap-overlaid images
- Click a node label → gallery of all submissions for that concept

---

## Running on Google Colab

Open `seeing_bias_run.ipynb` in Google Colab (GPU runtime required — A100 or L4 recommended).

The notebook has four cells:

**Cell 1 — Install dependencies**
```python
!pip install -q fastapi "uvicorn[standard]" pyngrok python-multipart \
               insightface onnxruntime-gpu opencv-python huggingface_hub
!git clone https://github.com/sbleeyouk/seeing-bias/content/seeing-bias
import sys; sys.path.insert(0, '/content/seeing-bias')
!pip install -q -e /content/seeing-bias
```

**Cell 2 — Load pipelines** (downloads Flux Schnell weights on first run, ~24 GB)
```python
from concept_attention.flux.pipeline import ConceptAttentionFluxPipeline
from concept_attention.flux.faceswap_pipeline import ConceptAttentionFaceSwapPipeline

concept_pipe  = ConceptAttentionFluxPipeline(model_name="flux-schnell", device="cuda:0")
faceswap_pipe = ConceptAttentionFaceSwapPipeline(concept_pipeline=concept_pipe)
```

**Cell 3 — Launch the web server**
```python
from web.launch import launch

NGROK_TOKEN = "YOUR_NGROK_TOKEN"   # free at https://dashboard.ngrok.com/
url = launch(concept_pipe, faceswap_pipeline=faceswap_pipe, ngrok_authtoken=NGROK_TOKEN)
# Prints the public URL — open it in any browser
```

**Cell 4 — Pull updates without restarting** (run after `git pull`)
```python
import importlib, subprocess, web.server, web.launch

subprocess.run(["git", "pull"], cwd="/content/seeing-bias")
importlib.reload(web.server)
importlib.reload(web.launch)
print("Modules reloaded — restart the server cell to apply changes")
```

> **Note:** After reloading modules you need to re-run Cell 3 (the launch cell). Because the uvicorn thread holds port 8000, pass `port=8001` on the second launch to avoid a bind conflict, or do a full runtime restart.

---

## Code Setup (local)

Requires Python 3.10+, CUDA GPU with ≥ 16 GB VRAM (24+ GB recommended for 1024×1024).

```bash
git clone https://github.com/sbleeyouk/seeing-bias
cd seeing-bias
python -m venv venv && source venv/bin/activate
pip install -e .
pip install fastapi "uvicorn[standard]" python-multipart insightface onnxruntime-gpu opencv-python
```

Run the server:
```python
from concept_attention.flux.pipeline import ConceptAttentionFluxPipeline
from concept_attention.flux.faceswap_pipeline import ConceptAttentionFaceSwapPipeline
from web.server import run_server

concept_pipe  = ConceptAttentionFluxPipeline(model_name="flux-schnell", device="cuda:0")
faceswap_pipe = ConceptAttentionFaceSwapPipeline(concept_pipeline=concept_pipe)

run_server(concept_pipe, faceswap_pipeline=faceswap_pipe, port=8000)
# Open http://localhost:8000
```

---

## Project Structure

```
seeing-bias/
├── concept_attention/
│   └── flux/
│       ├── pipeline.py          # ConceptAttentionFluxPipeline — encode_image, compare_images
│       ├── faceswap_pipeline.py # ConceptAttentionFaceSwapPipeline — swap_and_analyze
│       └── image_generator.py   # FluxGenerator — model loading, offload management
├── web/
│   ├── server.py                # FastAPI app — REST + WebSocket endpoints
│   ├── launch.py                # Colab/ngrok launcher
│   └── static/
│       ├── index.html           # Concept Attention Explorer UI
│       ├── faceswap.html        # Face-Swap Attention Explorer UI
│       └── brain.html           # Brain of AI — 3D force graph
├── seeing_bias_run.ipynb        # Google Colab notebook
├── requirements.txt
└── setup.py
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/analyze` | Start a concept attention job |
| `GET` | `/api/job/{id}` | Poll job status and results |
| `WS` | `/ws/{id}` | Stream per-step heatmaps in real time |
| `POST` | `/api/faceswap` | Start a face-swap + attention job |
| `WS` | `/ws/faceswap/{id}` | Stream face-swap progress |
| `POST` | `/api/brain/submit` | Submit a result to the Brain |
| `GET` | `/api/brain` | Fetch graph data (nodes, links, images) |

---

## Models and Downloads

| Model | Size | Source |
|-------|------|--------|
| Flux Schnell (transformer + VAE) | ~24 GB | HuggingFace `black-forest-labs/FLUX.1-schnell` |
| T5 text encoder | ~5 GB | Auto via Flux |
| CLIP text encoder | ~0.4 GB | Auto via Flux |
| InsightFace buffalo_l | ~0.3 GB | Auto via insightface |
| inswapper_128.onnx | ~0.5 GB | HuggingFace `Gourieff/ReActor` dataset |

All models download automatically on first use and cache in `~/.cache/`.

Flux Schnell requires accepting the license on HuggingFace before downloading:
```bash
huggingface-cli login
```

---

## Credits

- **ConceptAttention** — [Helblazer811 et al., 2025](https://arxiv.org/abs/2502.04320)
- **Flux** — [Black Forest Labs](https://github.com/black-forest-labs/flux)
- **InsightFace / inswapper** — [deepinsight](https://github.com/deepinsight/insightface)