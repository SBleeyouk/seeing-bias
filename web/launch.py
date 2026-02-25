"""
Colab launch script for the Concept Attention Explorer web UI.

Usage in a Colab cell:
    # ── Cell 1: install deps (run once) ──────────────────────────────────────
    !pip install -q fastapi "uvicorn[standard]" pyngrok python-multipart \\
                   insightface onnxruntime-gpu opencv-python

    # ── Cell 2: load pipelines ────────────────────────────────────────────────
    import sys; sys.path.insert(0, '/content/seeing-bias')
    from concept_attention.flux.pipeline import ConceptAttentionFluxPipeline
    from concept_attention.flux.faceswap_pipeline import ConceptAttentionFaceSwapPipeline

    concept_pipe  = ConceptAttentionFluxPipeline(model_name="flux-schnell", device="cuda:0")
    faceswap_pipe = ConceptAttentionFaceSwapPipeline(concept_pipeline=concept_pipe)

    # ── Cell 3: launch ────────────────────────────────────────────────────────
    from web.launch import launch
    launch(concept_pipe, faceswap_pipeline=faceswap_pipe)
"""
import os
import threading
import time


PORT = 8000


def launch(
    pipeline,
    faceswap_pipeline=None,
    port: int = PORT,
    ngrok_authtoken: str | None = None,
):
    """
    Start the FastAPI server in a background thread and expose it via ngrok.

    Args:
        pipeline:         ConceptAttentionFluxPipeline instance.
        faceswap_pipeline: ConceptAttentionFaceSwapPipeline instance (optional).
        port:             Local port to bind (default 8000).
        ngrok_authtoken:  ngrok auth token (free at https://dashboard.ngrok.com/).
    """
    from web.server import create_app
    import uvicorn

    app = create_app(pipeline, faceswap_pipeline=faceswap_pipeline)

    # ── Start uvicorn in a background thread ──────────────────────────────
    def _run():
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")

    server_thread = threading.Thread(target=_run, daemon=True)
    server_thread.start()
    time.sleep(2)  # let server warm up
    print(f"Server started on port {port}")

    # ── Expose via ngrok ──────────────────────────────────────────────────
    try:
        from pyngrok import ngrok, conf

        if ngrok_authtoken:
            conf.get_default().auth_token = ngrok_authtoken
        elif "NGROK_AUTHTOKEN" in os.environ:
            conf.get_default().auth_token = os.environ["NGROK_AUTHTOKEN"]

        public_url = ngrok.connect(port, "http")
        url = str(public_url)

        print("\n" + "=" * 60)
        print("  Concept Attention Explorer is LIVE!")
        print(f"  Open this URL in your browser:")
        print(f"\n      {url}\n")
        print("=" * 60 + "\n")
        return url

    except ImportError:
        print("\npyngrok not installed. Install with:  pip install pyngrok")
        print(f"Server is running locally at: http://localhost:{port}")
        print("Use Colab's built-in port forwarding or install pyngrok for external access.")
        return f"http://localhost:{port}"
    except Exception as e:
        print(f"\nngrok failed: {e}")
        print(f"Server is running locally at: http://localhost:{port}")
        return f"http://localhost:{port}"


def colab_quickstart():
    """Print a ready-to-paste Colab setup snippet."""
    print("""
# ── Cell 1: install dependencies ─────────────────────────────────────────────
!pip install -q fastapi "uvicorn[standard]" pyngrok python-multipart \\
               insightface onnxruntime-gpu opencv-python

# ── Cell 2: load pipelines ────────────────────────────────────────────────────
import sys
sys.path.insert(0, '/content/seeing-bias')

from concept_attention.flux.pipeline import ConceptAttentionFluxPipeline
from concept_attention.flux.faceswap_pipeline import ConceptAttentionFaceSwapPipeline

concept_pipe  = ConceptAttentionFluxPipeline(model_name="flux-schnell", device="cuda:0")
faceswap_pipe = ConceptAttentionFaceSwapPipeline(concept_pipeline=concept_pipe)

# ── Cell 3: launch the web server ─────────────────────────────────────────────
from web.launch import launch

NGROK_TOKEN = ""   # free at https://dashboard.ngrok.com/
url = launch(concept_pipe, faceswap_pipeline=faceswap_pipe, ngrok_authtoken=NGROK_TOKEN or None)
""")


if __name__ == "__main__":
    colab_quickstart()
