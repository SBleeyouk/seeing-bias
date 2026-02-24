"""
Colab launch script for the Concept Attention Explorer web UI.

Usage in a Colab cell:
    # 1. Install deps (run once)
    # !pip install fastapi uvicorn[standard] pyngrok python-multipart

    # 2. Load the pipeline
    import sys
    sys.path.insert(0, '/content/ConceptAttention')
    from concept_attention.flux2 import ConceptAttentionFlux2Pipeline
    pipe = ConceptAttentionFlux2Pipeline(model_name="flux.2-dev", device="cuda:0")

    # 3. Launch
    import sys
    sys.path.insert(0, '/content/ConceptAttention')
    from web.launch import launch
    launch(pipe)
"""
import os
import sys
import threading
import time


PORT = 8000


def launch(pipeline, port: int = PORT, ngrok_authtoken: str | None = None):
    """
    Start the FastAPI server and expose it via ngrok.

    Args:
        pipeline: An initialized ConceptAttentionFlux2Pipeline instance.
        port:     Local port to bind (default 8000).
        ngrok_authtoken: Your ngrok auth token (optional but recommended for
                  free tier — avoids session limits).
                  Get one free at https://dashboard.ngrok.com/
    """
    from web.server import create_app
    import uvicorn

    app = create_app(pipeline)

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
    """
    Print a ready-to-paste Colab setup snippet.
    """
    snippet = """
# ── Cell 1: install dependencies ─────────────────────────────────────────────
!pip install -q fastapi "uvicorn[standard]" pyngrok python-multipart

# ── Cell 2: load the pipeline ─────────────────────────────────────────────────
import sys
sys.path.insert(0, '/content/ConceptAttention')

from concept_attention.flux2 import ConceptAttentionFlux2Pipeline

pipe = ConceptAttentionFlux2Pipeline(
    model_name="flux.2-dev",
    device="cuda:0",
    offload_model=True,   # save VRAM by offloading text encoder between steps
)

# ── Cell 3: launch the web server ─────────────────────────────────────────────
from web.launch import launch

# Optional: set your ngrok auth token to avoid session limits
# (free at https://dashboard.ngrok.com/)
NGROK_TOKEN = ""   # paste your token here, or set env var NGROK_AUTHTOKEN

url = launch(pipe, ngrok_authtoken=NGROK_TOKEN or None)
"""
    print(snippet)


if __name__ == "__main__":
    colab_quickstart()
