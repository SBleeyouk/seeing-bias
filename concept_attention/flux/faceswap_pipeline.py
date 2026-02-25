"""
Face-swap + Concept Attention pipeline.

Uses InsightFace inswapper_128.onnx to replace the face in `original_image`
with the face from `target_image`, then runs Concept Attention encode_image()
on all three images (original, face source, swapped) to produce comparative
spatial heatmaps.

Dependencies (install once):
    pip install insightface onnxruntime-gpu opencv-python huggingface_hub

Model (auto-downloaded on first use, ~500 MB):
    inswapper_128.onnx — from Gourieff/ReActor dataset on HuggingFace

Usage (Colab):
    from concept_attention.flux.pipeline import ConceptAttentionFluxPipeline
    from concept_attention.flux.faceswap_pipeline import ConceptAttentionFaceSwapPipeline

    concept_pipe = ConceptAttentionFluxPipeline(model_name="flux-schnell", device="cuda:0")
    pipe = ConceptAttentionFaceSwapPipeline(concept_pipeline=concept_pipe)

    result = pipe.swap_and_analyze(
        original_image=img_to_modify,      # face here will be REPLACED
        target_image=img_with_donor_face,  # face here will be INSERTED
        prompt="a person in a park",
        concepts=["face", "hair", "skin", "background"],
    )
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import PIL.Image
import torch

from concept_attention.flux.pipeline import ConceptAttentionFluxPipeline


# ─────────────────────────────────────────────────────────────────────────────
# Output dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FaceSwapConceptOutput:
    """Output from swap_and_analyze()."""
    original_image: PIL.Image.Image       # original image (face replaced)
    target_image: PIL.Image.Image         # face source image (donor)
    swapped_image: PIL.Image.Image        # original with donor face inserted
    original_heatmaps: list              # list[PIL.Image.Image] per concept
    target_heatmaps: list                # list[PIL.Image.Image] per concept
    swapped_heatmaps: list               # list[PIL.Image.Image] per concept
    concepts: list[str]


# ─────────────────────────────────────────────────────────────────────────────
# Image conversion helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pil_to_cv2(img: PIL.Image.Image) -> np.ndarray:
    """PIL RGB → OpenCV BGR uint8."""
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def _cv2_to_pil(img: np.ndarray) -> PIL.Image.Image:
    """OpenCV BGR → PIL RGB."""
    return PIL.Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# ─────────────────────────────────────────────────────────────────────────────
# InsightFace face swapper
# ─────────────────────────────────────────────────────────────────────────────

class InsightFaceSwapper:
    """
    Thin wrapper around InsightFace + inswapper_128.onnx.

    Downloads the ONNX model from HuggingFace on first use and caches it at
    ~/.cache/insightface_swap/inswapper_128.onnx.
    """

    _CACHE_DIR = Path.home() / ".cache" / "insightface_swap"
    _HF_REPO   = "Gourieff/ReActor"
    _HF_FILE   = "models/inswapper_128.onnx"

    def __init__(self, device: str = "cuda:0"):
        import insightface
        from insightface.app import FaceAnalysis

        # Always use CPU for InsightFace ONNX inference to avoid CUBLAS/CUDA
        # allocation failures that occur when the Flux transformer is occupying
        # GPU VRAM.  Face detection and swapping are fast enough on CPU.
        providers = ["CPUExecutionProvider"]

        print("  Loading InsightFace face analyzer (buffalo_l)…")
        self.analyzer = FaceAnalysis(name="buffalo_l", providers=providers)
        self.analyzer.prepare(ctx_id=-1, det_size=(640, 640))

        print("  Loading inswapper_128 model…")
        model_path = self._get_model()
        self.swapper = insightface.model_zoo.get_model(
            str(model_path), providers=providers
        )

    def _get_model(self) -> Path:
        """Return path to inswapper_128.onnx, downloading if necessary."""
        dest = self._CACHE_DIR / "inswapper_128.onnx"
        if dest.exists():
            return dest
        self._CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print("  Downloading inswapper_128.onnx from HuggingFace (~500 MB)…")
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=self._HF_REPO,
            filename=self._HF_FILE,
            repo_type="dataset",
            local_dir=str(self._CACHE_DIR),
        )
        print(f"  Saved to: {path}")
        return Path(path)

    def _largest_face(self, img_bgr: np.ndarray):
        """Return the largest detected face in a BGR image, or None."""
        faces = self.analyzer.get(img_bgr)
        if not faces:
            return None
        # bbox is [x1, y1, x2, y2]; sort by area descending
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    def swap(
        self,
        original_img: PIL.Image.Image,   # image whose face will be REPLACED
        target_img: PIL.Image.Image,     # image containing the DONOR face
        width: int,
        height: int,
    ) -> PIL.Image.Image:
        """
        Replace the dominant face in `original_img` with the dominant face
        from `target_img`.  Both images are first resized to (width, height).

        Returns the modified image as PIL.
        Raises ValueError if no face is detected in either image.
        """
        orig_bgr  = _pil_to_cv2(original_img.resize((width, height), PIL.Image.LANCZOS))
        donor_bgr = _pil_to_cv2(target_img.resize((width, height), PIL.Image.LANCZOS))

        orig_face  = self._largest_face(orig_bgr)
        donor_face = self._largest_face(donor_bgr)

        if donor_face is None:
            raise ValueError(
                "No face detected in the Face Source image. "
                "Please provide an image with a clearly visible face."
            )
        if orig_face is None:
            raise ValueError(
                "No face detected in the Original image. "
                "Please provide an image with a clearly visible face."
            )

        result = orig_bgr.copy()
        result = self.swapper.get(result, orig_face, donor_face, paste_back=True)
        return _cv2_to_pil(result)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

class ConceptAttentionFaceSwapPipeline:
    """
    Face-swap + Concept Attention pipeline.

    Workflow:
        1. Swap the dominant face from `target_image` into `original_image`
           using InsightFace inswapper_128.
        2. Run Concept Attention encode_image() on all three images
           (original, face source, swapped result).
        3. Return spatial heatmaps for each concept across all three images.

    The three images are meant to be compared side-by-side to study how
    algorithmic attention changes before and after a face swap.
    """

    def __init__(self, concept_pipeline: ConceptAttentionFluxPipeline):
        self.pipe   = concept_pipeline
        self.device = concept_pipeline.device

        print("Loading InsightFace face swapper…")
        self.swapper = InsightFaceSwapper(device=self.device)
        print("Face swapper ready.")

    @torch.no_grad()
    def swap_and_analyze(
        self,
        original_image: PIL.Image.Image,
        target_image: PIL.Image.Image,
        prompt: str,
        concepts: list[str],
        width: int = 512,
        height: int = 512,
        layer_indices: list[int] | None = None,
        num_steps: int = 4,
        noise_timestep: int = 2,
        seed: int = 0,
        cmap: str = "plasma",
        on_progress: Callable | None = None,
    ) -> FaceSwapConceptOutput:
        """
        Perform face swap and compute Concept Attention heatmaps for all
        three images.

        Args:
            original_image: Image whose face will be REPLACED.
            target_image:   Image containing the DONOR face.
            prompt:         Text prompt for concept attention conditioning.
            concepts:       Concept tokens to generate heatmaps for.
            width / height: Processing resolution (512 recommended for speed).
            layer_indices:  Which transformer layers to extract attention from.
            num_steps:      Number of noise schedule steps for encode_image.
            noise_timestep: Which schedule step to add noise at (0 = less noise).
            seed:           Random seed for reproducibility.
            cmap:           Matplotlib colormap for heatmaps.
            on_progress:    Optional callback(stage_name, current, total).
        """
        total = 4
        current = [0]

        def _prog(name: str):
            current[0] += 1
            print(f"[{current[0]}/{total}] {name}")
            if on_progress:
                on_progress(name, current[0], total)

        common = dict(
            concepts=concepts,
            prompt=prompt,
            width=width,
            height=height,
            layer_indices=layer_indices,
            num_steps=num_steps,
            noise_timestep=noise_timestep,
            seed=seed,
            cmap=cmap,
        )

        # ── 1. Face swap ──────────────────────────────────────────────────────
        _prog("Swapping face…")
        swapped_image = self.swapper.swap(
            original_img=original_image,
            target_img=target_image,
            width=width,
            height=height,
        )

        # ── 2–4. Concept attention for each image ─────────────────────────────
        _prog("Computing attention maps for Original…")
        orig_out = self.pipe.encode_image(image=original_image, **common)

        _prog("Computing attention maps for Face Source…")
        tgt_out = self.pipe.encode_image(image=target_image, **common)

        _prog("Computing attention maps for Swapped…")
        swap_out = self.pipe.encode_image(image=swapped_image, **common)

        def _as_pil(out_image, fallback, w, h):
            if isinstance(out_image, PIL.Image.Image):
                return out_image
            return fallback.resize((w, h), PIL.Image.LANCZOS)

        return FaceSwapConceptOutput(
            original_image=_as_pil(orig_out.image, original_image, width, height),
            target_image=_as_pil(tgt_out.image, target_image, width, height),
            swapped_image=swapped_image,
            original_heatmaps=orig_out.concept_heatmaps,
            target_heatmaps=tgt_out.concept_heatmaps,
            swapped_heatmaps=swap_out.concept_heatmaps,
            concepts=concepts,
        )
