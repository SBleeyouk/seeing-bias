"""
    Wrapper pipeline for Flux 1 concept attention.
    Ported from the Flux 2 pipeline — includes proper img2img, temporal heatmaps,
    per-step streaming callbacks, and the comparison grid.
"""
from dataclasses import dataclass
import base64
import io
import os
from typing import Callable

import PIL
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import einops
from einops import rearrange
from tqdm import tqdm

from concept_attention.flux.flux.src.flux.sampling import prepare, get_schedule, get_noise, unpack
from concept_attention.segmentation import add_noise_to_image, encode_image as vae_encode_image
from concept_attention.utils import embed_concepts, linear_normalization
from concept_attention.flux.image_generator import FluxGenerator


# ─────────────────────────────────────────────────────────────────────────────
# Output dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConceptAttentionPipelineOutput:
    """Output from generate_image() or encode_image()."""
    image: PIL.Image.Image | np.ndarray
    concept_heatmaps: list[PIL.Image.Image]
    cross_attention_maps: list[PIL.Image.Image]
    concept_output_vectors: torch.Tensor | np.ndarray | None = None
    image_output_vectors: torch.Tensor | np.ndarray | None = None


@dataclass
class ConceptAttentionTemporalOutput:
    """Output from compare_images() — includes per-timestep heatmaps for animation."""
    original_image: PIL.Image.Image
    generated_image: PIL.Image.Image
    concepts: list[str]
    original_heatmaps: list[PIL.Image.Image]
    generated_heatmaps: list[PIL.Image.Image]
    # temporal_heatmaps[timestep_idx][concept_idx] → PIL image
    temporal_heatmaps: list[list[PIL.Image.Image]]
    num_denoising_steps: int


# ─────────────────────────────────────────────────────────────────────────────
# Heatmap utilities
# ─────────────────────────────────────────────────────────────────────────────

def compute_heatmaps_from_vectors(
    image_vectors: torch.Tensor,
    concept_vectors: torch.Tensor,
    layer_indices: list[int],
    timestep_indices: list[int],
    width: int = 1024,
    height: int = 1024,
    softmax: bool = True,
    normalize_concepts: bool = False,
) -> torch.Tensor:
    """
    Compute spatial heatmaps from cached image/concept vectors.

    Args:
        image_vectors:   (time, layers, batch, patches, dim)
        concept_vectors: (time, layers, batch, concepts, dim)
        layer_indices:   Which layers to average over
        timestep_indices: Which timesteps to average over
        width/height:    Image dimensions (used to infer spatial grid)

    Returns:
        Tensor of shape (batch, num_concepts, h, w)
    """
    # Collapse head dimension if present
    if len(image_vectors.shape) == 6:
        image_vectors = einops.rearrange(
            image_vectors,
            "time layers batch head patches dim -> time layers batch patches (head dim)"
        )
        concept_vectors = einops.rearrange(
            concept_vectors,
            "time layers batch head concepts dim -> time layers batch concepts (head dim)"
        )

    if normalize_concepts:
        concept_vectors = linear_normalization(concept_vectors, dim=-2)

    # Dot-product similarity: image patches × concept tokens
    heatmaps = einops.einsum(
        image_vectors,
        concept_vectors,
        "time layers batch patches dim, time layers batch concepts dim"
        " -> time layers batch concepts patches",
    )

    if softmax:
        heatmaps = torch.nn.functional.softmax(heatmaps, dim=-2)

    # Safe timestep indexing
    num_t = heatmaps.shape[0]
    safe_t = [t for t in timestep_indices if t < num_t] or list(range(num_t))
    heatmaps = heatmaps[safe_t]

    # Safe layer indexing
    num_l = heatmaps.shape[1]
    safe_l = [l for l in layer_indices if l < num_l] or list(range(num_l))
    heatmaps = heatmaps[:, safe_l]

    heatmaps = einops.reduce(
        heatmaps,
        "time layers batch concepts patches -> batch concepts patches",
        reduction="mean"
    )

    h = height // 16
    w = width // 16
    heatmaps = einops.rearrange(
        heatmaps,
        "batch concepts (h w) -> batch concepts h w",
        h=h, w=w
    )
    return heatmaps


def heatmaps_to_pil_images(
    heatmaps: torch.Tensor | np.ndarray,
    width: int,
    height: int,
    cmap: str = "plasma",
) -> list[PIL.Image.Image]:
    """Convert (num_concepts, h, w) tensor/array to colored PIL images."""
    if isinstance(heatmaps, torch.Tensor):
        heatmaps_np = heatmaps.cpu().float().numpy()
    else:
        heatmaps_np = heatmaps.astype(np.float32)

    global_min = heatmaps_np.min()
    global_max = heatmaps_np.max()
    colormap = plt.get_cmap(cmap)
    pil_images = []

    for concept_heatmap in heatmaps_np:
        normalized = (concept_heatmap - global_min) / (global_max - global_min + 1e-8)
        colored = colormap(normalized)
        rgb = (colored[:, :, :3] * 255).astype(np.uint8)
        pil_img = PIL.Image.fromarray(rgb)
        pil_img = pil_img.resize((width, height), PIL.Image.NEAREST)
        pil_images.append(pil_img)

    return pil_images


def heatmap_tensor_to_base64(
    heatmap_tensor: torch.Tensor,
    width: int,
    height: int,
    cmap: str = "plasma",
) -> list[str]:
    """Convert (num_concepts, h, w) tensor to base64-encoded PNG strings for streaming."""
    pil_images = heatmaps_to_pil_images(heatmap_tensor, width, height, cmap)
    result = []
    for img in pil_images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        result.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
    return result


def _overlay_heatmap(
    base_image: PIL.Image.Image,
    heatmap: PIL.Image.Image,
    alpha: float = 0.55,
) -> PIL.Image.Image:
    """Overlay a heatmap PIL image on top of a base image with transparency."""
    base = base_image.convert("RGBA")
    hm = heatmap.resize(base.size, PIL.Image.BILINEAR).convert("RGBA")
    hm_arr = np.array(hm)
    hm_arr[:, :, 3] = int(alpha * 255)
    blended = PIL.Image.alpha_composite(base, PIL.Image.fromarray(hm_arr))
    return blended.convert("RGB")


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

class ConceptAttentionFluxPipeline:
    """
    Pipeline for Flux 1 concept attention.

    Supports:
    - generate_image(): text-to-image or img2img with concept heatmaps
    - encode_image():   extract heatmaps from an existing image
    - compare_images(): side-by-side comparison with temporal heatmaps + streaming
    """

    def __init__(
        self,
        model_name: str = "flux-schnell",
        offload_model: bool = False,
        device: str = "cuda:0",
    ):
        self.model_name = model_name
        self.offload_model = offload_model
        self.device = device

        print("Loading Flux 1 models...")
        self.flux_generator = FluxGenerator(
            model_name=model_name,
            offload=offload_model,
            device=device,
        )
        print("Flux 1 models loaded successfully!")

    # ── Internal denoising loop ───────────────────────────────────────────────

    def _denoise(
        self,
        inp: dict,
        timesteps: list,
        guidance: float,
        cache_vectors: bool = True,
        layer_indices: list[int] | None = None,
        timestep_indices: list[int] | None = None,
        on_step_callback: Callable | None = None,
        callback_num_concepts: int | None = None,
        callback_width: int | None = None,
        callback_height: int | None = None,
        callback_softmax_temperature: float = 1.0,
        callback_cmap: str = "plasma",
        width: int = 1024,
        height: int = 1024,
    ) -> tuple[torch.Tensor, list]:
        """
        Denoising loop with per-step concept attention tracking.

        on_step_callback(step_idx, total_steps, b64_heatmaps_list) is called
        after each denoising step if provided. Heatmaps are base64-encoded PNGs,
        one per concept — ready to stream to a frontend.
        """
        img = inp["img"]
        guidance_vec = torch.full(
            (img.shape[0],), guidance,
            device=img.device, dtype=img.dtype
        )

        concept_attention_dicts_all = []
        total_steps = len(timesteps) - 1

        for step_idx, (t_curr, t_prev) in enumerate(tqdm(
            zip(timesteps[:-1], timesteps[1:]),
            total=total_steps,
            desc="Denoising",
        )):
            should_cache = (
                timestep_indices is None or step_idx in timestep_indices
            ) or (on_step_callback is not None)

            t_vec = torch.full(
                (img.shape[0],), t_curr,
                dtype=img.dtype, device=img.device
            )

            pred, step_dicts = self.flux_generator.model(
                img=img,
                img_ids=inp["img_ids"],
                txt=inp["txt"],
                txt_ids=inp["txt_ids"],
                concepts=inp["concepts"],
                concept_ids=inp["concept_ids"],
                concept_vec=inp["concept_vec"],
                y=inp["vec"],
                timesteps=t_vec,
                guidance=guidance_vec,
                stop_after_multimodal_attentions=False,
                joint_attention_kwargs=None,
                cache_vectors=should_cache,
                layer_indices=layer_indices,
            )

            # Euler step (flow matching: img_next = img + (t_prev - t_curr) * velocity)
            img = img + (t_prev - t_curr) * pred

            concept_attention_dicts_all.append(step_dicts)

            # Fire per-step callback with base64 heatmaps
            if on_step_callback is not None and callback_num_concepts is not None:
                # Build vectors for this step
                step_img_vecs = []
                step_con_vecs = []
                for d in step_dicts:
                    if "output_space_image_vectors" in d:
                        step_img_vecs.append(d["output_space_image_vectors"])
                    if "output_space_concept_vectors" in d:
                        step_con_vecs.append(d["output_space_concept_vectors"])

                if step_img_vecs and step_con_vecs:
                    iv = torch.stack(step_img_vecs).unsqueeze(0)   # (1, layers, batch, patches, dim)
                    cv = torch.stack(step_con_vecs).unsqueeze(0)   # (1, layers, batch, concepts, dim)
                    eff_layers = layer_indices if layer_indices else list(range(len(step_img_vecs)))
                    hm = compute_heatmaps_from_vectors(
                        iv, cv,
                        layer_indices=eff_layers,
                        timestep_indices=[0],
                        width=callback_width or width,
                        height=callback_height or height,
                        softmax=True,
                    )[0]  # (num_concepts, h, w)
                    b64 = heatmap_tensor_to_base64(hm, callback_width or width, callback_height or height, callback_cmap)
                    on_step_callback(step_idx, total_steps, b64)

        return img, concept_attention_dicts_all

    def _stack_vectors(self, concept_attention_dicts_all: list) -> dict:
        """
        Stack per-step, per-layer attention dicts into tensors.
        Returns dict with keys: output_space_image_vectors, output_space_concept_vectors,
        cross_attention_image_vectors, cross_attention_concept_vectors.
        Each tensor shape: (time, layers, batch, tokens, dim)
        """
        keys = [
            "output_space_image_vectors",
            "output_space_concept_vectors",
            "cross_attention_image_vectors",
            "cross_attention_concept_vectors",
        ]
        result = {}
        for key in keys:
            all_timesteps = []
            for step_dicts in concept_attention_dicts_all:
                layers = [d[key] for d in step_dicts if key in d]
                if layers:
                    all_timesteps.append(torch.stack(layers))
            if all_timesteps:
                result[key] = torch.stack(all_timesteps)  # (time, layers, batch, tokens, dim)
        return result

    # ── generate_image ────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate_image(
        self,
        prompt: str,
        concepts: list[str],
        width: int = 1024,
        height: int = 1024,
        layer_indices: list[int] | None = None,
        num_inference_steps: int = 4,
        guidance: float = 0.0,
        seed: int = 0,
        timesteps: list[int] | None = None,
        return_pil_heatmaps: bool = True,
        softmax: bool = True,
        cmap: str = "plasma",
        cache_vectors: bool = True,
        # img2img
        init_image: PIL.Image.Image | None = None,
        image2image_strength: float = 0.8,
        # streaming callback
        on_step_callback: Callable | None = None,
    ) -> ConceptAttentionPipelineOutput:
        """
        Generate an image with Flux 1, extracting concept attention heatmaps.

        For img2img pass init_image and image2image_strength (0.0–1.0).
        on_step_callback(step_idx, total_steps, b64_heatmaps) is called each step.
        """
        assert height == width, "Height and width must be equal"

        if layer_indices is None:
            layer_indices = list(range(15, 19))

        # Default aggregation timesteps
        if timesteps is None:
            start_idx = int(num_inference_steps * 0.7)
            timesteps = list(range(start_idx, num_inference_steps))

        # ── Build initial latent ──────────────────────────────────────────────
        x = get_noise(
            1, height, width,
            device=self.device,
            dtype=torch.bfloat16,
            seed=seed,
        )
        full_schedule = get_schedule(
            num_inference_steps,
            x.shape[-1] * x.shape[-2] // 4,
            shift=(not self.flux_generator.is_schnell),
        )

        if init_image is not None and image2image_strength > 0.0:
            # image2image_strength: fraction of original to PRESERVE
            #   1.0 = keep original exactly (no denoising)
            #   0.0 = full regeneration (ignore original, pure text2img)
            #   0.95 = add 5% noise, run 1 denoising step → mostly preserves original
            noise_level = 1.0 - image2image_strength   # e.g. strength=0.95 → noise=0.05
            t_start = float(noise_level)

            init_resized = init_image.resize((width, height), PIL.Image.LANCZOS).convert("RGB")
            img_np = np.array(init_resized).astype(np.float32) / 127.5 - 1.0  # [-1, 1]
            img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
            img_t = img_t.to(self.device, dtype=torch.float32)

            if self.offload_model:
                self.flux_generator.ae.encoder.to(self.device)
            encoded_init = self.flux_generator.ae.encode(img_t)
            if self.offload_model:
                self.flux_generator.ae = self.flux_generator.ae.cpu()
                torch.cuda.empty_cache()

            # Blend noise + image latent at chosen noise level
            x = t_start * x + (1.0 - t_start) * encoded_init.to(torch.bfloat16)

            # Build active schedule: [t_start, ...standard steps strictly below t_start..., 0.0]
            # This keeps the model seeing its trained timestep values for all but the first step.
            if t_start >= 1.0 - 1e-6:
                active_schedule = full_schedule
            else:
                active_schedule = [t_start] + [t for t in full_schedule[1:] if t < t_start]
                if not active_schedule or active_schedule[-1] > 1e-6:
                    active_schedule.append(0.0)
        else:
            active_schedule = full_schedule

        # ── Encode text + concepts ────────────────────────────────────────────
        if self.flux_generator.offload:
            self.flux_generator.t5 = self.flux_generator.t5.to(self.device)
            self.flux_generator.clip = self.flux_generator.clip.to(self.device)

        inp = prepare(
            t5=self.flux_generator.t5,
            clip=self.flux_generator.clip,
            img=x,
            prompt=prompt,
        )
        concept_embeddings, concept_ids, concept_vec = embed_concepts(
            self.flux_generator.clip,
            self.flux_generator.t5,
            concepts,
        )
        inp["concepts"] = concept_embeddings.to(x.device)
        inp["concept_ids"] = concept_ids.to(x.device)
        inp["concept_vec"] = concept_vec.to(x.device)

        if self.flux_generator.offload:
            self.flux_generator.t5 = self.flux_generator.t5.cpu()
            self.flux_generator.clip = self.flux_generator.clip.cpu()
            torch.cuda.empty_cache()
            self.flux_generator.model = self.flux_generator.model.to(self.device)

        # ── Denoising loop ────────────────────────────────────────────────────
        all_timestep_indices = list(range(len(active_schedule) - 1))

        img_out, concept_attention_dicts_all = self._denoise(
            inp=inp,
            timesteps=active_schedule,
            guidance=guidance,
            cache_vectors=cache_vectors,
            layer_indices=layer_indices,
            timestep_indices=all_timestep_indices,
            on_step_callback=on_step_callback,
            callback_num_concepts=len(concepts),
            callback_width=width,
            callback_height=height,
            callback_cmap=cmap,
            width=width,
            height=height,
        )

        # ── Decode ────────────────────────────────────────────────────────────
        self.flux_generator.model.cpu()
        torch.cuda.empty_cache()
        self.flux_generator.ae.decoder.to(img_out.device)

        img_out = unpack(img_out.float(), height, width)
        img_out = self.flux_generator.ae.decode(img_out.to(torch.float32).to(self.device))

        self.flux_generator.ae.decoder.cpu()
        torch.cuda.empty_cache()
        self.flux_generator.model.to(self.device)

        img_out = img_out.clamp(-1, 1)
        img_out = rearrange(img_out[0], "c h w -> h w c")
        pil_image = PIL.Image.fromarray((127.5 * (img_out + 1.0)).cpu().byte().numpy())

        # ── Compute heatmaps ──────────────────────────────────────────────────
        stacked = self._stack_vectors(concept_attention_dicts_all)

        # Map aggregation timestep indices to the active schedule
        effective_total = len(active_schedule) - 1
        agg_timesteps = [t for t in timesteps if t < effective_total] or list(range(effective_total))

        concept_heatmaps_tensor = compute_heatmaps_from_vectors(
            stacked["output_space_image_vectors"],
            stacked["output_space_concept_vectors"],
            layer_indices=layer_indices,
            timestep_indices=agg_timesteps,
            width=width, height=height,
            softmax=softmax,
        )[0]  # (num_concepts, h, w)

        cross_attention_tensor = compute_heatmaps_from_vectors(
            stacked["cross_attention_image_vectors"],
            stacked["cross_attention_concept_vectors"],
            layer_indices=layer_indices,
            timestep_indices=agg_timesteps,
            width=width, height=height,
            softmax=softmax,
        )[0]

        if return_pil_heatmaps:
            concept_heatmaps = heatmaps_to_pil_images(concept_heatmaps_tensor, width, height, cmap)
            cross_attention_maps = heatmaps_to_pil_images(cross_attention_tensor, width, height, cmap)
        else:
            concept_heatmaps = concept_heatmaps_tensor.cpu().numpy()
            cross_attention_maps = cross_attention_tensor.cpu().numpy()

        return ConceptAttentionPipelineOutput(
            image=pil_image,
            concept_heatmaps=concept_heatmaps,
            cross_attention_maps=cross_attention_maps,
            concept_output_vectors=stacked.get("output_space_concept_vectors"),
            image_output_vectors=stacked.get("output_space_image_vectors"),
        )

    # ── encode_image ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode_image(
        self,
        image: PIL.Image.Image,
        concepts: list[str],
        prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        layer_indices: list[int] | None = None,
        num_steps: int = 4,
        noise_timestep: int = 2,
        seed: int = 0,
        return_pil_heatmaps: bool = True,
        softmax: bool = True,
        cmap: str = "plasma",
        cache_vectors: bool = True,
    ) -> ConceptAttentionPipelineOutput:
        """
        Encode an existing image and extract concept attention heatmaps.
        Adds noise to the image at a specified timestep and runs one forward pass.
        """
        assert height == width, "Height and width must be equal"

        if layer_indices is None:
            layer_indices = list(range(15, 19))

        device = self.device
        print("Encoding image...")

        # ── VAE encode ────────────────────────────────────────────────────────
        image_resized = image.resize((width, height), PIL.Image.LANCZOS).convert("RGB")
        img_np = np.array(image_resized).astype(np.float32) / 127.5 - 1.0  # [-1, 1]
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        img_t = img_t.to(device, dtype=torch.float32)

        if self.flux_generator.offload:
            self.flux_generator.ae.encoder.to(device)
        encoded = self.flux_generator.ae.encode(img_t)
        if self.flux_generator.offload:
            self.flux_generator.ae = self.flux_generator.ae.cpu()
            torch.cuda.empty_cache()

        # ── Add noise ─────────────────────────────────────────────────────────
        schedule = get_schedule(
            num_steps,
            encoded.shape[-1] * encoded.shape[-2] // 4,
            shift=(not self.flux_generator.is_schnell),
        )
        t = schedule[noise_timestep]

        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn_like(encoded, generator=generator)
        noisy = (1.0 - t) * encoded + t * noise

        # ── Prepare inputs ────────────────────────────────────────────────────
        # Cast to bfloat16 to match transformer model weights
        noisy = noisy.to(torch.bfloat16)

        if self.flux_generator.offload:
            self.flux_generator.t5 = self.flux_generator.t5.to(device)
            self.flux_generator.clip = self.flux_generator.clip.to(device)

        inp = prepare(
            t5=self.flux_generator.t5,
            clip=self.flux_generator.clip,
            img=noisy,
            prompt=prompt,
        )
        concept_embeddings, concept_ids, concept_vec = embed_concepts(
            self.flux_generator.clip,
            self.flux_generator.t5,
            concepts,
        )
        inp["concepts"] = concept_embeddings.to(noisy.device)
        inp["concept_ids"] = concept_ids.to(noisy.device)
        inp["concept_vec"] = concept_vec.to(noisy.device)

        if self.flux_generator.offload:
            self.flux_generator.t5 = self.flux_generator.t5.cpu()
            self.flux_generator.clip = self.flux_generator.clip.cpu()
            torch.cuda.empty_cache()
            self.flux_generator.model = self.flux_generator.model.to(device)

        # ── Single forward pass ───────────────────────────────────────────────
        guidance_vec = torch.full((noisy.shape[0],), 0.0, device=noisy.device, dtype=torch.bfloat16)
        t_vec = torch.full((noisy.shape[0],), t, dtype=torch.bfloat16, device=noisy.device)

        _, step_dicts = self.flux_generator.model(
            img=inp["img"],
            img_ids=inp["img_ids"],
            txt=inp["txt"],
            txt_ids=inp["txt_ids"],
            concepts=inp["concepts"],
            concept_ids=inp["concept_ids"],
            concept_vec=inp["concept_vec"],
            y=inp["vec"],
            timesteps=t_vec,
            guidance=guidance_vec,
            stop_after_multimodal_attentions=True,
            joint_attention_kwargs=None,
            cache_vectors=cache_vectors,
            layer_indices=layer_indices,
        )

        # ── Compute heatmaps ──────────────────────────────────────────────────
        # Wrap as single-step for stacking: shape (1, layers, batch, tokens, dim)
        img_vecs = torch.stack([d["output_space_image_vectors"] for d in step_dicts if "output_space_image_vectors" in d]).unsqueeze(0)
        con_vecs = torch.stack([d["output_space_concept_vectors"] for d in step_dicts if "output_space_concept_vectors" in d]).unsqueeze(0)
        cross_img_vecs = torch.stack([d["cross_attention_image_vectors"] for d in step_dicts if "cross_attention_image_vectors" in d]).unsqueeze(0)
        cross_con_vecs = torch.stack([d["cross_attention_concept_vectors"] for d in step_dicts if "cross_attention_concept_vectors" in d]).unsqueeze(0)

        concept_heatmaps_tensor = compute_heatmaps_from_vectors(
            img_vecs, con_vecs,
            layer_indices=layer_indices,
            timestep_indices=[0],
            width=width, height=height,
            softmax=softmax,
        )[0]

        cross_attention_tensor = compute_heatmaps_from_vectors(
            cross_img_vecs, cross_con_vecs,
            layer_indices=layer_indices,
            timestep_indices=[0],
            width=width, height=height,
            softmax=softmax,
        )[0]

        if return_pil_heatmaps:
            concept_heatmaps = heatmaps_to_pil_images(concept_heatmaps_tensor, width, height, cmap)
            cross_attention_maps = heatmaps_to_pil_images(cross_attention_tensor, width, height, cmap)
        else:
            concept_heatmaps = concept_heatmaps_tensor.cpu().numpy()
            cross_attention_maps = cross_attention_tensor.cpu().numpy()

        return ConceptAttentionPipelineOutput(
            image=image_resized,
            concept_heatmaps=concept_heatmaps,
            cross_attention_maps=cross_attention_maps,
        )

    # ── compare_images ────────────────────────────────────────────────────────

    @torch.no_grad()
    def compare_images(
        self,
        original_image: PIL.Image.Image,
        prompt: str,
        concepts: list[str],
        width: int = 1024,
        height: int = 1024,
        layer_indices: list[int] | None = None,
        num_inference_steps: int = 4,
        noise_timestep: int = 2,
        seed: int = 0,
        softmax: bool = True,
        softmax_temperature: float = 1.0,
        cmap: str = "plasma",
        image2image_strength: float = 0.8,
        guidance: float = 0.0,
        save_path: str | None = None,
        on_step_callback: Callable | None = None,
    ) -> ConceptAttentionTemporalOutput:
        """
        Compare concept heatmaps between an original image and an img2img-generated image.

        image2image_strength: fraction of original image to preserve.
            1.0 = keep original exactly (no denoising)
            0.0 = full regeneration from prompt (original ignored)
            0.85 = add 15% noise, run 1–2 denoising steps → mostly preserves original

        on_step_callback(step_idx, total_steps, b64_heatmaps) is called each
        generation step — use this to stream heatmaps to a frontend in real time.

        Returns ConceptAttentionTemporalOutput with:
            - original_heatmaps: static heatmaps for the original image
            - temporal_heatmaps[step][concept]: heatmaps at each denoising step
        """
        print("=" * 50)
        print("Step 1/2: Encoding original image...")
        print("=" * 50)
        original_output = self.encode_image(
            image=original_image,
            concepts=concepts,
            prompt=prompt,
            width=width,
            height=height,
            layer_indices=layer_indices,
            noise_timestep=noise_timestep,
            seed=seed,
            softmax=softmax,
            cmap=cmap,
        )

        print("=" * 50)
        print(f"Step 2/2: Generating image (img2img strength={image2image_strength})...")
        print("=" * 50)

        # Collect per-step heatmaps for animation
        temporal_heatmaps: list[list[PIL.Image.Image]] = []

        def _collect_and_forward(step_idx, total_steps, b64_list):
            pil_list = []
            for b64 in b64_list:
                img_bytes = base64.b64decode(b64)
                pil_list.append(PIL.Image.open(io.BytesIO(img_bytes)))
            temporal_heatmaps.append(pil_list)
            if on_step_callback is not None:
                on_step_callback(step_idx, total_steps, b64_list)

        generated_output = self.generate_image(
            prompt=prompt,
            concepts=concepts,
            width=width,
            height=height,
            layer_indices=layer_indices,
            num_inference_steps=num_inference_steps,
            guidance=guidance,
            seed=seed,
            softmax=softmax,
            cmap=cmap,
            init_image=original_image,
            image2image_strength=image2image_strength,
            on_step_callback=_collect_and_forward,
        )

        # ── Build static comparison grid ──────────────────────────────────────
        fig = self._build_comparison_grid(
            original_image=original_output.image,
            generated_image=generated_output.image,
            original_heatmaps=original_output.concept_heatmaps,
            generated_heatmaps=generated_output.concept_heatmaps,
            concepts=concepts,
            prompt=prompt,
            width=width,
            height=height,
            cmap=cmap,
            save_path=save_path,
        )

        return ConceptAttentionTemporalOutput(
            original_image=original_output.image,
            generated_image=generated_output.image,
            concepts=concepts,
            original_heatmaps=original_output.concept_heatmaps,
            generated_heatmaps=generated_output.concept_heatmaps,
            temporal_heatmaps=temporal_heatmaps,
            num_denoising_steps=len(temporal_heatmaps),
        )

    def _build_comparison_grid(
        self,
        original_image: PIL.Image.Image,
        generated_image: PIL.Image.Image,
        original_heatmaps: list[PIL.Image.Image],
        generated_heatmaps: list[PIL.Image.Image],
        concepts: list[str],
        prompt: str,
        width: int,
        height: int,
        cmap: str = "plasma",
        save_path: str | None = None,
    ):
        """
        Build grid:
            2 columns: original | generated
            Row 0: complete images (red border)
            Rows 1..N: heatmap per concept overlaid on image (blue border)
        """
        num_rows = 1 + len(concepts)

        fig, axes = plt.subplots(
            num_rows, 2,
            figsize=(10, 4 * num_rows),
            gridspec_kw={"hspace": 0.45, "wspace": 0.08}
        )

        orig_resized = original_image.resize((width, height), PIL.Image.LANCZOS)
        images = [orig_resized, generated_image]
        col_titles = ["Original", "Generated"]
        heatmaps_cols = [original_heatmaps, generated_heatmaps]

        # Row 0: complete images
        for col in range(2):
            ax = axes[0, col]
            ax.imshow(images[col])
            ax.set_title(col_titles[col], fontsize=13, fontweight="bold",
                         color="#c0392b", pad=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#e74c3c")
                spine.set_linewidth(3)
            ax.set_xticks([]); ax.set_yticks([])
        axes[0, 0].set_ylabel("Complete\nImages", fontsize=10, fontweight="bold",
                               color="#c0392b", rotation=90, labelpad=10)

        # Rows 1..N: concept heatmaps
        for row, concept in enumerate(concepts, start=1):
            for col in range(2):
                ax = axes[row, col]
                blended = _overlay_heatmap(images[col], heatmaps_cols[col][row - 1], alpha=0.6)
                ax.imshow(blended)
                for spine in ax.spines.values():
                    spine.set_edgecolor("#2980b9")
                    spine.set_linewidth(2)
                ax.set_xticks([]); ax.set_yticks([])
            axes[row, 0].set_ylabel(f"({row}) {concept}", fontsize=10,
                                     fontweight="bold", color="#2980b9",
                                     rotation=90, labelpad=10)

        fig.suptitle(f'Concept Attention\nPrompt: "{prompt}"',
                     fontsize=12, fontweight="bold", y=1.01)

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"Saved to: {save_path}")

        plt.tight_layout()
        plt.show()
        return fig


# ─────────────────────────────────────────────────────────────────────────────
# Interactive Colab UI
# ─────────────────────────────────────────────────────────────────────────────

def run_interactive_ui(pipe: ConceptAttentionFluxPipeline):
    """
    Launch an interactive Colab widget UI.

    Usage:
        from concept_attention.flux.pipeline import run_interactive_ui
        run_interactive_ui(pipe)
    """
    import ipywidgets as widgets
    from IPython.display import display, clear_output

    title = widgets.HTML("<h2 style='color:#2c3e50'>🎨 Concept Attention Explorer</h2>")

    upload_btn = widgets.FileUpload(accept=".png,.jpg,.jpeg", multiple=False,
                                    description="Upload Image",
                                    layout=widgets.Layout(width="250px"))
    upload_label = widgets.Label("No image uploaded yet.")

    prompt_box = widgets.Textarea(
        value="A woman wearing an animal costume in a jungle",
        description="Prompt:",
        layout=widgets.Layout(width="100%", height="70px"),
        style={"description_width": "60px"}
    )
    concepts_box = widgets.Text(
        value="woman, animal, costume, jungle",
        description="Concepts:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "60px"}
    )
    concepts_hint = widgets.HTML("<small style='color:gray'>Comma-separated concept tokens</small>")
    size_dropdown = widgets.Dropdown(options=[512, 768, 1024], value=512,
                                     description="Image size:",
                                     style={"description_width": "80px"})
    seed_input = widgets.IntText(value=0, description="Seed:",
                                  layout=widgets.Layout(width="150px"),
                                  style={"description_width": "40px"})
    strength_slider = widgets.FloatSlider(
        value=0.85, min=0.1, max=1.0, step=0.05,
        description="Img2Img strength:",
        style={"description_width": "130px"},
        layout=widgets.Layout(width="420px"),
        readout_format=".2f",
    )
    strength_hint = widgets.HTML(
        "<small style='color:gray'>Lower = stays closer to original | Higher = follows prompt more</small>"
    )
    run_btn = widgets.Button(description="▶ Run", button_style="success",
                              layout=widgets.Layout(width="120px", height="36px"))
    save_checkbox = widgets.Checkbox(
        value=True,
        description="Save output to /content/results/comparison.png",
        indent=False
    )
    output_area = widgets.Output()

    state = {"image": None}

    def on_upload(change):
        if upload_btn.value:
            import io as _io
            file_info = list(upload_btn.value.values())[0]
            img = PIL.Image.open(_io.BytesIO(file_info["content"])).convert("RGB")
            state["image"] = img
            upload_label.value = f"✅ Loaded: {file_info['metadata']['name']} ({img.width}×{img.height})"

    upload_btn.observe(on_upload, names="value")

    def on_run(b):
        with output_area:
            clear_output(wait=True)
            if state["image"] is None:
                print("⚠️  Please upload an image first."); return
            raw_concepts = [c.strip() for c in concepts_box.value.split(",") if c.strip()]
            if not raw_concepts:
                print("⚠️  Please enter at least one concept token."); return
            prompt = prompt_box.value.strip()
            if not prompt:
                print("⚠️  Please enter a prompt."); return

            size = size_dropdown.value
            seed = seed_input.value
            strength = strength_slider.value
            save_path = "/content/results/comparison.png" if save_checkbox.value else None

            print(f"Prompt   : {prompt}")
            print(f"Concepts : {raw_concepts}")
            print(f"Size     : {size}×{size}  |  Seed: {seed}  |  Strength: {strength:.2f}")

            try:
                result = pipe.compare_images(
                    original_image=state["image"],
                    prompt=prompt,
                    concepts=raw_concepts,
                    width=size, height=size,
                    seed=seed,
                    save_path=save_path,
                    image2image_strength=strength,
                )
                if save_path:
                    print(f"\n✅ Saved to {save_path}")
            except Exception as e:
                print(f"❌ Error: {e}"); raise

    run_btn.on_click(on_run)

    ui = widgets.VBox([
        title,
        widgets.VBox([widgets.HTML("<b>1. Upload your image</b>"), upload_btn, upload_label]),
        widgets.HTML("<hr>"),
        widgets.VBox([
            widgets.HTML("<b>2. Set prompt & concepts</b>"),
            prompt_box, concepts_box, concepts_hint,
            widgets.HBox([size_dropdown, seed_input]),
            strength_slider, strength_hint,
            save_checkbox,
        ]),
        widgets.HTML("<hr>"),
        widgets.VBox([widgets.HTML("<b>3. Run</b>"), run_btn]),
        output_area,
    ], layout=widgets.Layout(padding="16px", max_width="750px"))

    display(ui)