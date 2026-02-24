"""
    Wrapper pipeline for Flux 2 concept attention.
"""
from dataclasses import dataclass, field
import base64
import io
import os
import sys
from typing import Callable

import PIL
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import einops
from torch import Tensor
from einops import rearrange
from PIL import Image
from safetensors.torch import load_file as load_sft
from tqdm import tqdm
import huggingface_hub

from concept_attention.flux2.flux2.src.flux2.model import Flux2Params
from concept_attention.flux2.flux2.src.flux2.sampling import (
    batched_prc_img,
    batched_prc_txt,
    get_schedule,
    scatter_ids,
)
from concept_attention.flux2.flux2.src.flux2.util import load_ae, load_mistral_small_embedder
from concept_attention.flux2.dit import ModifiedFlux2


@dataclass
class ConceptAttentionPipelineOutput:
    """Output from the ConceptAttentionFlux2Pipeline."""
    image: PIL.Image.Image | np.ndarray
    concept_heatmaps: list[PIL.Image.Image]
    cross_attention_maps: list[PIL.Image.Image]
    # Raw output vectors (only populated when cache_vectors=True)
    concept_output_vectors: torch.Tensor | np.ndarray | None = None
    image_output_vectors: torch.Tensor | np.ndarray | None = None


@dataclass
class ConceptAttentionTemporalOutput:
    """Output from compare_images() with per-timestep heatmaps for animation."""
    original_image: PIL.Image.Image
    generated_image: PIL.Image.Image
    concepts: list[str]
    # Static heatmaps for original image (from encode_image, one per concept)
    original_heatmaps: list[PIL.Image.Image]
    # Per-timestep heatmaps for generated image [timestep_idx][concept_idx]
    temporal_heatmaps: list[list[PIL.Image.Image]]
    num_denoising_steps: int


def compute_heatmaps_from_attention_dicts(
    concept_attention_dicts: list,
    num_concepts: int,
    width: int,
    height: int,
    layer_indices: list[int],
    timestep_indices: list[int],
    key: str = "concept_scores",
    softmax: bool = True,
    softmax_temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute spatial heatmaps from concept attention dictionaries.

    Args:
        concept_attention_dicts: List of attention dicts per timestep
        num_concepts: Number of concepts
        width: Image width in pixels
        height: Image height in pixels
        layer_indices: Which layers/blocks to average over
        timestep_indices: Which timesteps to average over
        key: Which attention key to use ("concept_scores" or "cross_attention_scores")
        softmax: Whether to apply softmax normalization
        softmax_temperature: Temperature for softmax

    Returns:
        Tensor of shape (num_concepts, h, w) with spatial heatmaps
    """
    # Stack concept attention over time and blocks
    selected_concept_attention = []
    for t_idx in timestep_indices:
        if t_idx >= len(concept_attention_dicts):
            continue
        time_step_dicts = concept_attention_dicts[t_idx]
        selected_blocks = []
        for b_idx in layer_indices:
            if b_idx >= len(time_step_dicts):
                continue
            selected_blocks.append(time_step_dicts[b_idx][key])
        if selected_blocks:
            selected_blocks = torch.stack(selected_blocks)
            selected_concept_attention.append(selected_blocks)

    if not selected_concept_attention:
        raise ValueError("No valid attention data found for the specified indices")

    selected_concept_attention = torch.stack(selected_concept_attention)

    # Average over time and blocks
    avg_concept_scores = einops.reduce(
        selected_concept_attention,
        "time blocks batch num_concepts num_image_tokens -> batch num_concepts num_image_tokens",
        "mean",
    )
    avg_concept_scores = avg_concept_scores[0]  # Remove batch dim

    # Reshape to spatial grid (16 pixels per latent token)
    num_image_tokens_h = height // 16
    num_image_tokens_w = width // 16
    avg_concept_scores = einops.rearrange(
        avg_concept_scores,
        "num_concepts (h w) -> num_concepts h w",
        h=num_image_tokens_h,
        w=num_image_tokens_w,
    )

    # Apply softmax normalization across concepts
    if softmax:
        avg_concept_scores = torch.softmax(avg_concept_scores / softmax_temperature, dim=0)

    return avg_concept_scores


def heatmaps_to_pil_images(
    heatmaps: torch.Tensor,
    width: int,
    height: int,
    cmap: str = "plasma",
) -> list[PIL.Image.Image]:
    """
    Convert tensor heatmaps to colored PIL images.

    Args:
        heatmaps: Tensor of shape (num_concepts, h, w)
        width: Target width
        height: Target height
        cmap: Matplotlib colormap name

    Returns:
        List of PIL images
    """
    heatmaps_np = heatmaps.cpu().float().numpy()
    global_min = heatmaps_np.min()
    global_max = heatmaps_np.max()

    colormap = plt.get_cmap(cmap)
    pil_images = []

    for concept_heatmap in heatmaps_np:
        # Normalize to [0, 1]
        normalized = (concept_heatmap - global_min) / (global_max - global_min + 1e-8)
        # Apply colormap
        colored = colormap(normalized)
        rgb_image = (colored[:, :, :3] * 255).astype(np.uint8)
        pil_img = PIL.Image.fromarray(rgb_image)
        # Resize to target dimensions
        pil_img = pil_img.resize((width, height), resample=PIL.Image.NEAREST)
        pil_images.append(pil_img)

    return pil_images


def compute_heatmaps_for_single_step(
    step_dicts: list,
    num_concepts: int,
    width: int,
    height: int,
    layer_indices: list[int],
    key: str = "concept_scores",
    softmax: bool = True,
    softmax_temperature: float = 1000.0,
) -> torch.Tensor:
    """
    Compute heatmaps for a single denoising step's attention dicts.

    Args:
        step_dicts: List of per-block attention dicts for one timestep
        (same format as concept_attention_dicts[t] in the full pipeline)

    Returns:
        Tensor of shape (num_concepts, h, w)
    """
    return compute_heatmaps_from_attention_dicts(
        [step_dicts],
        num_concepts=num_concepts,
        width=width,
        height=height,
        layer_indices=layer_indices,
        timestep_indices=[0],
        key=key,
        softmax=softmax,
        softmax_temperature=softmax_temperature,
    )


def heatmap_tensor_to_base64(
    heatmap_tensor: torch.Tensor,
    width: int,
    height: int,
    cmap: str = "plasma",
) -> list[str]:
    """
    Convert a (num_concepts, h, w) heatmap tensor to a list of base64-encoded PNG strings.
    Used for streaming heatmaps over WebSocket.
    """
    pil_images = heatmaps_to_pil_images(heatmap_tensor, width, height, cmap)
    result = []
    for img in pil_images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        result.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
    return result


def stack_output_vectors(
    concept_attention_dicts: list,
    key: str,
) -> torch.Tensor | None:
    """
    Stack output vectors from concept_attention_dicts.
    Vectors are already filtered to only requested layers/timesteps.

    Args:
        concept_attention_dicts: List of attention dicts per timestep,
                                 each containing a list of dicts per layer
        key: "concept_output_vectors" or "image_output_vectors"

    Returns:
        Tensor of shape (batch, time, layers, tokens, dim) or None if no vectors
    """
    all_timesteps = []
    for timestep_dicts in concept_attention_dicts:
        layers = []
        for layer_dict in timestep_dicts:
            if key in layer_dict:
                layers.append(layer_dict[key])
        if layers:
            all_timesteps.append(torch.stack(layers))  # (layers, batch, tokens, dim)

    if not all_timesteps:
        return None

    stacked = torch.stack(all_timesteps)  # (time, layers, batch, tokens, dim)
    return stacked.permute(2, 0, 1, 3, 4)  # (batch, time, layers, tokens, dim)


class ConceptAttentionFlux2Pipeline:
    """
    Pipeline for generating images with Flux 2 and extracting concept attention heatmaps.

    This mirrors the interface of ConceptAttentionFluxPipeline for Flux 1.
    """

    def __init__(
        self,
        model_name: str = "flux.2-dev",
        device: str = "cuda:0",
        offload_model: bool = False,
    ):
        """
        Initialize the Flux 2 pipeline.

        Args:
            model_name: Model name (currently only "flux.2-dev" supported)
            device: Device to run on
            offload_model: Whether to offload models to CPU when not in use
        """
        self.model_name = model_name
        self.device = device
        self.offload_model = offload_model

        print("Loading Flux 2 models...")

        # Load Mistral text embedder
        self.mistral = load_mistral_small_embedder()
        self.mistral.eval()

        # Load flow model
        self._load_flow_model()

        # Load VAE autoencoder
        self.ae = load_ae(model_name)
        self.ae.eval()

        print("Flux 2 models loaded successfully!")

    def _load_flow_model(self):
        """Load the ModifiedFlux2 flow model."""
        repo_id = "black-forest-labs/FLUX.2-dev"
        filename = "flux2-dev.safetensors"

        if "FLUX2_MODEL_PATH" in os.environ:
            weight_path = os.environ["FLUX2_MODEL_PATH"]
        else:
            try:
                weight_path = huggingface_hub.hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    repo_type="model",
                )
            except huggingface_hub.errors.RepositoryNotFoundError:
                print(f"Failed to access {repo_id}. Check your access permissions.")
                sys.exit(1)

        with torch.device("meta"):
            self.model = ModifiedFlux2(Flux2Params()).to(torch.bfloat16)

        print(f"Loading weights from {weight_path}")
        sd = load_sft(weight_path, device=str(self.device))
        self.model.load_state_dict(sd, strict=False, assign=True)
        self.model = self.model.to(self.device)

    def _denoise(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        concepts: Tensor,
        concept_ids: Tensor,
        timesteps: list[float],
        guidance: float,
        cache_vectors: bool = True,
        layer_indices: list[int] | None = None,
        timestep_indices: list[int] | None = None,
        # Callback for streaming per-step heatmaps:
        # on_step_callback(step_idx, total_steps, step_dicts) called each denoising step
        on_step_callback: Callable | None = None,
        # Parameters forwarded to the callback for heatmap computation
        callback_num_concepts: int | None = None,
        callback_width: int | None = None,
        callback_height: int | None = None,
        callback_softmax_temperature: float = 1000.0,
        callback_cmap: str = "plasma",
    ) -> tuple[Tensor, list]:
        """
        Denoising loop that tracks concept attention at each step.

        Args:
            cache_vectors: Whether to cache raw output vectors
            layer_indices: Which layers to cache vectors for (None = all)
            timestep_indices: Which timesteps to cache vectors for (None = all)
            on_step_callback: Optional callback(step_idx, total_steps, step_dicts)
                called after each denoising step with the raw attention dicts.
                If callback_num_concepts/width/height are also provided, the callback
                receives pre-computed base64 heatmaps instead of raw dicts.

        Returns:
            Tuple of (denoised_img, concept_attention_dicts)
        """
        guidance_vec = torch.full(
            (img.shape[0],), guidance, device=img.device, dtype=img.dtype
        )
        concept_attention_dicts = []
        total_steps = len(timesteps) - 1

        for step_idx, (t_curr, t_prev) in enumerate(tqdm(
            zip(timesteps[:-1], timesteps[1:]),
            total=total_steps,
            desc="Denoising"
        )):
            # Check if this timestep should be tracked
            should_track_timestep = (
                timestep_indices is None or step_idx in timestep_indices
            )

            t_vec = torch.full(
                (img.shape[0],), t_curr, dtype=img.dtype, device=img.device
            )

            pred, current_concept_attention_dict = self.model(
                x=img,
                x_ids=img_ids,
                timesteps=t_vec,
                ctx=txt,
                ctx_ids=txt_ids,
                guidance=guidance_vec,
                concepts=concepts,
                concept_ids=concept_ids,
                # Always cache at this step so the callback can read it
                cache_vectors=(cache_vectors and should_track_timestep) or (on_step_callback is not None),
                layer_indices=layer_indices,
                current_timestep=step_idx,
            )
            concept_attention_dicts.append(current_concept_attention_dict)
            img = img + (t_prev - t_curr) * pred

            # Fire callback with per-step heatmaps
            if on_step_callback is not None:
                if (
                    callback_num_concepts is not None
                    and callback_width is not None
                    and callback_height is not None
                ):
                    effective_layer_indices = layer_indices if layer_indices is not None else [5, 6, 7]
                    heatmap_tensor = compute_heatmaps_for_single_step(
                        current_concept_attention_dict,
                        num_concepts=callback_num_concepts,
                        width=callback_width,
                        height=callback_height,
                        layer_indices=effective_layer_indices,
                        key="concept_scores",
                        softmax=True,
                        softmax_temperature=callback_softmax_temperature,
                    )
                    b64_heatmaps = heatmap_tensor_to_base64(
                        heatmap_tensor, callback_width, callback_height, callback_cmap
                    )
                    on_step_callback(step_idx, total_steps, b64_heatmaps)
                else:
                    on_step_callback(step_idx, total_steps, current_concept_attention_dict)

        return img, concept_attention_dicts

    @torch.no_grad()
    def generate_image(
        self,
        prompt: str,
        concepts: list[str],
        width: int = 2048,
        height: int = 2048,
        num_inference_steps: int = 28,
        guidance: float = 4.0,
        seed: int = 0,
        layer_indices: list[int] = None,
        timesteps: list[int] = None,
        return_pil_heatmaps: bool = True,
        softmax: bool = True,
        softmax_temperature: float = 1000.0,
        cmap: str = "plasma",
        cache_vectors: bool = True,
        # img2img params
        init_image: PIL.Image.Image | None = None,
        image2image_strength: float = 0.8,
        # Streaming callback: called each denoising step with (step_idx, total_steps, b64_heatmaps)
        on_step_callback: Callable | None = None,
    ) -> ConceptAttentionPipelineOutput:
        """
        Generate an image with Flux 2 and extract concept attention heatmaps.

        Args:
            prompt: Text prompt for generation
            concepts: List of concept words to track attention for
            width: Output image width (should be divisible by 16)
            height: Output image height (should be divisible by 16)
            num_inference_steps: Number of denoising steps
            guidance: Guidance scale
            seed: Random seed
            layer_indices: Which transformer blocks to average over (default: last 3)
            timesteps: Which timesteps to average over (default: last 30%)
            return_pil_heatmaps: Whether to return PIL images (True) or numpy arrays
            softmax: Whether to apply softmax normalization
            softmax_temperature: Temperature for softmax
            cmap: Matplotlib colormap for heatmaps
            cache_vectors: Whether to cache raw output vectors (default: True)

        Returns:
            ConceptAttentionPipelineOutput with image, concept_heatmaps, cross_attention_maps,
            and optionally concept_output_vectors and image_output_vectors
        """
        # Default layer indices (last 3 of 8 double blocks)
        if layer_indices is None:
            layer_indices = [5, 6, 7]

        # Default timestep indices (last 30% of steps) — used for final heatmap aggregation
        if timesteps is None:
            start_idx = int(num_inference_steps * 0.7)
            timesteps = list(range(start_idx, num_inference_steps - 1))

        # Validate inputs
        assert all(0 <= idx < 8 for idx in layer_indices), "layer_indices must be in [0, 7]"

        # Encode text prompt
        ctx = self.mistral([prompt]).to(torch.bfloat16)
        ctx, ctx_ids = batched_prc_txt(ctx)

        # Encode concepts
        concept_embeddings = self.mistral(concepts).to(torch.bfloat16)
        concepts_tensor, concept_ids = batched_prc_txt(concept_embeddings)
        # Extract single token representation per concept (at position 510)
        concepts_tensor = concepts_tensor[:, 510].unsqueeze(0)
        concept_ids = concept_ids[:, 510].unsqueeze(0)

        # Offload text encoder if needed
        if self.offload_model:
            self.mistral = self.mistral.cpu()
            torch.cuda.empty_cache()
            self.model = self.model.to(self.device)

        # Prepare initial latents (pure noise or img2img noised latent)
        shape = (1, 128, height // 16, width // 16)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noise = torch.randn(shape, generator=generator, dtype=torch.bfloat16, device=self.device)

        seq_len = (height // 16) * (width // 16)
        full_schedule = get_schedule(num_inference_steps, seq_len)

        if init_image is not None and image2image_strength > 0.0:
            # img2img: start from a partially noised version of the init image
            init_resized = init_image.resize((width, height), PIL.Image.LANCZOS).convert("RGB")
            img_array = np.array(init_resized).astype(np.float32) / 127.5 - 1.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
            img_tensor = img_tensor.to(self.device, dtype=torch.bfloat16)
            encoded_init = self.ae.encode(img_tensor)

            # Truncate schedule: strength=1.0 → full noise (start from beginning)
            # strength=0.0 → no noise (skip denoising), strength=0.8 → 80% of diffusion
            start_step = int(num_inference_steps * (1.0 - image2image_strength))
            t_start = full_schedule[start_step]
            active_schedule = full_schedule[start_step:]

            noisy_init = (1.0 - t_start) * encoded_init + t_start * noise
            x, x_ids = batched_prc_img(noisy_init)
        else:
            x, x_ids = batched_prc_img(noise)
            active_schedule = full_schedule

        # For temporal callback: all steps are tracked (not just the aggregation window)
        all_timestep_indices = list(range(len(active_schedule) - 1))

        x, concept_attention_dicts = self._denoise(
            x, x_ids, ctx, ctx_ids,
            concepts=concepts_tensor,
            concept_ids=concept_ids,
            timesteps=active_schedule,
            guidance=guidance,
            cache_vectors=cache_vectors,
            layer_indices=layer_indices,
            timestep_indices=all_timestep_indices,
            on_step_callback=on_step_callback,
            callback_num_concepts=len(concepts),
            callback_width=width,
            callback_height=height,
            callback_softmax_temperature=softmax_temperature,
            callback_cmap=cmap,
        )

        # Map the aggregation timestep indices to the active schedule
        effective_total = len(active_schedule) - 1
        agg_timesteps = [t for t in timesteps if t < effective_total]
        if not agg_timesteps:
            agg_timesteps = list(range(effective_total))

        # Offload flow model, load VAE
        if self.offload_model:
            self.model = self.model.cpu()
            torch.cuda.empty_cache()

        # Decode latents to image
        x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
        x = self.ae.decode(x).float()

        # Convert to PIL image
        x = x.clamp(-1, 1)
        x = rearrange(x[0], "c h w -> h w c")
        image = PIL.Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

        # Compute concept heatmaps (output space attention) — averaged over agg window
        concept_heatmaps_tensor = compute_heatmaps_from_attention_dicts(
            concept_attention_dicts,
            num_concepts=len(concepts),
            width=width,
            height=height,
            layer_indices=layer_indices,
            timestep_indices=agg_timesteps,
            key="concept_scores",
            softmax=softmax,
            softmax_temperature=softmax_temperature,
        )

        # Compute cross-attention heatmaps
        cross_attention_tensor = compute_heatmaps_from_attention_dicts(
            concept_attention_dicts,
            num_concepts=len(concepts),
            width=width,
            height=height,
            layer_indices=layer_indices,
            timestep_indices=agg_timesteps,
            key="cross_attention_scores",
            softmax=softmax,
            softmax_temperature=softmax_temperature,
        )

        # Convert to PIL images if requested
        if return_pil_heatmaps:
            concept_heatmaps = heatmaps_to_pil_images(
                concept_heatmaps_tensor, width, height, cmap
            )
            cross_attention_maps = heatmaps_to_pil_images(
                cross_attention_tensor, width, height, cmap
            )
        else:
            concept_heatmaps = concept_heatmaps_tensor.cpu().numpy()
            cross_attention_maps = cross_attention_tensor.cpu().numpy()

        # Restore models if offloaded
        if self.offload_model:
            self.mistral = self.mistral.to(self.device)

        # Stack raw output vectors if caching is enabled
        concept_output_vectors = None
        image_output_vectors = None
        if cache_vectors:
            concept_output_vectors = stack_output_vectors(
                concept_attention_dicts, "concept_output_vectors"
            )
            image_output_vectors = stack_output_vectors(
                concept_attention_dicts, "image_output_vectors"
            )

        return ConceptAttentionPipelineOutput(
            image=image,
            concept_heatmaps=concept_heatmaps,
            cross_attention_maps=cross_attention_maps,
            concept_output_vectors=concept_output_vectors,
            image_output_vectors=image_output_vectors,
        )

    @torch.no_grad()
    def encode_image(
        self,
        image: PIL.Image.Image,
        concepts: list[str],
        prompt: str = "",
        width: int = 2048,
        height: int = 2048,
        layer_indices: list[int] = None,
        num_steps: int = 4,
        noise_timestep: int = 2,
        seed: int = 0,
        return_pil_heatmaps: bool = True,
        softmax: bool = True,
        softmax_temperature: float = 1000.0,
        cmap: str = "plasma",
        cache_vectors: bool = True,
    ) -> ConceptAttentionPipelineOutput:
        """
        Encode an existing image and extract concept attention heatmaps.

        Args:
            image: Input PIL image
            concepts: List of concept words to track attention for
            prompt: Optional text prompt describing the image
            width: Processing width (image will be resized)
            height: Processing height (image will be resized)
            layer_indices: Which transformer blocks to average over
            num_steps: Number of noise levels to use
            noise_timestep: Which noise level to add
            seed: Random seed for noise
            return_pil_heatmaps: Whether to return PIL images
            softmax: Whether to apply softmax normalization
            softmax_temperature: Temperature for softmax
            cmap: Matplotlib colormap for heatmaps
            cache_vectors: Whether to cache raw output vectors (default: True)

        Returns:
            ConceptAttentionPipelineOutput with original image, concept_heatmaps, cross_attention_maps,
            and optionally concept_output_vectors and image_output_vectors
        """
        # Default layer indices
        if layer_indices is None:
            layer_indices = [5, 6, 7]

        # Resize image to target dimensions
        image_resized = image.resize((width, height), PIL.Image.LANCZOS)

        # Encode text prompt and concepts
        ctx = self.mistral([prompt] if prompt else [""]).to(torch.bfloat16)
        ctx, ctx_ids = batched_prc_txt(ctx)

        concept_embeddings = self.mistral(concepts).to(torch.bfloat16)
        concepts_tensor, concept_ids = batched_prc_txt(concept_embeddings)
        concepts_tensor = concepts_tensor[:, 510].unsqueeze(0)
        concept_ids = concept_ids[:, 510].unsqueeze(0)

        if self.offload_model:
            self.mistral = self.mistral.cpu()
            torch.cuda.empty_cache()
            self.model = self.model.to(self.device)

        # Encode image to latent space
        img_array = np.array(image_resized).astype(np.float32) / 127.5 - 1.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(self.device, dtype=torch.bfloat16)

        # Encode with VAE
        encoded_image = self.ae.encode(img_tensor)

        # Add noise
        schedule = get_schedule(num_steps, encoded_image.shape[2] * encoded_image.shape[3])
        t = schedule[noise_timestep]

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noise = torch.randn_like(encoded_image, generator=generator)
        noisy_latent = (1 - t) * encoded_image + t * noise

        # Prepare for model
        x, x_ids = batched_prc_img(noisy_latent)

        # Run single forward pass
        t_vec = torch.full((1,), t, dtype=x.dtype, device=x.device)
        guidance_vec = torch.full((1,), 0.0, dtype=x.dtype, device=x.device)

        _, concept_attention_dict = self.model(
            x=x,
            x_ids=x_ids,
            timesteps=t_vec,
            ctx=ctx,
            ctx_ids=ctx_ids,
            guidance=guidance_vec,
            concepts=concepts_tensor,
            concept_ids=concept_ids,
            cache_vectors=cache_vectors,
            layer_indices=layer_indices,
            current_timestep=0,
        )

        # Wrap in list for compatibility with compute function
        concept_attention_dicts = [concept_attention_dict]

        # Compute heatmaps
        concept_heatmaps_tensor = compute_heatmaps_from_attention_dicts(
            concept_attention_dicts,
            num_concepts=len(concepts),
            width=width,
            height=height,
            layer_indices=layer_indices,
            timestep_indices=[0],
            key="concept_scores",
            softmax=softmax,
            softmax_temperature=softmax_temperature,
        )

        cross_attention_tensor = compute_heatmaps_from_attention_dicts(
            concept_attention_dicts,
            num_concepts=len(concepts),
            width=width,
            height=height,
            layer_indices=layer_indices,
            timestep_indices=[0],
            key="cross_attention_scores",
            softmax=softmax,
            softmax_temperature=softmax_temperature,
        )

        if return_pil_heatmaps:
            concept_heatmaps = heatmaps_to_pil_images(
                concept_heatmaps_tensor, width, height, cmap
            )
            cross_attention_maps = heatmaps_to_pil_images(
                cross_attention_tensor, width, height, cmap
            )
        else:
            concept_heatmaps = concept_heatmaps_tensor.cpu().numpy()
            cross_attention_maps = cross_attention_tensor.cpu().numpy()

        if self.offload_model:
            self.model = self.model.cpu()
            self.mistral = self.mistral.to(self.device)
            torch.cuda.empty_cache()

        # Stack raw output vectors if caching is enabled
        concept_output_vectors = None
        image_output_vectors = None
        if cache_vectors:
            concept_output_vectors = stack_output_vectors(
                concept_attention_dicts, "concept_output_vectors"
            )
            image_output_vectors = stack_output_vectors(
                concept_attention_dicts, "image_output_vectors"
            )

        return ConceptAttentionPipelineOutput(
            image=image_resized,
            concept_heatmaps=concept_heatmaps,
            cross_attention_maps=cross_attention_maps,
            concept_output_vectors=concept_output_vectors,
            image_output_vectors=image_output_vectors,
        )

    @torch.no_grad()
    def compare_images(
        self,
        original_image: PIL.Image.Image,
        prompt: str,
        concepts: list[str],
        width: int = 1024,
        height: int = 1024,
        layer_indices: list[int] = None,
        num_inference_steps: int = 28,
        noise_timestep: int = 1,
        seed: int = 0,
        softmax: bool = True,
        softmax_temperature: float = 1000.0,
        cmap: str = "plasma",
        image2image_strength: float = 0.8,
        guidance: float = 4.0,
        # Streaming: called each generation step with (step_idx, total_steps, b64_heatmaps_list)
        on_step_callback: Callable | None = None,
    ) -> "ConceptAttentionTemporalOutput":
        """
        Compare concept heatmaps between an original (encoded) image and a
        generated image conditioned on the original via img2img.

        Returns a ConceptAttentionTemporalOutput with:
        - Static heatmaps for the original image
        - Per-timestep heatmaps for the generated image (for animation)

        Grid layout:
            columns: concept tokens  (x-axis)
            rows:    original | generated  (y-axis)
            time:    animated via timestep slider

        Args:
            image2image_strength: 0.0 = stay close to original, 1.0 = generate freely
        """
        if layer_indices is None:
            layer_indices = [5, 6, 7]

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
            return_pil_heatmaps=True,
            softmax=softmax,
            softmax_temperature=softmax_temperature,
            cmap=cmap,
        )

        print("=" * 50)
        print(f"Step 2/2: Generating image (img2img strength={image2image_strength})...")
        print("=" * 50)

        # Collect per-timestep heatmaps for animation
        temporal_heatmaps: list[list[PIL.Image.Image]] = []
        temporal_b64: list[list[str]] = []

        def _collect_and_forward(step_idx, total_steps, b64_list):
            temporal_b64.append(b64_list)
            # Convert base64 back to PIL for local use
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
            softmax_temperature=softmax_temperature,
            cmap=cmap,
            init_image=original_image,
            image2image_strength=image2image_strength,
            on_step_callback=_collect_and_forward,
        )

        return ConceptAttentionTemporalOutput(
            original_image=original_output.image,
            generated_image=generated_output.image,
            concepts=concepts,
            original_heatmaps=original_output.concept_heatmaps,
            temporal_heatmaps=temporal_heatmaps,
            num_denoising_steps=len(temporal_heatmaps),
        )
