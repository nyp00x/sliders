#!/usr/bin/env python3
"""
FLUX Slider Training Script
Converts the FLUX concept sliders notebook into a runnable script with CLI arguments.
"""

import os
import sys
import torch
import gc
import copy
import argparse
import random
import logging
from pathlib import Path
from contextlib import ExitStack
from tqdm.auto import tqdm
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch.nn.functional as F
from torch.optim import AdamW
from transformers import (
    CLIPTokenizer,
    PretrainedConfig,
    T5TokenizerFast,
    CLIPTextModel,
    T5EncoderModel,
)

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling

# Add utils to path
sys.path.append(str(Path(__file__).parent))
from utils.custom_flux_pipeline import FluxPipeline
from utils.lora import (
    LoRANetwork,
    DEFAULT_TARGET_REPLACE,
    UNET_TARGET_REPLACE_MODULE_CONV,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def flush():
    """Clear CUDA cache and run garbage collection"""
    torch.cuda.empty_cache()
    gc.collect()


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str,
    subfolder: str = "text_encoder",
    device: str = "cuda:0",
):
    """Import the text encoder class from the model"""
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, device_map=device
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def load_text_encoders(
    pretrained_model_name_or_path, class_one, class_two, weight_dtype, device
):
    """Load both CLIP and T5 text encoders"""
    text_encoder_one = class_one.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        torch_dtype=weight_dtype,
        device_map=device,
    )
    text_encoder_two = class_two.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        torch_dtype=weight_dtype,
        device_map=device,
    )
    return text_encoder_one, text_encoder_two


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length=512,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
):
    """Encode prompt using T5 encoder"""
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError(
                "text_input_ids must be provided when the tokenizer is not specified"
            )

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    _, seq_len, _ = prompt_embeds.shape

    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    """Encode prompt using CLIP encoder"""
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError(
                "text_input_ids must be provided when the tokenizer is not specified"
            )

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    """Encode prompt using both CLIP and T5 encoders"""
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)
    dtype = text_encoders[0].dtype

    pooled_prompt_embeds = _encode_prompt_with_clip(
        text_encoder=text_encoders[0],
        tokenizer=tokenizers[0],
        prompt=prompt,
        device=device if device is not None else text_encoders[0].device,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[0] if text_input_ids_list else None,
    )

    prompt_embeds = _encode_prompt_with_t5(
        text_encoder=text_encoders[1],
        tokenizer=tokenizers[1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[1].device,
        text_input_ids=text_input_ids_list[1] if text_input_ids_list else None,
    )

    text_ids = torch.zeros(batch_size, prompt_embeds.shape[1], 3).to(
        device=device, dtype=dtype
    )
    text_ids = text_ids.repeat(num_images_per_prompt, 1, 1)

    return prompt_embeds, pooled_prompt_embeds, text_ids


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length):
    """Compute text embeddings for given prompt"""
    device = text_encoders[0].device
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length=max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
        text_ids = text_ids.to(device)
    return prompt_embeds, pooled_prompt_embeds, text_ids


def get_sigmas(
    timesteps, noise_scheduler_copy, n_dim=4, device="cuda:0", dtype=torch.bfloat16
):
    """Get sigma values for timesteps"""
    sigmas = noise_scheduler_copy.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler_copy.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def plot_history(history, save_path=None):
    """Plot training loss history"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
    ax1.plot(history["concept"])
    ax1.set_title("Concept Loss")

    # Moving average
    window_size = min(10, len(history["concept"]))
    if window_size > 1:
        window = np.ones(int(window_size)) / float(window_size)
        moving_avg = np.convolve(history["concept"], window, "same")
        ax2.plot(moving_avg)
        ax2.set_title("Moving Average Concept Loss")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Training history saved to {save_path}")
    plt.close()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train FLUX concept sliders")

    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        default="/workspace/models/FLUX.1-Krea-dev",
        help="Path to pretrained FLUX model (schnell or dev)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to use for training"
    )

    # Slider configuration
    parser.add_argument(
        "--target_prompt",
        type=str,
        required=True,
        help="Target/base prompt for the slider",
    )
    parser.add_argument(
        "--positive_prompt", type=str, required=True, help="Positive direction prompt"
    )
    parser.add_argument(
        "--negative_prompt", type=str, required=True, help="Negative direction prompt"
    )
    parser.add_argument(
        "--slider_name",
        type=str,
        required=True,
        help="Name for the slider (used in save path)",
    )

    # Training hyperparameters
    parser.add_argument(
        "--max_train_steps", type=int, default=1000, help="Number of training steps"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.002, help="Learning rate"
    )
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=1, help="LoRA alpha")
    parser.add_argument(
        "--eta",
        type=float,
        default=2.0,
        help="Training eta parameter for concept guidance",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")

    # Generation parameters
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--width", type=int, default=512, help="Image width")

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="fluxsliders",
        help="Output directory for trained sliders",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=0,
        help="Save checkpoint every N steps (0 = only save at end)",
    )

    # Advanced options
    parser.add_argument(
        "--train_method", type=str, default="xattn", help="LoRA training method"
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help="Timestep weighting scheme",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
        help="Learning rate scheduler",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=200,
        help="Number of warmup steps for lr scheduler",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup
    device = args.device
    weight_dtype = torch.bfloat16

    logger.info("=" * 50)
    logger.info("FLUX Slider Training")
    logger.info("=" * 50)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Slider: {args.slider_name}")
    logger.info(f"Target: {args.target_prompt}")
    logger.info(f"Positive: {args.positive_prompt}")
    logger.info(f"Negative: {args.negative_prompt}")
    logger.info(
        f"Steps: {args.max_train_steps}, LR: {args.learning_rate}, Rank: {args.rank}"
    )
    logger.info("=" * 50)

    # Determine if schnell or dev
    is_schnell = "schnell" in args.model_path.lower()
    num_inference_steps = 4 if is_schnell else 30
    guidance_scale = 0 if is_schnell else 3.5
    max_sequence_length = 256 if is_schnell else 512

    logger.info(f"Model type: {'Schnell' if is_schnell else 'Dev'}")
    logger.info(
        f"Inference steps: {num_inference_steps}, Guidance scale: {guidance_scale}"
    )

    # Setup output directory
    model_type = "schnell" if is_schnell else "dev"
    output_dir = Path(args.output_dir) / model_type
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Load tokenizers
    logger.info("Loading tokenizers...")
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.model_path, subfolder="tokenizer", torch_dtype=weight_dtype
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.model_path, subfolder="tokenizer_2", torch_dtype=weight_dtype
    )

    # Load scheduler
    logger.info("Loading scheduler...")
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.model_path, subfolder="scheduler", torch_dtype=weight_dtype
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    # Load text encoders
    logger.info("Loading text encoders...")
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.model_path, device=device
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.model_path, subfolder="text_encoder_2", device=device
    )
    text_encoder_one, text_encoder_two = load_text_encoders(
        args.model_path,
        text_encoder_cls_one,
        text_encoder_cls_two,
        weight_dtype,
        device,
    )

    # Load VAE and Transformer
    logger.info("Loading VAE and Transformer...")
    vae = AutoencoderKL.from_pretrained(
        args.model_path, subfolder="vae", torch_dtype=weight_dtype, device_map="auto"
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        args.model_path, subfolder="transformer", torch_dtype=weight_dtype
    )

    # Freeze models
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    # Move to device
    vae.to(device)
    transformer.to(device)
    text_encoder_one.to(device)
    text_encoder_two.to(device)

    logger.info("Models loaded successfully!")

    # Compute text embeddings
    logger.info("Computing text embeddings...")
    tokenizers = [tokenizer_one, tokenizer_two]
    text_encoders = [text_encoder_one, text_encoder_two]

    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = compute_text_embeddings(
            [args.target_prompt, args.positive_prompt, args.negative_prompt],
            text_encoders,
            tokenizers,
            max_sequence_length,
        )
        target_prompt_embeds, positive_prompt_embeds, negative_prompt_embeds = (
            prompt_embeds.chunk(3)
        )
        (
            target_pooled_prompt_embeds,
            positive_pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pooled_prompt_embeds.chunk(3)
        target_text_ids, positive_text_ids, negative_text_ids = text_ids.chunk(3)

    # Setup LoRA networks
    logger.info(f"Setting up LoRA network (rank={args.rank}, alpha={args.alpha})...")
    modules = DEFAULT_TARGET_REPLACE + UNET_TARGET_REPLACE_MODULE_CONV

    network = LoRANetwork(
        transformer,
        rank=args.rank,
        multiplier=1.0,
        alpha=args.alpha,
        train_method=args.train_method,
    ).to(device, dtype=weight_dtype)

    # Setup optimizer
    params = network.prepare_optimizer_params()
    optimizer = AdamW(params, lr=args.learning_rate)
    optimizer.zero_grad()

    # Setup learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Setup pipeline
    logger.info("Setting up pipeline...")
    pipe = FluxPipeline(
        noise_scheduler,
        vae,
        text_encoder_one,
        tokenizer_one,
        text_encoder_two,
        tokenizer_two,
        transformer,
    )
    pipe.set_progress_bar_config(disable=True)

    # Training loop
    logger.info("Starting training...")
    logger.info("=" * 50)

    progress_bar = tqdm(range(args.max_train_steps), desc="Training Steps")
    losses = {"concept": []}

    for step in range(args.max_train_steps):
        # Sample timesteps
        u = compute_density_for_timestep_sampling(
            weighting_scheme=args.weighting_scheme,
            batch_size=args.batch_size,
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.29,
        )
        indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
        timesteps = noise_scheduler_copy.timesteps[indices].to(device=device)

        # Get noisy latents
        timestep_to_infer = (
            (
                indices[0]
                * (
                    num_inference_steps
                    / noise_scheduler_copy.config.num_train_timesteps
                )
            )
            .long()
            .item()
        )
        with torch.no_grad():
            packed_noisy_model_input = pipe(
                args.target_prompt,
                height=args.height,
                width=args.width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                max_sequence_length=max_sequence_length,
                num_images_per_prompt=args.batch_size,
                generator=None,
                from_timestep=0,
                till_timestep=timestep_to_infer,
                output_type="latent",
            )

            vae_scale_factor = 2 ** (len(vae.config.block_out_channels))

            if step == 0:
                model_input = FluxPipeline._unpack_latents(
                    packed_noisy_model_input,
                    height=args.height,
                    width=args.width,
                    vae_scale_factor=vae_scale_factor,
                )

        # Prepare inputs
        latent_image_ids = FluxPipeline._prepare_latent_image_ids(
            model_input.shape[0],
            model_input.shape[2],
            model_input.shape[3],
            device,
            weight_dtype,
        )

        sigmas = get_sigmas(
            timesteps,
            noise_scheduler_copy,
            n_dim=model_input.ndim,
            device=device,
            dtype=model_input.dtype,
        )

        # Handle guidance
        if transformer.config.guidance_embeds:
            guidance = torch.tensor([guidance_scale], device=device)
            guidance = guidance.expand(model_input.shape[0])
        else:
            guidance = None

        # Forward pass with LoRA
        with network:
            model_pred = transformer(
                hidden_states=packed_noisy_model_input,
                timestep=timesteps / 1000,
                guidance=guidance,
                pooled_projections=target_pooled_prompt_embeds,
                encoder_hidden_states=target_prompt_embeds,
                txt_ids=target_text_ids,
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]

        model_pred = FluxPipeline._unpack_latents(
            model_pred,
            height=int(model_input.shape[2] * vae_scale_factor / 2),
            width=int(model_input.shape[3] * vae_scale_factor / 2),
            vae_scale_factor=vae_scale_factor,
        )

        # Compute target predictions
        with torch.no_grad():
            target_pred = transformer(
                hidden_states=packed_noisy_model_input,
                timestep=timesteps / 1000,
                guidance=guidance,
                pooled_projections=target_pooled_prompt_embeds,
                encoder_hidden_states=target_prompt_embeds,
                txt_ids=target_text_ids,
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]
            target_pred = FluxPipeline._unpack_latents(
                target_pred,
                height=int(model_input.shape[2] * vae_scale_factor / 2),
                width=int(model_input.shape[3] * vae_scale_factor / 2),
                vae_scale_factor=vae_scale_factor,
            )

            positive_pred = transformer(
                hidden_states=packed_noisy_model_input,
                timestep=timesteps / 1000,
                guidance=guidance,
                pooled_projections=positive_pooled_prompt_embeds,
                encoder_hidden_states=positive_prompt_embeds,
                txt_ids=positive_text_ids,
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]
            positive_pred = FluxPipeline._unpack_latents(
                positive_pred,
                height=int(model_input.shape[2] * vae_scale_factor / 2),
                width=int(model_input.shape[3] * vae_scale_factor / 2),
                vae_scale_factor=vae_scale_factor,
            )

            negative_pred = transformer(
                hidden_states=packed_noisy_model_input,
                timestep=timesteps / 1000,
                guidance=guidance,
                pooled_projections=negative_pooled_prompt_embeds,
                encoder_hidden_states=negative_prompt_embeds,
                txt_ids=negative_text_ids,
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]
            negative_pred = FluxPipeline._unpack_latents(
                negative_pred,
                height=int(model_input.shape[2] * vae_scale_factor / 2),
                width=int(model_input.shape[3] * vae_scale_factor / 2),
                vae_scale_factor=vae_scale_factor,
            )

            # Compute ground truth prediction
            gt_pred = target_pred + args.eta * (positive_pred - negative_pred)
            gt_pred = (gt_pred / gt_pred.norm()) * positive_pred.norm()

        # Compute loss
        concept_loss = torch.mean(
            ((model_pred.float() - gt_pred.float()) ** 2).reshape(gt_pred.shape[0], -1),
            1,
        )
        concept_loss = concept_loss.mean()

        # Backward pass
        concept_loss.backward()
        losses["concept"].append(concept_loss.item())

        # Optimizer step
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        # Update progress
        logs = {"loss": losses["concept"][-1], "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.update(1)
        progress_bar.set_postfix(**logs)

        # Save checkpoint periodically
        if args.save_every > 0 and (step + 1) % args.save_every == 0:
            checkpoint_name = f"flux-{args.slider_name}"
            checkpoint_path = output_dir / checkpoint_name
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving checkpoint at step {step + 1}...")
            network.save_weights(
                str(checkpoint_path / f"slider_step_{step+1}.pt"),
                dtype=weight_dtype,
            )
            logger.info(f"Checkpoint saved to {checkpoint_path / f'slider_step_{step+1}.pt'}")

    logger.info("Training completed!")

    # Save the trained model
    save_name = f"flux-{args.slider_name}"
    save_path = output_dir / save_name
    save_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving model to {save_path}...")
    network.save_weights(
        str(save_path / "slider_0.pt"),
        dtype=weight_dtype,
    )

    # Save training history plot
    plot_history(losses, save_path / "training_loss.png")

    # Save training config
    config_path = save_path / "training_config.txt"
    with open(config_path, "w") as f:
        f.write("FLUX Slider Training Configuration\n")
        f.write("=" * 50 + "\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

    logger.info(f"Model saved to: {save_path}")
    logger.info(f"Training config saved to: {config_path}")
    logger.info("=" * 50)
    logger.info("Training complete!")

    flush()


if __name__ == "__main__":
    main()
