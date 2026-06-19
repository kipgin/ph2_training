import os
import argparse
import yaml
import torch
import sys
import importlib.util
from tqdm.auto import tqdm

# Mock torchao check to bypass version incompatibility in peft under older Colab environments
_original_find_spec = importlib.util.find_spec
def _mocked_find_spec(name, package=None):
    if name == "torchao" or name.startswith("torchao."):
        return None
    return _original_find_spec(name, package)
importlib.util.find_spec = _mocked_find_spec

from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler

from src.dataset import DreamBoothDataset, DreamBoothCollator
from src.model_utils import load_base_models, inject_lora, save_lora_weights
from src.pipeline import TrainingPipeline

def parse_args():
    parser = argparse.ArgumentParser(description="DreamBooth LoRA Fine-Tuning Script")
    parser.add_argument(
        "--config",
        type=str,
        default="training_config.yaml",
        help="Path to the training configuration YAML file."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Load configuration
    if not os.path.exists(args.config):
        # Fallback to duplicate file path if needed
        fallback_config = "trainging_config.yaml"
        if os.path.exists(fallback_config):
            args.config = fallback_config
        else:
            raise FileNotFoundError(f"Configuration file {args.config} not found.")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Initialize Accelerator with native tracker (WandB) support
    accelerator = Accelerator(
        log_with=config["logging"].get("logger_type", "wandb"),
        project_dir=config["logging"].get("output_dir", "./checkpoints"),
        gradient_accumulation_steps=config["training"].get("gradient_accumulation_steps", 1),
        mixed_precision=config["training"].get("mixed_precision", "no")
    )

    # Set random seed
    set_seed(config["training"].get("seed", 42))

    # Determine weight dtype from mixed precision config
    mixed_precision = config["training"].get("mixed_precision", "no")
    if mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    # 1. Load tokenizer and base models (CLIP, VAE, U-Net)
    model_id = config["model"]["pretrained_model_name_or_path"]
    tokenizer, text_encoder, vae, unet = load_base_models(model_id, weight_dtype)

    # 2. Inject trainable LoRA matrices into U-Net & Text Encoder
    unet, text_encoder = inject_lora(unet, text_encoder, config)

    # 3. Load standard training noise scheduler (DDPMScheduler)
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    # 4. Prepare dataset and dataloader
    with_prior_preservation = config["training"].get("with_prior_preservation", False)
    
    train_dataset = DreamBoothDataset(
        instance_data_root=config["data"]["instance_data_dir"],
        instance_prompt=config["data"]["instance_prompt"],
        tokenizer=tokenizer,
        class_data_root=config["data"].get("class_data_dir") if with_prior_preservation else None,
        class_prompt=config["data"].get("class_prompt") if with_prior_preservation else None,
        size=config["data"].get("resolution", 512),
        center_crop=config["data"].get("center_crop", True)
    )

    collator = DreamBoothCollator(
        tokenizer=tokenizer,
        with_prior_preservation=with_prior_preservation
    )

    # Read num_workers from config (default 2). Background workers prefetch the next
    # batch while the GPU processes the current one, eliminating the idle gap that
    # caused ~11% GPU utilization with num_workers=0.
    num_workers = config["training"].get("num_workers", 2)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["training"].get("batch_size", 2),
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,             # Async CPU→GPU transfer via DMA (non-blocking)
        persistent_workers=num_workers > 0,  # Keep workers alive across epochs
        prefetch_factor=2 if num_workers > 0 else None,  # Queue 2 batches per worker
    )

    # 5. Define parameters to optimize (only those with requires_grad=True)
    params_to_optimize = [p for p in unet.parameters() if p.requires_grad]
    if config["training"].get("train_text_encoder", False) and text_encoder is not None:
        params_to_optimize += [p for p in text_encoder.parameters() if p.requires_grad]

    # Use 8-bit Adam optimization if configured & bitsandbytes is available
    if config["training"].get("use_8bit_adam", False):
        try:
            import bitsandbytes as bnb
            optimizer_class = bnb.optim.AdamW8bit
        except ImportError:
            print("bitsandbytes not found, falling back to standard torch.optim.AdamW")
            optimizer_class = torch.optim.AdamW
    else:
        optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class(
        params_to_optimize,
        lr=float(config["training"]["learning_rate"]),
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8
    )

    # 6. Define learning rate scheduler
    lr_scheduler = get_scheduler(
        config["training"].get("lr_scheduler", "constant"),
        optimizer=optimizer,
        num_warmup_steps=config["training"].get("lr_warmup_steps", 0),
        num_training_steps=config["training"]["max_train_steps"] * config["training"].get("gradient_accumulation_steps", 1),
    )

    # 7. Configure gradient checkpointing for memory efficiency
    if config["training"].get("gradient_checkpointing", False):
        if hasattr(unet, "enable_gradient_checkpointing"):
            unet.enable_gradient_checkpointing()
        elif hasattr(unet, "base_model") and hasattr(unet.base_model, "enable_gradient_checkpointing"):
            unet.base_model.enable_gradient_checkpointing()
            
        if config["training"].get("train_text_encoder", False) and text_encoder is not None:
            if hasattr(text_encoder, "gradient_checkpointing_enable"):
                text_encoder.gradient_checkpointing_enable()
            elif hasattr(text_encoder, "base_model") and hasattr(text_encoder.base_model, "gradient_checkpointing_enable"):
                text_encoder.base_model.gradient_checkpointing_enable()

    # 8. Prepare for distributed or single-device acceleration
    if config["training"].get("train_text_encoder", False) and text_encoder is not None:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )
        if text_encoder is not None:
            text_encoder.to(accelerator.device, dtype=weight_dtype)

    # VAE is frozen and doesn't need wrapping via prepare, move to device
    vae.to(accelerator.device, dtype=weight_dtype)

    # 9. Initialize Training Pipeline Step Abstraction
    pipeline = TrainingPipeline(
        accelerator=accelerator,
        unet=unet,
        text_encoder=text_encoder,
        vae=vae,
        noise_scheduler=noise_scheduler,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        with_prior_preservation=with_prior_preservation,
        prior_loss_weight=config["training"].get("prior_loss_weight", 1.0)
    )

    # 10. Instantiate trackers (like WandB)
    accelerator.init_trackers(config["logging"]["project_name"], config=config)

    # Determine epoch settings
    num_update_steps_per_epoch = len(train_dataloader)
    max_train_steps = config["training"]["max_train_steps"]
    num_train_epochs = (max_train_steps + num_update_steps_per_epoch - 1) // num_update_steps_per_epoch

    print("***** Running Training *****")
    print(f"  Num examples        = {len(train_dataset)}")
    print(f"  Num Epochs          = {num_train_epochs}")
    print(f"  Batch size / device = {config['training'].get('batch_size', 2)}")
    print(f"  Total steps         = {max_train_steps}")
    print(f"  Prior preservation  = {with_prior_preservation}")
    print()

    # Progress bar tracks completed optimizer steps across all epochs
    progress_bar = tqdm(
        total=max_train_steps,
        desc="Training",
        unit="step",
        dynamic_ncols=True,
        colour="green",
    )

    global_step = 0
    
    for epoch in range(num_train_epochs):
        unet.train()
        if config["training"].get("train_text_encoder", False) and text_encoder is not None:
            text_encoder.train()

        # Reset epoch-wise loss accumulation
        log_loss = 0.0
        log_instance_loss = 0.0
        log_prior_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Run step and calculate loss
                loss, loss_instance, loss_prior = pipeline.training_step(batch)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    params_to_clip = (
                        list(unet.parameters()) + list(text_encoder.parameters())
                        if config["training"].get("train_text_encoder", False) and text_encoder is not None
                        else unet.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Accumulate losses for logging
            log_loss += loss.detach().item()
            log_instance_loss += loss_instance.detach().item()
            log_prior_loss += loss_prior.detach().item() if loss_prior is not None else 0.0

            # Perform actions once gradient sync happens (an actual optimization step is complete)
            if accelerator.sync_gradients:
                global_step += 1

                current_lr = lr_scheduler.get_last_lr()[0]
                avg_loss = log_loss / (step + 1)
                avg_instance = log_instance_loss / (step + 1)
                avg_prior = log_prior_loss / (step + 1)

                # Update tqdm bar with live metrics
                progress_bar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    inst=f"{avg_instance:.4f}",
                    prior=f"{avg_prior:.4f}" if with_prior_preservation else "N/A",
                    lr=f"{current_lr:.2e}",
                )
                progress_bar.update(1)

                # Log metrics to tracker (WandB)
                log_dict = {
                    "train/loss": avg_loss,
                    "train/instance_loss": avg_instance,
                    "train/lr": current_lr,
                    "train/epoch": epoch,
                }
                if with_prior_preservation:
                    log_dict["train/prior_loss"] = avg_prior
                accelerator.log(log_dict, step=global_step)
                
                if global_step % config["logging"].get("save_steps", 200) == 0:
                    tqdm.write(f"[Step {global_step}] Saving checkpoint...")
                    if accelerator.is_main_process:
                        unwrapped_unet = accelerator.unwrap_model(unet)
                        unwrapped_text_encoder = (
                            accelerator.unwrap_model(text_encoder)
                            if config["training"].get("train_text_encoder", False) and text_encoder is not None
                            else None
                        )
                        save_lora_weights(
                            config["logging"]["output_dir"],
                            unwrapped_unet,
                            unwrapped_text_encoder,
                            global_step,
                            is_final=False
                        )

            if global_step >= max_train_steps:
                break

    # Save final LoRA adapters
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_unet = accelerator.unwrap_model(unet)
        unwrapped_text_encoder = (
            accelerator.unwrap_model(text_encoder)
            if config["training"].get("train_text_encoder", False) and text_encoder is not None
            else None
        )
        save_lora_weights(
            config["logging"]["output_dir"],
            unwrapped_unet,
            unwrapped_text_encoder,
            global_step,
            is_final=True
        )

    progress_bar.close()
    accelerator.end_training()
    print(f"\nTraining finished successfully! ({global_step} steps completed)")

if __name__ == "__main__":
    main()
