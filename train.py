"""
Training script for Concert2Studio
Implements training loop with Accelerate, checkpointing, and logging
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import os
import time
import math
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Accelerate for distributed training
from accelerate import Accelerator
from accelerate.utils import set_seed

# Local imports
from model import Concert2StudioModel, count_parameters
from dataset import create_dataloaders, verify_dataset


class EarlyStopping:
    """Early stopping to prevent overfitting"""

    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == "min":
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == 'max'
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop


class Trainer:
    """Main trainer class"""

    def __init__(self, config: dict, accelerator: Accelerator):
        self.config = config
        self.accelerator = accelerator

        # Set random seed
        set_seed(config["data"]["random_seed"])

        # Create directories
        self.create_directories()

        # Initialize model
        self.model = Concert2StudioModel(config)

        # Print model info
        if self.accelerator.is_main_process:
            param_count = count_parameters(self.model)
            print(f"🔧 Model parameters: {param_count:,}")
            if param_count > 12_000_000:
                print("⚠️  Model has >12M parameters, may exceed target")

        # Initialize optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=float(config["training"]["learning_rate"]),
            betas=(
                float(config["training"]["beta1"]),
                float(config["training"]["beta2"]),
            ),
            weight_decay=float(config["training"]["weight_decay"]),
        )

        # Create learning rate scheduler
        self.create_scheduler()

        # Initialize early stopping
        self.early_stopping = EarlyStopping(
            patience=int(config["training"]["early_stopping_patience"]), mode="min"
        )

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.last_checkpoint_time = time.time()

        # Compile model if requested
        if config["hardware"]["compile_model"]:
            try:
                self.model = torch.compile(self.model)
                if self.accelerator.is_main_process:
                    print("✅ Model compiled with torch.compile")
            except Exception as e:
                if self.accelerator.is_main_process:
                    print(f"⚠️  torch.compile failed: {e}")

        # Note: channels_last memory format is not applicable to audio models
        # Audio models work with 2D/3D tensors, not 4D image tensors

    def create_directories(self):
        """Create necessary directories"""
        if self.accelerator.is_main_process:
            for path_key in ["output_dir", "checkpoint_dir", "log_dir"]:
                path = Path(self.config["paths"][path_key])
                path.mkdir(parents=True, exist_ok=True)

    def create_scheduler(self):
        """Create learning rate scheduler with warmup"""
        warmup_steps = int(self.config["training"]["warmup_steps"])
        total_steps = int(self.config["training"]["num_epochs"]) * 1000  # Estimate
        scheduler_type = self.config["training"].get("scheduler", "cosine")

        # Linear warmup
        warmup_scheduler = LinearLR(
            self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
        )

        # Main scheduler based on config
        if scheduler_type == "linear":
            main_scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=total_steps - warmup_steps,
            )
        else:  # cosine
            main_scheduler = CosineAnnealingLR(
                self.optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6
            )

        # Sequential scheduler
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint"""
        if not os.path.exists(checkpoint_path):
            print(f"⚠️  Checkpoint {checkpoint_path} not found")
            return

        print(f"📂 Loading checkpoint from {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Load training state
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]

        print(f"✅ Resumed from epoch {self.current_epoch}, step {self.global_step}")

    def save_checkpoint(self, is_best: bool = False):
        """Save checkpoint"""
        if not self.accelerator.is_main_process:
            return

        checkpoint_dir = Path(self.config["paths"]["checkpoint_dir"])

        # Prepare checkpoint data
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.accelerator.unwrap_model(self.model).state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }

        # Save latest checkpoint
        latest_path = checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)

        # Save best checkpoint
        if is_best:
            best_path = checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
            print(f"💾 Saved best checkpoint (val_loss: {self.best_val_loss:.6f})")

        # Save epoch checkpoint
        epoch_path = checkpoint_dir / f"epoch_{self.current_epoch:03d}.pt"
        torch.save(checkpoint, epoch_path)

        # Clean up old checkpoints
        self.cleanup_checkpoints()

    def cleanup_checkpoints(self):
        """Remove old checkpoints to save space"""
        checkpoint_dir = Path(self.config["paths"]["checkpoint_dir"])
        max_checkpoints = self.config["checkpoint"]["max_checkpoints"]

        # Get all epoch checkpoints
        epoch_checkpoints = sorted(
            checkpoint_dir.glob("epoch_*.pt"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )

        # Remove old checkpoints
        for checkpoint in epoch_checkpoints[max_checkpoints:]:
            checkpoint.unlink()

    def should_save_checkpoint(self):
        """Check if we should save a checkpoint"""
        time_threshold = self.config["checkpoint"]["save_interval_minutes"] * 60
        step_threshold = self.config["checkpoint"]["save_interval_steps"]

        time_elapsed = time.time() - self.last_checkpoint_time

        return time_elapsed >= time_threshold or self.global_step % step_threshold == 0

    def train_step(self, batch):
        """Single training step"""
        concert_audio, studio_audio = batch

        # Note: channels_last is not applicable to audio tensors (2D/3D)
        # Audio tensors don't benefit from channels_last memory format

        # Forward pass
        with self.accelerator.autocast():
            enhanced_audio, losses = self.model(concert_audio, studio_audio)
            total_loss = losses["total"]

        # Backward pass
        self.accelerator.backward(total_loss)

        # Gradient clipping
        if self.accelerator.sync_gradients:
            max_norm = float(self.config["training"]["max_grad_norm"])
            self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)

        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        return losses

    def validate(self, val_loader):
        """Validation loop"""
        self.model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in tqdm(
                val_loader,
                desc="Validating",
                disable=not self.accelerator.is_main_process,
            ):
                concert_audio, studio_audio = batch

                # Note: channels_last is not applicable to audio tensors (2D/3D)
                # Audio tensors don't benefit from channels_last memory format

                with self.accelerator.autocast():
                    enhanced_audio, losses = self.model(concert_audio, studio_audio)

                val_losses.append(losses["total"].item())

        avg_val_loss = sum(val_losses) / len(val_losses)
        return avg_val_loss

    def train(self, train_loader, val_loader):
        """Main training loop"""

        # Prepare everything with accelerator
        self.model, self.optimizer, train_loader, val_loader, self.scheduler = (
            self.accelerator.prepare(
                self.model, self.optimizer, train_loader, val_loader, self.scheduler
            )
        )

        # Resume from checkpoint if specified
        if self.config["checkpoint"]["resume_from"]:
            self.load_checkpoint(self.config["checkpoint"]["resume_from"])

        if self.accelerator.is_main_process:
            print(f"🚀 Starting training from epoch {self.current_epoch}")
            print(
                f"📊 Train batches: {len(train_loader)}, Val batches: {len(val_loader)}"
            )

        # Training loop
        for epoch in range(self.current_epoch, self.config["training"]["num_epochs"]):
            self.current_epoch = epoch

            # Handle vocoder freezing/unfreezing
            freeze_epochs = self.config["model"]["vocoder"].get("freeze_epochs", 0)
            if freeze_epochs > 0:
                if epoch < freeze_epochs:
                    if self.accelerator.is_main_process and epoch == 0:
                        print(f"🔒 Freezing vocoder for first {freeze_epochs} epochs")
                    # Access the actual model through accelerator wrapper
                    actual_model = self.accelerator.unwrap_model(self.model)
                    actual_model.vocoder.freeze_parameters()
                elif epoch == freeze_epochs:
                    if self.accelerator.is_main_process:
                        print(f"🔓 Unfreezing vocoder at epoch {epoch+1}")
                    # Access the actual model through accelerator wrapper
                    actual_model = self.accelerator.unwrap_model(self.model)
                    actual_model.vocoder.unfreeze_parameters()

            # Training phase
            self.model.train()
            epoch_losses = []

            progress_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{self.config['training']['num_epochs']}",
                disable=not self.accelerator.is_main_process,
            )

            for batch in progress_bar:
                losses = self.train_step(batch)
                epoch_losses.append(losses["total"].item())
                self.global_step += 1

                # Update progress bar
                if self.accelerator.is_main_process and self.global_step % 10 == 0:
                    progress_bar.set_postfix(
                        {
                            "loss": f"{losses['total'].item():.6f}",
                            "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                        }
                    )

                # Save checkpoint if needed
                if self.should_save_checkpoint():
                    self.save_checkpoint()
                    self.last_checkpoint_time = time.time()

            # Validation phase
            if self.accelerator.is_main_process:
                avg_train_loss = sum(epoch_losses) / len(epoch_losses)
                avg_val_loss = self.validate(val_loader)

                print(
                    f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
                )

                # Check for best model
                is_best = avg_val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = avg_val_loss

                # Save checkpoint
                self.save_checkpoint(is_best=is_best)

                # Early stopping check
                if self.early_stopping(avg_val_loss):
                    print(f"🛑 Early stopping triggered at epoch {epoch+1}")
                    break

            # Synchronize processes
            self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            print("🎉 Training completed!")
            print(f"📊 Best validation loss: {self.best_val_loss:.6f}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train Concert2Studio model")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to configuration file"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None, help="Override data directory"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Override output directory"
    )

    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Override config with command line arguments
    if args.resume:
        config["checkpoint"]["resume_from"] = args.resume
    if args.data_dir:
        config["data"]["data_dir"] = args.data_dir
    if args.output_dir:
        config["paths"]["output_dir"] = args.output_dir

    # Initialize accelerator
    # Determine mixed precision based on available hardware
    mixed_precision = "no"  # Default to no mixed precision

    if torch.cuda.is_available():
        print(f"🚀 CUDA detected: {torch.cuda.get_device_name()}")
        print(f"🚀 CUDA version: {torch.version.cuda}")
        # Check for bfloat16 support on modern GPUs
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            mixed_precision = "bf16"
            print("✅ Using bfloat16 mixed precision")
        else:
            mixed_precision = "fp16"
            print("✅ Using float16 mixed precision")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("🍎 Apple Metal (MPS) detected")
        mixed_precision = "no"  # MPS doesn't support mixed precision yet
    else:
        print("💻 Using CPU")

    accelerator = Accelerator(
        gradient_accumulation_steps=int(
            config["training"]["gradient_accumulation_steps"]
        ),
        mixed_precision=mixed_precision,
        log_with="tensorboard",
        project_dir=config["paths"]["log_dir"],
    )

    # Verify dataset
    if accelerator.is_main_process:
        print("🔍 Verifying dataset...")
        verify_dataset(config["data"]["data_dir"])

    # Create dataloaders
    try:
        train_loader, val_loader = create_dataloaders(config)
        if accelerator.is_main_process:
            print(
                f"📚 Created dataloaders: {len(train_loader)} train, {len(val_loader)} val batches"
            )
    except Exception as e:
        if accelerator.is_main_process:
            print(f"❌ Error creating dataloaders: {e}")
            print(
                "Make sure you have audio files in the correct format in the data directory"
            )
        return

    # Initialize trainer
    trainer = Trainer(config, accelerator)

    # Start training
    try:
        trainer.train(train_loader, val_loader)
    except KeyboardInterrupt:
        if accelerator.is_main_process:
            print("\n⏹️  Training interrupted by user")
            trainer.save_checkpoint()
    except Exception as e:
        if accelerator.is_main_process:
            print(f"❌ Training failed: {e}")
            import traceback

            traceback.print_exc()
    finally:
        # Cleanup
        accelerator.end_training()


if __name__ == "__main__":
    main()
