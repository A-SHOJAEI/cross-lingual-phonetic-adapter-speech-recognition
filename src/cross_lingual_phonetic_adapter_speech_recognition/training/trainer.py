"""Training loop with curriculum learning and early stopping."""

import logging
from pathlib import Path
from typing import Dict, Optional, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from cross_lingual_phonetic_adapter_speech_recognition.models.model import PhoneticAdapterASR
from cross_lingual_phonetic_adapter_speech_recognition.models.components import (
    PhoneticContrastiveLoss,
    CurriculumScheduler,
    AdapterRegularization,
)


class ASRTrainer:
    """Trainer for phonetic adapter ASR model."""

    def __init__(
        self,
        model: PhoneticAdapterASR,
        config: Dict[str, Any],
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
    ) -> None:
        """Initialize trainer.

        Args:
            model: ASR model to train
            config: Training configuration dictionary
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device for training
        """
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Training config
        train_config = config.get('training', {})
        self.num_epochs = train_config.get('num_epochs', 50)
        self.learning_rate = train_config.get('learning_rate', 0.0001)
        self.weight_decay = train_config.get('weight_decay', 0.01)
        self.max_grad_norm = train_config.get('max_grad_norm', 1.0)
        self.gradient_accumulation_steps = train_config.get('gradient_accumulation_steps', 1)
        self.mixed_precision = train_config.get('mixed_precision', True)

        # Loss weights
        loss_config = config.get('loss_weights', {})
        self.asr_weight = loss_config.get('asr_loss', 1.0)
        self.contrastive_weight = loss_config.get('phonetic_contrastive_loss', 0.5)
        self.regularization_weight = loss_config.get('adapter_regularization', 0.01)

        # Setup optimizer
        self.optimizer = self._setup_optimizer()

        # Setup scheduler
        self.scheduler = self._setup_scheduler()

        # Setup losses
        self.asr_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.contrastive_criterion = PhoneticContrastiveLoss(
            temperature=config.get('model', {}).get('phonetic_contrastive_temp', 0.07)
        )
        self.regularization_criterion = AdapterRegularization()

        # Setup curriculum learning
        curriculum_config = config.get('curriculum', {})
        if curriculum_config.get('enabled', False):
            self.curriculum = CurriculumScheduler(
                stages=curriculum_config.get('stages', []),
                total_epochs=self.num_epochs,
            )
        else:
            self.curriculum = None

        # Setup mixed precision
        self.scaler = GradScaler() if self.mixed_precision else None

        # Early stopping
        early_stop_config = config.get('early_stopping', {})
        self.early_stopping_enabled = early_stop_config.get('enabled', True)
        self.patience = early_stop_config.get('patience', 10)
        self.min_delta = early_stop_config.get('min_delta', 0.001)
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Checkpointing
        checkpoint_config = config.get('checkpoint', {})
        self.save_dir = Path(checkpoint_config.get('save_dir', './models'))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        logging_config = config.get('logging', {})
        self.log_interval = logging_config.get('log_interval', 100)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.train_losses = []
        self.val_losses = []

        logging.info("Initialized ASRTrainer")

    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer.

        Returns:
            Configured optimizer
        """
        optimizer_name = self.config.get('training', {}).get('optimizer', 'adamw')

        if optimizer_name.lower() == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif optimizer_name.lower() == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9,
            )

        return optimizer

    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler.

        Returns:
            Configured scheduler or None
        """
        scheduler_name = self.config.get('training', {}).get('scheduler', 'cosine')
        scheduler_params = self.config.get('training', {}).get('scheduler_params', {})

        if scheduler_name.lower() == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_params.get('T_max', self.num_epochs),
                eta_min=scheduler_params.get('eta_min', 0.000001),
            )
        elif scheduler_name.lower() == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_params.get('step_size', 10),
                gamma=scheduler_params.get('gamma', 0.1),
            )
        elif scheduler_name.lower() == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_params.get('factor', 0.5),
                patience=scheduler_params.get('patience', 5),
            )
        else:
            scheduler = None

        return scheduler

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Update curriculum
        if self.curriculum is not None:
            self.curriculum.step(epoch)

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.num_epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            # Apply curriculum filtering
            if self.curriculum is not None:
                batch = self.curriculum.filter_batch_by_complexity(batch, epoch)

            # Move to device
            audio = batch['audio'].to(self.device)
            phonetic_features = batch['phonetic_features'].to(self.device)
            languages = batch['language']

            # Forward pass with mixed precision
            if self.mixed_precision and self.scaler is not None:
                with autocast():
                    outputs = self.model(audio, phonetic_features, languages)
                    loss = self._compute_loss(outputs, batch, phonetic_features)
                    loss = loss / self.gradient_accumulation_steps

                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(audio, phonetic_features, languages)
                loss = self._compute_loss(outputs, batch, phonetic_features)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.mixed_precision and self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.global_step += 1

            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item() * self.gradient_accumulation_steps})

            # Logging
            if batch_idx % self.log_interval == 0:
                logging.info(
                    f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {loss.item() * self.gradient_accumulation_steps:.4f}"
                )

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def _compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict,
        phonetic_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute total loss.

        Args:
            outputs: Model outputs
            batch: Input batch
            phonetic_features: Phonetic features

        Returns:
            Total loss
        """
        total_loss = 0.0

        # ASR loss using CTC loss with actual text targets
        logits = outputs['logits']
        batch_size, seq_len = logits.shape[0], logits.shape[1]

        if logits.dim() == 3:
            # Build character-level targets from batch text
            texts = batch.get('text', [])
            if texts:
                # Create a simple character vocabulary for CTC targets
                # blank=0, space=1, a-z=2-27
                char_to_id = {' ': 1}
                for i, c in enumerate('abcdefghijklmnopqrstuvwxyz'):
                    char_to_id[c] = i + 2

                # Encode text targets
                encoded_targets = []
                target_lengths = []
                for text in texts:
                    text_lower = text.lower().strip()
                    encoded = []
                    for c in text_lower:
                        if c in char_to_id:
                            encoded.append(char_to_id[c])
                        # Skip characters not in vocab
                    encoded_targets.extend(encoded)
                    target_lengths.append(len(encoded))

                if encoded_targets:
                    targets = torch.tensor(encoded_targets, dtype=torch.long, device=logits.device)
                    target_lengths = torch.tensor(target_lengths, dtype=torch.long, device=logits.device)

                    # CTC loss requires (seq_len, batch, vocab_size) log-probabilities
                    log_probs = nn.functional.log_softmax(logits, dim=-1).permute(1, 0, 2)
                    input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=logits.device)

                    # Clamp target lengths to not exceed input lengths
                    target_lengths = torch.clamp(target_lengths, max=seq_len)

                    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
                    asr_loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
                else:
                    asr_loss = torch.tensor(0.0, device=logits.device)
            else:
                asr_loss = torch.tensor(0.0, device=logits.device)
        else:
            asr_loss = torch.tensor(0.0, device=logits.device)
        total_loss += self.asr_weight * asr_loss

        # Phonetic contrastive loss
        if self.contrastive_weight > 0:
            phonetic_embedding = outputs['phonetic_embedding']
            language_ids = torch.tensor(
                [self.model.language_to_id.get(lang, 0) for lang in batch['language']],
                device=phonetic_embedding.device
            )
            contrastive_loss = self.contrastive_criterion(
                phonetic_embedding,
                phonetic_features,
                language_ids,
            )
            total_loss += self.contrastive_weight * contrastive_loss

        # Adapter regularization
        if self.regularization_weight > 0 and len(self.model.adapters) > 0:
            adapter_params = self.model.get_adapter_parameters()
            reg_loss = self.regularization_criterion(adapter_params)
            total_loss += self.regularization_weight * reg_loss

        return total_loss

    @torch.no_grad()
    def validate(self) -> float:
        """Run validation.

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validation"):
            audio = batch['audio'].to(self.device)
            phonetic_features = batch['phonetic_features'].to(self.device)
            languages = batch['language']

            outputs = self.model(audio, phonetic_features, languages)
            loss = self._compute_loss(outputs, batch, phonetic_features)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def train(self) -> Dict[str, list]:
        """Run full training loop.

        Returns:
            Dictionary with training history
        """
        logging.info("Starting training...")

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)

            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)

            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']
            logging.info(
                f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, lr={current_lr:.6f}"
            )

            # Save best model
            if val_loss < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(is_best=True)
                logging.info(f"New best validation loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1

            # Early stopping
            if self.early_stopping_enabled and self.patience_counter >= self.patience:
                logging.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

        logging.info("Training completed!")
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }

    def save_checkpoint(self, is_best: bool = False) -> None:
        """Save model checkpoint.

        Args:
            is_best: Whether this is the best model so far
        """
        checkpoint_path = self.save_dir / "best_model.pt" if is_best else self.save_dir / "last_model.pt"
        self.model.save_pretrained(str(checkpoint_path))
        logging.info(f"Saved checkpoint to {checkpoint_path}")
