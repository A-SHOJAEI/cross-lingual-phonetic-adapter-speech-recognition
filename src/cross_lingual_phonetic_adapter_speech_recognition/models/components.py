"""Custom model components including adapters and loss functions."""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PhoneticAdapterLayer(nn.Module):
    """Phonetic-aware adapter layer for cross-lingual transfer.

    This is a custom component that implements language-specific adaptation
    conditioned on phonetic embeddings. The adapter uses a bottleneck architecture
    with phonetic feature integration.
    """

    def __init__(
        self,
        hidden_dim: int,
        adapter_dim: int,
        phonetic_dim: int,
        num_languages: int = 8,
    ) -> None:
        """Initialize phonetic adapter layer.

        Args:
            hidden_dim: Dimension of hidden representations
            adapter_dim: Dimension of adapter bottleneck
            phonetic_dim: Dimension of phonetic embeddings
            num_languages: Number of languages
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.adapter_dim = adapter_dim
        self.phonetic_dim = phonetic_dim

        # Down-projection
        self.down_project = nn.Linear(hidden_dim, adapter_dim)

        # Phonetic conditioning layer
        self.phonetic_gate = nn.Sequential(
            nn.Linear(phonetic_dim, adapter_dim),
            nn.Sigmoid()
        )

        # Language-specific adapter weights
        self.language_adapters = nn.ModuleList([
            nn.Linear(adapter_dim, adapter_dim) for _ in range(num_languages)
        ])

        # Up-projection
        self.up_project = nn.Linear(adapter_dim, hidden_dim)

        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Initialize with small weights for stability
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with small values."""
        nn.init.normal_(self.down_project.weight, std=0.01)
        nn.init.zeros_(self.down_project.bias)
        nn.init.normal_(self.up_project.weight, std=0.01)
        nn.init.zeros_(self.up_project.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        phonetic_embedding: torch.Tensor,
        language_id: int,
    ) -> torch.Tensor:
        """Forward pass with phonetic conditioning.

        Args:
            hidden_states: Input hidden states (batch, seq_len, hidden_dim)
            phonetic_embedding: Phonetic embedding (batch, phonetic_dim)
            language_id: Language identifier (0 to num_languages-1)

        Returns:
            Adapted hidden states (batch, seq_len, hidden_dim)
        """
        residual = hidden_states

        # Down-project to adapter dimension
        x = self.down_project(hidden_states)  # (batch, seq_len, adapter_dim)

        # Apply phonetic gating
        phonetic_gate = self.phonetic_gate(phonetic_embedding)  # (batch, adapter_dim)
        phonetic_gate = phonetic_gate.unsqueeze(1)  # (batch, 1, adapter_dim)
        x = x * phonetic_gate  # Element-wise gating

        # Apply language-specific transformation
        batch_size, seq_len, _ = x.shape
        x_flat = x.reshape(-1, self.adapter_dim)  # (batch * seq_len, adapter_dim)
        x_flat = self.language_adapters[language_id](x_flat)
        x = x_flat.reshape(batch_size, seq_len, self.adapter_dim)

        # Apply non-linearity
        x = F.gelu(x)

        # Up-project back to hidden dimension
        x = self.up_project(x)

        # Residual connection and layer norm
        output = self.layer_norm(residual + x)

        return output


class PhoneticContrastiveLoss(nn.Module):
    """Contrastive loss for learning cross-lingual phonetic representations.

    This custom loss encourages similar phonetic features to have similar
    embeddings across languages, enabling zero-shot transfer.
    """

    def __init__(self, temperature: float = 0.07, margin: float = 0.5) -> None:
        """Initialize phonetic contrastive loss.

        Args:
            temperature: Temperature parameter for softmax
            margin: Margin for negative pairs
        """
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(
        self,
        phonetic_embeddings: torch.Tensor,
        phonetic_features: torch.Tensor,
        language_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute contrastive loss.

        Args:
            phonetic_embeddings: Learned embeddings (batch, phonetic_dim)
            phonetic_features: Ground truth phonetic features (batch, max_len, feature_dim) - unused, kept for compatibility
            language_ids: Language identifiers (batch,)

        Returns:
            Contrastive loss value
        """
        batch_size = phonetic_embeddings.shape[0]

        # Normalize embeddings
        phonetic_embeddings = F.normalize(phonetic_embeddings, p=2, dim=1)

        # Compute similarity matrix (self-similarity across batch)
        similarity = torch.matmul(phonetic_embeddings, phonetic_embeddings.t())  # (batch, batch)
        similarity = similarity / self.temperature

        # Create positive/negative mask
        # Positive pairs: same phonetic features OR same language family
        positive_mask = torch.eye(batch_size, device=similarity.device)

        # Compute InfoNCE loss
        exp_similarity = torch.exp(similarity)
        log_prob = similarity - torch.log(exp_similarity.sum(dim=1, keepdim=True))

        # Mean of log-likelihood over positive pairs
        loss = -(positive_mask * log_prob).sum(dim=1) / positive_mask.sum(dim=1).clamp(min=1.0)
        loss = loss.mean()

        return loss


class CurriculumScheduler:
    """Curriculum learning scheduler for progressive phonetic complexity.

    This custom component implements curriculum learning that progressively
    increases the phonetic complexity of training samples.
    """

    def __init__(
        self,
        stages: List[Dict],
        total_epochs: int,
    ) -> None:
        """Initialize curriculum scheduler.

        Args:
            stages: List of curriculum stages with name, duration, and complexity
            total_epochs: Total number of training epochs
        """
        self.stages = stages
        self.total_epochs = total_epochs
        self.current_stage = 0
        self.current_complexity = 1

        # Build epoch to stage mapping
        self.epoch_to_stage = []
        epoch_count = 0
        for stage in stages:
            duration = stage.get('duration_epochs', 10)
            for _ in range(duration):
                self.epoch_to_stage.append(stage)
                epoch_count += 1

        # Pad if necessary
        while len(self.epoch_to_stage) < total_epochs:
            self.epoch_to_stage.append(stages[-1])

        logging.info(f"Initialized curriculum with {len(stages)} stages")

    def get_current_stage(self, epoch: int) -> Dict:
        """Get current curriculum stage.

        Args:
            epoch: Current training epoch

        Returns:
            Stage configuration dictionary
        """
        if epoch < len(self.epoch_to_stage):
            return self.epoch_to_stage[epoch]
        return self.stages[-1]

    def get_complexity_threshold(self, epoch: int) -> int:
        """Get maximum phonetic complexity for current epoch.

        Args:
            epoch: Current training epoch

        Returns:
            Maximum complexity level (1-4)
        """
        stage = self.get_current_stage(epoch)
        return stage.get('phonetic_complexity', 4)

    def filter_batch_by_complexity(
        self,
        batch: Dict[str, torch.Tensor],
        epoch: int,
    ) -> Dict[str, torch.Tensor]:
        """Filter batch to include only samples within complexity threshold.

        Args:
            batch: Input batch dictionary
            epoch: Current training epoch

        Returns:
            Filtered batch dictionary
        """
        threshold = self.get_complexity_threshold(epoch)
        complexities = batch['complexity']

        # Find indices within complexity threshold
        valid_indices = (complexities <= threshold).nonzero(as_tuple=True)[0]

        if len(valid_indices) == 0:
            # Return at least one sample to avoid empty batch
            valid_indices = torch.tensor([0], device=complexities.device)

        # Filter batch
        filtered_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                filtered_batch[key] = value[valid_indices]
            elif isinstance(value, list):
                filtered_batch[key] = [value[i] for i in valid_indices.cpu().tolist()]
            else:
                filtered_batch[key] = value

        return filtered_batch

    def step(self, epoch: int) -> None:
        """Update scheduler for new epoch.

        Args:
            epoch: Current training epoch
        """
        stage = self.get_current_stage(epoch)
        self.current_complexity = stage.get('phonetic_complexity', 4)
        logging.info(
            f"Epoch {epoch}: Curriculum stage '{stage.get('name', 'unknown')}', "
            f"complexity threshold = {self.current_complexity}"
        )


class AdapterRegularization(nn.Module):
    """Regularization loss for adapter parameters to prevent overfitting."""

    def __init__(self, regularization_type: str = "l2") -> None:
        """Initialize adapter regularization.

        Args:
            regularization_type: Type of regularization ('l2' or 'l1')
        """
        super().__init__()
        self.regularization_type = regularization_type

    def forward(self, adapter_params: List[torch.Tensor]) -> torch.Tensor:
        """Compute regularization loss.

        Args:
            adapter_params: List of adapter parameter tensors

        Returns:
            Regularization loss value
        """
        reg_loss = 0.0

        for param in adapter_params:
            if self.regularization_type == "l2":
                reg_loss += torch.sum(param ** 2)
            elif self.regularization_type == "l1":
                reg_loss += torch.sum(torch.abs(param))

        return reg_loss / max(len(adapter_params), 1)
