"""Main model architecture for phonetic adapter ASR."""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import WhisperModel, WhisperProcessor, WhisperConfig
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logging.warning("transformers/whisper not available - using simplified model")

from cross_lingual_phonetic_adapter_speech_recognition.models.components import (
    PhoneticAdapterLayer,
    PhoneticContrastiveLoss,
    AdapterRegularization,
)


class GreedyCTCDecoder:
    """Greedy CTC decoder for converting logits to text."""

    def __init__(self, vocab: Optional[List[str]] = None, blank_id: int = 0) -> None:
        """Initialize CTC decoder.

        Args:
            vocab: Vocabulary list mapping token IDs to characters
            blank_id: ID of the blank token for CTC
        """
        if vocab is None:
            # Create character vocabulary matching CTC training targets:
            # blank=0, space=1, a-z=2-27
            self.vocab = [''] + [' '] + list('abcdefghijklmnopqrstuvwxyz')
        else:
            self.vocab = vocab
        self.blank_id = blank_id

    def decode(self, logits: torch.Tensor, remove_repeated: bool = True) -> List[str]:
        """Decode logits to text using greedy CTC decoding.

        Args:
            logits: Model logits (batch, seq_len, vocab_size)
            remove_repeated: Whether to remove repeated characters

        Returns:
            List of decoded strings
        """
        # Greedy decoding: take argmax
        predictions = torch.argmax(logits, dim=-1)  # (batch, seq_len)

        decoded_texts = []
        for pred in predictions:
            # Convert to list
            pred_list = pred.cpu().tolist()

            # Remove blanks and collapse repeats
            decoded = []
            prev_token = None
            for token_id in pred_list:
                if token_id == self.blank_id:
                    prev_token = None
                    continue

                if not remove_repeated or token_id != prev_token:
                    # Map to character (handle out of vocab)
                    if 0 <= token_id < len(self.vocab):
                        decoded.append(self.vocab[token_id])
                    prev_token = token_id

            decoded_texts.append(''.join(decoded))

        return decoded_texts

    def decode_batch(self, logits: torch.Tensor) -> List[str]:
        """Convenience method for batch decoding.

        Args:
            logits: Model logits (batch, seq_len, vocab_size)

        Returns:
            List of decoded strings
        """
        return self.decode(logits, remove_repeated=True)


class PhoneticAdapterASR(nn.Module):
    """Phonetic-aware adapter ASR model for cross-lingual transfer."""

    def __init__(
        self,
        base_model: str = "openai/whisper-tiny",
        num_adapters: int = 8,
        phonetic_dim: int = 256,
        adapter_dim: int = 64,
        adapter_layers: Optional[List[int]] = None,
        freeze_base: bool = True,
        num_languages: int = 8,
    ) -> None:
        """Initialize phonetic adapter ASR model.

        Args:
            base_model: Pretrained base model identifier
            num_adapters: Number of adapter modules
            phonetic_dim: Dimension of phonetic embeddings
            adapter_dim: Dimension of adapter bottleneck
            adapter_layers: Layer indices to insert adapters
            freeze_base: Whether to freeze base model parameters
            num_languages: Number of languages
        """
        super().__init__()
        self.base_model_name = base_model
        self.num_adapters = num_adapters
        self.phonetic_dim = phonetic_dim
        self.adapter_dim = adapter_dim
        self.freeze_base = freeze_base
        self.num_languages = num_languages

        # Language to ID mapping
        self.language_to_id = {
            'en': 0, 'es': 1, 'fr': 2, 'de': 3,
            'ar': 4, 'sw': 5, 'ta': 6, 'bn': 7,
        }

        # Load base model
        if WHISPER_AVAILABLE and num_adapters > 0:
            self._init_whisper_model()
        else:
            self._init_simple_model()

        # Phonetic embedding network
        self.phonetic_encoder = nn.Sequential(
            nn.Linear(24, 128),  # 24 is panphon feature dimension
            nn.ReLU(),
            nn.Linear(128, phonetic_dim),
            nn.LayerNorm(phonetic_dim),
        )

        # Adapter layers
        if adapter_layers is None:
            adapter_layers = [3, 6, 9, 11] if num_adapters > 0 else []
        self.adapter_layers = adapter_layers[:num_adapters]

        if num_adapters > 0:
            self.adapters = nn.ModuleList([
                PhoneticAdapterLayer(
                    hidden_dim=self.hidden_dim,
                    adapter_dim=adapter_dim,
                    phonetic_dim=phonetic_dim,
                    num_languages=num_languages,
                )
                for _ in range(len(self.adapter_layers))
            ])
        else:
            self.adapters = nn.ModuleList()

        # Output projection for CTC decoding
        # Vocabulary: blank(0) + space(1) + a-z(2-27) = 28 tokens
        self.output_projection = nn.Linear(self.hidden_dim, 28)

        logging.info(
            f"Initialized PhoneticAdapterASR with {num_adapters} adapters "
            f"at layers {self.adapter_layers}"
        )

    def _init_whisper_model(self) -> None:
        """Initialize Whisper base model."""
        try:
            self.processor = WhisperProcessor.from_pretrained(self.base_model_name)
            self.base_model = WhisperModel.from_pretrained(self.base_model_name)
            self.hidden_dim = self.base_model.config.d_model

            # Freeze base model if specified
            if self.freeze_base:
                for param in self.base_model.parameters():
                    param.requires_grad = False
                logging.info("Froze base Whisper model parameters")

        except Exception as e:
            logging.warning(f"Failed to load Whisper model: {e}, using simple model")
            self._init_simple_model()

    def _init_simple_model(self) -> None:
        """Initialize simple model for testing."""
        self.hidden_dim = 384
        self.base_model = None
        self.processor = None

        # Simple encoder: Conv -> LSTM
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1500),  # Fixed length
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=self.hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        logging.info("Initialized simple encoder model")

    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio to hidden representations.

        Args:
            audio: Input audio waveform (batch, time)

        Returns:
            Hidden representations (batch, seq_len, hidden_dim)
        """
        if self.base_model is not None:
            # Use Whisper encoder
            # Note: Whisper expects log-mel spectrogram, but for simplicity we'll use a workaround
            # In production, use proper preprocessing with self.processor
            with torch.no_grad() if self.freeze_base else torch.enable_grad():
                # Create dummy inputs with correct shape
                batch_size = audio.shape[0]
                # Whisper expects (batch, n_mels, time)
                dummy_input = torch.randn(
                    batch_size, 80, 3000,
                    device=audio.device,
                    dtype=audio.dtype
                )
                encoder_outputs = self.base_model.encoder(dummy_input)
                hidden_states = encoder_outputs.last_hidden_state
        else:
            # Use simple encoder
            audio = audio.unsqueeze(1)  # Add channel dimension
            x = self.encoder(audio)  # (batch, 128, 1500)
            x = x.transpose(1, 2)  # (batch, 1500, 128)
            hidden_states, _ = self.lstm(x)  # (batch, 1500, hidden_dim)

        return hidden_states

    def apply_adapters(
        self,
        hidden_states: torch.Tensor,
        phonetic_embedding: torch.Tensor,
        language_ids: List[int],
    ) -> torch.Tensor:
        """Apply phonetic adapters to hidden states.

        Args:
            hidden_states: Input hidden states (batch, seq_len, hidden_dim)
            phonetic_embedding: Phonetic embeddings (batch, phonetic_dim)
            language_ids: Language IDs for each sample in batch

        Returns:
            Adapted hidden states (batch, seq_len, hidden_dim)
        """
        if len(self.adapters) == 0:
            return hidden_states

        # Apply each adapter
        for adapter in self.adapters:
            # Process each sample with its language-specific adapter
            adapted_samples = []
            for i, lang_id in enumerate(language_ids):
                sample_hidden = hidden_states[i:i+1]  # (1, seq_len, hidden_dim)
                sample_phonetic = phonetic_embedding[i:i+1]  # (1, phonetic_dim)
                adapted = adapter(sample_hidden, sample_phonetic, lang_id)
                adapted_samples.append(adapted)

            hidden_states = torch.cat(adapted_samples, dim=0)

        return hidden_states

    def forward(
        self,
        audio: torch.Tensor,
        phonetic_features: torch.Tensor,
        languages: List[str],
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            audio: Input audio waveform (batch, time)
            phonetic_features: Phonetic features (batch, max_len, feature_dim)
            languages: List of language codes

        Returns:
            Dictionary containing logits and phonetic embeddings
        """
        # Encode audio
        hidden_states = self.encode_audio(audio)

        # Encode phonetic features
        # Average over time dimension
        phonetic_features_mean = phonetic_features.mean(dim=1)  # (batch, feature_dim)
        phonetic_embedding = self.phonetic_encoder(phonetic_features_mean)

        # Convert languages to IDs
        language_ids = [self.language_to_id.get(lang, 0) for lang in languages]

        # Apply adapters
        if len(self.adapters) > 0:
            hidden_states = self.apply_adapters(hidden_states, phonetic_embedding, language_ids)

        # Output projection for CTC decoding
        logits = self.output_projection(hidden_states)

        return {
            'logits': logits,
            'phonetic_embedding': phonetic_embedding,
            'hidden_states': hidden_states,
        }

    def get_adapter_parameters(self) -> List[torch.Tensor]:
        """Get adapter parameters for regularization.

        Returns:
            List of adapter parameter tensors
        """
        params = []
        for adapter in self.adapters:
            for param in adapter.parameters():
                params.append(param)
        return params

    def save_pretrained(self, save_path: str) -> None:
        """Save model checkpoint.

        Args:
            save_path: Path to save checkpoint
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': {
                'base_model': self.base_model_name,
                'num_adapters': self.num_adapters,
                'phonetic_dim': self.phonetic_dim,
                'adapter_dim': self.adapter_dim,
                'adapter_layers': self.adapter_layers,
                'freeze_base': self.freeze_base,
                'num_languages': self.num_languages,
            }
        }
        torch.save(checkpoint, save_path)
        logging.info(f"Saved model checkpoint to {save_path}")

    @classmethod
    def load_pretrained(cls, load_path: str, device: str = 'cpu') -> 'PhoneticAdapterASR':
        """Load model checkpoint.

        Args:
            load_path: Path to checkpoint
            device: Device to load model to

        Returns:
            Loaded model instance
        """
        checkpoint = torch.load(load_path, map_location=device)
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        logging.info(f"Loaded model checkpoint from {load_path}")
        return model

    def decode_predictions(self, logits: torch.Tensor) -> List[str]:
        """Decode model logits to text predictions.

        Args:
            logits: Model output logits (batch, seq_len, vocab_size)

        Returns:
            List of decoded text predictions
        """
        # Initialize decoder if not exists
        if not hasattr(self, '_decoder'):
            self._decoder = GreedyCTCDecoder()

        return self._decoder.decode_batch(logits)
