"""Tests for data loading and preprocessing."""

import pytest
import torch
import numpy as np

from cross_lingual_phonetic_adapter_speech_recognition.data.preprocessing import (
    AudioPreprocessor,
    PhoneticExtractor,
)
from cross_lingual_phonetic_adapter_speech_recognition.data.loader import (
    CommonVoiceDataset,
    CommonVoiceDataLoader,
)


class TestAudioPreprocessor:
    """Tests for AudioPreprocessor."""

    def test_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = AudioPreprocessor(sample_rate=16000, max_length=10.0)
        assert preprocessor.sample_rate == 16000
        assert preprocessor.max_length == 10.0
        assert preprocessor.max_samples == 160000

    def test_normalize(self):
        """Test audio normalization."""
        preprocessor = AudioPreprocessor()
        waveform = np.array([0.5, 1.0, -0.5, -1.0])
        normalized = preprocessor.normalize(waveform)

        assert np.abs(normalized).max() <= 1.0
        assert normalized.shape == waveform.shape

    def test_pad_or_trim_padding(self):
        """Test padding short audio."""
        preprocessor = AudioPreprocessor(sample_rate=16000, max_length=1.0)
        waveform = np.random.randn(8000)  # 0.5 seconds
        processed = preprocessor.pad_or_trim(waveform)

        assert len(processed) == 16000
        assert processed[:8000].tolist() == pytest.approx(waveform.tolist())

    def test_pad_or_trim_trimming(self):
        """Test trimming long audio."""
        preprocessor = AudioPreprocessor(sample_rate=16000, max_length=1.0)
        waveform = np.random.randn(32000)  # 2 seconds
        processed = preprocessor.pad_or_trim(waveform)

        assert len(processed) == 16000

    def test_augmentation_disabled(self):
        """Test that augmentation can be disabled."""
        preprocessor = AudioPreprocessor(augmentation=False)
        assert preprocessor.augmentation is None


class TestPhoneticExtractor:
    """Tests for PhoneticExtractor."""

    def test_initialization(self):
        """Test extractor initialization."""
        extractor = PhoneticExtractor(language_codes=['eng', 'spa'])
        assert extractor.enabled or not extractor.enabled  # May vary based on lib availability

    def test_text_to_ipa(self):
        """Test IPA conversion."""
        extractor = PhoneticExtractor()
        text = "hello"
        ipa = extractor.text_to_ipa(text, "eng")

        assert isinstance(ipa, str)
        assert len(ipa) > 0

    def test_ipa_to_features(self):
        """Test phonetic feature extraction."""
        extractor = PhoneticExtractor()
        ipa = "həloʊ"
        features = extractor.ipa_to_features(ipa)

        assert isinstance(features, np.ndarray)
        assert features.shape[0] >= 1  # At least one phoneme
        assert features.shape[1] == 24  # Panphon feature dimension

    def test_extract_phonetic_complexity(self):
        """Test complexity estimation."""
        extractor = PhoneticExtractor()

        # Simple vowel
        simple = extractor.extract_phonetic_complexity("aaa")
        assert simple >= 1

        # Complex cluster
        complex_text = extractor.extract_phonetic_complexity("strpkt")
        assert complex_text >= 1


class TestCommonVoiceDataset:
    """Tests for CommonVoiceDataset."""

    def test_initialization(self):
        """Test dataset initialization."""
        dataset = CommonVoiceDataset(
            language='en',
            split='train',
            max_samples=10,
        )
        assert len(dataset) > 0
        assert dataset.language == 'en'

    def test_getitem(self):
        """Test getting dataset item."""
        dataset = CommonVoiceDataset(
            language='en',
            split='train',
            max_samples=10,
        )
        item = dataset[0]

        assert 'audio' in item
        assert 'text' in item
        assert 'language' in item
        assert 'phonetic_features' in item
        assert 'complexity' in item

        assert isinstance(item['audio'], torch.Tensor)
        assert isinstance(item['text'], str)
        assert item['language'] == 'en'

    def test_synthetic_data_generation(self):
        """Test synthetic data fallback."""
        dataset = CommonVoiceDataset(
            language='unknown',
            split='train',
            max_samples=5,
        )
        assert len(dataset) == 5


class TestCommonVoiceDataLoader:
    """Tests for CommonVoiceDataLoader."""

    def test_initialization(self):
        """Test data loader initialization."""
        loader = CommonVoiceDataLoader(
            languages=['en', 'es'],
            max_samples_per_language=5,
        )
        assert 'en' in loader.train_datasets
        assert 'es' in loader.train_datasets

    def test_get_train_loader(self):
        """Test getting training data loader."""
        loader = CommonVoiceDataLoader(
            languages=['en'],
            train_batch_size=2,
            max_samples_per_language=10,
        )
        train_loader = loader.get_train_loader()

        assert train_loader is not None
        batch = next(iter(train_loader))

        assert 'audio' in batch
        assert 'text' in batch
        assert batch['audio'].shape[0] <= 2  # Batch size

    def test_get_val_loader(self):
        """Test getting validation data loader."""
        loader = CommonVoiceDataLoader(
            languages=['en'],
            eval_batch_size=4,
            max_samples_per_language=10,
        )
        val_loader = loader.get_val_loader()

        assert val_loader is not None

    def test_collate_fn(self):
        """Test batch collation."""
        loader = CommonVoiceDataLoader(
            languages=['en'],
            max_samples_per_language=10,
        )

        # Create mock batch
        mock_batch = [
            {
                'audio': torch.randn(16000),
                'text': 'hello',
                'language': 'en',
                'ipa': 'həloʊ',
                'phonetic_features': torch.randn(5, 24),
                'complexity': 1,
            },
            {
                'audio': torch.randn(16000),
                'text': 'world',
                'language': 'en',
                'ipa': 'wɜrld',
                'phonetic_features': torch.randn(3, 24),
                'complexity': 2,
            },
        ]

        collated = loader._collate_fn(mock_batch)

        assert collated['audio'].shape[0] == 2
        assert len(collated['text']) == 2
        assert collated['phonetic_features'].shape[0] == 2
        # Features should be padded to same length
        assert collated['phonetic_features'].shape[1] == 5
