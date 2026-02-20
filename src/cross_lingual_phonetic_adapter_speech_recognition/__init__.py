"""Cross-Lingual Phonetic Adapter Speech Recognition.

A novel multilingual ASR system using phonetic-aware adapter modules
for zero-shot transfer to low-resource languages.
"""

__version__ = "0.1.0"
__author__ = "Alireza Shojaei"

from cross_lingual_phonetic_adapter_speech_recognition.models.model import PhoneticAdapterASR
from cross_lingual_phonetic_adapter_speech_recognition.data.loader import CommonVoiceDataLoader

__all__ = ["PhoneticAdapterASR", "CommonVoiceDataLoader"]
