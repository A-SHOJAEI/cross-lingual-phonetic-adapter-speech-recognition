"""Evaluation metrics for ASR."""

import logging
from typing import Dict, List, Tuple

import numpy as np

try:
    import jiwer
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False
    logging.warning("jiwer not available - WER/CER metrics will use fallback")


def compute_wer(references: List[str], hypotheses: List[str]) -> float:
    """Compute Word Error Rate.

    Args:
        references: List of reference transcriptions
        hypotheses: List of hypothesis transcriptions

    Returns:
        Word Error Rate (0-100)
    """
    if not references or not hypotheses:
        return 100.0

    if JIWER_AVAILABLE:
        try:
            wer = jiwer.wer(references, hypotheses)
            return wer * 100.0
        except Exception as e:
            logging.warning(f"jiwer failed: {e}, using fallback")

    # Fallback: simple word-level accuracy
    total_words = 0
    total_errors = 0

    for ref, hyp in zip(references, hypotheses):
        ref_words = ref.split()
        hyp_words = hyp.split()

        total_words += len(ref_words)
        # Simple error count (not true edit distance)
        errors = abs(len(ref_words) - len(hyp_words))
        for r, h in zip(ref_words, hyp_words):
            if r != h:
                errors += 1
        total_errors += errors

    if total_words == 0:
        return 100.0

    return (total_errors / total_words) * 100.0


def compute_cer(references: List[str], hypotheses: List[str]) -> float:
    """Compute Character Error Rate.

    Args:
        references: List of reference transcriptions
        hypotheses: List of hypothesis transcriptions

    Returns:
        Character Error Rate (0-100)
    """
    if not references or not hypotheses:
        return 100.0

    if JIWER_AVAILABLE:
        try:
            cer = jiwer.cer(references, hypotheses)
            return cer * 100.0
        except Exception as e:
            logging.warning(f"jiwer CER failed: {e}, using fallback")

    # Fallback: character-level accuracy
    total_chars = 0
    total_errors = 0

    for ref, hyp in zip(references, hypotheses):
        total_chars += len(ref)
        errors = abs(len(ref) - len(hyp))
        for r, h in zip(ref, hyp):
            if r != h:
                errors += 1
        total_errors += errors

    if total_chars == 0:
        return 100.0

    return (total_errors / total_chars) * 100.0


def edit_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Edit distance
    """
    if len(s1) < len(s2):
        return edit_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def compute_per(
    reference_phonemes: List[List[str]],
    hypothesis_phonemes: List[List[str]],
) -> float:
    """Compute Phoneme Error Rate.

    Args:
        reference_phonemes: List of reference phoneme sequences
        hypothesis_phonemes: List of hypothesis phoneme sequences

    Returns:
        Phoneme Error Rate (0-100)
    """
    if not reference_phonemes or not hypothesis_phonemes:
        return 100.0

    total_phonemes = 0
    total_errors = 0

    for ref, hyp in zip(reference_phonemes, hypothesis_phonemes):
        ref_str = ' '.join(ref)
        hyp_str = ' '.join(hyp)

        total_phonemes += len(ref)
        distance = edit_distance(ref_str, hyp_str)
        total_errors += distance

    if total_phonemes == 0:
        return 100.0

    return (total_errors / total_phonemes) * 100.0


class ASRMetrics:
    """Container for ASR evaluation metrics."""

    def __init__(self) -> None:
        """Initialize metrics container."""
        self.wer_scores: List[float] = []
        self.cer_scores: List[float] = []
        self.per_scores: List[float] = []
        self.language_metrics: Dict[str, Dict[str, float]] = {}

    def add_sample(
        self,
        reference: str,
        hypothesis: str,
        language: str = 'unknown',
        reference_phonemes: List[str] = None,
        hypothesis_phonemes: List[str] = None,
    ) -> None:
        """Add a sample for evaluation.

        Args:
            reference: Reference transcription
            hypothesis: Hypothesis transcription
            language: Language code
            reference_phonemes: Reference phoneme sequence
            hypothesis_phonemes: Hypothesis phoneme sequence
        """
        # Compute WER for this sample
        wer = compute_wer([reference], [hypothesis])
        self.wer_scores.append(wer)

        # Compute CER for this sample
        cer = compute_cer([reference], [hypothesis])
        self.cer_scores.append(cer)

        # Compute PER if phonemes provided
        if reference_phonemes and hypothesis_phonemes:
            per = compute_per([reference_phonemes], [hypothesis_phonemes])
            self.per_scores.append(per)

        # Track per-language metrics
        if language not in self.language_metrics:
            self.language_metrics[language] = {
                'wer': [],
                'cer': [],
                'per': [],
            }

        self.language_metrics[language]['wer'].append(wer)
        self.language_metrics[language]['cer'].append(cer)
        if reference_phonemes and hypothesis_phonemes:
            self.language_metrics[language]['per'].append(
                compute_per([reference_phonemes], [hypothesis_phonemes])
            )

    def compute_aggregate_metrics(self) -> Dict[str, float]:
        """Compute aggregate metrics.

        Returns:
            Dictionary of aggregate metrics
        """
        metrics = {}

        if self.wer_scores:
            metrics['wer_mean'] = np.mean(self.wer_scores)
            metrics['wer_std'] = np.std(self.wer_scores)
            metrics['wer_median'] = np.median(self.wer_scores)

        if self.cer_scores:
            metrics['cer_mean'] = np.mean(self.cer_scores)
            metrics['cer_std'] = np.std(self.cer_scores)
            metrics['cer_median'] = np.median(self.cer_scores)

        if self.per_scores:
            metrics['per_mean'] = np.mean(self.per_scores)
            metrics['per_std'] = np.std(self.per_scores)
            metrics['per_median'] = np.median(self.per_scores)

        return metrics

    def compute_language_metrics(self) -> Dict[str, Dict[str, float]]:
        """Compute per-language metrics.

        Returns:
            Dictionary of per-language metrics
        """
        language_results = {}

        for lang, scores in self.language_metrics.items():
            language_results[lang] = {}

            if scores['wer']:
                language_results[lang]['wer_mean'] = np.mean(scores['wer'])
                language_results[lang]['wer_std'] = np.std(scores['wer'])

            if scores['cer']:
                language_results[lang]['cer_mean'] = np.mean(scores['cer'])
                language_results[lang]['cer_std'] = np.std(scores['cer'])

            if scores['per']:
                language_results[lang]['per_mean'] = np.mean(scores['per'])
                language_results[lang]['per_std'] = np.std(scores['per'])

        return language_results

    def get_summary(self) -> str:
        """Get summary string.

        Returns:
            Formatted summary string
        """
        agg_metrics = self.compute_aggregate_metrics()
        lang_metrics = self.compute_language_metrics()

        summary = "=== ASR Evaluation Results ===\n\n"
        summary += "Aggregate Metrics:\n"
        for key, value in agg_metrics.items():
            summary += f"  {key}: {value:.2f}\n"

        summary += "\nPer-Language Metrics:\n"
        for lang, metrics in lang_metrics.items():
            summary += f"\n  {lang}:\n"
            for key, value in metrics.items():
                summary += f"    {key}: {value:.2f}\n"

        return summary
