"""Results analysis and visualization."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_results(
    results: Dict[str, Any],
    output_dir: str = "./results",
) -> Dict[str, Any]:
    """Analyze evaluation results and generate statistics.

    Args:
        results: Dictionary containing evaluation results
        output_dir: Directory to save analysis outputs

    Returns:
        Dictionary with analysis statistics
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    analysis = {
        'summary_statistics': {},
        'language_comparison': {},
        'complexity_analysis': {},
    }

    # Compute summary statistics
    if 'wer_scores' in results:
        wer_scores = results['wer_scores']
        analysis['summary_statistics']['wer'] = {
            'mean': float(np.mean(wer_scores)),
            'std': float(np.std(wer_scores)),
            'median': float(np.median(wer_scores)),
            'min': float(np.min(wer_scores)),
            'max': float(np.max(wer_scores)),
            'percentile_25': float(np.percentile(wer_scores, 25)),
            'percentile_75': float(np.percentile(wer_scores, 75)),
        }

    if 'cer_scores' in results:
        cer_scores = results['cer_scores']
        analysis['summary_statistics']['cer'] = {
            'mean': float(np.mean(cer_scores)),
            'std': float(np.std(cer_scores)),
            'median': float(np.median(cer_scores)),
            'min': float(np.min(cer_scores)),
            'max': float(np.max(cer_scores)),
        }

    if 'per_scores' in results:
        per_scores = results['per_scores']
        analysis['summary_statistics']['per'] = {
            'mean': float(np.mean(per_scores)),
            'std': float(np.std(per_scores)),
            'median': float(np.median(per_scores)),
        }

    # Language comparison
    if 'language_metrics' in results:
        for lang, metrics in results['language_metrics'].items():
            analysis['language_comparison'][lang] = {
                'wer_mean': float(np.mean(metrics.get('wer', []))),
                'cer_mean': float(np.mean(metrics.get('cer', []))),
                'per_mean': float(np.mean(metrics.get('per', []))),
            }

    # Save analysis to JSON
    analysis_file = output_path / 'analysis.json'
    with open(analysis_file, 'w') as f:
        json.dump(analysis, indent=2, fp=f)
    logging.info(f"Saved analysis to {analysis_file}")

    return analysis


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    output_path: str = "./results/training_curves.png",
    metric_name: str = "Loss",
) -> None:
    """Plot training and validation curves.

    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch
        output_path: Path to save plot
        metric_name: Name of metric being plotted
    """
    plt.figure(figsize=(10, 6))

    epochs = list(range(1, len(train_losses) + 1))

    plt.plot(epochs, train_losses, label=f'Training {metric_name}', marker='o', linewidth=2)
    plt.plot(epochs, val_losses, label=f'Validation {metric_name}', marker='s', linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.title(f'Training and Validation {metric_name}', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # Create output directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"Saved training curves to {output_path}")


def plot_language_comparison(
    language_metrics: Dict[str, Dict[str, float]],
    output_path: str = "./results/language_comparison.png",
) -> None:
    """Plot comparison of metrics across languages.

    Args:
        language_metrics: Dictionary of per-language metrics
        output_path: Path to save plot
    """
    if not language_metrics:
        logging.warning("No language metrics to plot")
        return

    languages = list(language_metrics.keys())
    wer_means = [metrics.get('wer_mean', 0) for metrics in language_metrics.values()]
    cer_means = [metrics.get('cer_mean', 0) for metrics in language_metrics.values()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # WER comparison
    ax1.bar(languages, wer_means, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Language', fontsize=12)
    ax1.set_ylabel('Word Error Rate (%)', fontsize=12)
    ax1.set_title('WER by Language', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=45)

    # CER comparison
    ax2.bar(languages, cer_means, color='coral', alpha=0.7)
    ax2.set_xlabel('Language', fontsize=12)
    ax2.set_ylabel('Character Error Rate (%)', fontsize=12)
    ax2.set_title('CER by Language', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    # Create output directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"Saved language comparison to {output_path}")


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    labels: List[str],
    output_path: str = "./results/confusion_matrix.png",
) -> None:
    """Plot confusion matrix.

    Args:
        confusion_matrix: Confusion matrix array
        labels: Label names
        output_path: Path to save plot
    """
    plt.figure(figsize=(10, 8))

    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count'},
    )

    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.tight_layout()

    # Create output directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"Saved confusion matrix to {output_path}")


def compute_statistical_significance(
    baseline_scores: List[float],
    proposed_scores: List[float],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
) -> Dict[str, float]:
    """Compute statistical significance using bootstrap resampling.

    Args:
        baseline_scores: Baseline model scores
        proposed_scores: Proposed model scores
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for intervals

    Returns:
        Dictionary with significance statistics
    """
    baseline_scores = np.array(baseline_scores)
    proposed_scores = np.array(proposed_scores)

    # Compute observed difference
    observed_diff = np.mean(proposed_scores) - np.mean(baseline_scores)

    # Bootstrap resampling
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        baseline_sample = np.random.choice(baseline_scores, size=len(baseline_scores), replace=True)
        proposed_sample = np.random.choice(proposed_scores, size=len(proposed_scores), replace=True)
        diff = np.mean(proposed_sample) - np.mean(baseline_sample)
        bootstrap_diffs.append(diff)

    bootstrap_diffs = np.array(bootstrap_diffs)

    # Compute confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    ci_lower = np.percentile(bootstrap_diffs, lower_percentile)
    ci_upper = np.percentile(bootstrap_diffs, upper_percentile)

    # Compute p-value (two-tailed test)
    p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))

    return {
        'observed_difference': float(observed_diff),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'p_value': float(p_value),
        'significant': p_value < (1 - confidence_level),
    }
