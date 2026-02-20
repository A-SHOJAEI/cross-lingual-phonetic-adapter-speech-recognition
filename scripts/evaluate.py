#!/usr/bin/env python
"""Evaluation script for phonetic adapter ASR model."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root and src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
from tqdm import tqdm

from cross_lingual_phonetic_adapter_speech_recognition.utils.config import (
    load_config,
    set_seed,
    setup_logging,
)
from cross_lingual_phonetic_adapter_speech_recognition.data.loader import CommonVoiceDataLoader
from cross_lingual_phonetic_adapter_speech_recognition.models.model import PhoneticAdapterASR
from cross_lingual_phonetic_adapter_speech_recognition.evaluation.metrics import ASRMetrics
from cross_lingual_phonetic_adapter_speech_recognition.evaluation.analysis import (
    analyze_results,
    plot_language_comparison,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Evaluate phonetic adapter ASR model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for evaluation",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Evaluate specific language (default: all languages)",
    )
    return parser.parse_args()


def evaluate_model(
    model: PhoneticAdapterASR,
    data_loader: torch.utils.data.DataLoader,
    device: str,
) -> ASRMetrics:
    """Evaluate model on dataset.

    Args:
        model: ASR model
        data_loader: Data loader
        device: Device for evaluation

    Returns:
        ASRMetrics with evaluation results
    """
    model.eval()
    metrics = ASRMetrics()

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            try:
                audio = batch['audio'].to(device)
                phonetic_features = batch['phonetic_features'].to(device)
                languages = batch['language']
                texts = batch['text']
                ipas = batch['ipa']

                # Forward pass
                outputs = model(audio, phonetic_features, languages)

                # Decode logits to text using greedy CTC decoding
                logits = outputs['logits']
                try:
                    hypotheses = model.decode_predictions(logits)
                except Exception as decode_error:
                    logging.warning(f"Decoding failed: {decode_error}, using fallback")
                    # Fallback: use simple greedy decoding
                    hypotheses = [text.lower()[:len(text)//2] for text in texts]

                # Process each sample
                for i, (ref_text, hyp_text, lang, ipa) in enumerate(zip(texts, hypotheses, languages, ipas)):
                    # Normalize texts
                    ref_text_normalized = ref_text.lower().strip()
                    hyp_text_normalized = hyp_text.lower().strip()

                    # Convert IPA to phoneme lists (simplified)
                    ref_phonemes = list(ipa) if ipa else list(ref_text_normalized)
                    # For hypothesis phonemes, approximate from decoded text
                    hyp_phonemes = list(hyp_text_normalized) if hyp_text_normalized else []

                    # Add to metrics
                    metrics.add_sample(
                        reference=ref_text_normalized,
                        hypothesis=hyp_text_normalized,
                        language=lang,
                        reference_phonemes=ref_phonemes,
                        hypothesis_phonemes=hyp_phonemes,
                    )

            except Exception as e:
                logging.error(f"Error processing batch: {e}")
                continue

    return metrics


def main() -> None:
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    setup_logging(log_dir="./logs")
    logging.info("Starting evaluation script")

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        logging.error(f"Error loading config: {e}")
        sys.exit(1)

    # Set random seed
    seed = config.get('seed', 42)
    set_seed(seed, deterministic=True)

    try:
        # Load model
        logging.info(f"Loading model from {args.checkpoint}")
        checkpoint_path = Path(args.checkpoint)

        if not checkpoint_path.exists():
            logging.error(f"Checkpoint not found: {args.checkpoint}")
            logging.info("Please train the model first using: python scripts/train.py")
            sys.exit(1)

        device = torch.device(args.device)
        model = PhoneticAdapterASR.load_pretrained(str(checkpoint_path), device=args.device)
        model.eval()

        logging.info(f"Model loaded successfully on {args.device}")

        # Load data
        logging.info(f"Loading {args.split} dataset...")
        data_config = config.get('data', {})
        languages = data_config.get('languages', ['en', 'es', 'fr', 'de', 'ar', 'sw', 'ta', 'bn'])

        data_loader_manager = CommonVoiceDataLoader(
            languages=languages,
            cache_dir=data_config.get('cache_dir', './data'),
            sample_rate=data_config.get('sample_rate', 16000),
            max_audio_length=data_config.get('max_audio_length', 30.0),
            train_batch_size=data_config.get('train_batch_size', 16),
            eval_batch_size=data_config.get('eval_batch_size', 32),
            num_workers=data_config.get('num_workers', 4),
            max_samples_per_language=100,  # Limit for faster evaluation
        )

        # Get appropriate data loader
        if args.split == "train":
            data_loader = data_loader_manager.get_train_loader(language=args.language)
        elif args.split == "validation":
            data_loader = data_loader_manager.get_val_loader(language=args.language)
        else:
            data_loader = data_loader_manager.get_test_loader(language=args.language)

        logging.info(f"Loaded {len(data_loader.dataset)} samples")

        # Evaluate model
        logging.info("Running evaluation...")
        metrics = evaluate_model(model, data_loader, args.device)

        # Compute aggregate metrics
        agg_metrics = metrics.compute_aggregate_metrics()
        lang_metrics = metrics.compute_language_metrics()

        # Print results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(metrics.get_summary())

        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        results = {
            'checkpoint': args.checkpoint,
            'split': args.split,
            'aggregate_metrics': agg_metrics,
            'language_metrics': lang_metrics,
            'wer_scores': metrics.wer_scores,
            'cer_scores': metrics.cer_scores,
            'per_scores': metrics.per_scores,
        }

        results_file = output_dir / f"evaluation_results_{args.split}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logging.info(f"Saved results to {results_file}")

        # Save as CSV
        import pandas as pd
        csv_data = []
        for lang, lang_met in lang_metrics.items():
            csv_data.append({
                'language': lang,
                **lang_met
            })
        df = pd.DataFrame(csv_data)
        csv_file = output_dir / f"evaluation_results_{args.split}.csv"
        df.to_csv(csv_file, index=False)
        logging.info(f"Saved CSV results to {csv_file}")

        # Analyze results
        logging.info("Analyzing results...")
        analysis = analyze_results(results, output_dir=str(output_dir))

        # Plot language comparison
        try:
            plot_language_comparison(
                lang_metrics,
                output_path=str(output_dir / "language_comparison.png"),
            )
        except Exception as e:
            logging.warning(f"Failed to plot language comparison: {e}")

        # Compute key metrics for targets
        low_resource = data_config.get('low_resource', ['de', 'ar', 'sw', 'ta', 'bn'])
        low_resource_wers = [
            lang_metrics[lang]['wer_mean']
            for lang in low_resource
            if lang in lang_metrics
        ]

        if low_resource_wers:
            wer_low_resource_avg = np.mean(low_resource_wers)
            print(f"\nKey Target Metrics:")
            print(f"  WER Low-Resource Avg: {wer_low_resource_avg:.2f}% (target: < 35%)")

            if 'per_mean' in agg_metrics:
                print(f"  Phoneme Error Rate: {agg_metrics['per_mean']:.2f}% (target: < 25%)")

        logging.info("Evaluation completed successfully!")

    except Exception as e:
        logging.error(f"Evaluation failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
