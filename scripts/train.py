#!/usr/bin/env python
"""Training script for phonetic adapter ASR model."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root and src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from cross_lingual_phonetic_adapter_speech_recognition.utils.config import (
    load_config,
    set_seed,
    setup_logging,
)
from cross_lingual_phonetic_adapter_speech_recognition.data.loader import CommonVoiceDataLoader
from cross_lingual_phonetic_adapter_speech_recognition.models.model import PhoneticAdapterASR
from cross_lingual_phonetic_adapter_speech_recognition.training.trainer import ASRTrainer
from cross_lingual_phonetic_adapter_speech_recognition.evaluation.analysis import plot_training_curves


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Train phonetic adapter ASR model")
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
        help="Device to use for training",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with reduced dataset size",
    )
    return parser.parse_args()


def main() -> None:
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    # Setup logging
    log_dir = config.get('logging', {}).get('log_dir', './logs')
    setup_logging(log_dir=log_dir)
    logging.info("Starting training script")
    logging.info(f"Using device: {args.device}")
    logging.info(f"Configuration: {args.config}")

    # Set random seed
    seed = config.get('seed', 42)
    deterministic = config.get('deterministic', True)
    set_seed(seed, deterministic)

    # Initialize MLflow (wrapped in try/except)
    use_mlflow = config.get('logging', {}).get('use_mlflow', False)
    if use_mlflow:
        try:
            import mlflow
            mlflow.set_experiment(config.get('logging', {}).get('experiment_name', 'phonetic_adapter_asr'))
            mlflow.start_run()
            mlflow.log_params({
                'config_file': args.config,
                'seed': seed,
                'device': args.device,
            })
            logging.info("MLflow tracking enabled")
        except Exception as e:
            logging.warning(f"MLflow initialization failed: {e}")
            use_mlflow = False

    try:
        # Load data
        logging.info("Loading datasets...")
        data_config = config.get('data', {})
        languages = data_config.get('languages', ['en', 'es', 'fr', 'de', 'ar', 'sw', 'ta', 'bn'])

        # Use smaller dataset in debug mode
        max_samples = 100 if args.debug else None

        data_loader = CommonVoiceDataLoader(
            languages=languages,
            cache_dir=data_config.get('cache_dir', './data'),
            sample_rate=data_config.get('sample_rate', 16000),
            max_audio_length=data_config.get('max_audio_length', 30.0),
            train_batch_size=data_config.get('train_batch_size', 16),
            eval_batch_size=data_config.get('eval_batch_size', 32),
            num_workers=data_config.get('num_workers', 4),
            max_samples_per_language=max_samples,
        )

        train_loader = data_loader.get_train_loader()
        val_loader = data_loader.get_val_loader()

        logging.info(f"Loaded {len(train_loader.dataset)} training samples")
        logging.info(f"Loaded {len(val_loader.dataset)} validation samples")

        # Initialize model
        logging.info("Initializing model...")
        model_config = config.get('model', {})
        model = PhoneticAdapterASR(
            base_model=model_config.get('base_model', 'openai/whisper-tiny'),
            num_adapters=model_config.get('num_adapters', 8),
            phonetic_dim=model_config.get('phonetic_dim', 256),
            adapter_dim=model_config.get('adapter_dim', 64),
            adapter_layers=model_config.get('adapter_layers', [3, 6, 9, 11]),
            freeze_base=model_config.get('freeze_base', True),
            num_languages=len(languages),
        )

        # Move model to device
        device = torch.device(args.device)
        model = model.to(device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Trainable parameters: {trainable_params:,}")

        # Initialize trainer
        logging.info("Initializing trainer...")
        trainer = ASRTrainer(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            device=args.device,
        )

        # Train model
        logging.info("Starting training...")
        history = trainer.train()

        # Save training history
        results_dir = Path("./results")
        results_dir.mkdir(parents=True, exist_ok=True)

        history_file = results_dir / "training_history.json"
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        logging.info(f"Saved training history to {history_file}")

        # Plot training curves
        try:
            plot_training_curves(
                train_losses=history['train_losses'],
                val_losses=history['val_losses'],
                output_path=str(results_dir / "training_curves.png"),
            )
        except Exception as e:
            logging.warning(f"Failed to plot training curves: {e}")

        # Log to MLflow
        if use_mlflow:
            try:
                import mlflow
                mlflow.log_metrics({
                    'final_train_loss': history['train_losses'][-1],
                    'final_val_loss': history['val_losses'][-1],
                    'best_val_loss': trainer.best_val_loss,
                })
                mlflow.log_artifact(str(history_file))
                mlflow.end_run()
            except Exception as e:
                logging.warning(f"MLflow logging failed: {e}")

        logging.info("Training completed successfully!")
        logging.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
        logging.info(f"Model saved to: {trainer.save_dir / 'best_model.pt'}")

    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
        if use_mlflow:
            try:
                import mlflow
                mlflow.end_run(status='KILLED')
            except:
                pass
        sys.exit(1)

    except Exception as e:
        logging.error(f"Training failed with error: {e}", exc_info=True)
        if use_mlflow:
            try:
                import mlflow
                mlflow.end_run(status='FAILED')
            except:
                pass
        sys.exit(1)


if __name__ == "__main__":
    main()
