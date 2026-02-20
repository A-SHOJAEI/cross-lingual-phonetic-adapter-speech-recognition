#!/usr/bin/env python
"""Script to download and prepare CommonVoice dataset for training."""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Add project root and src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("ERROR: 'datasets' library not installed. Install with: pip install datasets")
    sys.exit(1)

from cross_lingual_phonetic_adapter_speech_recognition.utils.config import setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Download CommonVoice dataset for cross-lingual ASR training"
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en", "es", "fr", "de", "ar", "sw", "ta", "bn"],
        help="List of language codes to download (default: en es fr de ar sw ta bn)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Directory to save downloaded data (default: ./data)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="11.0",
        help="CommonVoice version to download (default: 11.0)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per language (default: None for all samples)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation", "test"],
        help="Dataset splits to download (default: train validation test)",
    )
    return parser.parse_args()


def download_language(
    language: str,
    version: str,
    cache_dir: str,
    splits: List[str],
    max_samples: Optional[int] = None,
) -> bool:
    """Download CommonVoice data for a specific language.

    Args:
        language: Language code (e.g., 'en', 'es')
        version: CommonVoice version
        cache_dir: Directory to cache downloaded data
        splits: List of splits to download
        max_samples: Maximum samples per split

    Returns:
        True if download successful, False otherwise
    """
    try:
        logging.info(f"Downloading CommonVoice {version} for language: {language}")

        for split in splits:
            try:
                logging.info(f"  Downloading {split} split...")

                dataset = load_dataset(
                    "mozilla-foundation/common_voice_11_0",
                    language,
                    split=split,
                    cache_dir=cache_dir,
                    trust_remote_code=True,
                )

                # Limit samples if specified
                if max_samples is not None and len(dataset) > max_samples:
                    dataset = dataset.select(range(max_samples))

                num_samples = len(dataset)
                logging.info(f"    Downloaded {num_samples} samples for {split}")

            except Exception as e:
                logging.error(f"    Failed to download {split} for {language}: {e}")
                return False

        logging.info(f"Successfully downloaded all splits for {language}")
        return True

    except Exception as e:
        logging.error(f"Failed to download {language}: {e}")
        return False


def main() -> None:
    """Main download function."""
    args = parse_args()

    # Setup logging
    setup_logging(log_dir="./logs")
    logging.info("Starting CommonVoice dataset download")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")

    # Download each language
    successful_downloads = []
    failed_downloads = []

    for language in args.languages:
        logging.info(f"\n{'='*60}")
        logging.info(f"Processing language: {language}")
        logging.info(f"{'='*60}")

        try:
            success = download_language(
                language=language,
                version=args.version,
                cache_dir=str(output_dir),
                splits=args.splits,
                max_samples=args.max_samples,
            )

            if success:
                successful_downloads.append(language)
            else:
                failed_downloads.append(language)

        except KeyboardInterrupt:
            logging.info("\nDownload interrupted by user")
            sys.exit(1)
        except Exception as e:
            logging.error(f"Error processing {language}: {e}")
            failed_downloads.append(language)

    # Print summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print(f"Successful downloads ({len(successful_downloads)}):")
    for lang in successful_downloads:
        print(f"  ✓ {lang}")

    if failed_downloads:
        print(f"\nFailed downloads ({len(failed_downloads)}):")
        for lang in failed_downloads:
            print(f"  ✗ {lang}")

    print("="*60)

    # Save download info
    info_file = output_dir / "download_info.txt"
    try:
        with open(info_file, 'w') as f:
            f.write(f"CommonVoice Version: {args.version}\n")
            f.write(f"Downloaded Languages: {', '.join(successful_downloads)}\n")
            f.write(f"Splits: {', '.join(args.splits)}\n")
            if args.max_samples:
                f.write(f"Max Samples per Language: {args.max_samples}\n")
        logging.info(f"Saved download info to {info_file}")
    except Exception as e:
        logging.warning(f"Failed to save download info: {e}")

    if failed_downloads:
        sys.exit(1)


if __name__ == "__main__":
    main()
