"""
Batch processing pipeline: Parse all CVs and extract NER entities.

Usage:
    python batch_process.py [--skip-parsed] [--delay SECONDS]

Options:
    --skip-parsed   Skip CVs that already have NER output
    --delay         Delay between Gemini API calls (default: 1.0 seconds)
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.parsing.pdf_parser import parse_pdf
from src.ner.gemini_ner import extract_entities
from config.settings import DATA_RAW_CV, DATA_PROCESSED_CV, DATA_NER_CV


def main():
    parser = argparse.ArgumentParser(description="Batch process CV PDFs")
    parser.add_argument("--skip-parsed", action="store_true", help="Skip already processed CVs")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between API calls")
    args = parser.parse_args()

    cv_dir = Path(DATA_RAW_CV)
    parsed_dir = Path(DATA_PROCESSED_CV)
    ner_dir = Path(DATA_NER_CV)

    # Ensure output directories exist
    parsed_dir.mkdir(parents=True, exist_ok=True)
    ner_dir.mkdir(parents=True, exist_ok=True)

    # Get list of PDF files
    pdf_files = sorted(cv_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {cv_dir}")
        return

    print(f"Found {len(pdf_files)} PDF files in {cv_dir}")

    results = {"success": 0, "failed": 0, "skipped": 0, "errors": []}

    for filename in tqdm(pdf_files, desc="Processing CVs"):
        cv_id = filename.stem
        ner_output = ner_dir / f"{cv_id}.json"

        # Skip if already processed
        if args.skip_parsed and ner_output.exists():
            results["skipped"] += 1
            continue

        # Step 1: Parse PDF
        text, error = parse_pdf(str(filename))
        if error:
            results["failed"] += 1
            results["errors"].append({"cv_id": cv_id, "stage": "parsing", "error": error})
            continue

        # Save parsed text
        with open(parsed_dir / f"{cv_id}.txt", "w", encoding="utf-8") as f:
            f.write(text)

        # Step 2: NER with Gemini
        entities = extract_entities(text)
        if entities is None:
            results["failed"] += 1
            results["errors"].append({"cv_id": cv_id, "stage": "ner", "error": "NER returned None"})
            continue

        # Save NER result
        with open(ner_output, "w", encoding="utf-8") as f:
            json.dump(entities, f, ensure_ascii=False, indent=2)

        results["success"] += 1

        # Rate limit delay
        time.sleep(args.delay)

    # Print summary
    print(f"\n{'='*50}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*50}")
    print(f"Success: {results['success']}")
    print(f"Failed:  {results['failed']}")
    print(f"Skipped: {results['skipped']}")
    print(f"Total:   {len(pdf_files)}")

    if results["errors"]:
        print(f"\nErrors:")
        for err in results["errors"]:
            print(f"  - {err['cv_id']}: [{err['stage']}] {err['error']}")

        # Save error log
        error_log = ner_dir.parent / "batch_errors.json"
        with open(error_log, "w", encoding="utf-8") as f:
            json.dump(results["errors"], f, ensure_ascii=False, indent=2)
        print(f"\nError log saved to {error_log}")


if __name__ == "__main__":
    main()
