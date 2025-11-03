"""Run evaluation in batches of 10 questions.

This allows incremental progress saving and easier debugging.
"""

import sys
import os
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simple_agent import run_evaluation, load_evaluation_data


def run_batch_evaluation(batch_size=10, start_batch=0, total_batches=10):
    """Run evaluation in batches.

    Args:
        batch_size: Number of questions per batch (default: 10)
        start_batch: Which batch to start from (default: 0)
        total_batches: Total number of batches to run (default: 10 = 100 questions)
    """
    print("=" * 80)
    print("BATCH EVALUATION")
    print("=" * 80)
    print(f"Batch size: {batch_size} questions")
    print(f"Total batches: {total_batches}")
    print(f"Starting from batch: {start_batch}")
    print(f"Total questions: {batch_size * total_batches}")
    print("=" * 80)
    print()

    all_scores = []
    batch_results = []

    for batch_num in range(start_batch, start_batch + total_batches):
        start_index = batch_num * batch_size

        print(f"\n{'#'*80}")
        print(f"BATCH {batch_num + 1}/{start_batch + total_batches}")
        print(f"Questions {start_index + 1} - {start_index + batch_size}")
        print(f"{'#'*80}\n")

        try:
            # Run evaluation for this batch
            scores = run_evaluation(
                num_questions=batch_size,
                start_index=start_index,
                save_results=True  # Saves individual batch results
            )

            all_scores.extend(scores)

            # Save batch summary
            batch_summary = {
                'batch_number': batch_num + 1,
                'start_index': start_index,
                'end_index': start_index + batch_size,
                'batch_average': sum(scores) / len(scores) if scores else 0,
                'scores': scores,
                'timestamp': datetime.now().isoformat()
            }
            batch_results.append(batch_summary)

            print(f"\n{'='*80}")
            print(f"BATCH {batch_num + 1} COMPLETE")
            print(f"Batch average: {batch_summary['batch_average']:.2f}/5.0")
            print(f"Running average: {sum(all_scores)/len(all_scores):.2f}/5.0")
            print(f"{'='*80}\n")

        except Exception as e:
            print(f"\nERROR in batch {batch_num + 1}: {e}")
            import traceback
            traceback.print_exc()

            # Save progress so far
            save_aggregate_results(all_scores, batch_results, partial=True)
            print("\nProgress saved. You can resume from batch", batch_num + 1)
            break

    # Save final aggregate results
    save_aggregate_results(all_scores, batch_results, partial=False)

    # Print final summary
    print_final_summary(all_scores, batch_results)


def save_aggregate_results(all_scores, batch_results, partial=False):
    """Save aggregate results across all batches."""
    results_file = os.path.join(
        os.path.dirname(__file__),
        f'evaluation_aggregate_{"partial_" if partial else ""}{len(all_scores)}q.json'
    )

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_questions': len(all_scores),
            'average_score': sum(all_scores) / len(all_scores) if all_scores else 0,
            'score_distribution': {str(i): all_scores.count(i) for i in range(0, 6)},
            'batches': batch_results,
            'all_scores': all_scores,
            'partial': partial,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    print(f"\n{'Partial results' if partial else 'Final results'} saved to: {results_file}")


def print_final_summary(all_scores, batch_results):
    """Print final evaluation summary."""
    print("\n" + "=" * 80)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total questions evaluated: {len(all_scores)}")
    print(f"Total batches completed: {len(batch_results)}")
    print(f"\nOverall average score: {sum(all_scores)/len(all_scores):.2f}/5.0")

    print(f"\nScore distribution:")
    for score in range(0, 6):
        count = all_scores.count(score)
        percentage = (count / len(all_scores) * 100) if all_scores else 0
        print(f"  {score}/5: {count:3d} questions ({percentage:5.1f}%)")

    print(f"\nBatch averages:")
    for batch in batch_results:
        print(f"  Batch {batch['batch_number']:2d}: {batch['batch_average']:.2f}/5.0")

    print("=" * 80)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Run evaluation in batches')
    parser.add_argument('--batch-size', type=int, default=10, help='Questions per batch (default: 10)')
    parser.add_argument('--start-batch', type=int, default=0, help='Starting batch number (default: 0)')
    parser.add_argument('--num-batches', type=int, default=10, help='Number of batches to run (default: 10)')

    args = parser.parse_args()

    run_batch_evaluation(
        batch_size=args.batch_size,
        start_batch=args.start_batch,
        total_batches=args.num_batches
    )


if __name__ == "__main__":
    main()
