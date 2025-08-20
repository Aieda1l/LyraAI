# evaluate.py
# Modified for LyraAI to evaluate word-level alignment against TTML ground truth.

import json
from pathlib import Path
import warnings
from typing import List, Dict
from xml.etree import ElementTree

import numpy as np
import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import track

from modules.utils import label  # We still use this for parsing TextGrid files

CONSOLE = Console()


class WordBoundaryMetrics:
    """
    Calculates and stores metrics for word boundary alignment accuracy.
    """

    def __init__(self, tolerances: List[float] = [0.05, 0.1, 0.2]):
        """
        Initializes the metrics tracker.

        Args:
            tolerances (List[float]): A list of time tolerances in seconds for calculating
                                      Boundary Error Rate (BER).
        """
        self.tolerances = sorted(tolerances)
        self.total_word_count = 0
        self.absolute_errors = []
        self.boundary_hits = {tol: 0 for tol in self.tolerances}

    def update(self, pred_timings: List[Dict], target_timings: List[Dict]):
        """
        Updates metrics with data from a single file.

        Args:
            pred_timings (List[Dict]): List of {'word': str, 'start': float, 'end': float} from the model.
            target_timings (List[Dict]): List of {'word': str, 'start': float, 'end': float} from TTML.
        """
        # Simple word sequence matching. A more advanced version could use sequence alignment.
        if [p['word'] for p in pred_timings] != [t['word'] for t in target_timings]:
            warnings.warn(
                f"Word sequence mismatch. Skipping file. "
                f"Pred: '{' '.join([p['word'] for p in pred_timings][:5])}...', "
                f"Target: '{' '.join([t['word'] for t in target_timings][:5])}...'"
            )
            return

        num_words = len(target_timings)
        self.total_word_count += num_words

        for i in range(num_words):
            start_error = abs(pred_timings[i]['start'] - target_timings[i]['start'])
            end_error = abs(pred_timings[i]['end'] - target_timings[i]['end'])

            self.absolute_errors.extend([start_error, end_error])

            for tol in self.tolerances:
                if start_error <= tol:
                    self.boundary_hits[tol] += 1
                if end_error <= tol:
                    self.boundary_hits[tol] += 1

    def compute(self) -> Dict:
        """
        Computes the final metrics across all updated files.
        """
        if self.total_word_count == 0:
            return {}

        total_boundaries = self.total_word_count * 2

        mean_absolute_error = np.mean(self.absolute_errors) if self.absolute_errors else 0.0

        boundary_error_rates = {
            f"BER_{int(tol * 1000)}ms": 1.0 - (self.boundary_hits[tol] / total_boundaries)
            for tol in self.tolerances
        }

        return {
            "Mean_Absolute_Error_seconds": round(mean_absolute_error, 4),
            **boundary_error_rates,
            "Total_Words_Evaluated": self.total_word_count,
            "Total_Boundaries_Evaluated": total_boundaries
        }


def parse_prediction_textgrid(pred_file: Path) -> List[Dict] | None:
    """Parses the 'words' tier from a predicted TextGrid file."""
    try:
        textgrid = label.textgrid_from_file(pred_file)
        # The original script converts IntervalTiers to PointTiers, so we reverse it.
        word_tier_points = next((tier for tier in textgrid if tier.name == 'words'), None)
        if not word_tier_points:
            return None

        word_tier_intervals = label.point_tier_to_interval_tier(word_tier_points)

        timings = []
        for interval in word_tier_intervals:
            word = interval.mark.strip()
            if word:  # Ignore empty intervals
                timings.append({'word': word, 'start': interval.minTime, 'end': interval.maxTime})
        return timings
    except Exception as e:
        warnings.warn(f"Could not parse prediction file {pred_file.name}: {e}")
        return None


def parse_target_ttml(target_file: Path) -> List[Dict] | None:
    """Parses word timings from a ground-truth TTML file."""
    try:
        tree = ElementTree.parse(target_file)
        root = tree.getroot()
        ns = {'ttml': 'http://www.w3.org/ns/ttml'}
        timings = []
        for span in root.findall('.//ttml:span', ns):
            word = (span.text or "").strip()
            start_str = span.attrib.get('begin')
            end_str = span.attrib.get('end')
            if word and start_str and end_str:
                timings.append({'word': word, 'start': float(start_str), 'end': float(end_str)})
        return timings
    except Exception as e:
        warnings.warn(f"Could not parse target file {target_file.name}: {e}")
        return None


@click.command(help="Calculate word-level alignment metrics between LyraAI predictions and TTML ground truth.")
@click.argument("pred_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True))
@click.argument("target_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True))
def main(pred_dir: str, target_dir: str):
    """Main evaluation function."""
    pred_path = Path(pred_dir)
    target_path = Path(target_dir)  # This should be the 'data/raw/lyrics' directory

    CONSOLE.print(Panel("[bold magenta]LyraAI Word-Level Alignment Evaluation[/bold magenta]", border_style="magenta"))
    CONSOLE.print(f"Comparing predictions in: [cyan]{pred_path}[/cyan]")
    CONSOLE.print(f"Against ground truth in: [cyan]{target_path}[/cyan]")

    metrics = WordBoundaryMetrics(tolerances=[0.05, 0.1, 0.2, 0.3, 0.5])  # 50ms, 100ms, etc.

    # We look for TextGrid files in the prediction directory
    pred_files = list(pred_path.rglob("*.TextGrid"))
    if not pred_files:
        CONSOLE.print("[bold red]Error:[/bold red] No prediction (.TextGrid) files found in the specified directory.")
        return

    for pred_file in track(pred_files, description="Evaluating files..."):
        # Find the matching ground truth TTML file
        target_file = target_path / f"{pred_file.stem}.ttml"
        if not target_file.exists():
            warnings.warn(f'Prediction file "{pred_file.name}" has no matching TTML target file.')
            continue

        pred_timings = parse_prediction_textgrid(pred_file)
        target_timings = parse_target_ttml(target_file)

        if pred_timings and target_timings:
            metrics.update(pred_timings, target_timings)

    # Compute and display final results
    results = metrics.compute()
    if not results:
        CONSOLE.print("[bold red]Error:[/bold red] No valid files could be compared.")
        return

    results_str = json.dumps(results, indent=4)
    CONSOLE.print(Panel(
        results_str,
        title="[bold blue]Evaluation Results[/bold blue]",
        border_style="blue"
    ))


if __name__ == "__main__":
    main()