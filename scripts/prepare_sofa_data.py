# scripts/prepare_sofa_data.py

import re
import pandas as pd
from pathlib import Path
from xml.etree import ElementTree
from concurrent.futures import ProcessPoolExecutor, as_completed

from pydub import AudioSegment
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from modules.g2p.ollama_g2p import OllamaG2P

# --- Configuration ---
# These paths should align with your dataset_config.yaml
RAW_AUDIO_DIR = Path("data/raw/audio")
RAW_LYRICS_DIR = Path("data/raw/lyrics")
SOFA_PREPARED_DIR = Path("data/sofa_prepared/full_label/AppleMusic")
SOFA_WAVS_DIR = SOFA_PREPARED_DIR / "wavs"
TRANSCRIPTIONS_CSV_PATH = SOFA_PREPARED_DIR / "transcriptions.csv"
OLLAMA_MODEL = 'deepseek-coder'  # The LLM model to use for G2P

# SOFA's expected audio format
TARGET_SAMPLE_RATE = 44100
TARGET_CHANNELS = 1  # Mono

CONSOLE = Console()


def parse_ttml(ttml_path: Path) -> tuple[str, str, list[dict]] | None:
    """
    Parses a TTML file to extract language, full text, and word timings.

    Returns:
        A tuple (language, full_text, word_timings) or None if parsing fails.
        word_timings is a list of dicts: [{'word': str, 'start': float, 'end': float}]
    """
    try:
        tree = ElementTree.parse(ttml_path)
        root = tree.getroot()

        # Extract language code from the root element's xml:lang attribute
        lang_code = root.get('{http://www.w3.org/XML/1998/namespace}lang', 'en')

        # Map Apple's language codes to a more general name for the LLM
        lang_map = {
            'en': 'English',
            'ko': 'Korean',
            'zh-Hant': 'Traditional Chinese',
            'zh-Hans': 'Simplified Chinese',
            'ja': 'Japanese'
        }
        language = lang_map.get(lang_code, lang_code)  # Default to the code if not in map

        word_timings = []
        full_text_list = []

        # Namespace for TTML parsing
        ns = {'ttml': 'http://www.w3.org/ns/ttml'}

        for span in root.findall('.//ttml:span', ns):
            word = (span.text or "").strip()
            start_str = span.attrib.get('begin')
            end_str = span.attrib.get('end')

            if word and start_str and end_str:
                word_timings.append({
                    'word': word,
                    'start': float(start_str),
                    'end': float(end_str)
                })
                full_text_list.append(word)

        if not word_timings:
            return None

        return language, " ".join(full_text_list), word_timings

    except ElementTree.ParseError as e:
        CONSOLE.log(f"[yellow]Warning:[/yellow] Could not parse TTML file {ttml_path.name}: {e}")
        return None


def process_file(ttml_path: Path, g2p_converter: OllamaG2P) -> dict | None:
    """
    Processes a single TTML file to generate a row for the transcriptions.csv.
    """
    file_stem = ttml_path.stem

    # 1. Check for corresponding audio file
    m4a_path = RAW_AUDIO_DIR / f"{file_stem}.m4a"
    if not m4a_path.exists():
        return None  # Skip if no audio

    # 2. Parse TTML for lyrics, language, and timings
    parsed_data = parse_ttml(ttml_path)
    if not parsed_data:
        return None
    language, full_text, word_timings = parsed_data

    # 3. Convert text to IPA phonemes using the LLM
    try:
        ph_seq_list, word_seq_list, _ = g2p_converter._g2p(full_text, lang=language)
    except Exception as e:
        CONSOLE.log(f"[red]Error during G2P for {file_stem}: {e}[/red]")
        return None

    # We need a clean word-to-phoneme mapping from the G2P result
    # The G2P output `word_seq_list` is the source of truth for words now.
    word_to_phonemes = {}
    current_word_phonemes = []
    word_iter = iter(word_seq_list)
    current_word = next(word_iter, None)

    # Reconstruct word-to-phoneme mapping from the flat phoneme list
    for ph in ph_seq_list:
        if ph == 'SP':
            if current_word and current_word_phonemes:
                word_to_phonemes[current_word] = current_word_phonemes
                current_word_phonemes = []
                current_word = next(word_iter, None)
        else:
            current_word_phonemes.append(ph)

    # 4. Calculate phoneme durations
    final_phonemes = []
    final_durations = []
    last_end_time = 0.0

    for timing_info in word_timings:
        word = timing_info['word']
        start_time = timing_info['start']
        end_time = timing_info['end']

        # Add SP duration for gaps between words
        if start_time > last_end_time:
            final_phonemes.append("SP")
            final_durations.append(round(start_time - last_end_time, 6))

        phonemes_for_word = word_to_phonemes.get(word)
        if phonemes_for_word:
            word_duration = end_time - start_time
            if word_duration > 0 and len(phonemes_for_word) > 0:
                # Distribute duration equally among the word's phonemes
                duration_per_phoneme = round(word_duration / len(phonemes_for_word), 6)
                final_phonemes.extend(phonemes_for_word)
                final_durations.extend([duration_per_phoneme] * len(phonemes_for_word))

        last_end_time = end_time

    if not final_phonemes:
        CONSOLE.log(f"[yellow]Warning:[/yellow] No valid phonemes generated for {file_stem}.")
        return None

    # 5. Convert audio to WAV format for SOFA
    wav_path = SOFA_WAVS_DIR / f"{file_stem}.wav"
    try:
        audio = AudioSegment.from_file(m4a_path, format="m4a")
        audio = audio.set_frame_rate(TARGET_SAMPLE_RATE).set_channels(TARGET_CHANNELS)
        audio.export(wav_path, format="wav")
    except Exception as e:
        CONSOLE.log(f"[red]Error converting audio for {file_stem}: {e}[/red]")
        return None

    return {
        "name": file_stem,
        "ph_seq": " ".join(final_phonemes),
        "ph_dur": " ".join(map(str, final_durations))
    }


def main():
    """Main function to orchestrate the data preparation process."""
    CONSOLE.print(Panel("[bold magenta]LyraAI Data Preparation for SOFA[/bold magenta]", border_style="magenta"))

    # Create necessary directories
    SOFA_WAVS_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize G2P converter
    try:
        g2p_converter = OllamaG2P(model=OLLAMA_MODEL)
    except Exception:
        return  # Error is handled in the G2P module constructor

    ttml_files = sorted(list(RAW_LYRICS_DIR.glob("*.ttml")))
    if not ttml_files:
        CONSOLE.print("[bold red]Error:[/bold red] No .ttml files found in 'data/raw/lyrics'.")
        CONSOLE.print("Please run the 'scripts/download_dataset.py' script first.")
        return

    all_results = []

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[bold green]{task.completed} / {task.total}"),
        TimeElapsedColumn(),
        console=CONSOLE
    )

    with progress:
        task = progress.add_task("[cyan]Processing files...", total=len(ttml_files))
        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor() as executor:
            future_to_ttml = {executor.submit(process_file, ttml, g2p_converter): ttml for ttml in ttml_files}

            for future in as_completed(future_to_ttml):
                result = future.result()
                if result:
                    all_results.append(result)
                progress.update(task, advance=1)

    if not all_results:
        CONSOLE.print("[bold red]Error:[/bold red] Failed to process any files. No data was generated.")
        return

    # Create and save the final DataFrame
    df = pd.DataFrame(all_results)
    df.to_csv(TRANSCRIPTIONS_CSV_PATH, index=False)

    CONSOLE.print(Panel(
        f"[bold green]✅ Success![/bold green]\n"
        f"Processed [cyan]{len(all_results)}[/cyan] files.\n"
        f"WAV files saved to: [magenta]{SOFA_WAVS_DIR}[/magenta]\n"
        f"Transcription data saved to: [magenta]{TRANSCRIPTIONS_CSV_PATH}[/magenta]",
        title="Preparation Complete",
        border_style="green"
    ))


if __name__ == "__main__":
    main()