import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from xml.etree import ElementTree

import google.generativeai as genai
from google.generativeai import types
from langdetect import detect, LangDetectException
from pydantic import BaseModel, Field, ValidationError

# Import project-specific configurations and utilities
import config
from utils.console_utils import console, print_panel, print_success, print_error, print_warning


# --- Pydantic model for structured Gemini output ---
# This forces Gemini to return JSON in a predictable format, which is crucial for reliability.
class PhonemeResponse(BaseModel):
    phonemes: list[str] = Field(description="A list of phonemes in the International Phonetic Alphabet (IPA).")


def configure_gemini():
    """Configures the Gemini client with the API key from the config file."""
    try:
        genai.configure(api_key=config.GEMINI_API_KEY)
        return True
    except Exception as e:
        print_error(f"Failed to configure Gemini API: {e}")
        print_warning("Please ensure your GEMINI_API_KEY is correct in the .env file.")
        return False


def parse_ttml(ttml_path: Path) -> list[dict] | None:
    """
    Parses a TTML file to extract a list of syllables/characters with their start and end times.
    """
    try:
        tree = ElementTree.parse(ttml_path)
        root = tree.getroot()
        # Namespace is often present in TTML files
        ns = {'ttml': 'http://www.w3.org/ns/ttml'}

        syllables = []
        # Find all paragraph <p> tags, which usually contain lines of lyrics
        for p in root.findall('.//ttml:p', ns):
            # Find all span <span> tags within a paragraph, which often contain syllables
            for span in p.findall('.//ttml:span', ns):
                start_time_str = span.attrib.get('begin')
                end_time_str = span.attrib.get('end')
                text = (span.text or "").strip()

                if text and start_time_str and end_time_str:
                    # Robustly parse different time formats (e.g., '12.345s', '00:01:02.345')
                    syllables.append({
                        'text': text,
                        'start_time': _parse_time(start_time_str),
                        'end_time': _parse_time(end_time_str),
                    })
        return syllables
    except ElementTree.ParseError:
        print_warning(f"Could not parse XML for {ttml_path.name}")
        return None
    except Exception as e:
        print_error(f"An unexpected error occurred while parsing {ttml_path.name}: {e}")
        return None


def _parse_time(time_str: str) -> float:
    """Helper function to parse various TTML time formats into seconds."""
    if 's' in time_str:
        return float(time_str.replace('s', ''))
    parts = time_str.split(':')
    total_seconds = 0.0
    if len(parts) == 3:  # HH:MM:SS.ms
        h, m, s = parts
        total_seconds = int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:  # MM:SS.ms
        m, s = parts
        total_seconds = int(m) * 60 + float(s)
    elif len(parts) == 1:  # SS.ms
        total_seconds = float(parts[0])
    return total_seconds


def get_phonemes_from_gemini(word: str, language: str, model) -> list[str] | None:
    """
    Sends a request to the Gemini API to get the IPA phonemes for a word.
    Retries on failure.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            prompt = config.GEMINI_G2P_PROMPT_TEMPLATE.format(word=word, language=language)
            response = model.generate_content(
                contents=prompt,
                generation_config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=PhonemeResponse,
                    temperature=0.0  # We want deterministic output
                )
            )
            # The .parsed attribute automatically validates and instantiates the Pydantic model
            parsed_response: PhonemeResponse = response.parsed
            return parsed_response.phonemes
        except (ValidationError, AttributeError) as e:
            print_warning(
                f"Gemini output validation failed for '{word}' (Attempt {attempt + 1}): {e}. Gemini response: {response.text}")
        except Exception as e:
            print_warning(f"Gemini API call failed for '{word}' (Attempt {attempt + 1}): {e}")

        time.sleep(2 ** attempt)  # Exponential backoff

    print_error(f"Failed to get phonemes for '{word}' after {max_retries} attempts.")
    return None


def process_lyrics_file(ttml_path: Path, model, lang_cache: dict) -> tuple[str, list | None]:
    """
    Worker function to process a single lyrics file:
    1. Parses TTML.
    2. Detects language.
    3. Converts each syllable/word to phonemes using Gemini.
    4. Returns the processed data structure.
    """
    song_stem = ttml_path.stem
    output_path = config.PHONEME_ANNOTATION_DIR / f"{song_stem}.json"
    if output_path.exists():
        return song_stem, "skipped"

    syllables = parse_ttml(ttml_path)
    if not syllables:
        return song_stem, None

    # Detect language from the first few syllables for efficiency
    try:
        sample_text = " ".join([s['text'] for s in syllables[:10]])
        # Use a simple cache to avoid re-detecting language for songs from the same album/artist
        lang_code = lang_cache.get(song_stem.split(' - ')[0])
        if not lang_code:
            lang_code = detect(sample_text)
            lang_cache[song_stem.split(' - ')[0]] = lang_code

        # Map detected lang code (e.g., 'ja') to full name (e.g., 'Japanese')
        language_name = next((name for name, code in config.LANGUAGES.items() if code == lang_code), "Unknown")
        if language_name == "Unknown":
            return song_stem, None

    except LangDetectException:
        return song_stem, None

    phoneme_annotations = []
    for syl in syllables:
        word = syl['text']
        phonemes = get_phonemes_from_gemini(word, language_name, model)
        if phonemes:
            phoneme_annotations.append({
                "text": word,
                "start_time": syl['start_time'],
                "end_time": syl['end_time'],
                "phonemes": phonemes
            })
        else:
            # If a single word fails, we fail the whole song to ensure data integrity
            return song_stem, None

    # Save the final processed data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(phoneme_annotations, f, indent=2, ensure_ascii=False)

    return song_stem, phoneme_annotations


def run_phonemizer():
    """
    Main function to orchestrate the phonemization of all downloaded lyrics files.
    """
    if not configure_gemini():
        return

    config.PHONEME_ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

    print_panel("Starting Phonemization Process", style="magenta")
    console.print(f"Input TTML directory:  '{config.RAW_LYRICS_DIR}'")
    console.print(f"Output JSON directory: '{config.PHONEME_ANNOTATION_DIR}'")

    ttml_files = list(config.RAW_LYRICS_DIR.glob("*.ttml"))
    if not ttml_files:
        print_error("No TTML files found in the raw lyrics directory. Run the downloader first.")
        return

    model = genai.GenerativeModel(config.GEMINI_MODEL)
    successful_files, failed_files, skipped_files = 0, 0, 0
    lang_cache = {}  # Simple cache for language detection

    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TextColumn(
                "• [green]{task.completed} ✓[/green] [yellow]{task.fields[skipped]} ⏭[/yellow] [red]{task.fields[failed]} ✗[/red]"),
            transient=False
    ) as progress:
        task = progress.add_task(
            "[magenta]Converting lyrics to phonemes...",
            total=len(ttml_files),
            skipped=0,
            failed=0
        )

        with ThreadPoolExecutor(max_workers=16) as executor:
            future_to_file = {executor.submit(process_lyrics_file, ttml_file, model, lang_cache): ttml_file for
                              ttml_file in ttml_files}

            for future in as_completed(future_to_file):
                song_stem, result = future.result()
                if result == "skipped":
                    skipped_files += 1
                elif result is not None:
                    successful_files += 1
                else:
                    failed_files += 1
                    console.print(f"❌ [red]Failed processing:[/] {song_stem}")

                progress.update(task, advance=1, skipped=skipped_files, failed=failed_files)

    summary = (
        f"\n"
        f"  [bold green]Successful[/]: {successful_files}\n"
        f"  [bold yellow]Skipped[/]:    {skipped_files} (Already processed)\n"
        f"  [bold red]Failed[/]:      {failed_files}"
    )
    print_panel(summary, title="[bold magenta]Phonemization Summary[/bold magenta]", style="magenta")


if __name__ == '__main__':
    run_phonemizer()