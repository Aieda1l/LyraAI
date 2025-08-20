# lyra_aligner.py

import click
import torch
import torchaudio
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

from modules.task.forced_alignment import LitForcedAlignmentTask
from modules.g2p.ollama_g2p import OllamaG2P

CONSOLE = Console()


def seconds_to_srt_time(seconds: float) -> str:
    """Converts seconds to SRT timestamp format (HH:MM:SS,ms)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds * 1000) % 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def save_srt(output_path: Path, word_intervals: list, word_seq: list):
    """Saves the word-level alignment to an SRT file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, (word, interval) in enumerate(zip(word_seq, word_intervals)):
            if word.upper() in ["SP", "AP"]:  # Skip silence markers
                continue

            start_time = seconds_to_srt_time(interval[0])
            end_time = seconds_to_srt_time(interval[1])

            f.write(f"{i + 1}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{word}\n\n")


@click.command(help="Aligns audio with unsynced lyrics using the LyraAI model.")
@click.option('--audio', '-a', required=True, type=click.Path(exists=True, dir_okay=False),
              help='Path to the input audio file (e.g., .wav, .mp3, .m4a).')
@click.option('--lyrics', '-l', required=True, type=click.Path(exists=True, dir_okay=False),
              help='Path to a plain text file containing the unsynced lyrics.')
@click.option('--lang', '-ln', required=True, type=str,
              help='Language of the lyrics (e.g., English, Japanese, Korean, Chinese).')
@click.option('--ckpt', '-c', required=True, type=click.Path(exists=True, dir_okay=False),
              help='Path to the trained LyraAI model checkpoint (.ckpt).')
@click.option('--output', '-o', type=click.Path(dir_okay=False),
              help='Path to the output SRT file. Defaults to the audio filename with an .srt extension.')
@click.option('--g2p_model', default='deepseek-coder',
              help='The Ollama model to use for Grapheme-to-Phoneme conversion.')
def align(audio: str, lyrics: str, lang: str, ckpt: str, output: str, g2p_model: str):
    """
    Main function for the LyraAI Aligner.
    """
    audio_path = Path(audio)
    lyrics_path = Path(lyrics)
    ckpt_path = Path(ckpt)

    if output:
        output_path = Path(output)
    else:
        output_path = audio_path.with_suffix('.srt')

    output_path.parent.mkdir(parents=True, exist_ok=True)

    CONSOLE.print(Panel("[bold magenta]LyraAI: Word-level Forced Aligner[/bold magenta]", border_style="magenta"))
    CONSOLE.print(f"▶️ [bold]Audio:[/bold] [cyan]{audio_path.name}[/cyan]")
    CONSOLE.print(f"📄 [bold]Lyrics:[/bold] [cyan]{lyrics_path.name}[/cyan]")
    CONSOLE.print(f"🌐 [bold]Language:[/bold] [cyan]{lang}[/cyan]")
    CONSOLE.print(f"🧠 [bold]Model:[/bold] [cyan]{ckpt_path.name}[/cyan]")

    # --- Step 1: Initialize G2P and Model ---
    with CONSOLE.status("[cyan]Initializing models...", spinner="dots"):
        try:
            g2p_converter = OllamaG2P(model=g2p_model)
            model = LitForcedAlignmentTask.load_from_checkpoint(ckpt_path)
            model.set_inference_mode('force')  # Use 'force' for aligning full lyrics
            if torch.cuda.is_available():
                model.cuda()
            model.eval()
        except Exception as e:
            CONSOLE.print(f"\n[bold red]Error during model initialization:[/bold red] {e}")
            return

    # --- Step 2: Convert Lyrics to Phonemes ---
    try:
        with open(lyrics_path, 'r', encoding='utf-8') as f:
            unsynced_text = f.read().replace('\n', ' ').strip()

        ph_seq, word_seq, ph_idx_to_word_idx = g2p_converter(unsynced_text, lang=lang)
        CONSOLE.print("✅ [green]Lyrics converted to phonemes successfully.[/green]")
    except Exception as e:
        CONSOLE.print(f"\n[bold red]Error during G2P conversion:[/bold red] {e}")
        return

    # --- Step 3: Load and Preprocess Audio ---
    with CONSOLE.status("[cyan]Processing audio...", spinner="dots"):
        try:
            # Use torchaudio which handles various formats and resampling
            waveform, sr = torchaudio.load(audio_path)

            # Resample if necessary and convert to mono
            target_sr = model.hparams.melspec_config['sample_rate']
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(sr, target_sr)
                waveform = resampler(waveform)

            if waveform.shape[0] > 1:  # If stereo, convert to mono
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            waveform = waveform.squeeze(0)  # Remove batch dimension if present
            if torch.cuda.is_available():
                waveform = waveform.cuda()

        except Exception as e:
            CONSOLE.print(f"\n[bold red]Error loading or processing audio:[/bold red] {e}")
            return

    # --- Step 4: Perform Alignment ---
    CONSOLE.print("⏳ [bold]Performing alignment...[/bold] (This may take a moment)")
    try:
        with torch.no_grad():
            # The model's predict_step expects a batch-like structure
            batch = (str(audio_path), ph_seq, word_seq, ph_idx_to_word_idx)

            # Manually call the core inference logic from the model
            wav_length = len(waveform) / model.hparams.melspec_config["sample_rate"]
            melspec = model.get_melspec(waveform).detach().unsqueeze(0)
            melspec = (melspec - melspec.mean()) / melspec.std()
            melspec = torch.nn.functional.interpolate(
                melspec,
                scale_factor=(1, model.hparams.melspec_config["scale_factor"]),
                mode='bilinear',
                align_corners=False
            ).squeeze(2)

            (
                pred_ph_seq,
                pred_ph_intervals,
                pred_word_seq,
                pred_word_intervals,
                confidence,
                _,
                _,
            ) = model._infer_once(
                melspec, wav_length, ph_seq, word_seq, ph_idx_to_word_idx
            )
    except Exception as e:
        CONSOLE.print(f"\n[bold red]An error occurred during the alignment process:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Step 5: Save the Output ---
    save_srt(output_path, pred_word_intervals, pred_word_seq)

    CONSOLE.print(Panel(
        f"[bold green]✅ Alignment Complete![/bold green]\n"
        f"Confidence Score: [yellow]{confidence:.4f}[/yellow]\n"
        f"Synced lyrics saved to: [magenta]{output_path}[/magenta]",
        title="Success",
        border_style="green"
    ))


if __name__ == '__main__':
    align()