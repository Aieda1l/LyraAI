# binarize.py
# Adapted from the original SOFA codebase for the LyraAI project.

import pathlib
import warnings
import click
import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from modules.utils.get_melspec import MelSpecExtractor
from modules.utils.load_wav import load_wav

CONSOLE = Console()


class LyraBinarizer:
    def __init__(
            self,
            data_folder,
            valid_set_size,
            valid_set_preferred_folders,
            data_augmentation,
            ignored_phonemes,
            melspec_config,
            max_length,
    ):
        self.data_folder = pathlib.Path(data_folder)
        self.binary_output_dir = self.data_folder / "binary"
        self.valid_set_size = valid_set_size
        self.valid_set_preferred_folders = valid_set_preferred_folders
        self.data_augmentation = data_augmentation
        self.data_augmentation["key_shift_choices"] = np.array(
            self.data_augmentation.get("key_shift_choices", [])
        )
        self.ignored_phonemes = ignored_phonemes
        self.melspec_config = melspec_config
        self.scale_factor = melspec_config["scale_factor"]
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if str(self.device) == "cuda":
            CONSOLE.print("CUDA is available, using GPU for Mel spectrogram extraction.")
        else:
            CONSOLE.print("CUDA not available, using CPU for Mel spectrogram extraction.")

        self.sample_rate = self.melspec_config["sample_rate"]
        self.frame_length = self.melspec_config["hop_length"] / self.sample_rate

        self.get_melspec = MelSpecExtractor(**melspec_config, device=self.device)

    @staticmethod
    def get_vocab(data_folder_path, ignored_phonemes):
        CONSOLE.print("Generating vocabulary from transcriptions...")
        phonemes = set()
        trans_path_list = list(data_folder_path.rglob("transcriptions.csv"))

        if not trans_path_list:
            raise FileNotFoundError(f"No 'transcriptions.csv' found in {data_folder_path} or its subdirectories.")

        for trans_path in trans_path_list:
            df = pd.read_csv(trans_path)
            if "ph_seq" in df.columns:
                all_phonemes = " ".join(df["ph_seq"].dropna()).split(" ")
                phonemes.update(p for p in all_phonemes if p)  # Add non-empty phonemes

        for p in ignored_phonemes:
            if p in phonemes:
                phonemes.remove(p)

        phonemes = sorted(list(phonemes))
        if "SP" not in phonemes:
            phonemes.insert(0, "SP")
        else:  # Ensure SP is always at index 0
            phonemes.remove("SP")
            phonemes.insert(0, "SP")

        vocab = {ph: i for i, ph in enumerate(phonemes)}
        vocab.update({i: ph for i, ph in enumerate(phonemes)})
        vocab.update({p: vocab["SP"] for p in ignored_phonemes})  # Map ignored to SP
        vocab["<vocab_size>"] = len(phonemes)

        CONSOLE.print(f"Vocabulary generated with [cyan]{len(phonemes)}[/cyan] unique phonemes.")
        return vocab

    def process(self):
        self.binary_output_dir.mkdir(parents=True, exist_ok=True)

        vocab = self.get_vocab(self.data_folder, self.ignored_phonemes)
        with open(self.binary_output_dir / "vocab.yaml", "w", encoding="utf-8") as file:
            yaml.dump(vocab, file, allow_unicode=True)

        meta_data_df = self.get_meta_data(self.data_folder, vocab)

        # Split train and valid sets
        meta_data_valid = (
            meta_data_df[meta_data_df["label_type"] != "no_label"]
            .sample(frac=1)
            .sort_values(by="preferred", ascending=False)
            .head(self.valid_set_size)
        )
        meta_data_train = meta_data_df.drop(meta_data_valid.index).reset_index(drop=True)
        meta_data_valid = meta_data_valid.reset_index(drop=True)

        self.binarize_set("valid", meta_data_valid, vocab)
        self.binarize_set("train", meta_data_train, vocab)

    def binarize_set(self, prefix: str, meta_data: pd.DataFrame, vocab: dict):
        h5py_file_path = self.binary_output_dir / f"{prefix}.h5py"

        progress = Progress(
            SpinnerColumn(),
            TextColumn(f"[cyan]Binarizing '{prefix}' set...[/cyan]"),
            BarColumn(),
            TextColumn("[bold green]{task.completed} / {task.total}"),
            TimeElapsedColumn(),
            console=CONSOLE
        )

        with h5py.File(h5py_file_path, "w") as h5py_file:
            h5py_meta_data = h5py_file.create_group("meta_data")
            items_meta_data = {"label_types": [], "wav_lengths": []}
            h5py_items = h5py_file.create_group("items")
            label_type_to_id = {"no_label": 0, "weak_label": 1, "full_label": 2}

            idx = 0
            total_time = 0.0

            with progress:
                task = progress.add_task("Processing items", total=len(meta_data))
                for _, item in meta_data.iterrows():
                    try:
                        waveform = load_wav(item.wav_path, self.device, self.sample_rate)
                        wav_length = len(waveform) / self.sample_rate

                        if wav_length > self.max_length:
                            warnings.warn(f"Item {item.wav_path} is too long ({wav_length:.2f}s), skipping.")
                            progress.update(task, advance=1)
                            continue

                        input_feature = self.get_melspec(waveform).unsqueeze(0)  # Add batch dim
                        input_feature = (input_feature - input_feature.mean()) / input_feature.std()
                        T = input_feature.shape[-1] * self.scale_factor

                        h5py_item_data = h5py_items.create_group(str(idx))
                        h5py_item_data["input_feature"] = input_feature.cpu().numpy().astype(
                            "float16")  # Use float16 to save space

                        label_type_id = label_type_to_id[item.label_type]
                        if label_type_id == 2 and (len(item.ph_dur) != len(item.ph_seq) or not item.ph_seq):
                            label_type_id = 1  # Downgrade to weak label if inconsistent or empty

                        h5py_item_data["label_type"] = label_type_id
                        items_meta_data["label_types"].append(label_type_id)
                        items_meta_data["wav_lengths"].append(wav_length)

                        ph_seq_ids = np.array(item.ph_seq).astype("int32")

                        # ph_mask: [vocab_size]
                        ph_mask = np.zeros(vocab["<vocab_size>"], dtype="bool")
                        ph_mask[ph_seq_ids] = 1
                        ph_mask[vocab["SP"]] = 1  # Always allow SP

                        # ph_edge and ph_frame (for full labels)
                        if label_type_id == 2:
                            ph_dur = np.array(item.ph_dur).astype("float32")
                            ph_time = np.concatenate(([0], ph_dur)).cumsum() / (self.frame_length / self.scale_factor)

                            ph_edge = np.zeros(T, dtype="float32")
                            ph_frame = np.zeros(T, dtype="int32")

                            ph_time_int = np.round(ph_time).astype("int32")

                            for i in range(len(ph_time_int) - 1):
                                ph_id = ph_seq_ids[i]
                                st = ph_time_int[i]
                                ed = ph_time_int[i + 1]
                                if st >= ed: continue  # Skip zero-duration phonemes

                                ph_frame[st:ed] = ph_id

                                if ed < T:
                                    time_frac = ph_time[i + 1] - ph_time_int[i + 1]
                                    ph_edge[ed] = 0.5 + time_frac
                                    if ed > 0:
                                        ph_edge[ed - 1] = 0.5 - time_frac

                            ph_edge = np.clip(ph_edge, 0, 1) * 0.8 + 0.1
                            h5py_item_data["ph_edge"] = ph_edge.astype("float16")
                            h5py_item_data["ph_frame"] = ph_frame.astype("int32")

                        h5py_item_data["ph_seq"] = ph_seq_ids
                        h5py_item_data["ph_mask"] = ph_mask

                        idx += 1
                        total_time += wav_length
                    except Exception as e:
                        warnings.warn(f"Error processing {item.wav_path}: {e}")
                    finally:
                        progress.update(task, advance=1)

            for k, v in items_meta_data.items():
                h5py_meta_data.create_dataset(k, data=np.array(v))

        CONSOLE.print(Panel(
            f"[bold green]✅ Binarization complete for '{prefix}' set.[/bold green]\n"
            f"Processed [cyan]{idx}[/cyan] items.\n"
            f"Total duration: [cyan]{total_time / 3600:.2f} hours[/cyan].\n"
            f"Output file: [magenta]{h5py_file_path}[/magenta]",
            title="Binarization Summary"
        ))

    def get_meta_data(self, data_folder, vocab):
        trans_path = next(data_folder.rglob("transcriptions.csv"), None)
        if trans_path is None:
            raise FileNotFoundError(f"Could not find 'transcriptions.csv' in {data_folder}")

        CONSOLE.print(f"Loading metadata from [magenta]{trans_path}[/magenta]...")
        meta_data_df = pd.read_csv(trans_path, dtype={"name": str})

        # All our data is full_label
        meta_data_df["label_type"] = "full_label"
        meta_data_df["wav_path"] = meta_data_df["name"].apply(lambda name: str(SOFA_WAVS_DIR / f"{name}.wav"))
        meta_data_df["preferred"] = meta_data_df["wav_path"].str.contains("|".join(self.valid_set_preferred_folders))

        # Convert phoneme sequences to integer IDs
        meta_data_df["ph_seq"] = meta_data_df["ph_seq"].apply(
            lambda x: [vocab.get(p, vocab["SP"]) for p in str(x).split()] if pd.notna(x) else []
        )
        meta_data_df["ph_dur"] = meta_data_df["ph_dur"].apply(
            lambda x: [float(d) for d in str(x).split()] if pd.notna(x) else []
        )

        return meta_data_df.dropna(subset=['wav_path']).reset_index(drop=True)


@click.command(help="Binarize the prepared LyraAI dataset for SOFA training.")
@click.option(
    "--config_path",
    "-c",
    type=str,
    default="configs/lyra_binarize_config.yaml",
    show_default=True,
    help="Path to the binarization config YAML file.",
)
def binarize(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Save a global config for the training script to reference
    binary_dir = pathlib.Path(config["data_folder"]) / "binary"
    binary_dir.mkdir(exist_ok=True)
    global_config = {
        "melspec_config": config["melspec_config"],
        "data_augmentation_size": config["data_augmentation"]["size"],
    }
    with open(binary_dir / "global_config.yaml", "w") as file:
        yaml.dump(global_config, file)

    LyraBinarizer(**config).process()


if __name__ == "__main__":
    binarize()