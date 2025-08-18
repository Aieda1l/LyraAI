import os
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console

# --- Load environment variables from .env file ---
# Create a .env file in the root directory and add your GEMINI_API_KEY
load_dotenv()

# ==============================================================================
# CONSOLE & LOGGING CONFIGURATION
# ==============================================================================
# A single rich console instance to be used across the project for consistent styling
console = Console(highlight=False)

# ==============================================================================
# PATHS CONFIGURATION
# ==============================================================================
# Define the root directory of the project
ROOT_DIR = Path(__file__).parent.resolve()

# --- Raw Data Paths (from downloader) ---
DATA_DIR = ROOT_DIR / "dataset"
RAW_AUDIO_DIR = DATA_DIR / "1_raw_audio"
RAW_LYRICS_DIR = DATA_DIR / "2_raw_lyrics_ttml"
RAW_META_DIR = DATA_DIR / "3_raw_meta"
TEMP_PATH_BASE = DATA_DIR / "temp"  # Base directory for temporary download files

# --- Processed Data Paths (for training) ---
PROCESSED_DIR = DATA_DIR / "processed"
PHONEME_ANNOTATION_DIR = PROCESSED_DIR / "phoneme_annotations"
TRAIN_SPLITS_DIR = PROCESSED_DIR / "train_splits"

# --- Model & Output Paths ---
OUTPUT_DIR = ROOT_DIR / "output"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"
PREDICTION_DIR = OUTPUT_DIR / "predictions"

# ==============================================================================
# DATASET DOWNLOAD CONFIGURATION (for dataset_download.py)
# ==============================================================================
# Path to your Netscape cookies file for Apple Music authentication
COOKIES_PATH = ROOT_DIR / "cookies.txt"

# --- Download Strategy ---
# 'playlists': Download all tracks from the specified PLAYLIST_URLS.
# 'genres': Search for playlists matching DATASET_GENRES and download up to NUM_SONGS_TO_DOWNLOAD.
DOWNLOAD_MODE = 'playlists'  # 'playlists' or 'genres'

# List of Apple Music playlist URLs to scrape (only used if DOWNLOAD_MODE is 'playlists')
PLAYLIST_URLS = [
    "https://music.apple.com/us/playlist/haken-anisong/pl.89d598cd24cb4cf89895085a0fa278ec",
    "https://music.apple.com/us/playlist/anime-rewind/pl.434e71c6da7f4dbf8c4e8d130177c809",
    "https://music.apple.com/us/playlist/ado-essentials/pl.a91ae9bfb69543348b54c2ba5fca4bd6",
    "https://music.apple.com/us/playlist/japan-hits-2024/pl.15fc72e800354792b242704defc35828",
    "https://music.apple.com/us/playlist/karaoke-hits/pl.ad831dcfc24f456987b4011c1ddfdd39"
    "https://music.apple.com/us/playlist/the-a-list-mandopop/pl.beb783da7712481fbeed35be144bd48c",
    "https://music.apple.com/us/playlist/10s-mandopop-essentials/pl.3e52664b20ea45c394a37f4a7f3a8451",
    "https://music.apple.com/us/playlist/breaking-mandopop/pl.80e13199d5db46c7b519b25cf6e5816a",
    "https://music.apple.com/us/playlist/sing-mandopop/pl.7094457cee324d6ebb28388ccaeca7f3",
    "https://music.apple.com/us/playlist/viral-c-pop/pl.2a0a202d08c3439e95d22a73126f44170",
    "https://music.apple.com/us/playlist/k-pop-essentials/pl.6a3c854a49a542739e5d57291b27e122",
    "https://music.apple.com/us/playlist/viral-k-pop/pl.5b7698bbcd01407a92b8457f650bebaf",
    "https://music.apple.com/us/playlist/kpopwrld/pl.48229b41bbfc47d7af39dae8e8b5276e",
    "https://music.apple.com/us/playlist/new-in-k-pop/pl.a784f95d4f504d579647523ff95433be",
    "https://music.apple.com/us/playlist/k-pop-hits-2024/pl.b3c079fea5704332baed562d8f90d3a8",
    "https://music.apple.com/us/playlist/a-list-k-pop-2021/pl.454b6b29fa994a93bdc74efbe7a2c465",
    "https://music.apple.com/us/playlist/pop-throwback/pl.c21556629e97453f9672feb9d8f228a3",
    "https://music.apple.com/us/playlist/radio-chart-rock/pl.ca17374227094d6fb2b27a1e6f144c8a",
    "https://music.apple.com/us/playlist/hip-hop-r-b-throwback/pl.674abcd261d04582b58d6388394cd047",
    "https://music.apple.com/us/playlist/hip-hop-hits/pl.87c7af5767764860a0e3368d0bef9a6f",
    "https://music.apple.com/us/playlist/todays-hits/pl.f4d106fed2bd41149aaacabb233eb5eb",
]

# List of genres to search for (only used if DOWNLOAD_MODE is 'genres')
DATASET_GENRES = ["j-pop", "k-pop", "c-pop", "anime"]

# Target number of songs to download (only used if DOWNLOAD_MODE is 'genres')
NUM_SONGS_TO_DOWNLOAD = 2000

# Number of parallel workers for downloading
DOWNLOAD_WORKERS = 16

# The format for synced lyrics. We are targeting syllable-synced TTML.
# The downloader library will convert this to our desired intermediate format.
LYRICS_FORMAT = "ttml"

# ==============================================================================
# PHONEMIZATION CONFIGURATION (using Gemini)
# ==============================================================================
# --- Gemini API ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# The model to use for grapheme-to-phoneme conversion.
# 2.5-flash is fast and cost-effective for this task.
GEMINI_MODEL = "gemini-2.5-flash"

# --- Language Configuration ---
# Define the languages you are targeting and their corresponding codes for G2P
# This helps in constructing the correct prompt for Gemini.
LANGUAGES = {
    "Japanese": "ja",
    "Chinese": "zh",
    "Korean": "ko",
    "English": "en",
}

# --- Prompting ---
# The system instruction tells Gemini its role and the expected output format.
# A structured JSON output is crucial for reliable parsing.
GEMINI_G2P_PROMPT_TEMPLATE = """
You are a linguistic expert specializing in phonetics. Your task is to convert a given word from a specified language into its International Phonetic Alphabet (IPA) representation.

Provide the output as a JSON object with a single key "phonemes", which contains a list of strings, where each string is a single phoneme.

Word: "{word}"
Language: "{language}"
"""

# ==============================================================================
# DATA PROCESSING & AUDIO CONFIGURATION
# ==============================================================================
# Target sampling rate for all audio files. The original model used 22050.
SR = 22050

# The length of audio segments (in samples) fed into the model during training.
# Default from the paper is 5.6 seconds * 22050 Hz = 123904
INPUT_SAMPLE_LENGTH = 123904

# Ratios for splitting the dataset into training, validation, and testing sets.
TRAIN_SPLIT_RATIO = 0.9
VALIDATION_SPLIT_RATIO = 0.05
# TEST_SPLIT_RATIO is the remainder (0.05)

# ==============================================================================
# MODEL & TRAINING HYPERPARAMETERS
# ==============================================================================
# --- Model Architecture ---
# 'baseline': Standard phoneme recognition model.
# 'MTL': Multi-Task Learning model with joint pitch detection.
MODEL_TYPE = 'MTL'

# Dimensions of the acoustic model
CNN_LAYERS = 1
RNN_DIM = 256

# --- Training Hyperparameters ---
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
NUM_WORKERS = 4  # Number of threads for the data loader
EARLY_STOPPING_PATIENCE = 20  # Stop training if validation loss doesn't improve for this many epochs

# Weight for the melody loss in the Multi-Task Learning (MTL) model
LOSS_WEIGHT_MELODY = 0.5 # Increased from 0.1 for potentially stronger regularization

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
def create_directories():
    """Creates all necessary directories defined in the config."""
    console.print("[bold cyan]Initializing project directories...[/bold cyan]")
    paths_to_create = [
        RAW_AUDIO_DIR, RAW_LYRICS_DIR, RAW_META_DIR, TEMP_PATH_BASE,
        PHONEME_ANNOTATION_DIR, TRAIN_SPLITS_DIR, CHECKPOINT_DIR,
        LOG_DIR, PREDICTION_DIR
    ]
    for path in paths_to_create:
        path.mkdir(parents=True, exist_ok=True)
    console.print("[bold green]...directories created successfully.[/bold green]")

def validate_config():
    """Validates critical configuration settings before running a script."""
    console.print("[bold cyan]Validating configuration...[/bold cyan]")
    if not GEMINI_API_KEY:
        console.print("[bold red]ERROR: GEMINI_API_KEY is not set.[/bold red]")
        console.print("Please create a '.env' file in the root directory and add your key:")
        console.print("GEMINI_API_KEY='Your-API-Key-Here'")
        exit(1)
    if not COOKIES_PATH.exists():
        console.print(f"[bold red]ERROR: Apple Music cookies file not found at '{COOKIES_PATH}'[/bold red]")
        console.print("Please make sure the cookies.txt file is in the project root.")
        exit(1)
    console.print("[bold green]...configuration validated.[/bold green]")