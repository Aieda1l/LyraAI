# scripts/download_dataset.py

import json
import random
import traceback
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

# Use the module path for the downloader library
from modules.applemusic_dl import (
    AppleMusicApi,
    Downloader,
    DownloaderSong,
    DownloaderSongLegacy,
    SongCodec,
    SyncedLyricsFormat,
)

CONSOLE = Console()


def load_config(config_path: Path) -> dict:
    """Loads the dataset configuration from a YAML file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        CONSOLE.print(f"[bold red]Error:[/bold red] Configuration file not found at '{config_path}'.")
        exit(1)
    except Exception as e:
        CONSOLE.print(f"[bold red]Error:[/bold red] Failed to parse configuration file: {e}")
        exit(1)


def create_directories(config: dict):
    """Creates all necessary output directories defined in the config."""
    Path(config["RAW_AUDIO_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(config["RAW_LYRICS_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(config["RAW_META_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(config["TEMP_PATH_BASE"]).mkdir(parents=True, exist_ok=True)
    CONSOLE.print("✅ [green]All necessary directories are created.[/green]")


def validate_config(config: dict):
    """Validates the presence of essential configuration keys."""
    if not Path(config["COOKIES_PATH"]).exists():
        CONSOLE.print(f"[bold red]Error:[/bold red] cookies.txt file not found at '{config['COOKIES_PATH']}'.")
        CONSOLE.print("Please export your cookies from music.apple.com and place the file in the root directory.")
        exit(1)
    if config["DOWNLOAD_MODE"] != 'playlists':
        CONSOLE.print(
            f"[bold red]Error:[/bold red] Invalid DOWNLOAD_MODE in config. For LyraAI, please use 'playlists'.")
        exit(1)


def fetch_song_urls(api: AppleMusicApi, config: dict) -> list[str]:
    """Fetches song URLs from the playlists defined in the config."""
    song_urls = set()
    CONSOLE.print(
        Panel(f"Fetching song URLs using [cyan]{config['DOWNLOAD_MODE']}[/cyan] mode...",
              title="[bold blue]Step 1: Song Discovery[/bold blue]",
              border_style="blue")
    )

    playlist_urls = config.get('PLAYLIST_URLS', [])
    if not playlist_urls:
        CONSOLE.print("[bold red]Error:[/bold red] No playlists found in the configuration file.")
        return []

    CONSOLE.print(f"-> Reading [magenta]{len(playlist_urls)}[/magenta] playlist(s) from config.")
    for url in playlist_urls:
        try:
            playlist_id = url.split('/')[-1].split('?')[0]
            playlist_data = api.get_playlist(playlist_id)
            playlist_name = playlist_data['attributes']['name']
            CONSOLE.print(f"  -> Fetching tracks from playlist: [cyan]{playlist_name}[/cyan]")
            tracks = playlist_data['relationships']['tracks']
            for track in tracks.get("data", []):
                if track["attributes"].get("url"):
                    song_urls.add(track["attributes"]["url"])
        except Exception as e:
            CONSOLE.print(f"[yellow]Warning:[/yellow] Could not fetch playlist '{url}': {e}")

    if not song_urls:
        CONSOLE.print("[bold red]Error:[/bold red] Could not find any songs to download from the provided playlists.")
        return []

    shuffled_urls = list(song_urls)
    random.shuffle(shuffled_urls)

    CONSOLE.print(f"[bold green]Success:[/bold green] Found [cyan]{len(shuffled_urls)}[/cyan] unique songs to process.")
    return shuffled_urls


def download_worker(song_url: str, api: AppleMusicApi, config: dict) -> tuple[bool, str]:
    """
    Worker function to download a single song, its TTML lyrics, and metadata.
    """
    raw_audio_dir = Path(config["RAW_AUDIO_DIR"])
    raw_lyrics_dir = Path(config["RAW_LYRICS_DIR"])
    raw_meta_dir = Path(config["RAW_META_DIR"])
    temp_path_base = Path(config["TEMP_PATH_BASE"])
    lyrics_format = SyncedLyricsFormat[config["LYRICS_FORMAT"]]

    with Downloader(
            apple_music_api=api,
            output_path=raw_audio_dir,
            temp_path_base=temp_path_base,
            silent=True
    ) as downloader:
        try:
            downloader.set_cdm()
            downloader_song = DownloaderSong(
                downloader=downloader,
                codec=SongCodec.AAC_LEGACY,
                synced_lyrics_format=lyrics_format,
            )
            downloader_song_legacy = DownloaderSongLegacy(
                downloader=downloader,
                codec=SongCodec.AAC_LEGACY,
            )

            url_info = downloader.get_url_info(song_url)
            queue = downloader.get_download_queue(url_info)
            if not queue.medias_metadata:
                return False, "Failed to get media metadata."

            media_metadata = queue.medias_metadata[0]
            track_id = downloader.get_media_id(media_metadata)

            # Specifically fetch syllable/word-timed TTML lyrics
            lyrics = downloader_song.get_syllable_lyrics_only(track_id, media_metadata)
            if not lyrics or not lyrics.synced:
                return False, "No word/syllable-synced TTML lyrics available."

            webplayback = api.get_webplayback(track_id)
            tags = downloader_song.get_tags(webplayback, lyrics.unsynced if lyrics else None)

            sanitized_stem = downloader.get_sanitized_string(f"{tags['artist']} - {tags['title']}", is_folder=False)
            final_audio_path = raw_audio_dir / f"{sanitized_stem}.m4a"

            if final_audio_path.exists():
                return False, "Song already exists."

            stream_info = downloader_song_legacy.get_stream_info(webplayback)
            decryption_key = downloader_song_legacy.get_decryption_key(
                stream_info.audio_track.widevine_pssh, track_id
            )
            encrypted_path = downloader_song.get_encrypted_path(track_id)
            decrypted_path = downloader_song.get_decrypted_path(track_id)
            remuxed_path = downloader_song.get_remuxed_path(track_id, stream_info.file_format)

            downloader.download(encrypted_path, stream_info.audio_track.stream_url)
            downloader_song_legacy.remux(encrypted_path, decrypted_path, remuxed_path, decryption_key)

            downloader.apply_tags(remuxed_path, tags)
            downloader.move_to_output_path(remuxed_path, final_audio_path)

            lyrics_path = raw_lyrics_dir / f"{sanitized_stem}.ttml"
            downloader_song.save_lyrics_synced(lyrics_path, lyrics.synced)

            meta_path = raw_meta_dir / f"{sanitized_stem}.json"
            meta_data = {
                "song_url": song_url,
                "track_id": track_id,
                "title": tags['title'],
                "artist": tags['artist'],
                "album": tags['album'],
                "genre": tags.get('genre', 'Unknown')
            }
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta_data, f, indent=4)

            return True, sanitized_stem

        except Exception:
            tb_str = traceback.format_exc()
            return False, f"A worker process failed unexpectedly. Traceback:\n{tb_str}"


def run_dataset_downloader(config_path: str):
    """Orchestrates the entire data acquisition process for the LyraAI dataset."""
    config = load_config(Path(config_path))

    CONSOLE.print(Panel(f"Welcome to the [bold magenta]{config['PROJECT_NAME']}[/bold magenta] Dataset Downloader!",
                        border_style="magenta"))

    create_directories(config)
    validate_config(config)

    try:
        api = AppleMusicApi.from_netscape_cookies(cookies_path=Path(config["COOKIES_PATH"]))
    except Exception as e:
        CONSOLE.print(f"[bold red]Error:[/bold red] Failed to initialize Apple Music API: {e}")
        return

    all_urls = fetch_song_urls(api, config)
    if not all_urls:
        return

    target_downloads = len(all_urls)
    successful_downloads, failed_downloads, skipped_downloads = 0, 0, 0

    CONSOLE.print(
        Panel(f"Starting download of up to [cyan]{target_downloads}[/cyan] songs.",
              title="[bold blue]Step 2: Download Phase[/bold blue]",
              border_style="blue")
    )

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("([green]{task.completed}[/green]/"
                   "[cyan]{task.total}[/cyan])"),
        TimeElapsedColumn(),
        console=CONSOLE,
        transient=False,
    )

    with progress:
        task = progress.add_task("[green]Downloading songs...", total=target_downloads)

        with ThreadPoolExecutor(max_workers=16) as executor:
            future_to_url = {executor.submit(download_worker, url, api, config): url for url in all_urls}

            for future in as_completed(future_to_url):
                try:
                    success, message = future.result()
                    if success:
                        successful_downloads += 1
                        progress.console.print(f"✅ [green]Success:[/green] {message}")
                    else:
                        if "already exists" in message or "No word/syllable-synced" in message:
                            skipped_downloads += 1
                        else:
                            failed_downloads += 1
                            progress.console.print(
                                f"❌ [yellow]Failed:[/yellow] {Path(future_to_url[future]).name} | Reason: {message.splitlines()[0][:150]}")
                except Exception as e:
                    failed_downloads += 1
                    progress.console.print(
                        f"❌ [bold red]Error:[/bold red] Worker crashed for {Path(future_to_url[future]).name}: {e}")

                progress.update(task, advance=1)

    total_processed = successful_downloads + failed_downloads + skipped_downloads
    summary_text = (
        f"[bold green]Successful: {successful_downloads}[/bold green]\n"
        f"[bold yellow]Skipped (exists/no lyrics): {skipped_downloads}[/bold yellow]\n"
        f"[bold red]Failed (errors): {failed_downloads}[/bold red]\n"
        f"--------------------------\n"
        f"Total Processed: {total_processed}"
    )
    CONSOLE.print(Panel(summary_text, title="[bold blue]Download Complete![/bold blue]", border_style="blue"))


if __name__ == '__main__':
    # The script is now run by providing the path to the config file.
    # Example: python scripts/download_dataset.py
    import argparse

    parser = argparse.ArgumentParser(description="LyraAI Dataset Downloader")
    parser.add_argument(
        '-c', '--config',
        type=str,
        default='configs/dataset_config.yaml',
        help='Path to the dataset configuration YAML file.'
    )
    args = parser.parse_args()
    run_dataset_downloader(args.config)