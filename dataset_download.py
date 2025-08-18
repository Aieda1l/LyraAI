import json
import random
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

# Import project-specific configurations and utilities
import config
from applemusic_dl import (
    AppleMusicApi,
    Downloader,
    DownloaderSong,
    DownloaderSongLegacy,
    SongCodec,
    SyncedLyricsFormat
)
from utils.console_utils import console, print_panel, print_success, print_error, print_warning


def fetch_song_urls(api: AppleMusicApi) -> list[str]:
    """
    Fetches song URLs based on the DOWNLOAD_MODE in the config, using the provided API client.
    """
    song_urls = set()

    if config.DOWNLOAD_MODE == 'playlists':
        print_panel(f"Fetching all songs from {len(config.PLAYLIST_URLS)} specified playlist(s)...",
                    title="[bold blue]Playlist Mode[/bold blue]", style="blue")
        for url in config.PLAYLIST_URLS:
            try:
                # Extract playlist ID from URL
                playlist_id = url.split('/')[-1].split('?')[0]
                playlist_data = api.get_playlist(playlist_id, fetch_all=True) # Ensure all tracks are fetched
                playlist_name = playlist_data['attributes']['name']
                console.print(f"  -> Fetching tracks from playlist: [cyan]{playlist_name}[/cyan]")

                tracks = playlist_data['relationships'].get('tracks', {}).get("data", [])
                for track in tracks:
                    if track["type"] == 'songs' and track["attributes"].get("url"):
                        song_urls.add(track["attributes"]["url"])
                console.print(f"     Found {len(tracks)} tracks in this playlist.")
            except Exception as e:
                print_warning(f"Could not fetch playlist '{url}': {e}")

    elif config.DOWNLOAD_MODE == 'genres':
        print_panel(f"Searching for playlists by genres: {', '.join(config.DATASET_GENRES)}",
                    title="[bold blue]Genre Mode[/bold blue]", style="blue")
        # Search for a larger number of playlists to get a diverse pool of songs
        search_limit = max(25, config.NUM_SONGS_TO_DOWNLOAD // 20)
        for genre in config.DATASET_GENRES:
            try:
                console.print(f"  -> Searching for playlists in genre: [cyan]{genre}[/cyan]")
                search_results = api.search(term=genre, storefront="us", types=["playlists"], limit=search_limit)
                if "playlists" in search_results.get("results", {}):
                    for playlist in search_results["results"]["playlists"]["data"]:
                        # Fetch tracks for each found playlist
                        playlist_data = api.get_playlist(playlist["id"], limit_tracks=100, fetch_all=True)
                        tracks = playlist_data['relationships'].get('tracks', {}).get("data", [])
                        for track in tracks:
                            if track["type"] == 'songs' and track["attributes"].get("url"):
                                song_urls.add(track["attributes"]["url"])
            except Exception as e:
                print_warning(f"Could not fetch playlists for genre '{genre}': {e}")
    else:
        print_error(f"Invalid DOWNLOAD_MODE: '{config.DOWNLOAD_MODE}'. Must be 'genres' or 'playlists'.")
        return []

    if not song_urls:
        print_error("Could not find any songs to download.")
        return []

    shuffled_urls = list(song_urls)
    random.shuffle(shuffled_urls)

    print_success(f"Found {len(shuffled_urls)} unique songs to process.")
    return shuffled_urls


def download_worker(song_url: str, api: AppleMusicApi) -> tuple[bool, str]:
    """
    A worker that downloads a song, its syllable-synced TTML lyrics, and a metadata file.
    Returns a tuple (success_boolean, message).
    """
    # Each worker gets its own Downloader instance to manage its own unique temp directory
    with Downloader(
            apple_music_api=api,
            output_path=config.RAW_AUDIO_DIR,
            temp_path_base=config.TEMP_PATH_BASE,
            silent=True
    ) as downloader:
        try:
            downloader.set_cdm()
            downloader_song = DownloaderSong(
                downloader=downloader,
                codec=SongCodec.AAC_LEGACY,  # Use a common, high-compatibility codec
                synced_lyrics_format=SyncedLyricsFormat.TTML, # Explicitly request TTML
            )
            downloader_song_legacy = DownloaderSongLegacy(
                downloader=downloader,
                codec=SongCodec.AAC_LEGACY,
            )

            # --- Main Download Logic ---
            url_info = downloader.get_url_info(song_url)
            queue = downloader.get_download_queue(url_info)
            if not queue.medias_metadata:
                return False, "Failed to get media metadata."

            media_metadata = queue.medias_metadata[0]
            track_id = downloader.get_media_id(media_metadata)

            # This is the crucial check for syllable-synced lyrics
            syllable_lyrics_ttml = api.get_syllable_lyrics(track_id)
            if not syllable_lyrics_ttml:
                return False, "No syllable-synced lyrics available."

            webplayback = api.get_webplayback(track_id)
            # We don't need unsynced lyrics for tags
            tags = downloader_song.get_tags(webplayback, lyrics_unsynced=None)

            # Create a clean, safe filename stem
            sanitized_stem = downloader.get_sanitized_string(f"{tags['artist']} - {tags['title']}", is_folder=False)
            final_audio_path = config.RAW_AUDIO_DIR / f"{sanitized_stem}.m4a"

            if final_audio_path.exists():
                return False, "Song already exists."

            # --- Audio Download and Remuxing ---
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

            # --- Save Syllable-Synced TTML Lyrics ---
            lyrics_path = config.RAW_LYRICS_DIR / f"{sanitized_stem}.ttml"
            lyrics_path.write_text(syllable_lyrics_ttml, encoding='utf-8')

            # --- Save Metadata ---
            meta_path = config.RAW_META_DIR / f"{sanitized_stem}.json"
            meta_data = {
                "track_id": track_id,
                "song_url": song_url,
                "title": tags['title'],
                "artist": tags['artist'],
                "album": tags['album'],
                "genre": tags.get('genre'),
                "release_date": tags.get('date')
            }
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta_data, f, indent=2, ensure_ascii=False)

            return True, sanitized_stem

        except Exception:
            # Capture the full traceback for detailed debugging of worker failures
            tb_str = traceback.format_exc()
            return False, f"Worker Exception: {tb_str}"


def run_dataset_downloader():
    """
    Orchestrates the entire data acquisition process:
    1. Validates config and creates directories.
    2. Initializes the Apple Music API.
    3. Fetches a list of song URLs.
    4. Downloads songs with syllable-synced lyrics in parallel.
    """
    config.create_directories()
    config.validate_config()

    try:
        api = AppleMusicApi.from_netscape_cookies(cookies_path=config.COOKIES_PATH)
    except Exception as e:
        print_error(f"Failed to initialize Apple Music API: {e}")
        return

    all_urls = fetch_song_urls(api)
    if not all_urls:
        return

    target_downloads = len(all_urls) if config.DOWNLOAD_MODE == 'playlists' else config.NUM_SONGS_TO_DOWNLOAD
    successful_downloads, failed_downloads, skipped_downloads = 0, 0, 0

    print_panel(f"Beginning download phase. Processing up to {len(all_urls)} songs. Target: {target_downloads}",
                title="[bold green]Download Phase[/bold green]", style="green")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        TextColumn("•"),
        TextColumn("[green]{task.completed} ✓[/green]"),
        TextColumn("[yellow]{task.fields[skipped]} ⏭[/yellow]"),
        TextColumn("[red]{task.fields[failed]} ✗[/red]"),
        transient=False,
    ) as progress:
        task = progress.add_task(
            "[green]Downloading songs...",
            total=target_downloads,
            failed=0,
            skipped=0
        )

        with ThreadPoolExecutor(max_workers=config.DOWNLOAD_WORKERS) as executor:
            future_to_url = {executor.submit(download_worker, url, api): url for url in all_urls}

            for future in as_completed(future_to_url):
                # Stop submitting new tasks if the target is reached in 'genres' mode
                if successful_downloads >= target_downloads and config.DOWNLOAD_MODE == 'genres':
                    # This doesn't stop running futures, but prevents new ones from starting effectively.
                    # We'll just skip processing the results of any remaining futures.
                    continue

                try:
                    success, message = future.result()
                    if success:
                        successful_downloads += 1
                        progress.console.print(f"✅ [green]Success:[/] {message}")
                        progress.update(task, advance=1)
                    else:
                        # Categorize failures for cleaner logging
                        if "already exists" in message:
                            skipped_downloads += 1
                        elif "No syllable-synced lyrics" in message:
                            skipped_downloads += 1
                        else:
                            failed_downloads += 1
                            url_path = Path(future_to_url[future]).name
                            progress.console.print(
                                f"❌ [red]Failed:[/] {url_path} | [yellow]Reason:[/] {message[:200]}"
                            )
                except Exception as e:
                    failed_downloads += 1
                    url_path = Path(future_to_url[future]).name
                    progress.console.print(f"❌ [bold red]Worker crashed for {url_path}:[/] {e}")

                progress.update(task, failed=failed_downloads, skipped=skipped_downloads)


    # Final summary report
    remaining_urls = len(all_urls) - (successful_downloads + failed_downloads + skipped_downloads)
    total_skipped = skipped_downloads + remaining_urls

    summary = (
        f"\n"
        f"  [bold green]Successful[/]: {successful_downloads}\n"
        f"  [bold yellow]Skipped[/]:    {total_skipped} (Already exists, no synced lyrics, or not processed)\n"
        f"  [bold red]Failed[/]:      {failed_downloads}"
    )
    print_panel(summary, title="[bold blue]Download Summary[/bold blue]", style="blue")


if __name__ == '__main__':
    run_dataset_downloader()