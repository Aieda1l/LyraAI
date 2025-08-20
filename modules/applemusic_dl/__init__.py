# modules/applemusic_dl/__init__.py

"""
LyraAI Apple Music Downloader Module.

This package contains the necessary tools to download songs, lyrics, and metadata
from Apple Music. It is adapted for use within the LyraAI project.
"""

__version__ = "3.0.0-lyra"

# Import key classes to make them directly accessible from the package namespace.
# e.g., `from modules.applemusic_dl import Downloader`
from .apple_music_api import AppleMusicApi
from .downloader import Downloader
from .downloader_song import DownloaderSong
from .downloader_song_legacy import DownloaderSongLegacy
from .enums import SongCodec, SyncedLyricsFormat

# Define what gets imported with a wildcard import `from modules.applemusic_dl import *`
__all__ = [
    "Downloader",
    "DownloaderSong",
    "DownloaderSongLegacy",
    "AppleMusicApi",
    "SongCodec",
    "SyncedLyricsFormat",
]