__version__ = "3.0.0-lyra"

from .apple_music_api import AppleMusicApi
from .downloader import Downloader
from .downloader_song import DownloaderSong
from .downloader_song_legacy import DownloaderSongLegacy
from .enums import SongCodec, SyncedLyricsFormat

__all__ = [
    "Downloader",
    "DownloaderSong",
    "DownloaderSongLegacy",
    "AppleMusicApi",
    "SongCodec",
    "SyncedLyricsFormat",
]