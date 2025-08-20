from __future__ import annotations

import base64
import datetime
import re
import shutil
import subprocess
import typing
import uuid
from pathlib import Path

from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from mutagen.mp4 import MP4
from pywidevine import Cdm, Device, PSSH
from yt_dlp import YoutubeDL

from .apple_music_api import AppleMusicApi
from .constants import MP4_TAGS_MAP
from .enums import DownloadMode, RemuxMode
from .hardcoded_wvd import HARDCODED_WVD
from .models import DownloadQueue, UrlInfo


class Downloader:
    ILLEGAL_CHARS_RE = r'[\\/:*?"<>|;]'
    ILLEGAL_CHAR_REPLACEMENT = "_"
    VALID_URL_RE = (
        r"(/(?P<storefront>[a-z]{2})/(?P<type>artist|album|playlist|song)/(?P<slug>[^/]*)(?:/(?P<id>[^/?]*))?(?:\?i=)?(?P<sub_id>[0-9a-z]*)?)|"
        r"(/library/(?P<library_type>|playlist|albums)/(?P<library_id>[a-z]\.[0-9a-zA-Z]*))"
    )

    def __init__(
            self,
            apple_music_api: AppleMusicApi,
            output_path: Path = Path("./Apple Music"),
            temp_path_base: Path = Path("./temp"),  # Base path for temp dirs
            wvd_path: Path = None,
            nm3u8dlre_path: str = "N_m3u8DL-RE",
            mp4decrypt_path: str = "mp4decrypt",
            ffmpeg_path: str = "ffmpeg",
            mp4box_path: str = "MP4Box",
            download_mode: DownloadMode = DownloadMode.YTDLP,
            remux_mode: RemuxMode = RemuxMode.FFMPEG,
            template_folder_album: str = "{album_artist}/{album}",
            template_folder_compilation: str = "Compilations/{album}",
            template_file_single_disc: str = "{track:02d} {title}",
            template_file_multi_disc: str = "{disc}-{track:02d} {title}",
            template_folder_no_album: str = "{artist}/Unknown Album",
            template_file_no_album: str = "{title}",
            template_date: str = "%Y-%m-%dT%H:%M:%SZ",
            exclude_tags: str = None,
            truncate: int = None,
            silent: bool = False,
    ):
        self.apple_music_api = apple_music_api
        self.output_path = output_path
        # --- MODIFIED: Create a unique temp directory for this instance ---
        self.temp_path = temp_path_base / f"temp_{uuid.uuid4().hex}"
        self.temp_path.mkdir(parents=True, exist_ok=True)

        self.wvd_path = wvd_path
        self.nm3u8dlre_path = nm3u8dlre_path
        self.mp4decrypt_path = mp4decrypt_path
        self.ffmpeg_path = ffmpeg_path
        self.mp4box_path = mp4box_path
        self.download_mode = download_mode
        self.remux_mode = remux_mode
        self.template_folder_album = template_folder_album
        self.template_folder_compilation = template_folder_compilation
        self.template_file_single_disc = template_file_single_disc
        self.template_file_multi_disc = template_file_multi_disc
        self.template_folder_no_album = template_folder_no_album
        self.template_file_no_album = template_file_no_album
        self.template_date = template_date
        self.exclude_tags = exclude_tags
        self.truncate = truncate
        self.silent = silent
        self._set_binaries_path_full()
        self._set_exclude_tags_list()
        self._set_truncate()
        self._set_subprocess_additional_args()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup_temp_path()

    def _set_binaries_path_full(self):
        self.nm3u8dlre_path_full = shutil.which(self.nm3u8dlre_path)
        self.ffmpeg_path_full = shutil.which(self.ffmpeg_path)
        self.mp4box_path_full = shutil.which(self.mp4box_path)
        self.mp4decrypt_path_full = shutil.which(self.mp4decrypt_path)

    def _set_exclude_tags_list(self):
        self.exclude_tags_list = (
            [i.lower() for i in self.exclude_tags.split(",")]
            if self.exclude_tags is not None
            else []
        )

    def _set_truncate(self):
        if self.truncate is not None:
            self.truncate = None if self.truncate < 4 else self.truncate

    def _set_subprocess_additional_args(self):
        self.subprocess_additional_args = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        } if self.silent else {}

    def set_cdm(self):
        if self.wvd_path:
            self.cdm = Cdm.from_device(Device.load(self.wvd_path))
        else:
            self.cdm = Cdm.from_device(Device.loads(HARDCODED_WVD))

    def get_url_info(self, url: str) -> UrlInfo:
        url_info = UrlInfo()
        url_regex_result = re.search(self.VALID_URL_RE, url)
        is_library = url_regex_result.group("library_type") is not None
        if is_library:
            url_info.type = url_regex_result.group("library_type")
            url_info.id = url_regex_result.group("library_id")
        else:
            url_info.storefront = url_regex_result.group("storefront")
            url_info.type = "song" if url_regex_result.group("sub_id") else url_regex_result.group("type")
            url_info.id = url_regex_result.group("sub_id") or url_regex_result.group("id")
        url_info.is_library = is_library
        return url_info

    def get_download_queue(self, url_info: UrlInfo) -> DownloadQueue:
        download_queue = DownloadQueue()
        url_type = url_info.type
        _id = url_info.id
        is_library = url_info.is_library

        if url_type == "artist":
            artist = self.apple_music_api.get_artist(_id)
            download_queue.medias_metadata = list(self.get_download_queue_from_artist(artist))
        elif url_type == "song":
            download_queue.medias_metadata = [self.apple_music_api.get_song(_id)]
        elif url_type in ("album", "albums"):
            album = self.apple_music_api.get_library_album(_id) if is_library else self.apple_music_api.get_album(_id)
            download_queue.medias_metadata = album["relationships"]["tracks"]["data"]
        elif url_type == "playlist":
            playlist = self.apple_music_api.get_library_playlist(
                _id) if is_library else self.apple_music_api.get_playlist(_id)
            download_queue.medias_metadata = playlist["relationships"]["tracks"]["data"]
        return download_queue

    def get_download_queue_from_artist(self, artist: dict) -> typing.Generator[dict, None, None]:
        albums = artist["relationships"]["albums"]["data"]
        choices = [
            Choice(
                name=f'{album["attributes"]["releaseDate"]:<10} | {album["attributes"]["name"]}',
                value=album,
            ) for album in albums
        ]
        selected = inquirer.select(
            message=f'Select albums to download for "{artist["attributes"]["name"]}":',
            choices=choices,
            multiselect=True,
        ).execute()
        for album in selected:
            for track in self.apple_music_api.get_album(album["id"])["relationships"]["tracks"]["data"]:
                yield track

    def get_media_id(self, media_metadata: dict) -> str | None:
        play_params = media_metadata["attributes"].get("playParams", {})
        return play_params.get("catalogId") or play_params.get("id")

    def sanitize_date(self, date: str) -> datetime.datetime:
        return datetime.datetime.fromisoformat(date[:-1]).strftime(self.template_date)

    def get_decryption_key(self, pssh: str, track_id: str) -> str:
        try:
            cdm_session = self.cdm.open()
            pssh_obj = PSSH(pssh.split(",")[-1])
            challenge = base64.b64encode(self.cdm.get_license_challenge(cdm_session, pssh_obj)).decode()
            license_data = self.apple_music_api.get_widevine_license(track_id, pssh, challenge)
            self.cdm.parse_license(cdm_session, license_data)
            return next(k for k in self.cdm.get_keys(cdm_session) if k.type == "CONTENT").key.hex()
        finally:
            self.cdm.close(cdm_session)

    def download(self, path: Path, stream_url: str):
        if self.download_mode == DownloadMode.YTDLP:
            self.download_ytdlp(path, stream_url)
        elif self.download_mode == DownloadMode.NM3U8DLRE:
            self.download_nm3u8dlre(path, stream_url)

    def download_ytdlp(self, path: Path, stream_url: str):
        with YoutubeDL({
            "quiet": True,
            "no_warnings": True,
            "outtmpl": str(path),
            "allow_unplayable_formats": True,
            "fixup": "never",
            "noprogress": self.silent,
        }) as ydl:
            ydl.download(stream_url)

    def download_nm3u8dlre(self, path: Path, stream_url: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            self.nm3u8dlre_path_full, stream_url,
            "--binary-merge", "--no-log", "--log-level", "off",
            "--ffmpeg-binary-path", self.ffmpeg_path_full,
            "--save-name", path.stem,
            "--save-dir", str(path.parent),
            "--tmp-dir", str(path.parent),
        ], check=True, **self.subprocess_additional_args)

    def get_sanitized_string(self, dirty_string: str, is_folder: bool) -> str:
        dirty_string = re.sub(self.ILLEGAL_CHARS_RE, self.ILLEGAL_CHAR_REPLACEMENT, dirty_string)
        if is_folder:
            dirty_string = dirty_string[:self.truncate]
            if dirty_string.endswith("."):
                dirty_string = dirty_string[:-1] + self.ILLEGAL_CHAR_REPLACEMENT
        elif self.truncate is not None:
            dirty_string = dirty_string[:self.truncate - 4]
        return dirty_string.strip()

    def get_final_path(self, tags: dict, file_extension: str) -> Path:
        if tags.get("album"):
            template_folder = self.template_folder_compilation.split("/") if tags.get(
                "compilation") else self.template_folder_album.split("/")
            template_file = self.template_file_multi_disc.split("/") if tags.get("disc_total",
                                                                                 0) > 1 else self.template_file_single_disc.split(
                "/")
        else:
            template_folder = self.template_folder_no_album.split("/")
            template_file = self.template_file_no_album.split("/")

        template_final = template_folder + template_file
        return Path(
            self.output_path,
            *[self.get_sanitized_string(i.format(**tags), True) for i in template_final[:-1]],
            self.get_sanitized_string(template_final[-1].format(**tags), False) + file_extension
        )

    def apply_tags(self, path: Path, tags: dict):
        to_apply_tags = [tag for tag in tags if tag not in self.exclude_tags_list]
        mp4_tags = {}
        for tag_name in to_apply_tags:
            if tag_name in ("disc", "disc_total"):
                if "disk" not in mp4_tags: mp4_tags["disk"] = [[0, 0]]
                mp4_tags["disk"][0][0 if tag_name == "disc" else 1] = tags[tag_name]
            elif tag_name in ("track", "track_total"):
                if "trkn" not in mp4_tags: mp4_tags["trkn"] = [[0, 0]]
                mp4_tags["trkn"][0][0 if tag_name == "track" else 1] = tags[tag_name]
            elif tag_name == "compilation":
                mp4_tags["cpil"] = tags["compilation"]
            elif tag_name == "gapless":
                mp4_tags["pgap"] = tags["gapless"]
            elif MP4_TAGS_MAP.get(tag_name) and tags.get(tag_name):
                mp4_tags[MP4_TAGS_MAP[tag_name]] = [tags[tag_name]]

        mp4 = MP4(path)
        mp4.clear()
        mp4.update(mp4_tags)
        mp4.save()

    def move_to_output_path(self, remuxed_path: Path, final_path: Path):
        final_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(remuxed_path, final_path)

    def cleanup_temp_path(self):
        if self.temp_path.exists():
            shutil.rmtree(self.temp_path)