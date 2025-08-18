from __future__ import annotations

import base64
import json
import re
import subprocess
from pathlib import Path
from xml.dom import minidom
from xml.etree import ElementTree

import m3u8
from InquirerPy import inquirer
from InquirerPy.base.control import Choice

from .constants import SONG_CODEC_REGEX_MAP, SYNCED_LYRICS_FILE_EXTENSION_MAP
from .downloader import Downloader
from .enums import MediaFileFormat, RemuxMode, SongCodec, SyncedLyricsFormat
from .models import Lyrics, StreamInfo, StreamInfoAv


class DownloaderSong:
    DEFAULT_DECRYPTION_KEY = "32b8ade1769e26b1ffb8986352793fc6"
    MP4_FORMAT_CODECS = ["ec-3"]

    def __init__(
            self,
            downloader: Downloader,
            codec: SongCodec = SongCodec.AAC_LEGACY,
            synced_lyrics_format: SyncedLyricsFormat = SyncedLyricsFormat.LRC,
    ):
        self.downloader = downloader
        self.codec = codec
        self.synced_lyrics_format = synced_lyrics_format

    def get_lyrics(self, track_id: str, track_metadata: dict) -> Lyrics | None:
        if not track_metadata["attributes"].get("hasLyrics"):
            return None

        syllable_ttml = None
        if self.synced_lyrics_format == SyncedLyricsFormat.ENHANCED_LRC:
            syllable_ttml = self.downloader.apple_music_api.get_syllable_lyrics(track_id)

        if syllable_ttml:
            return self._parse_lyrics_ttml(syllable_ttml, is_syllable_timed=True)

        if (
                track_metadata.get("relationships", {})
                        .get("lyrics", {})
                        .get("data")
        ):
            line_synced_ttml = track_metadata["relationships"]["lyrics"]["data"][0]["attributes"]["ttml"]
            return self._parse_lyrics_ttml(line_synced_ttml, is_syllable_timed=False)

        return None

    # --- CORRECTED TIMESTAMP PARSING LOGIC ---
    def _parse_ttml_time(self, time_str: str) -> float:
        """
        Robustly parses TTML time strings which can be in formats like:
        '1:02.345', '7.209', '00:01:02.345'.
        Returns total seconds as a float.
        """
        if 's' in time_str:  # Handles '12.345s' format
            return float(time_str.replace('s', ''))

        parts = time_str.split(':')
        total_seconds = 0.0

        if len(parts) == 1:  # Format '7.209'
            total_seconds = float(parts[0])
        elif len(parts) == 2:  # Format '01:02.345'
            m, s = parts
            total_seconds = int(m) * 60 + float(s)
        elif len(parts) == 3:  # Format '00:01:02.345'
            h, m, s = parts
            total_seconds = int(h) * 3600 + int(m) * 60 + float(s)

        return total_seconds

    def _seconds_to_lrc(self, total_seconds: float) -> str:
        """Converts total seconds to LRC time format (MM:SS.xx)."""
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60
        return f"{minutes:02d}:{seconds:05.2f}".replace('.', ':')

    def _seconds_to_srt(self, total_seconds: float) -> str:
        """Converts total seconds to SRT time format (HH:MM:SS,ms)."""
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace('.', ',')

    def _parse_lyrics_ttml(self, lyrics_ttml: str, is_syllable_timed: bool) -> Lyrics:
        lyrics = Lyrics(unsynced="", synced="")
        lyrics_ttml_et = ElementTree.fromstring(lyrics_ttml)
        ns = {"ttml": "http://www.w3.org/ns/ttml"}

        if self.synced_lyrics_format == SyncedLyricsFormat.TTML:
            lyrics.synced = minidom.parseString(lyrics_ttml).toprettyxml()
            return lyrics

        srt_index = 1
        for p in lyrics_ttml_et.findall(".//ttml:p", ns):
            line_begin_str = p.attrib.get("begin")
            line_end_str = p.attrib.get("end")

            line_text = "".join(p.itertext()).strip()
            if line_text:
                lyrics.unsynced += line_text + "\n"

            if not line_begin_str:
                continue

            line_begin_sec = self._parse_ttml_time(line_begin_str)

            if self.synced_lyrics_format in [SyncedLyricsFormat.LRC, SyncedLyricsFormat.ENHANCED_LRC]:
                lrc_line = f"[{self._seconds_to_lrc(line_begin_sec)}]"

                spans = p.findall("ttml:span", ns)
                # Check if syllable data actually exists within the line
                if is_syllable_timed and spans:
                    for span in spans:
                        span_begin_str = span.attrib.get("begin")
                        span_text = (span.text or "").strip()
                        if span_begin_str and span_text:
                            span_begin_sec = self._parse_ttml_time(span_begin_str)
                            lrc_line += f"<{self._seconds_to_lrc(span_begin_sec)}>{span_text}"
                else:
                    # Fallback for regular LRC or if enhanced LRC has no syllables
                    lrc_line += line_text

                lyrics.synced += lrc_line + "\n"

            elif self.synced_lyrics_format == SyncedLyricsFormat.SRT:
                if line_end_str:
                    line_end_sec = self._parse_ttml_time(line_end_str)
                    start_srt = self._seconds_to_srt(line_begin_sec)
                    end_srt = self._seconds_to_srt(line_end_sec)
                    lyrics.synced += f"{srt_index}\n{start_srt} --> {end_srt}\n{line_text}\n\n"
                    srt_index += 1

        lyrics.unsynced = lyrics.unsynced.strip()
        return lyrics

    def get_tags(self, webplayback: dict, lyrics_unsynced: str) -> dict:
        tags_raw = webplayback["assets"][0]["metadata"]
        tags = {
            "album": tags_raw["playlistName"],
            "album_artist": tags_raw["playlistArtistName"],
            "album_id": int(tags_raw.get("playlistId", tags_raw.get("itemId", 0))),
            "album_sort": tags_raw["sort-album"],
            "artist": tags_raw["artistName"],
            "artist_id": int(tags_raw["artistId"]),
            "artist_sort": tags_raw["sort-artist"],
            "compilation": tags_raw["compilation"],
            "copyright": tags_raw.get("copyright"),
            "date": (
                self.downloader.sanitize_date(tags_raw["releaseDate"])
                if tags_raw.get("releaseDate")
                else None
            ),
            "disc": tags_raw["discNumber"],
            "disc_total": tags_raw["discCount"],
            "gapless": tags_raw["gapless"],
            "genre": tags_raw.get("genre"),
            "lyrics": lyrics_unsynced if lyrics_unsynced else None,
            "media_type": 1,
            "rating": tags_raw["explicit"],
            "storefront": tags_raw["s"],
            "title": tags_raw["itemName"],
            "title_id": int(tags_raw["itemId"]),
            "title_sort": tags_raw["sort-name"],
            "track": tags_raw["trackNumber"],
            "track_total": tags_raw.get("trackCount", 1),
        }
        return tags

    def get_lyrics_synced_path(self, final_path: Path) -> Path:
        return final_path.with_suffix(
            SYNCED_LYRICS_FILE_EXTENSION_MAP[self.synced_lyrics_format]
        )

    def save_lyrics_synced(self, lyrics_synced_path: Path, lyrics_synced: str):
        lyrics_synced_path.parent.mkdir(parents=True, exist_ok=True)
        lyrics_synced_path.write_text(lyrics_synced, encoding="utf8")

    # --- The rest of the methods are for audio stream handling and are simplified ---
    # --- from the original files. They remain largely the same functionally. ---

    def get_drm_infos(self, m3u8_data: dict) -> dict:
        drm_info_raw = next(
            (sd for sd in m3u8_data["session_data"] if sd["data_id"] == "com.apple.hls.AudioSessionKeyInfo"), None)
        if not drm_info_raw: return None
        return json.loads(base64.b64decode(drm_info_raw["value"]).decode("utf-8"))

    def get_asset_infos(self, m3u8_data: dict) -> dict:
        return json.loads(base64.b64decode(
            next(sd for sd in m3u8_data["session_data"] if sd["data_id"] == "com.apple.hls.audioAssetMetadata")[
                "value"]).decode("utf-8"))

    def get_playlist_from_codec(self, m3u8_data: dict) -> dict | None:
        playlists = [p for p in m3u8_data["playlists"] if
                     re.fullmatch(SONG_CODEC_REGEX_MAP[self.codec], p["stream_info"]["audio"])]
        if not playlists: return None
        playlists.sort(key=lambda x: x["stream_info"]["average_bandwidth"])
        return playlists[-1]

    def get_playlist_from_user(self, m3u8_data: dict) -> dict | None:
        choices = [Choice(name=p["stream_info"]["audio"], value=p) for p in m3u8_data["playlists"]]
        return inquirer.select(message="Select which codec to download:", choices=choices).execute()

    def _get_drm_data(self, drm_infos: dict, drm_ids: list, drm_key: str) -> str | None:
        info = next((drm_infos[did] for did in drm_ids if drm_infos[did].get(drm_key) and did != "1"), None)
        return info[drm_key]["URI"] if info else None

    def get_widevine_pssh(self, drm_infos: dict, drm_ids: list) -> str | None:
        return self._get_drm_data(drm_infos, drm_ids, "urn:uuid:edef8ba9-79d6-4ace-a3c8-27dcd51d21ed")

    def get_stream_info(self, track_metadata: dict) -> StreamInfoAv | None:
        m3u8_url = track_metadata["attributes"]["extendedAssetUrls"].get("enhancedHls")
        if not m3u8_url: return None
        return self._get_stream_info(m3u8_url)

    def _get_stream_info(self, m3u8_url: str) -> StreamInfoAv | None:
        stream_info = StreamInfo()
        m3u8_obj = m3u8.load(m3u8_url)
        m3u8_data = m3u8_obj.data
        drm_infos = self.get_drm_infos(m3u8_data)
        if not drm_infos: return None
        asset_infos = self.get_asset_infos(m3u8_data)
        playlist = self.get_playlist_from_user(
            m3u8_data) if self.codec == SongCodec.ASK else self.get_playlist_from_codec(m3u8_data)
        if playlist is None: return None
        stream_info.stream_url = m3u8_obj.base_uri + playlist["uri"]
        variant_id = playlist["stream_info"]["stable_variant_id"]
        drm_ids = asset_infos[variant_id]["AUDIO-SESSION-KEY-IDS"]
        stream_info.widevine_pssh = self.get_widevine_pssh(drm_infos, drm_ids)
        stream_info.codec = playlist["stream_info"]["codecs"]
        is_mp4 = any(stream_info.codec.startswith(c) for c in self.MP4_FORMAT_CODECS)
        return StreamInfoAv(audio_track=stream_info, file_format=MediaFileFormat.MP4 if is_mp4 else MediaFileFormat.M4A)

    def get_encrypted_path(self, track_id: str) -> Path:
        return self.downloader.temp_path / f"{track_id}_encrypted.m4a"

    def get_decrypted_path(self, track_id: str) -> Path:
        return self.downloader.temp_path / f"{track_id}_decrypted.m4a"

    def get_remuxed_path(self, track_id: str, file_format: MediaFileFormat) -> Path:
        suffix = "m4a" if file_format == MediaFileFormat.M4A else "mp4"
        return self.downloader.temp_path / f"{track_id}_remuxed.{suffix}"

    def fix_key_id(self, encrypted_path: Path):
        count = 0
        with open(encrypted_path, "rb+") as file:
            while data := file.read(4096):
                pos = file.tell()
                i = 0
                while tenc := max(0, data.find(b"tenc", i)):
                    kid = tenc + 12
                    file.seek(max(0, pos - 4096) + kid, 0)
                    file.write(bytes.fromhex(f"{count:032}"))
                    count += 1
                    i = kid + 1
                file.seek(pos, 0)

    def decrypt(self, encrypted_path: Path, decrypted_path: Path, decryption_key: str):
        self.fix_key_id(encrypted_path)
        subprocess.run([self.downloader.mp4decrypt_path_full, str(encrypted_path), "--key",
                        f"00000000000000000000000000000001:{decryption_key}", "--key",
                        f"00000000000000000000000000000000:{self.DEFAULT_DECRYPTION_KEY}", str(decrypted_path)],
                       check=True, **self.downloader.subprocess_additional_args)

    def remux(self, decrypted_path: Path, remuxed_path: Path):
        if self.downloader.remux_mode == RemuxMode.MP4BOX:
            subprocess.run([self.downloader.mp4box_path_full, "-quiet", "-add", str(decrypted_path), "-itags",
                            "artist=placeholder", "-keep-utc", "-new", str(remuxed_path)], check=True,
                           **self.downloader.subprocess_additional_args)
        elif self.downloader.remux_mode == RemuxMode.FFMPEG:
            subprocess.run(
                [self.downloader.ffmpeg_path_full, "-loglevel", "error", "-y", "-i", str(decrypted_path), "-c", "copy",
                 "-movflags", "+faststart", str(remuxed_path)], check=True,
                **self.downloader.subprocess_additional_args)