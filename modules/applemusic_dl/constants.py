from .enums import SongCodec, SyncedLyricsFormat

MP4_TAGS_MAP = {
    "album": "\xa9alb",
    "album_artist": "aART",
    "album_id": "plID",
    "album_sort": "soal",
    "artist": "\xa9ART",
    "artist_id": "atID",
    "artist_sort": "soar",
    "comment": "\xa9cmt",
    "copyright": "cprt",
    "date": "\xa9day",
    "genre": "\xa9gen",
    "lyrics": "\xa9lyr",
    "media_type": "stik",
    "rating": "rtng",
    "storefront": "sfID",
    "title": "\xa9nam",
    "title_id": "cnID",
    "title_sort": "sonm",
}

SONG_CODEC_REGEX_MAP = {
    SongCodec.AAC: r"audio-stereo-\d+",
    SongCodec.AAC_HE: r"audio-HE-stereo-\d+",
    SongCodec.AAC_BINAURAL: r"audio-stereo-\d+-binaural",
    SongCodec.AAC_DOWNMIX: r"audio-stereo-\d+-downmix",
    SongCodec.AAC_HE_BINAURAL: r"audio-HE-stereo-\d+-binaural",
    SongCodec.AAC_HE_DOWNMIX: r"audio-HE-stereo-\d+-downmix",
    SongCodec.ATMOS: r"audio-atmos-.*",
    SongCodec.AC3: r"audio-ac3-.*",
    SongCodec.ALAC: r"audio-alac-.*",
}

SYNCED_LYRICS_FILE_EXTENSION_MAP = {
    SyncedLyricsFormat.LRC: ".lrc",
    SyncedLyricsFormat.ENHANCED_LRC: ".lrc", # Enhanced LRC still uses the .lrc extension
    SyncedLyricsFormat.SRT: ".srt",
    SyncedLyricsFormat.TTML: ".ttml",
}

LEGACY_CODECS = [
    SongCodec.AAC_LEGACY,
    SongCodec.AAC_HE_LEGACY,
]