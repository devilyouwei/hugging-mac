"""Parse SenseVoice rich tokens without depending on FunASR."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

_LANGUAGE_TAGS = {
    "<|zh|>": "zh",
    "<|en|>": "en",
    "<|yue|>": "yue",
    "<|ja|>": "ja",
    "<|ko|>": "ko",
    "<|nospeech|>": "nospeech",
}
_EMOTION_TAGS = {
    "<|HAPPY|>": ("happy", "😊"),
    "<|SAD|>": ("sad", "😔"),
    "<|ANGRY|>": ("angry", "😡"),
    "<|NEUTRAL|>": ("neutral", ""),
    "<|FEARFUL|>": ("fearful", "😰"),
    "<|DISGUSTED|>": ("disgusted", "🤢"),
    "<|SURPRISED|>": ("surprised", "😮"),
    "<|EMO_UNKNOWN|>": ("unknown", ""),
}
_EVENT_TAGS = {
    "<|BGM|>": ("bgm", "🎼"),
    "<|Speech|>": ("speech", ""),
    "<|Applause|>": ("applause", "👏"),
    "<|Laughter|>": ("laughter", "😀"),
    "<|Cry|>": ("cry", "😭"),
    "<|Sneeze|>": ("sneeze", "🤧"),
    "<|Breath|>": ("breath", ""),
    "<|Cough|>": ("cough", "😷"),
    "<|Sing|>": ("sing", ""),
    "<|Speech_Noise|>": ("speech-noise", ""),
    "<|Event_UNK|>": ("unknown", ""),
}
_CONTROL_TAGS = {"<|withitn|>", "<|woitn|>", "<|GBG|>"}
_SPECIAL_TOKEN = re.compile(r"<\|[^|]+\|>")


@dataclass(frozen=True, slots=True)
class RichTranscript:
    text: str
    raw_text: str
    languages: tuple[str, ...]
    emotion: str | None
    events: tuple[str, ...]


def parse_rich_transcript(raw_text: str) -> RichTranscript:
    """Extract structured annotations and produce the official-style rich text."""

    tags = _SPECIAL_TOKEN.findall(raw_text)
    languages = _unique(_LANGUAGE_TAGS[tag] for tag in tags if tag in _LANGUAGE_TAGS)
    events = _unique(_EVENT_TAGS[tag][0] for tag in tags if tag in _EVENT_TAGS)
    emotion_tags = [tag for tag in tags if tag in _EMOTION_TAGS]
    emotion_tag = _most_common(emotion_tags)
    emotion = _EMOTION_TAGS[emotion_tag][0] if emotion_tag is not None else None

    body = raw_text
    for tag in _LANGUAGE_TAGS:
        body = body.replace(tag, "<|lang|>")
    segments = body.split("<|lang|>")
    rendered = [_format_segment(segment) for segment in segments]
    text = " ".join(segment for segment in rendered if segment).strip()
    text = text.replace("The.", " ").strip()
    return RichTranscript(
        text=text,
        raw_text=raw_text,
        languages=languages,
        emotion=emotion,
        events=events,
    )


def _format_segment(segment: str) -> str:
    counts = Counter(_SPECIAL_TOKEN.findall(segment))
    clean = segment
    known = set(_LANGUAGE_TAGS) | set(_EMOTION_TAGS) | set(_EVENT_TAGS) | _CONTROL_TAGS
    for tag in known:
        clean = clean.replace(tag, "")
    clean = _SPECIAL_TOKEN.sub("", clean)

    event_prefix = "".join(
        emoji for tag, (_, emoji) in _EVENT_TAGS.items() if emoji and counts[tag] > 0
    )
    emotion_tag = _most_common(
        [tag for tag in _EMOTION_TAGS if counts[tag] > 0],
        counts=counts,
    )
    emotion_suffix = _EMOTION_TAGS[emotion_tag][1] if emotion_tag is not None else ""
    return f"{event_prefix}{clean.strip()}{emotion_suffix}".strip()


def _most_common(
    values: list[str],
    *,
    counts: Counter[str] | None = None,
) -> str | None:
    if not values:
        return None
    frequencies = counts or Counter(values)
    return max(values, key=lambda value: (frequencies[value], -values.index(value)))


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))
