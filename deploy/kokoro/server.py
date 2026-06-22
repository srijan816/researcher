from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import threading
import wave
from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.request import Request
from urllib.request import urlopen

import numpy as np
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from fastapi.responses import Response
from kokoro_onnx import Kokoro
from pydantic import BaseModel
from pydantic import Field

MODEL_FILE = "kokoro-v1.0.onnx"
VOICES_FILE = "voices-v1.0.bin"
DEFAULT_VOICE = "am_adam"
DEFAULT_MAX_CHARS = 20000
DEFAULT_MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
DEFAULT_VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

app = FastAPI(title="Kokoro TTS", version="1.0.0")
_asset_lock = threading.Lock()
_model_lock = threading.Lock()


class SpeechRequest(BaseModel):
    model: str | None = None
    input: str = Field(min_length=1)
    voice: str | None = None
    response_format: Literal["mp3", "wav"] = "mp3"
    speed: float = 1.0
    lang: str = "en-us"


def _env_bool(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off"}


def _model_dir() -> Path:
    return Path(os.environ.get("KOKORO_MODEL_DIR", "/data/kokoro-models"))


def _max_chars() -> int:
    raw = os.environ.get("KOKORO_MAX_TTS_CHARS")
    if not raw:
        return DEFAULT_MAX_CHARS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_CHARS
    return value if value > 0 else DEFAULT_MAX_CHARS


def _normalize_voice(value: str | None) -> str:
    if not value or not value.strip():
        return DEFAULT_VOICE
    normalized = value.strip().lower()
    if normalized == "onyx":
        return DEFAULT_VOICE
    return normalized


def _trim_at_speech_boundary(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value

    head = value[:max_chars]
    boundary_floor = int(max_chars * 0.6)
    sentence_boundary = max(
        head.rfind(". "),
        head.rfind("! "),
        head.rfind("? "),
        head.rfind(".\n"),
        head.rfind("!\n"),
        head.rfind("?\n"),
    )
    if sentence_boundary >= boundary_floor:
        return head[: sentence_boundary + 1].strip()

    word_boundary = head.rfind(" ")
    if word_boundary >= boundary_floor:
        return head[:word_boundary].strip()

    return head.strip()


def _clean_text(value: str) -> str:
    value = re.sub(r"https?://\S+", "", value)
    value = re.sub(r"\[(\d+)\]", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _split_text(value: str, max_chars: int = 420) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", value) if part.strip()]
    if not paragraphs:
        paragraphs = [value]

    chunks: list[str] = []
    current = ""

    def push(piece: str) -> None:
        nonlocal current
        piece = piece.strip()
        if not piece:
            return
        if len(piece) > max_chars:
            words = piece.split()
            buffer = ""
            for word in words:
                if len(buffer) + len(word) + 1 > max_chars:
                    if buffer:
                        push(buffer)
                    buffer = word
                else:
                    buffer = f"{buffer} {word}".strip()
            if buffer:
                push(buffer)
            return
        if not current:
            current = piece
        elif len(current) + len(piece) + 1 <= max_chars:
            current = f"{current} {piece}"
        else:
            chunks.append(current)
            current = piece

    for paragraph in paragraphs:
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", paragraph) if part.strip()]
        for sentence in sentences or [paragraph]:
            push(sentence)

    if current:
        chunks.append(current)
    return chunks


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + f".tmp-{os.getpid()}")
    if temp_path.exists():
        temp_path.unlink()

    request = Request(url, headers={"User-Agent": "Kokoro-TTS/1.0"})
    with urlopen(request, timeout=600) as response, open(temp_path, "wb") as handle:
        shutil.copyfileobj(response, handle)
    temp_path.replace(destination)


def _ensure_assets() -> tuple[Path, Path]:
    model_dir = _model_dir()
    model_path = model_dir / MODEL_FILE
    voices_path = model_dir / VOICES_FILE

    with _asset_lock:
        missing: list[tuple[Path, str]] = []
        if not model_path.exists():
            missing.append((model_path, os.environ.get("KOKORO_MODEL_URL", DEFAULT_MODEL_URL)))
        if not voices_path.exists():
            missing.append((voices_path, os.environ.get("KOKORO_VOICES_URL", DEFAULT_VOICES_URL)))

        if missing and not _env_bool("KOKORO_AUTO_DOWNLOAD", True):
            missing_names = ", ".join(path.name for path, _ in missing)
            raise FileNotFoundError(f"Missing Kokoro assets: {missing_names}")

        for destination, url in missing:
            _download_file(url, destination)

    return model_path, voices_path


@lru_cache(maxsize=1)
def _get_model() -> Kokoro:
    model_path, voices_path = _ensure_assets()
    return Kokoro(str(model_path), str(voices_path))


def _audio_content_type(audio: bytes) -> str:
    return "audio/wav" if audio[:4] == b"RIFF" else "audio/mpeg"


def _synthesize(text: str, voice: str, speed: float, lang: str, response_format: str) -> bytes:
    clean_text = _clean_text(text)
    if not clean_text:
        raise HTTPException(status_code=400, detail="Input text is empty")

    model = _get_model()
    audio_parts: list[np.ndarray] = []
    sample_rate = 24000

    with _model_lock:
        for chunk in _split_text(clean_text):
            audio, sample_rate = model.create(chunk, voice=voice, speed=speed, lang=lang)
            audio_parts.append(np.asarray(audio, dtype=np.float32))
            audio_parts.append(np.zeros(int(sample_rate * 0.18), dtype=np.float32))

    if not audio_parts:
        raise HTTPException(status_code=500, detail="Kokoro produced no audio")

    audio = np.concatenate(audio_parts)
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)

    with tempfile.TemporaryDirectory(prefix="kokoro-tts-") as tmp:
        wav_path = Path(tmp) / "speech.wav"
        mp3_path = Path(tmp) / "speech.mp3"
        with wave.open(str(wav_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm.tobytes())

        if response_format == "wav":
            return wav_path.read_bytes()

        ffmpeg = shutil.which(os.environ.get("FFMPEG_BIN", "ffmpeg"))
        if ffmpeg:
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(wav_path), str(mp3_path)], check=True)
            return mp3_path.read_bytes()

        return wav_path.read_bytes()


@app.on_event("startup")
def _startup() -> None:
    _get_model()


@app.get("/health")
def health() -> JSONResponse:
    model_path = _model_dir() / MODEL_FILE
    voices_path = _model_dir() / VOICES_FILE
    ready = model_path.exists() and voices_path.exists()
    return JSONResponse(
        {
            "status": "ready" if ready else "not_ready",
            "modelDir": str(_model_dir()),
            "modelFile": MODEL_FILE,
            "voicesFile": VOICES_FILE,
            "defaultVoice": DEFAULT_VOICE,
            "maxTtsChars": _max_chars(),
            "autoDownload": _env_bool("KOKORO_AUTO_DOWNLOAD", True),
        }
    )


@app.post("/v1/audio/speech")
def audio_speech(payload: SpeechRequest) -> Response:
    text = _trim_at_speech_boundary(payload.input, _max_chars())
    voice = _normalize_voice(payload.voice)
    audio = _synthesize(
        text=text,
        voice=voice,
        speed=payload.speed,
        lang=payload.lang,
        response_format=payload.response_format,
    )
    return Response(content=audio, media_type=_audio_content_type(audio))
