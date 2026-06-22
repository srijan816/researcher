// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NextResponse } from 'next/server'
import { spawn } from 'node:child_process'
import { createWriteStream, existsSync } from 'node:fs'
import { mkdir, rename, rm } from 'node:fs/promises'
import path from 'node:path'
import { Readable } from 'node:stream'
import { pipeline } from 'node:stream/promises'

export const runtime = 'nodejs'

const DEFAULT_MAX_TTS_CHARS = 20000
const DEFAULT_KOKORO_VOICE = 'am_adam'
const KOKORO_MODEL_FILE = 'kokoro-v1.0.onnx'
const KOKORO_VOICES_FILE = 'voices-v1.0.bin'
const DEFAULT_KOKORO_MODEL_URL =
  'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx'
const DEFAULT_KOKORO_VOICES_URL =
  'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin'

const getKokoroUrl = (): string | null => {
  const url = process.env.KOKORO_TTS_URL
  return url ? url.replace(/\/$/, '') : null
}

const getRepoRoot = (): string => {
  return process.env.AIQ_RESEARCH_ROOT || path.resolve(process.cwd(), '..', '..')
}

const getLocalPython = (): string => {
  const repoPython = path.join(getRepoRoot(), '.venv', 'bin', 'python')
  if (process.env.KOKORO_PYTHON) return process.env.KOKORO_PYTHON
  if (existsSync(repoPython)) return repoPython
  return 'python3'
}

const getModelDir = (): string => {
  return process.env.KOKORO_MODEL_DIR || path.join(getRepoRoot(), 'data', 'kokoro-models')
}

const getModelPaths = () => {
  const modelDir = getModelDir()
  return {
    modelDir,
    modelPath: path.join(modelDir, KOKORO_MODEL_FILE),
    voicesPath: path.join(modelDir, KOKORO_VOICES_FILE),
  }
}

const isAutoDownloadEnabled = (): boolean => {
  return process.env.KOKORO_AUTO_DOWNLOAD !== '0' && process.env.KOKORO_AUTO_DOWNLOAD !== 'false'
}

const audioContentType = (audio: Buffer): string => {
  if (audio.subarray(0, 4).toString('ascii') === 'RIFF') return 'audio/wav'
  return 'audio/mpeg'
}

const getMaxTtsChars = (): number => {
  const configured = Number(process.env.KOKORO_MAX_TTS_CHARS)
  if (Number.isFinite(configured) && configured > 0) return Math.floor(configured)
  return DEFAULT_MAX_TTS_CHARS
}

const trimAtSpeechBoundary = (value: string, maxChars: number): string => {
  if (value.length <= maxChars) return value

  const head = value.slice(0, maxChars)
  const boundaryFloor = Math.floor(maxChars * 0.6)
  const sentenceBoundary = Math.max(
    head.lastIndexOf('. '),
    head.lastIndexOf('! '),
    head.lastIndexOf('? '),
    head.lastIndexOf('.\n'),
    head.lastIndexOf('!\n'),
    head.lastIndexOf('?\n')
  )

  if (sentenceBoundary >= boundaryFloor) {
    return head.slice(0, sentenceBoundary + 1).trim()
  }

  const wordBoundary = head.lastIndexOf(' ')
  if (wordBoundary >= boundaryFloor) {
    return head.slice(0, wordBoundary).trim()
  }

  return head.trim()
}

const normalizeVoice = (value: unknown): string => {
  if (typeof value !== 'string' || !value.trim()) return DEFAULT_KOKORO_VOICE
  const normalized = value.trim().toLowerCase()
  if (normalized === 'onyx') return DEFAULT_KOKORO_VOICE
  return normalized
}

const downloadFile = async (url: string, destination: string): Promise<void> => {
  const tempPath = `${destination}.tmp-${process.pid}-${Date.now()}`
  await rm(tempPath, { force: true })

  const response = await fetch(url, {
    method: 'GET',
    signal: AbortSignal.timeout(10 * 60 * 1000),
  })

  if (!response.ok || !response.body) {
    throw new Error(`Unable to download ${path.basename(destination)} (${response.status})`)
  }

  try {
    await pipeline(
      Readable.fromWeb(response.body as unknown as Parameters<typeof Readable.fromWeb>[0]),
      createWriteStream(tempPath, { flags: 'wx' })
    )
    await rename(tempPath, destination)
  } catch (error) {
    await rm(tempPath, { force: true })
    throw error
  }
}

const ensureLocalKokoroAssets = async (): Promise<void> => {
  const { modelDir, modelPath, voicesPath } = getModelPaths()
  await mkdir(modelDir, { recursive: true })

  const missingFiles = [
    { path: modelPath, url: process.env.KOKORO_MODEL_URL || DEFAULT_KOKORO_MODEL_URL },
    { path: voicesPath, url: process.env.KOKORO_VOICES_URL || DEFAULT_KOKORO_VOICES_URL },
  ].filter((file) => !existsSync(file.path))

  if (missingFiles.length === 0) return

  if (!isAutoDownloadEnabled()) {
    throw new Error(`Kokoro model files are missing in ${modelDir}`)
  }

  for (const file of missingFiles) {
    await downloadFile(file.url, file.path)
  }
}

const synthesizeWithLocalPython = async (
  text: string,
  voice: string,
  speed: number,
  lang: string
): Promise<Buffer> => {
  await ensureLocalKokoroAssets()
  const python = getLocalPython()
  if (path.isAbsolute(python) && !existsSync(python)) {
    throw new Error(`Local Python fallback failed: Python executable at ${python} does not exist.`)
  }
  const modelDir = getModelDir()
  const script = String.raw`
import os
import re
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
from kokoro_onnx import Kokoro

text = sys.stdin.read()
voice = os.environ.get("KOKORO_VOICE", "af_heart")
speed = float(os.environ.get("KOKORO_SPEED", "1.0"))
lang = os.environ.get("KOKORO_LANG", "en-us")
model_dir = Path(os.environ["KOKORO_MODEL_DIR"])
model_path = model_dir / "kokoro-v1.0.onnx"
voices_path = model_dir / "voices-v1.0.bin"
ffmpeg = os.environ.get("FFMPEG_BIN", "ffmpeg")

if not model_path.exists() or not voices_path.exists():
    raise SystemExit("Kokoro model files are missing")

text = re.sub(r"https?://\S+", "", text)
text = re.sub(r"\[(\d+)\]", "", text)
text = re.sub(r"\s+", " ", text).strip()
if not text:
    raise SystemExit("Input text is empty")

def split_text(value: str, max_chars: int = 420) -> list[str]:
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

kokoro = Kokoro(str(model_path), str(voices_path))
audio_parts = []
sample_rate = 24000
for chunk in split_text(text):
    audio, sample_rate = kokoro.create(chunk, voice=voice, speed=speed, lang=lang)
    audio_parts.append(np.asarray(audio, dtype=np.float32))
    audio_parts.append(np.zeros(int(sample_rate * 0.18), dtype=np.float32))

if not audio_parts:
    raise SystemExit("Input text produced no speakable chunks")

audio = np.concatenate(audio_parts)

with tempfile.TemporaryDirectory(prefix="deep-research-tts-") as tmp:
    wav_path = Path(tmp) / "speech.wav"
    mp3_path = Path(tmp) / "speech.mp3"
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    ffmpeg_path = shutil.which(ffmpeg) or (ffmpeg if Path(ffmpeg).exists() else None)
    if ffmpeg_path:
        subprocess.run([ffmpeg_path, "-y", "-loglevel", "error", "-i", str(wav_path), str(mp3_path)], check=True)
        sys.stdout.buffer.write(mp3_path.read_bytes())
    else:
        sys.stdout.buffer.write(wav_path.read_bytes())
`

  return new Promise((resolve, reject) => {
    const child = spawn(python, ['-c', script], {
      env: {
        ...process.env,
        KOKORO_MODEL_DIR: modelDir,
        KOKORO_VOICE: voice,
        KOKORO_SPEED: String(speed),
        KOKORO_LANG: lang,
      },
      stdio: ['pipe', 'pipe', 'pipe'],
    })
    const stdout: Buffer[] = []
    const stderr: Buffer[] = []
    const timeout = setTimeout(() => {
      child.kill('SIGKILL')
      reject(new Error('Kokoro synthesis timed out'))
    }, 120000)

    child.stdout.on('data', (chunk: Buffer) => stdout.push(chunk))
    child.stderr.on('data', (chunk: Buffer) => stderr.push(chunk))
    child.on('error', (error) => {
      clearTimeout(timeout)
      reject(error)
    })
    child.on('close', (code) => {
      clearTimeout(timeout)
      if (code === 0) {
        const audio = Buffer.concat(stdout)
        if (audio.length === 0) {
          reject(new Error('Kokoro produced no audio'))
          return
        }
        resolve(audio)
        return
      }
      reject(new Error(Buffer.concat(stderr).toString('utf8') || `Kokoro exited with ${code}`))
    })
    child.stdin.end(text)
  })
}

export async function GET(): Promise<Response> {
  const { modelDir, modelPath, voicesPath } = getModelPaths()
  const localStatus = () => {
    const ready = existsSync(modelPath) && existsSync(voicesPath)
    return NextResponse.json(
      {
        status: ready ? 'ready' : 'not_ready',
        mode: 'local',
        python: getLocalPython(),
        modelDir,
        modelFile: KOKORO_MODEL_FILE,
        voicesFile: KOKORO_VOICES_FILE,
        defaultVoice: DEFAULT_KOKORO_VOICE,
        maxTtsChars: getMaxTtsChars(),
        autoDownload: isAutoDownloadEnabled(),
      },
      { status: ready || isAutoDownloadEnabled() ? 200 : 503 }
    )
  }

  try {
    const kokoroUrl = getKokoroUrl()
    if (kokoroUrl) {
      const response = await fetch(`${kokoroUrl}/health`, {
        method: 'GET',
        signal: AbortSignal.timeout(2500),
      })

      if (!response.ok) {
        return localStatus()
      }

      return NextResponse.json(await response.json())
    }
    return localStatus()
  } catch {
    return localStatus()
  }
}

export async function POST(req: Request): Promise<Response> {
  try {
    const payload = await req.json()
    const text = typeof payload.text === 'string' ? payload.text.trim() : ''

    if (!text) {
      return NextResponse.json({ error: 'Missing text' }, { status: 400 })
    }

    const voice = normalizeVoice(payload.voice)
    const speed = payload.speed || 1
    const lang = payload.lang || 'en-us'
    const input = trimAtSpeechBoundary(text, getMaxTtsChars())

    const kokoroUrl = getKokoroUrl()
    const response = kokoroUrl
      ? await fetch(`${kokoroUrl}/v1/audio/speech`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model: payload.model || 'kokoro',
            input,
            voice,
            response_format: payload.format || 'mp3',
            speed,
          }),
          signal: AbortSignal.timeout(120000),
        }).catch(() => null)
      : null

    if (!response?.ok || !response.body) {
      const remoteDetail = response ? await response.text().catch(() => '') : ''
      const audio = await synthesizeWithLocalPython(input, voice, speed, lang).catch((error) => {
        const message = error instanceof Error ? error.message : 'Unable to synthesize audio'
        throw new Error(remoteDetail ? `${message}; remote Kokoro returned: ${remoteDetail}` : message)
      })
      return new NextResponse(new Uint8Array(audio), {
        status: 200,
        headers: {
          'Content-Type': audioContentType(audio),
          'Cache-Control': 'no-store',
        },
      })
    }

    return new NextResponse(response.body, {
      status: 200,
      headers: {
        'Content-Type': response.headers.get('Content-Type') || 'audio/mpeg',
        'Cache-Control': 'no-store',
      },
    })
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unable to synthesize audio'
    return NextResponse.json({ error: message }, { status: 502 })
  }
}
