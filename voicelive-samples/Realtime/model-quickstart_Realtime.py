# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""
GPT Realtime Model Inference - Azure AI Foundry
================================================
A quickstart program to inference GPT Realtime models (e.g., gpt-4o-realtime-preview)
deployed on Microsoft Azure AI Foundry. Supports real-time voice conversations
with microphone input and speaker output via WebSocket.

Prerequisites:
    pip install openai pyaudio azure-identity python-dotenv

Environment variables (or .env file):
    AZURE_OPENAI_ENDPOINT   - Your Azure AI Foundry endpoint (e.g., https://<resource>.openai.azure.com/)
    AZURE_OPENAI_API_KEY    - (Optional) API key, if not using Entra ID auth
    AZURE_OPENAI_DEPLOYMENT - Deployment name for your GPT Realtime model (e.g., gpt-4o-realtime-preview)
    AZURE_OPENAI_API_VERSION - API version (default: 2025-04-01-preview)
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import queue
import signal
import sys
from datetime import datetime
from typing import Optional, Union, cast
import system_instructions

import pyaudio
from dotenv import load_dotenv

# Change to the directory where this script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
if not os.path.exists("logs"):
    os.makedirs("logs")

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

logging.basicConfig(
    filename=f"logs/{timestamp}_realtime.log",
    filemode="w",
    format="%(asctime)s:%(name)s:%(levelname)s:%(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audio Processor – handles microphone capture and speaker playback
# ---------------------------------------------------------------------------
class AudioProcessor:
    """
    Handles real-time audio capture from the microphone and playback
    through the speakers using PyAudio with callback-based threading.

    Threading:
      - Capture callback: reads mic data → sends to Realtime API
      - Playback callback: reads from queue → writes to speaker
    """

    class _Packet:
        """Sequenced audio packet for ordered playback."""

        def __init__(self, seq: int, data: Optional[bytes]):
            self.seq = seq
            self.data = data

    def __init__(self, send_audio_coro, sample_rate: int = 24000):
        """
        Args:
            send_audio_coro: An async callable(base64_audio: str) that sends
                             audio data to the Realtime API.
            sample_rate: Audio sample rate (16000 or 24000 Hz).
        """
        self._send_audio = send_audio_coro
        self._pa = pyaudio.PyAudio()
        self._format = pyaudio.paInt16
        self._channels = 1
        self._rate = sample_rate
        self._chunk = int(sample_rate * 0.05)  # 50 ms frames

        self._input_stream: Optional[pyaudio.Stream] = None
        self._output_stream: Optional[pyaudio.Stream] = None

        self._playback_queue: queue.Queue[AudioProcessor._Packet] = queue.Queue()
        self._playback_base = 0
        self._next_seq = 0
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        logger.info("AudioProcessor initialised – %d Hz PCM16 mono", sample_rate)

    # -- capture ---------------------------------------------------------------

    def start_capture(self):
        """Open the microphone stream (non-blocking callback)."""
        if self._input_stream:
            return
        self._loop = asyncio.get_event_loop()

        def _on_capture(in_data, _frame_count, _time_info, _flags):
            b64 = base64.b64encode(in_data).decode("utf-8")
            asyncio.run_coroutine_threadsafe(self._send_audio(b64), self._loop)
            return (None, pyaudio.paContinue)

        self._input_stream = self._pa.open(
            format=self._format,
            channels=self._channels,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            stream_callback=_on_capture,
        )
        logger.info("Microphone capture started")

    # -- playback --------------------------------------------------------------

    def start_playback(self):
        """Open the speaker stream (non-blocking callback)."""
        if self._output_stream:
            return

        remaining = bytes()

        def _on_playback(_in_data, frame_count, _time_info, _flags):
            nonlocal remaining
            need = frame_count * pyaudio.get_sample_size(pyaudio.paInt16)
            out = remaining[:need]
            remaining = remaining[need:]

            while len(out) < need:
                try:
                    pkt = self._playback_queue.get_nowait()
                except queue.Empty:
                    out += bytes(need - len(out))
                    continue
                if pkt is None or pkt.data is None:
                    break
                if pkt.seq < self._playback_base:
                    remaining = bytes()
                    continue
                take = need - len(out)
                out += pkt.data[:take]
                remaining = pkt.data[take:]

            if len(out) >= need:
                return (out, pyaudio.paContinue)
            return (out, pyaudio.paComplete)

        self._output_stream = self._pa.open(
            format=self._format,
            channels=self._channels,
            rate=self._rate,
            output=True,
            frames_per_buffer=self._chunk,
            stream_callback=_on_playback,
        )
        logger.info("Speaker playback started")

    def queue_audio(self, data: Optional[bytes]):
        """Enqueue decoded PCM audio for playback."""
        seq = self._next_seq
        self._next_seq += 1
        self._playback_queue.put(self._Packet(seq, data))

    def skip_pending(self):
        """Discard queued audio (e.g. user barge-in)."""
        self._playback_base = self._next_seq
        self._next_seq += 1

    # -- cleanup ---------------------------------------------------------------

    def shutdown(self):
        """Release all audio resources."""
        if self._input_stream:
            self._input_stream.stop_stream()
            self._input_stream.close()
            self._input_stream = None
        if self._output_stream:
            self.skip_pending()
            self.queue_audio(None)
            self._output_stream.stop_stream()
            self._output_stream.close()
            self._output_stream = None
        if self._pa:
            self._pa.terminate()
        logger.info("AudioProcessor shut down")


# ---------------------------------------------------------------------------
# Realtime Voice Assistant
# ---------------------------------------------------------------------------
class RealtimeVoiceAssistant:
    """
    Connects to a GPT Realtime model deployed on Azure AI Foundry and runs
    a full-duplex voice conversation over WebSocket.
    """

    def __init__(
        self,
        endpoint: str,
        deployment: str,
        api_version: str = "2025-04-01-preview",
        api_key: Optional[str] = None,
        voice: str = "alloy",
        instructions: str = (
            "You are a helpful AI assistant. Always respond in English. "
            "Respond naturally and conversationally. "
            "Keep responses concise but engaging. "
            "If you hear silence or unclear audio, ask the user to repeat."
        ),
        sample_rate: int = 24000,
        temperature: float = 0.8,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.deployment = deployment
        self.api_version = api_version
        self.api_key = api_key
        self.voice = voice
        self.instructions = instructions
        self.sample_rate = sample_rate
        self.temperature = temperature

        self._connection = None
        self._audio: Optional[AudioProcessor] = None
        self._active_response = False
        self._response_done = False
        self._greeting_done = False

        # Latency tracking
        self._speech_stopped_time: Optional[float] = None
        self._latency_logged = False
        self._latency_records: list[dict] = []

        # Transcript tracking
        self._transcript: list[dict] = []
        self._current_assistant_text = ""

    # -- public entry point ----------------------------------------------------

    async def start(self):
        """Connect and run the voice conversation loop."""
        from openai import AsyncAzureOpenAI
        from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider

        try:
            if self.api_key:
                client = AsyncAzureOpenAI(
                    azure_endpoint=self.endpoint,
                    api_key=self.api_key,
                    api_version=self.api_version,
                )
                logger.info("Using API key authentication")
                print("🔑 Using API key authentication...")
            else:
                credential = DefaultAzureCredential()
                token_provider = get_bearer_token_provider(
                    credential, "https://cognitiveservices.azure.com/.default"
                )
                client = AsyncAzureOpenAI(
                    azure_endpoint=self.endpoint,
                    azure_ad_token_provider=token_provider,
                    api_version=self.api_version,
                )
                logger.info("Using Entra ID (DefaultAzureCredential) authentication")
                print("🔐 Using Azure Identity (Entra ID) authentication...")

            logger.info("Connecting to Realtime API – deployment: %s", self.deployment)

            async with client.beta.realtime.connect(
                model=self.deployment,
            ) as conn:
                self._connection = conn

                # Build send-audio helper for AudioProcessor
                async def _send_audio(b64_audio: str):
                    await conn.input_audio_buffer.append(audio=b64_audio)

                self._audio = AudioProcessor(_send_audio, self.sample_rate)

                # Configure session
                await self._configure_session()

                # Start speaker first, then mic
                self._audio.start_playback()

                print("\n" + "=" * 60)
                print("🎤 REALTIME VOICE ASSISTANT READY")
                print(f"   Model deployment : {self.deployment}")
                print(f"   Voice            : {self.voice}")
                print("   Start speaking to begin conversation")
                print("   Press Ctrl+C to exit")
                print("=" * 60 + "\n")

                # Kick off a proactive greeting
                # NOTE: Mic capture is delayed until greeting finishes to prevent
                # speaker audio from being picked up by the mic (feedback loop).
                await asyncio.sleep(1.5)
                await conn.response.create()

                # Main event loop (mic capture starts after greeting completes)
                await self._event_loop()

        finally:
            self._save_session_data()
            if self._audio:
                self._audio.shutdown()

    # -- session configuration -------------------------------------------------

    async def _configure_session(self):
        """Send session configuration to the Realtime API."""
        conn = self._connection
        assert conn is not None

        await conn.session.update(
            session={
                "modalities": ["text", "audio"],
                "instructions": self.instructions,
                "voice": self.voice,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {"model": "whisper-1"},
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.9,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500,
                },
                "temperature": self.temperature,
            }
        )
        logger.info("Session configured")

    # -- event processing ------------------------------------------------------

    async def _event_loop(self):
        """Iterate over server-sent events from the Realtime connection."""
        conn = self._connection
        assert conn is not None

        async for event in conn:
            await self._handle(event)

    async def _handle(self, event):
        """Dispatch a single server event."""
        etype = event.type
        conn = self._connection
        ap = self._audio
        assert conn is not None and ap is not None

        # -- session lifecycle -------------------------------------------------
        if etype == "session.updated":
            logger.info("Session updated – id: %s", getattr(event.session, "id", "?"))

        # -- user speech -------------------------------------------------------
        elif etype == "input_audio_buffer.speech_started":
            logger.info("User started speaking")
            print("🎤 Listening...")
            ap.skip_pending()
            if self._active_response and not self._response_done:
                try:
                    await conn.response.cancel()
                except Exception:
                    pass

        elif etype == "input_audio_buffer.speech_stopped":
            self._speech_stopped_time = asyncio.get_event_loop().time()
            self._latency_logged = False
            logger.info("User stopped speaking")
            print("🤔 Processing...")

        elif etype == "input_audio_buffer.committed":
            logger.debug("Audio buffer committed")

        # -- user transcript ---------------------------------------------------
        elif etype == "conversation.item.input_audio_transcription.completed":
            transcript = getattr(event, "transcript", "") or ""
            if transcript.strip():
                print(f"👤 You: {transcript.strip()}")
                self._transcript.append(
                    {"role": "user", "text": transcript.strip(), "ts": datetime.now().isoformat()}
                )

        # -- assistant response ------------------------------------------------
        elif etype == "response.created":
            self._active_response = True
            self._response_done = False
            self._current_assistant_text = ""

        elif etype == "response.audio.delta":
            # First audio chunk → measure latency
            if self._speech_stopped_time and not self._latency_logged:
                now = asyncio.get_event_loop().time()
                latency_ms = (now - self._speech_stopped_time) * 1000
                self._latency_records.append(
                    {"ts": datetime.now().isoformat(), "latency_ms": round(latency_ms, 2)}
                )
                print(f"⏱️  Latency: {latency_ms:.0f} ms")
                self._latency_logged = True

            # event.delta is base64-encoded; decode to raw PCM bytes for playback
            audio_bytes = base64.b64decode(event.delta) if isinstance(event.delta, str) else event.delta
            ap.queue_audio(audio_bytes)

        elif etype == "response.audio_transcript.delta":
            delta = getattr(event, "delta", "") or ""
            self._current_assistant_text += delta

        elif etype == "response.audio_transcript.done":
            text = self._current_assistant_text.strip()
            if text:
                print(f"🤖 Assistant: {text}")
                self._transcript.append(
                    {"role": "assistant", "text": text, "ts": datetime.now().isoformat()}
                )

        elif etype == "response.audio.done":
            logger.info("Assistant audio complete")
            print("🎤 Ready for next input...")

        elif etype == "response.done":
            self._active_response = False
            self._response_done = True

            # Start mic capture after the first response (greeting) finishes
            # This prevents the mic from picking up the greeting audio
            if not self._greeting_done:
                self._greeting_done = True
                # Wait briefly for speaker to fully drain before opening mic
                await asyncio.sleep(0.5)
                ap.start_capture()
                logger.info("Greeting complete – microphone capture started")

        elif etype == "response.text.done":
            text = getattr(event, "text", "") or ""
            if text.strip():
                print(f"🤖 Assistant (text): {text.strip()}")

        # -- errors ------------------------------------------------------------
        elif etype == "error":
            msg = getattr(event, "message", str(event))
            if "no active response" not in str(msg).lower():
                logger.error("API error: %s", msg)
                print(f"❌ Error: {msg}")

        else:
            logger.debug("Unhandled event: %s", etype)

    # -- persistence -----------------------------------------------------------

    def _save_session_data(self):
        """Persist transcript and latency data to disk."""
        if not self._transcript and not self._latency_records:
            return

        # Latency summary
        if self._latency_records:
            vals = [r["latency_ms"] for r in self._latency_records]
            avg = sum(vals) / len(vals)
            print(
                f"\n📊 Latency – min: {min(vals):.0f} ms | max: {max(vals):.0f} ms | avg: {avg:.0f} ms"
            )

        payload = {
            "session_timestamp": timestamp,
            "deployment": self.deployment,
            "transcript": self._transcript,
            "latency": self._latency_records,
        }
        out_path = f"logs/{timestamp}_session.json"
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            print(f"💾 Session data saved to {out_path}")
        except Exception as e:
            logger.error("Failed to save session data: %s", e)

        # Plain-text transcript
        if self._transcript:
            txt_path = f"logs/{timestamp}_transcript.txt"
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write("=" * 60 + "\n")
                    f.write("CONVERSATION TRANSCRIPT\n")
                    f.write(f"Deployment: {self.deployment}\n")
                    f.write(f"Session   : {timestamp}\n")
                    f.write("=" * 60 + "\n\n")
                    for entry in self._transcript:
                        role = entry["role"].upper()
                        f.write(f"[{entry['ts']}] {role}:\n{entry['text']}\n\n")
                print(f"📝 Transcript saved to {txt_path}")
            except Exception as e:
                logger.error("Failed to save transcript: %s", e)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="GPT Realtime Model Inference – Azure AI Foundry",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=os.environ.get(
            "AZURE_OPENAI_ENDPOINT", "https://<your-resource>.openai.azure.com/"
        ),
        help="Azure OpenAI endpoint URL",
    )
    parser.add_argument(
        "--deployment",
        type=str,
        default=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-realtime-preview"),
        help="Model deployment name",
    )
    parser.add_argument(
        "--api-version",
        type=str,
        default=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
        help="Azure OpenAI API version",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("AZURE_OPENAI_API_KEY"),
        help="API key (omit to use Entra ID / DefaultAzureCredential)",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=os.environ.get("AZURE_OPENAI_VOICE", "alloy"),
        help="Voice: alloy, echo, fable, onyx, nova, shimmer",
    )
    parser.add_argument(
        "--instructions",
        type=str,
        default=system_instructions.KOTAK_BOT
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=int(os.environ.get("AZURE_OPENAI_SAMPLE_RATE", "24000")),
        choices=[16000, 24000],
        help="Audio sample rate (Hz)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ.get("AZURE_OPENAI_TEMPERATURE", "0.8")),
        help="Model temperature (0.0 – 2.0)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    assistant = RealtimeVoiceAssistant(
        endpoint=args.endpoint,
        deployment=args.deployment,
        api_version=args.api_version,
        api_key=args.api_key,
        voice=args.voice,
        instructions=args.instructions,
        sample_rate=args.sample_rate,
        temperature=args.temperature,
    )

    # Graceful shutdown
    def _sig_handler(_sig, _frame):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    try:
        asyncio.run(assistant.start())
    except KeyboardInterrupt:
        print("\n👋 Session ended. Goodbye!")
    except Exception as e:
        print(f"Fatal error: {e}")
        logger.exception("Fatal error")


if __name__ == "__main__":
    # Pre-flight: verify audio devices are available
    try:
        p = pyaudio.PyAudio()
        has_input = any(
            cast(Union[int, float], p.get_device_info_by_index(i).get("maxInputChannels", 0) or 0) > 0
            for i in range(p.get_device_count())
        )
        has_output = any(
            cast(Union[int, float], p.get_device_info_by_index(i).get("maxOutputChannels", 0) or 0) > 0
            for i in range(p.get_device_count())
        )
        p.terminate()

        if not has_input:
            print("❌ No microphone found.")
            sys.exit(1)
        if not has_output:
            print("❌ No speakers found.")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Audio check failed: {e}")
        sys.exit(1)

    print("🎙️  GPT Realtime Model Inference – Azure AI Foundry")
    print("=" * 50)
    main()
