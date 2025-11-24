# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
from __future__ import annotations
import os
import sys
import argparse
import asyncio
import base64
from datetime import datetime
import logging
import queue
import signal
from typing import Union, Optional, TYPE_CHECKING, cast

from azure.core.credentials import AzureKeyCredential
from azure.core.credentials_async import AsyncTokenCredential
from azure.identity.aio import AzureCliCredential, DefaultAzureCredential

from azure.ai.voicelive.aio import connect
from azure.ai.voicelive.models import (
    AudioEchoCancellation,
    AudioNoiseReduction,
    AzureSemanticVad,
    AzureStandardVoice,
    EouDetection,
    InputAudioFormat,
    Modality,
    OutputAudioFormat,
    RequestSession,
    ServerEventType,
    ServerVad
)

from dotenv import load_dotenv
import pyaudio

import system_instructions

if TYPE_CHECKING:
    # Only needed for type checking; avoids runtime import issues
    from azure.ai.voicelive.aio import VoiceLiveConnection

## Change to the directory where this script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Environment variable loading
load_dotenv('./.env', override=True)

# Set up logging
## Add folder for logging
if not os.path.exists('logs'):
    os.makedirs('logs')

## Add timestamp for logfiles
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

## Set up logging
logging.basicConfig(
    filename=f'logs/{timestamp}_voicelive.log',
    filemode="w",
    format='%(asctime)s:%(name)s:%(levelname)s:%(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class AudioProcessor:
    """
    Handles real-time audio capture and playback for the voice assistant.

    Threading Architecture:
    - Main thread: Event loop and UI
    - Capture thread: PyAudio input stream reading
    - Send thread: Async audio data transmission to VoiceLive
    - Playback thread: PyAudio output stream writing
    """
    
    loop: asyncio.AbstractEventLoop
    
    class AudioPlaybackPacket:
        """Represents a packet that can be sent to the audio playback queue."""
        def __init__(self, seq_num: int, data: Optional[bytes]):
            self.seq_num = seq_num
            self.data = data

    def __init__(self, connection, sample_rate=24000):
        self.connection = connection
        self.audio = pyaudio.PyAudio()

        # Audio configuration - PCM16, mono as specified
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = sample_rate
        self.chunk_size = int(sample_rate * 0.05) # 50ms chunks

        # Capture and playback state
        self.input_stream = None

        self.playback_queue: queue.Queue[AudioProcessor.AudioPlaybackPacket] = queue.Queue()
        self.playback_base = 0
        self.next_seq_num = 0
        self.output_stream: Optional[pyaudio.Stream] = None

        logger.info("AudioProcessor initialized with 24kHz PCM16 mono audio")

    def start_capture(self):
        """Start capturing audio from microphone."""
        def _capture_callback(
            in_data,      # data
            _frame_count,  # number of frames
            _time_info,    # dictionary
            _status_flags):
            """Audio capture thread - runs in background."""
            audio_base64 = base64.b64encode(in_data).decode("utf-8")
            asyncio.run_coroutine_threadsafe(
                self.connection.input_audio_buffer.append(audio=audio_base64), self.loop
            )
            return (None, pyaudio.paContinue)

        if self.input_stream:
            return

        # Store the current event loop for use in threads
        self.loop = asyncio.get_event_loop()

        try:
            self.input_stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=_capture_callback,
            )
            logger.info("Started audio capture")

        except Exception:
            logger.exception("Failed to start audio capture")
            raise

    def start_playback(self):
        """Initialize audio playback system."""
        if self.output_stream:
            return

        remaining = bytes()
        def _playback_callback(
            _in_data,
            frame_count,  # number of frames
            _time_info,
            _status_flags):

            nonlocal remaining
            frame_count *= pyaudio.get_sample_size(pyaudio.paInt16)

            out = remaining[:frame_count]
            remaining = remaining[frame_count:]

            while len(out) < frame_count:
                try:
                    packet = self.playback_queue.get_nowait()
                except queue.Empty:
                    out = out + bytes(frame_count - len(out))
                    continue
                except Exception:
                    logger.exception("Error in audio playback")
                    raise

                if not packet or not packet.data:
                    # None packet indicates end of stream
                    logger.info("End of playback queue.")
                    break

                if packet.seq_num < self.playback_base:
                    # skip requested
                    # ignore skipped packet and clear remaining
                    if len(remaining) > 0:
                        remaining = bytes()
                    continue

                num_to_take = frame_count - len(out)
                out = out + packet.data[:num_to_take]
                remaining = packet.data[num_to_take:]

            if len(out) >= frame_count:
                return (out, pyaudio.paContinue)
            else:
                return (out, pyaudio.paComplete)

        try:
            self.output_stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                output=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=_playback_callback
            )
            logger.info("Audio playback system ready")
        except Exception:
            logger.exception("Failed to initialize audio playback")
            raise

    def _get_and_increase_seq_num(self):
        seq = self.next_seq_num
        self.next_seq_num += 1
        return seq

    def queue_audio(self, audio_data: Optional[bytes]) -> None:
        """Queue audio data for playback."""
        self.playback_queue.put(
            AudioProcessor.AudioPlaybackPacket(
                seq_num=self._get_and_increase_seq_num(),
                data=audio_data))

    def skip_pending_audio(self):
        """Skip current audio in playback queue."""
        self.playback_base = self._get_and_increase_seq_num()

    def shutdown(self):
        """Clean up audio resources."""
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
            self.input_stream = None

        logger.info("Stopped audio capture")

        # Inform thread to complete
        if self.output_stream:
            self.skip_pending_audio()
            self.queue_audio(None)
            self.output_stream.stop_stream()
            self.output_stream.close()
            self.output_stream = None

        logger.info("Stopped audio playback")

        if self.audio:
            self.audio.terminate()

        logger.info("Audio processor cleaned up")

class BasicVoiceAssistant:
    """Basic voice assistant implementing the VoiceLive SDK patterns."""

    def __init__(
        self,
        endpoint: str,
        credential: Union[AzureKeyCredential, AsyncTokenCredential],
        model: str,
        voice: str,
        instructions: str,
        voice_temperature: Optional[float] = None,
        voice_rate: Optional[str] = None,
        vad_type: str = "server_vad",
        vad_threshold: float = 0.8,
        vad_prefix_padding_ms: int = 300,
        vad_silence_duration_ms: int = 900,
        vad_speech_duration_ms: int = 80,
        vad_remove_filler_words: bool = False,
        vad_interrupt_response: bool = False,
        end_of_utterance_enabled: bool = False,
        end_of_utterance_model: str = "semantic_detection_v1",
        end_of_utterance_threshold_level: str = "default",
        end_of_utterance_timeout_ms: int = 1000,
        audio_sample_rate: int = 24000,
        noise_reduction_type: str = "azure_deep_noise_suppression",
        echo_cancellation_enabled: bool = True,
        greeting_delay: float = 2.0,
    ):

        self.endpoint = endpoint
        self.credential = credential
        self.model = model
        self.voice = voice
        self.instructions = instructions
        self.voice_temperature = voice_temperature
        self.voice_rate = voice_rate
        self.vad_type = vad_type
        self.vad_threshold = vad_threshold
        self.vad_prefix_padding_ms = vad_prefix_padding_ms
        self.vad_silence_duration_ms = vad_silence_duration_ms
        self.vad_speech_duration_ms = vad_speech_duration_ms
        self.vad_remove_filler_words = vad_remove_filler_words
        self.vad_interrupt_response = vad_interrupt_response
        self.end_of_utterance_enabled = end_of_utterance_enabled
        self.end_of_utterance_model = end_of_utterance_model
        self.end_of_utterance_threshold_level = end_of_utterance_threshold_level
        self.end_of_utterance_timeout_ms = end_of_utterance_timeout_ms
        self.audio_sample_rate = audio_sample_rate
        self.noise_reduction_type = noise_reduction_type
        self.echo_cancellation_enabled = echo_cancellation_enabled
        self.greeting_delay = greeting_delay
        self.connection: Optional["VoiceLiveConnection"] = None
        self.audio_processor: Optional[AudioProcessor] = None
        self.session_ready = False
        self._active_response = False
        self._response_api_done = False

    async def start(self):
        """Start the voice assistant session."""
        try:
            logger.info("Connecting to VoiceLive API with model %s", self.model)

            # Connect to VoiceLive WebSocket API
            async with connect(
                endpoint=self.endpoint,
                credential=self.credential,
                model=self.model,
            ) as connection:
                conn = connection
                self.connection = conn

                # Initialize audio processor
                ap = AudioProcessor(conn, sample_rate=self.audio_sample_rate)
                self.audio_processor = ap

                # Configure session for voice conversation
                await self._setup_session()

                # Start audio systems
                ap.start_playback()

                logger.info("Voice assistant ready! Start speaking...")
                print("\n" + "=" * 60)
                print("🎤 VOICE ASSISTANT READY")
                print("Start speaking to begin conversation")
                print("Press Ctrl+C to exit")
                print("=" * 60 + "\n")

                # Process events
                await self._process_events()
        finally:
            if self.audio_processor:
                self.audio_processor.shutdown()

    async def _setup_session(self):
        """Configure the VoiceLive session for audio conversation."""
        logger.info("Setting up voice conversation session...")

        # Create voice configuration
        voice_config: Union[AzureStandardVoice, str]
        if self.voice.startswith("en-US-") or self.voice.startswith("en-CA-") or "-" in self.voice:
            # Azure voice - create with optional temperature and rate
            voice_kwargs = {"name": self.voice}
            if self.voice_temperature is not None:
                voice_kwargs["temperature"] = self.voice_temperature
            if self.voice_rate is not None:
                voice_kwargs["rate"] = self.voice_rate
            voice_config = AzureStandardVoice(**voice_kwargs)
            logger.info(f"Azure voice configured: {self.voice} (temp={self.voice_temperature}, rate={self.voice_rate})")
        else:
            # OpenAI voice (alloy, echo, fable, onyx, nova, shimmer)
            voice_config = self.voice
            logger.info(f"OpenAI voice configured: {self.voice}")

        # Create turn detection configuration based on VAD type
        if self.vad_type == "azure_semantic_vad" or self.vad_type == "azure_multilingual_semantic_vad":
            # Azure Semantic VAD with advanced features
            vad_kwargs = {
                "threshold": self.vad_threshold,
                "prefix_padding_ms": self.vad_prefix_padding_ms,
                "silence_duration_ms": self.vad_silence_duration_ms,
                "speech_duration_ms": self.vad_speech_duration_ms,
                "remove_filler_words": self.vad_remove_filler_words,
                "interrupt_response": self.vad_interrupt_response,
            }
            
            # Add end of utterance detection if enabled
            if self.end_of_utterance_enabled:
                vad_kwargs["end_of_utterance_detection"] = EouDetection(
                    model=self.end_of_utterance_model
                )
                logger.info(f"End-of-utterance detection enabled: {self.end_of_utterance_model}")
            
            turn_detection_config = AzureSemanticVad(**vad_kwargs)
            logger.info(f"Using {self.vad_type} with threshold={self.vad_threshold}")
        else:
            # Standard Server VAD
            turn_detection_config = ServerVad(
                threshold=self.vad_threshold,
                prefix_padding_ms=self.vad_prefix_padding_ms,
                silence_duration_ms=self.vad_silence_duration_ms,
            )
            logger.info(f"Using server_vad with threshold={self.vad_threshold}")

        # Configure noise reduction
        noise_reduction = None
        if self.noise_reduction_type and self.noise_reduction_type != "none":
            noise_reduction = AudioNoiseReduction(type=self.noise_reduction_type)
            logger.info(f"Noise reduction enabled: {self.noise_reduction_type}")

        # Configure echo cancellation
        echo_cancellation = None
        if self.echo_cancellation_enabled:
            echo_cancellation = AudioEchoCancellation()
            logger.info("Echo cancellation enabled")

        # Create session configuration
        session_config = RequestSession(
            modalities=[Modality.TEXT, Modality.AUDIO],
            instructions=self.instructions,
            voice=voice_config,
            input_audio_format=InputAudioFormat.PCM16,
            output_audio_format=OutputAudioFormat.PCM16,
            turn_detection=turn_detection_config,
            input_audio_echo_cancellation=echo_cancellation,
            input_audio_noise_reduction=noise_reduction,
        )

        conn = self.connection
        assert conn is not None, "Connection must be established before setting up session"
        await conn.session.update(session=session_config)

        logger.info("Session configuration sent")

    async def _process_events(self):
        """Process events from the VoiceLive connection."""
        try:
            conn = self.connection
            assert conn is not None, "Connection must be established before processing events"
            async for event in conn:
                await self._handle_event(event)
        except Exception:
            logger.exception("Error processing events")
            raise

    async def _handle_event(self, event):
        """Handle different types of events from VoiceLive."""
        logger.debug("Received event: %s", event.type)
        ap = self.audio_processor
        conn = self.connection
        assert ap is not None, "AudioProcessor must be initialized"
        assert conn is not None, "Connection must be established"
        # assert keyword is used for debugging purposes to ensure that certain conditions are met or it will log the error.

        if event.type == ServerEventType.SESSION_UPDATED:
            logger.info("Session ready: %s", event.session.id)
            self.session_ready = True

            # Add this for proactive greeting:
            # if not hasattr(self, 'conversation_started') or not self.conversation_started:
            #     self.conversation_started = True
            await asyncio.sleep(self.greeting_delay)
            print("Agent initiated conversation...")
            await conn.response.create()

            # Start audio capture once session is ready
            ap.start_capture()

        elif event.type == ServerEventType.INPUT_AUDIO_BUFFER_SPEECH_STARTED:
            logger.info("User started speaking - stopping playback")
            print("🎤 Listening...")

            ap.skip_pending_audio()

            # Only cancel if response is active and not already done
            if self._active_response and not self._response_api_done:
                try:
                    await conn.response.cancel()
                    logger.debug("Cancelled in-progress response due to barge-in")
                except Exception as e:
                    if "no active response" in str(e).lower():
                        logger.debug("Cancel ignored - response already completed")
                    else:
                        logger.warning("Cancel failed: %s", e)

        elif event.type == ServerEventType.INPUT_AUDIO_BUFFER_SPEECH_STOPPED:
            logger.info("🎤 User stopped speaking")
            print("🤔 Processing...")

        elif event.type == ServerEventType.RESPONSE_CREATED:
            logger.info("🤖 Assistant response created")
            self._active_response = True
            self._response_api_done = False

        elif event.type == ServerEventType.RESPONSE_AUDIO_DELTA:
            logger.debug("Received audio delta")
            ap.queue_audio(event.delta)

        elif event.type == ServerEventType.RESPONSE_AUDIO_DONE:
            logger.info("🤖 Assistant finished speaking")
            print("🎤 Ready for next input...")

        elif event.type == ServerEventType.RESPONSE_DONE:
            logger.info("✅ Response complete")
            self._active_response = False
            self._response_api_done = True

        elif event.type == ServerEventType.ERROR:
            msg = event.error.message
            if "Cancellation failed: no active response" in msg:
                logger.debug("Benign cancellation error: %s", msg)
            else:
                logger.error("❌ VoiceLive error: %s", msg)
                print(f"Error: {msg}")

        elif event.type == ServerEventType.CONVERSATION_ITEM_CREATED:
            logger.debug("Conversation item created: %s", event.item.id)

        else:
            logger.debug("Unhandled event type: %s", event.type)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Basic Voice Assistant using Azure VoiceLive SDK",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--api-key",
        help="Azure VoiceLive API key. If not provided, will use AZURE_VOICELIVE_API_KEY environment variable.",
        type=str,
        default=os.environ.get("AZURE_VOICELIVE_API_KEY"),
    )

    parser.add_argument(
        "--endpoint",
        help="Azure VoiceLive endpoint",
        type=str,
        default=os.environ.get("AZURE_VOICELIVE_ENDPOINT", "https://your-resource-name.services.ai.azure.com/"),
    )

    parser.add_argument(
        "--model",
        help="VoiceLive model to use",
        type=str,
        default=os.environ.get("AZURE_VOICELIVE_MODEL", "gpt-realtime"),
    )

    parser.add_argument(
        "--voice",
        help="Voice to use for the assistant. E.g. alloy, echo, fable, en-US-AvaNeural, en-US-GuyNeural",
        type=str,
        default=os.environ.get("AZURE_VOICELIVE_VOICE", "en-US-Ava:DragonHDLatestNeural"),
    )

    # Option 1 :- Fetching system messages from .env file

    # parser.add_argument(
    #     "--instructions",
    #     help="System instructions for the AI assistant",
    #     type=str,
    #     default=os.environ.get(
    #         "AZURE_VOICELIVE_INSTRUCTIONS",
    #         "You are a helpful AI assistant. Respond naturally and conversationally. "
    #         "Keep your responses concise but engaging.",
    #     ),
    # )

    # Option 2 :- Fetching system messages from system_instructions.py file

    parser.add_argument(
        "--instructions",
        help="System instructions for the AI assistant",
        type=str,
        default=(
            system_instructions.AZURE_VOICELIVE_INSTRUCTIONS.strip()
            or "You are a helpful AI assistant. Respond naturally and conversationally. "
            "Keep your responses concise but engaging."
    ),
    )

    # Option 3 :- Priority to system_instructions.py file, fallback to .env file

    # parser.add_argument(
    # "--instructions",
    # help="System instructions for the AI assistant",
    # type=str,
    # default=(
    #     system_instructions.AZURE_VOICELIVE_INSTRUCTIONS.strip() 
    #     or os.environ.get("AZURE_VOICELIVE_INSTRUCTIONS")
    #     or "You are a helpful AI assistant. Respond naturally and conversationally. "
    #        "Keep your responses concise but engaging."
    # ),
    # )

    # ============================================
    # Voice Configuration Parameters
    # ============================================
    parser.add_argument(
        "--voice-temperature",
        help="Voice temperature for HD voices (0.0-2.0, controls variability)",
        type=float,
        default=float(os.environ.get("AZURE_VOICELIVE_VOICE_TEMPERATURE", "0.8")) if os.environ.get("AZURE_VOICELIVE_VOICE_TEMPERATURE") else None,
    )

    parser.add_argument(
        "--voice-rate",
        help="Speaking rate (0.5-1.5, controls speed)",
        type=str,
        default=os.environ.get("AZURE_VOICELIVE_VOICE_RATE", "1.0") if os.environ.get("AZURE_VOICELIVE_VOICE_RATE") else None,
    )

    # ============================================
    # Turn Detection (VAD) Configuration
    # ============================================
    parser.add_argument(
        "--vad-type",
        help="VAD type: server_vad, azure_semantic_vad, or azure_multilingual_semantic_vad",
        type=str,
        default=os.environ.get("AZURE_VOICELIVE_VAD_TYPE", "server_vad"),
        choices=["server_vad", "azure_semantic_vad", "azure_multilingual_semantic_vad"],
    )

    parser.add_argument(
        "--vad-threshold",
        help="VAD threshold (0.0-1.0, higher = less sensitive)",
        type=float,
        default=float(os.environ.get("AZURE_VOICELIVE_VAD_THRESHOLD", "0.8")),
    )

    parser.add_argument(
        "--vad-prefix-padding-ms",
        help="Audio captured before speech start (milliseconds)",
        type=int,
        default=int(os.environ.get("AZURE_VOICELIVE_VAD_PREFIX_PADDING_MS", "300")),
    )

    parser.add_argument(
        "--vad-silence-duration-ms",
        help="Silence duration to detect speech end (milliseconds)",
        type=int,
        default=int(os.environ.get("AZURE_VOICELIVE_VAD_SILENCE_DURATION_MS", "900")),
    )

    parser.add_argument(
        "--vad-speech-duration-ms",
        help="Minimum speech duration to start detection (milliseconds)",
        type=int,
        default=int(os.environ.get("AZURE_VOICELIVE_VAD_SPEECH_DURATION_MS", "80")),
    )

    parser.add_argument(
        "--vad-remove-filler-words",
        help="Remove filler words like 'umm', 'ah'",
        action="store_true",
        default=os.environ.get("AZURE_VOICELIVE_VAD_REMOVE_FILLER_WORDS", "false").lower() == "true",
    )

    parser.add_argument(
        "--vad-interrupt-response",
        help="Enable barge-in interruption",
        action="store_true",
        default=os.environ.get("AZURE_VOICELIVE_VAD_INTERRUPT_RESPONSE", "false").lower() == "true",
    )

    # ============================================
    # End of Utterance Detection
    # ============================================
    parser.add_argument(
        "--end-of-utterance-enabled",
        help="Enable advanced end-of-utterance detection",
        action="store_true",
        default=os.environ.get("AZURE_VOICELIVE_END_OF_UTTERANCE_ENABLED", "false").lower() == "true",
    )

    parser.add_argument(
        "--end-of-utterance-model",
        help="End-of-utterance model: semantic_detection_v1 or semantic_detection_v1_multilingual",
        type=str,
        default=os.environ.get("AZURE_VOICELIVE_END_OF_UTTERANCE_MODEL", "semantic_detection_v1"),
        choices=["semantic_detection_v1", "semantic_detection_v1_multilingual"],
    )

    parser.add_argument(
        "--end-of-utterance-threshold-level",
        help="Threshold level: low, medium, high, or default",
        type=str,
        default=os.environ.get("AZURE_VOICELIVE_END_OF_UTTERANCE_THRESHOLD_LEVEL", "default"),
        choices=["low", "medium", "high", "default"],
    )

    parser.add_argument(
        "--end-of-utterance-timeout-ms",
        help="Maximum wait time for detection (milliseconds)",
        type=int,
        default=int(os.environ.get("AZURE_VOICELIVE_END_OF_UTTERANCE_TIMEOUT_MS", "1000")),
    )

    # ============================================
    # Audio Configuration
    # ============================================
    parser.add_argument(
        "--audio-sample-rate",
        help="Audio sample rate: 16000 or 24000",
        type=int,
        default=int(os.environ.get("AZURE_VOICELIVE_AUDIO_SAMPLE_RATE", "24000")),
        choices=[16000, 24000],
    )

    parser.add_argument(
        "--noise-reduction-type",
        help="Noise reduction type: azure_deep_noise_suppression or none",
        type=str,
        default=os.environ.get("AZURE_VOICELIVE_NOISE_REDUCTION_TYPE", "azure_deep_noise_suppression"),
        choices=["azure_deep_noise_suppression", "none"],
    )

    parser.add_argument(
        "--echo-cancellation-enabled",
        help="Enable echo cancellation",
        action="store_true",
        default=os.environ.get("AZURE_VOICELIVE_ECHO_CANCELLATION_ENABLED", "true").lower() == "true",
    )

    parser.add_argument(
        "--no-echo-cancellation",
        help="Disable echo cancellation",
        action="store_true",
        default=False,
    )

    # ============================================
    # Additional Features
    # ============================================
    parser.add_argument(
        "--greeting-delay",
        help="Proactive greeting delay in seconds",
        type=float,
        default=float(os.environ.get("AZURE_VOICELIVE_GREETING_DELAY", "2.0")),
    )

    parser.add_argument(
        "--use-token-credential", help="Use Azure token credential instead of API key", action="store_true", default=False
    )

    parser.add_argument("--verbose", help="Enable verbose logging", action="store_true")

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate credentials
    if not args.api_key and not args.use_token_credential:
        print("❌ Error: No authentication provided")
        print("Please provide an API key using --api-key or set AZURE_VOICELIVE_API_KEY environment variable,")
        print("or use --use-token-credential for Azure authentication.")
        sys.exit(1)

    # Create client with appropriate credential
    credential: Union[AzureKeyCredential, AsyncTokenCredential]
    if args.use_token_credential:
        credential = AzureCliCredential()  # or DefaultAzureCredential() if needed
        logger.info("Using Azure token credential")
    else:
        credential = AzureKeyCredential(args.api_key)
        logger.info("Using API key credential")

    # Handle echo cancellation flag override
    echo_cancellation_enabled = args.echo_cancellation_enabled and not args.no_echo_cancellation

    # Create and start voice assistant
    assistant = BasicVoiceAssistant(
        endpoint=args.endpoint,
        credential=credential,
        model=args.model,
        voice=args.voice,
        instructions=args.instructions,
        voice_temperature=args.voice_temperature,
        voice_rate=args.voice_rate,
        vad_type=args.vad_type,
        vad_threshold=args.vad_threshold,
        vad_prefix_padding_ms=args.vad_prefix_padding_ms,
        vad_silence_duration_ms=args.vad_silence_duration_ms,
        vad_speech_duration_ms=args.vad_speech_duration_ms,
        vad_remove_filler_words=args.vad_remove_filler_words,
        vad_interrupt_response=args.vad_interrupt_response,
        end_of_utterance_enabled=args.end_of_utterance_enabled,
        end_of_utterance_model=args.end_of_utterance_model,
        end_of_utterance_threshold_level=args.end_of_utterance_threshold_level,
        end_of_utterance_timeout_ms=args.end_of_utterance_timeout_ms,
        audio_sample_rate=args.audio_sample_rate,
        noise_reduction_type=args.noise_reduction_type,
        echo_cancellation_enabled=echo_cancellation_enabled,
        greeting_delay=args.greeting_delay,
    )

    # Setup signal handlers for graceful shutdown
    def signal_handler(_sig, _frame):
        logger.info("Received shutdown signal")
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start the assistant
    try:
        asyncio.run(assistant.start())
    except KeyboardInterrupt:
        print("\n👋 Voice assistant shut down. Goodbye!")
    except Exception as e:
        print("Fatal Error: ", e)

if __name__ == "__main__":
    # Check audio system
    try:
        p = pyaudio.PyAudio()
        # Check for input devices
        input_devices = [
            i
            for i in range(p.get_device_count())
            if cast(Union[int, float], p.get_device_info_by_index(i).get("maxInputChannels", 0) or 0) > 0
        ]
        # Check for output devices
        output_devices = [
            i
            for i in range(p.get_device_count())
            if cast(Union[int, float], p.get_device_info_by_index(i).get("maxOutputChannels", 0) or 0) > 0
        ]
        p.terminate()

        if not input_devices:
            print("❌ No audio input devices found. Please check your microphone.")
            sys.exit(1)
        if not output_devices:
            print("❌ No audio output devices found. Please check your speakers.")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Audio system check failed: {e}")
        sys.exit(1)

    print("🎙️  Basic Voice Assistant with Azure VoiceLive SDK")
    print("=" * 50)

    # Run the assistant
    main()
