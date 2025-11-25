# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""
Voice Live MCP Integration Example

This example demonstrates how to integrate MCP (Model Context Protocol) servers
with Azure Voice Live API for real-time voice interactions with tool calling.
"""
from __future__ import annotations
import os
import sys
import argparse
import asyncio
import base64
from datetime import datetime
import json
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
    from azure.ai.voicelive.aio import VoiceLiveConnection

## Change to the directory where this script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Environment variable loading
load_dotenv('./.env', override=True)

# Set up logging
if not os.path.exists('logs'):
    os.makedirs('logs')

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

logging.basicConfig(
    filename=f'logs/{timestamp}_mcp_voicelive.log',
    filemode="w",
    format='%(asctime)s:%(name)s:%(levelname)s:%(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class AudioProcessor:
    """Handles real-time audio capture and playback for the voice assistant."""
    
    loop: asyncio.AbstractEventLoop
    
    class AudioPlaybackPacket:
        """Represents a packet that can be sent to the audio playback queue."""
        def __init__(self, seq_num: int, data: Optional[bytes]):
            self.seq_num = seq_num
            self.data = data

    def __init__(self, connection, sample_rate=24000):
        self.connection = connection
        self.audio = pyaudio.PyAudio()

        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = sample_rate
        self.chunk_size = int(sample_rate * 0.05)

        self.input_stream = None
        self.playback_queue: queue.Queue[AudioProcessor.AudioPlaybackPacket] = queue.Queue()
        self.playback_base = 0
        self.next_seq_num = 0
        self.output_stream: Optional[pyaudio.Stream] = None

        logger.info("AudioProcessor initialized with %dkHz PCM16 mono audio", sample_rate // 1000)

    def start_capture(self):
        """Start capturing audio from microphone."""
        def _capture_callback(in_data, _frame_count, _time_info, _status_flags):
            audio_base64 = base64.b64encode(in_data).decode("utf-8")
            asyncio.run_coroutine_threadsafe(
                self.connection.input_audio_buffer.append(audio=audio_base64), self.loop
            )
            return (None, pyaudio.paContinue)

        if self.input_stream:
            return

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
        def _playback_callback(_in_data, frame_count, _time_info, _status_flags):
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
                    logger.info("End of playback queue.")
                    break

                if packet.seq_num < self.playback_base:
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


class MCPVoiceAssistant:
    """Voice assistant with MCP (Model Context Protocol) integration."""

    def __init__(
        self,
        endpoint: str,
        credential: Union[AzureKeyCredential, AsyncTokenCredential],
        model: str,
        voice: str,
        instructions: str,
        mcp_server_url: Optional[str] = None,
        mcp_server_label: Optional[str] = None,
        mcp_allowed_tools: Optional[list[str]] = None,
        mcp_require_approval: str = "always",
        mcp_authorization: Optional[str] = None,
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
        
        # MCP Configuration
        self.mcp_server_url = mcp_server_url
        self.mcp_server_label = mcp_server_label or "mcp_server"
        self.mcp_allowed_tools = mcp_allowed_tools
        self.mcp_require_approval = mcp_require_approval
        self.mcp_authorization = mcp_authorization
        
        # Voice and VAD configuration
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
        self.audio_sample_rate = audio_sample_rate
        self.noise_reduction_type = noise_reduction_type
        self.echo_cancellation_enabled = echo_cancellation_enabled
        self.greeting_delay = greeting_delay
        
        self.connection: Optional["VoiceLiveConnection"] = None
        self.audio_processor: Optional[AudioProcessor] = None
        self.session_ready = False
        self._active_response = False
        self._response_api_done = False
        
        # MCP tracking
        self.pending_mcp_approvals = {}

    async def start(self):
        """Start the voice assistant session."""
        try:
            logger.info("Connecting to VoiceLive API with model %s", self.model)
            if self.mcp_server_url:
                logger.info("MCP Server configured: %s (label: %s)", self.mcp_server_url, self.mcp_server_label)

            async with connect(
                endpoint=self.endpoint,
                credential=self.credential,
                model=self.model,
            ) as connection:
                conn = connection
                self.connection = conn

                ap = AudioProcessor(conn, sample_rate=self.audio_sample_rate)
                self.audio_processor = ap

                await self._setup_session()

                ap.start_playback()

                logger.info("Voice assistant ready! Start speaking...")
                print("\n" + "=" * 60)
                print("🎤 VOICE ASSISTANT WITH MCP INTEGRATION")
                if self.mcp_server_url:
                    print(f"🔧 MCP Server: {self.mcp_server_label}")
                    print(f"   URL: {self.mcp_server_url}")
                    print(f"   Approval: {self.mcp_require_approval}")
                print("Start speaking to begin conversation")
                print("Press Ctrl+C to exit")
                print("=" * 60 + "\n")

                await self._process_events()
        finally:
            if self.audio_processor:
                self.audio_processor.shutdown()

    async def _setup_session(self):
        """Configure the VoiceLive session with MCP tools."""
        logger.info("Setting up voice conversation session with MCP integration...")

        # Create voice configuration
        voice_config: Union[AzureStandardVoice, str]
        if self.voice.startswith("en-US-") or self.voice.startswith("en-CA-") or "-" in self.voice:
            voice_kwargs = {"name": self.voice}
            if self.voice_temperature is not None:
                voice_kwargs["temperature"] = self.voice_temperature
            if self.voice_rate is not None:
                voice_kwargs["rate"] = self.voice_rate
            voice_config = AzureStandardVoice(**voice_kwargs)
            logger.info(f"Azure voice configured: {self.voice}")
        else:
            voice_config = self.voice
            logger.info(f"OpenAI voice configured: {self.voice}")

        # Create turn detection configuration
        if self.vad_type in ["azure_semantic_vad", "azure_multilingual_semantic_vad"]:
            vad_kwargs = {
                "threshold": self.vad_threshold,
                "prefix_padding_ms": self.vad_prefix_padding_ms,
                "silence_duration_ms": self.vad_silence_duration_ms,
                "speech_duration_ms": self.vad_speech_duration_ms,
                "remove_filler_words": self.vad_remove_filler_words,
                "interrupt_response": self.vad_interrupt_response,
            }
            
            if self.end_of_utterance_enabled:
                vad_kwargs["end_of_utterance_detection"] = EouDetection(
                    model=self.end_of_utterance_model
                )
                logger.info(f"End-of-utterance detection enabled: {self.end_of_utterance_model}")
            
            turn_detection_config = AzureSemanticVad(**vad_kwargs)
            logger.info(f"Using {self.vad_type}")
        else:
            turn_detection_config = ServerVad(
                threshold=self.vad_threshold,
                prefix_padding_ms=self.vad_prefix_padding_ms,
                silence_duration_ms=self.vad_silence_duration_ms,
            )
            logger.info("Using server_vad")

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

        # Configure MCP tools if server URL is provided
        tools = []
        if self.mcp_server_url:
            # MCP tool configuration as a dictionary (per Voice Live API spec)
            mcp_tool = {
                "type": "mcp",
                "server_label": self.mcp_server_label,
                "server_url": self.mcp_server_url,
                "require_approval": self.mcp_require_approval,
            }
            
            if self.mcp_allowed_tools:
                mcp_tool["allowed_tools"] = self.mcp_allowed_tools
                logger.info(f"MCP allowed tools: {self.mcp_allowed_tools}")
            
            if self.mcp_authorization:
                mcp_tool["authorization"] = self.mcp_authorization
                logger.info("MCP authorization configured")
            
            tools.append(mcp_tool)
            logger.info(f"MCP tool configured: {self.mcp_server_label}")

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
            tools=tools if tools else None,
        )

        conn = self.connection
        assert conn is not None, "Connection must be established before setting up session"
        await conn.session.update(session=session_config)

        logger.info("Session configuration sent with MCP tools")

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

        if event.type == ServerEventType.SESSION_UPDATED:
            logger.info("Session ready: %s", event.session.id)
            self.session_ready = True

            await asyncio.sleep(self.greeting_delay)
            print("Agent initiated conversation...")
            await conn.response.create()

            ap.start_capture()

        elif event.type == ServerEventType.INPUT_AUDIO_BUFFER_SPEECH_STARTED:
            logger.info("User started speaking - stopping playback")
            print("🎤 Listening...")

            ap.skip_pending_audio()

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

        # MCP-specific events
        elif event.type == ServerEventType.MCP_LIST_TOOLS_IN_PROGRESS:
            logger.info("🔧 MCP: Listing tools from server...")
            print("🔧 Discovering MCP tools...")

        elif event.type == ServerEventType.MCP_LIST_TOOLS_COMPLETED:
            logger.info("✅ MCP: Tool listing completed")
            print("✅ MCP tools ready")

        elif event.type == ServerEventType.MCP_LIST_TOOLS_FAILED:
            logger.error("❌ MCP: Tool listing failed")
            print("❌ Failed to list MCP tools")

        elif event.type == ServerEventType.RESPONSE_MCP_CALL_ARGUMENTS_DELTA:
            logger.debug("MCP call arguments streaming")

        elif event.type == ServerEventType.RESPONSE_MCP_CALL_ARGUMENTS_DONE:
            logger.info("🔧 MCP call arguments complete")
            print(f"🔧 MCP Tool Call: {event.item_id}")

        elif event.type == ServerEventType.RESPONSE_MCP_CALL_IN_PROGRESS:
            logger.info("⚙️  MCP call executing...")
            print("⚙️  Executing MCP tool...")

        elif event.type == ServerEventType.RESPONSE_MCP_CALL_COMPLETED:
            logger.info("✅ MCP call completed successfully")
            print("✅ MCP tool execution complete")

        elif event.type == ServerEventType.RESPONSE_MCP_CALL_FAILED:
            logger.error("❌ MCP call failed")
            print("❌ MCP tool execution failed")

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
        description="Voice Assistant with MCP Integration using Azure VoiceLive SDK",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Authentication
    parser.add_argument(
        "--api-key",
        help="Azure VoiceLive API key",
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

    # MCP Configuration
    parser.add_argument(
        "--mcp-server-url",
        help="MCP server URL (e.g., https://your-mcp-server.com)",
        type=str,
        default=os.environ.get("MCP_SERVER_URL"),
    )

    parser.add_argument(
        "--mcp-server-label",
        help="Label for the MCP server",
        type=str,
        default=os.environ.get("MCP_SERVER_LABEL", "mcp_server"),
    )

    parser.add_argument(
        "--mcp-allowed-tools",
        help="Comma-separated list of allowed MCP tools",
        type=str,
        default=os.environ.get("MCP_ALLOWED_TOOLS"),
    )

    parser.add_argument(
        "--mcp-require-approval",
        help="MCP approval mode: always, never, or JSON config",
        type=str,
        default=os.environ.get("MCP_REQUIRE_APPROVAL", "always"),
    )

    parser.add_argument(
        "--mcp-authorization",
        help="Authorization token for MCP requests",
        type=str,
        default=os.environ.get("MCP_AUTHORIZATION"),
    )

    # Voice Configuration
    parser.add_argument(
        "--voice",
        help="Voice to use (e.g., alloy, echo, en-US-AvaNeural)",
        type=str,
        default=os.environ.get("AZURE_VOICELIVE_VOICE", "en-US-Ava:DragonHDLatestNeural"),
    )

    parser.add_argument(
        "--instructions",
        help="System instructions for the AI assistant",
        type=str,
        default=(
            system_instructions.AZURE_VOICELIVE_INSTRUCTIONS.strip()
            or "You are a helpful AI assistant with access to external tools via MCP. "
            "Use the available tools when appropriate to help the user."
        ),
    )

    parser.add_argument(
        "--voice-temperature",
        help="Voice temperature (0.0-2.0)",
        type=float,
        default=float(os.environ.get("AZURE_VOICELIVE_VOICE_TEMPERATURE", "0.8")) if os.environ.get("AZURE_VOICELIVE_VOICE_TEMPERATURE") else None,
    )

    parser.add_argument(
        "--voice-rate",
        help="Speaking rate (0.5-1.5)",
        type=str,
        default=os.environ.get("AZURE_VOICELIVE_VOICE_RATE"),
    )

    # VAD Configuration
    parser.add_argument(
        "--vad-type",
        help="VAD type",
        type=str,
        default=os.environ.get("AZURE_VOICELIVE_VAD_TYPE", "server_vad"),
        choices=["server_vad", "azure_semantic_vad", "azure_multilingual_semantic_vad"],
    )

    parser.add_argument(
        "--vad-threshold",
        help="VAD threshold (0.0-1.0)",
        type=float,
        default=float(os.environ.get("AZURE_VOICELIVE_VAD_THRESHOLD", "0.8")),
    )

    parser.add_argument(
        "--vad-prefix-padding-ms",
        help="Audio captured before speech start (ms)",
        type=int,
        default=int(os.environ.get("AZURE_VOICELIVE_VAD_PREFIX_PADDING_MS", "300")),
    )

    parser.add_argument(
        "--vad-silence-duration-ms",
        help="Silence duration to detect speech end (ms)",
        type=int,
        default=int(os.environ.get("AZURE_VOICELIVE_VAD_SILENCE_DURATION_MS", "900")),
    )

    parser.add_argument(
        "--vad-speech-duration-ms",
        help="Minimum speech duration (ms)",
        type=int,
        default=int(os.environ.get("AZURE_VOICELIVE_VAD_SPEECH_DURATION_MS", "80")),
    )

    parser.add_argument(
        "--vad-remove-filler-words",
        help="Remove filler words",
        action="store_true",
        default=os.environ.get("AZURE_VOICELIVE_VAD_REMOVE_FILLER_WORDS", "false").lower() == "true",
    )

    parser.add_argument(
        "--vad-interrupt-response",
        help="Enable barge-in interruption",
        action="store_true",
        default=os.environ.get("AZURE_VOICELIVE_VAD_INTERRUPT_RESPONSE", "false").lower() == "true",
    )

    # End of Utterance Detection
    parser.add_argument(
        "--end-of-utterance-enabled",
        help="Enable end-of-utterance detection",
        action="store_true",
        default=os.environ.get("AZURE_VOICELIVE_END_OF_UTTERANCE_ENABLED", "false").lower() == "true",
    )

    parser.add_argument(
        "--end-of-utterance-model",
        help="End-of-utterance model",
        type=str,
        default=os.environ.get("AZURE_VOICELIVE_END_OF_UTTERANCE_MODEL", "semantic_detection_v1"),
        choices=["semantic_detection_v1", "semantic_detection_v1_multilingual"],
    )

    # Audio Configuration
    parser.add_argument(
        "--audio-sample-rate",
        help="Audio sample rate",
        type=int,
        default=int(os.environ.get("AZURE_VOICELIVE_AUDIO_SAMPLE_RATE", "24000")),
        choices=[16000, 24000],
    )

    parser.add_argument(
        "--noise-reduction-type",
        help="Noise reduction type",
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

    # Additional Features
    parser.add_argument(
        "--greeting-delay",
        help="Proactive greeting delay (seconds)",
        type=float,
        default=float(os.environ.get("AZURE_VOICELIVE_GREETING_DELAY", "2.0")),
    )

    parser.add_argument(
        "--use-token-credential",
        help="Use Azure token credential",
        action="store_true",
        default=False
    )

    parser.add_argument("--verbose", help="Enable verbose logging", action="store_true")

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.api_key and not args.use_token_credential:
        print("❌ Error: No authentication provided")
        print("Please provide an API key using --api-key or set AZURE_VOICELIVE_API_KEY,")
        print("or use --use-token-credential for Azure authentication.")
        sys.exit(1)

    if not args.mcp_server_url:
        print("⚠️  Warning: No MCP server configured")
        print("Use --mcp-server-url or set MCP_SERVER_URL environment variable")
        print("Continuing without MCP integration...\n")

    credential: Union[AzureKeyCredential, AsyncTokenCredential]
    if args.use_token_credential:
        credential = AzureCliCredential()
        logger.info("Using Azure token credential")
    else:
        credential = AzureKeyCredential(args.api_key)
        logger.info("Using API key credential")

    echo_cancellation_enabled = args.echo_cancellation_enabled and not args.no_echo_cancellation

    # Parse MCP allowed tools if provided
    mcp_allowed_tools = None
    if args.mcp_allowed_tools:
        mcp_allowed_tools = [tool.strip() for tool in args.mcp_allowed_tools.split(",")]

    assistant = MCPVoiceAssistant(
        endpoint=args.endpoint,
        credential=credential,
        model=args.model,
        voice=args.voice,
        instructions=args.instructions,
        mcp_server_url=args.mcp_server_url,
        mcp_server_label=args.mcp_server_label,
        mcp_allowed_tools=mcp_allowed_tools,
        mcp_require_approval=args.mcp_require_approval,
        mcp_authorization=args.mcp_authorization,
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
        audio_sample_rate=args.audio_sample_rate,
        noise_reduction_type=args.noise_reduction_type,
        echo_cancellation_enabled=echo_cancellation_enabled,
        greeting_delay=args.greeting_delay,
    )

    def signal_handler(_sig, _frame):
        logger.info("Received shutdown signal")
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        asyncio.run(assistant.start())
    except KeyboardInterrupt:
        print("\n👋 Voice assistant shut down. Goodbye!")
    except Exception as e:
        logger.exception("Fatal error")
        print(f"Fatal Error: {e}")


if __name__ == "__main__":
    try:
        p = pyaudio.PyAudio()
        input_devices = [
            i
            for i in range(p.get_device_count())
            if cast(Union[int, float], p.get_device_info_by_index(i).get("maxInputChannels", 0) or 0) > 0
        ]
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

    print("🎙️  Voice Assistant with MCP Integration")
    print("=" * 50)

    main()
