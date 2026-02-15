# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""
Human-in-the-Loop (HITL) Voice Assistant with Agent Escalation

This implementation adds functionality to:
1. Detect when the AI cannot answer a query
2. Route the call to a human agent
3. Gracefully close the VoiceLive WebSocket connection
4. Provide handoff context to the human agent

The AI can trigger escalation via:
- Explicit function call: transfer_to_human_agent()
- Detection of uncertainty keywords
- Repeated failed attempts to answer
"""
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
import json
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
    ServerVad,
    FunctionTool,
    # ToolChoice
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
## Add folder for logging
if not os.path.exists('logs'):
    os.makedirs('logs')

## Add timestamp for logfiles
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

## Set up logging
logging.basicConfig(
    filename=f'logs/{timestamp}_hitl_voicelive.log',
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


class HITLVoiceAssistant:
    """
    Human-in-the-Loop Voice Assistant with agent escalation capabilities.
    
    Features:
    - Function calling to detect when to transfer to human
    - Conversation history tracking for handoff context
    - Graceful WebSocket closure
    - Integration hooks for human agent systems
    """

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
        human_agent_endpoint: Optional[str] = None,
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
        self.human_agent_endpoint = human_agent_endpoint
        
        self.connection: Optional["VoiceLiveConnection"] = None
        self.audio_processor: Optional[AudioProcessor] = None
        self.session_ready = False
        self._active_response = False
        self._response_api_done = False
        self.conversation_history = []
        self._current_user_transcript = ""
        self._current_assistant_response = ""
        
        # HITL-specific state
        self._escalation_requested = False
        self._escalation_reason = ""
        self._should_exit = False

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

                # Configure session for voice conversation with HITL tools
                await self._setup_session()

                # Start audio systems
                ap.start_playback()

                logger.info("Voice assistant ready! Start speaking...")
                print("\n" + "=" * 60)
                print("🎤 HITL VOICE ASSISTANT READY")
                print("AI will route to human agent when needed")
                print("Start speaking to begin conversation")
                print("Press Ctrl+C to exit")
                print("=" * 60 + "\n")

                # Process events
                await self._process_events()
                
        finally:
            # Save conversation history and handle escalation
            self._save_conversation_history()
            
            if self._escalation_requested:
                await self._handle_human_agent_transfer()
            
            if self.audio_processor:
                self.audio_processor.shutdown()

    def _save_conversation_history(self):
        """Save conversation history to a JSON file."""
        if not self.conversation_history:
            return
        
        try:
            history_file = f'logs/{timestamp}_hitl_conversation_history.json'
            
            # Add escalation metadata if applicable
            metadata = {
                "session_id": timestamp,
                "escalation_requested": self._escalation_requested,
                "escalation_reason": self._escalation_reason,
                "conversation": self.conversation_history
            }
            
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"Conversation history saved to {history_file}")
            print(f"\n💾 Conversation history saved to {history_file}")
        except Exception as e:
            logger.error(f"Failed to save conversation history: {e}")

    async def _handle_human_agent_transfer(self):
        """
        Handle the transfer to a human agent.
        
        This is a framework method that you would customize to integrate with your
        actual call center/human agent system (e.g., Twilio, Azure Communication Services,
        custom telephony system, etc.)
        """
        logger.info("=" * 60)
        logger.info("ESCALATING TO HUMAN AGENT")
        logger.info(f"Reason: {self._escalation_reason}")
        logger.info("=" * 60)
        
        print("\n" + "=" * 60)
        print("🚀 TRANSFERRING TO HUMAN AGENT")
        print(f"📋 Reason: {self._escalation_reason}")
        print("=" * 60)
        
        # Prepare handoff context for human agent
        handoff_context = {
            "timestamp": datetime.now().isoformat(),
            "session_id": timestamp,
            "reason": self._escalation_reason,
            "conversation_summary": self._generate_conversation_summary(),
            "last_user_message": self._current_user_transcript,
            "conversation_history": self.conversation_history
        }
        
        # Save handoff context
        handoff_file = f'logs/{timestamp}_handoff_context.json'
        try:
            with open(handoff_file, 'w', encoding='utf-8') as f:
                json.dump(handoff_context, f, indent=2, ensure_ascii=False)
            logger.info(f"Handoff context saved to {handoff_file}")
            print(f"📄 Handoff context saved: {handoff_file}")
        except Exception as e:
            logger.error(f"Failed to save handoff context: {e}")
        
        # TODO: Integrate with your human agent system here
        # Examples:
        # - API call to your call center routing system
        # - Create a ticket in your support system
        # - Send notification to available agents
        # - Transfer the audio stream to human agent
        
        if self.human_agent_endpoint:
            logger.info(f"Would connect to human agent endpoint: {self.human_agent_endpoint}")
            print(f"🔗 Human agent endpoint: {self.human_agent_endpoint}")
            # Add your integration code here **Note - custom code to route the call**
            # await self._connect_to_human_agent(handoff_context)
        else:
            print("\n⚠️  No human agent endpoint configured.")
            print("To integrate with your call center:")
            print("1. Set --human-agent-endpoint parameter")
            print("2. Implement integration in _handle_human_agent_transfer()")
            print(f"3. Use handoff context from: {handoff_file}")

    def _generate_conversation_summary(self) -> str:
        """Generate a brief summary of the conversation for human agent context."""
        if not self.conversation_history:
            return "No conversation history available."
        
        summary_parts = []
        for item in self.conversation_history[-5:]:  # Last 5 items
            role = item.get('role', 'unknown')
            content = item.get('content', [])
            
            for content_item in content:
                if content_item.get('type') == 'input_text' and 'transcript' in content_item:
                    summary_parts.append(f"User: {content_item['transcript']}")
                elif content_item.get('type') == 'text' and 'text' in content_item:
                    summary_parts.append(f"Assistant: {content_item['text']}")
        
        return " | ".join(summary_parts[-10:]) if summary_parts else "Brief conversation"

    async def _setup_session(self):
        """Configure the VoiceLive session for audio conversation with HITL tools."""
        logger.info("Setting up HITL voice conversation session...")

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
        if self.vad_type == "azure_semantic_vad" or self.vad_type == "azure_multilingual_semantic_vad":
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
            
            turn_detection_config = AzureSemanticVad(**vad_kwargs)
        else:
            turn_detection_config = ServerVad(
                threshold=self.vad_threshold,
                prefix_padding_ms=self.vad_prefix_padding_ms,
                silence_duration_ms=self.vad_silence_duration_ms,
            )

        # Configure noise reduction
        noise_reduction = None
        if self.noise_reduction_type and self.noise_reduction_type != "none":
            noise_reduction = AudioNoiseReduction(type=self.noise_reduction_type)

        # Configure echo cancellation
        echo_cancellation = None
        if self.echo_cancellation_enabled:
            echo_cancellation = AudioEchoCancellation()

        # Define human agent transfer function
        transfer_function = FunctionTool(
            name="transfer_to_human_agent",
            description=(
                "Transfer the call to a human agent when the AI cannot adequately help the customer. "
                "Use this when:\n"
                "- The query is too complex or outside your knowledge\n"
                "- The customer explicitly requests a human agent\n"
                "- You've failed to resolve the issue after multiple attempts\n"
                "- The situation requires human judgment or empathy\n"
                "- Legal, financial, or sensitive matters that need human oversight"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Brief explanation of why human agent transfer is needed"
                    },
                    "urgency": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "critical"],
                        "description": "Urgency level of the transfer"
                    },
                    "category": {
                        "type": "string",
                        "description": "Category of the issue (e.g., technical, billing, complaint, etc.)"
                    }
                },
                "required": ["reason", "urgency"]
            }
        )

        # Enhanced instructions with HITL guidance
        enhanced_instructions = self.instructions + """

IMPORTANT - Human Agent Escalation:
You have access to a function called 'transfer_to_human_agent'. Use it when:
1. You genuinely cannot answer the customer's question
2. The customer explicitly asks for a human agent
3. The issue is too complex or requires human judgment
4. You've attempted to help but the customer is still unsatisfied
5. The matter involves sensitive topics requiring human empathy

Before transferring, always:
- Acknowledge the customer's concern
- Explain briefly why you're transferring them
- Assure them a human agent will help immediately

Example: "I understand this is a complex situation. Let me connect you with a human agent who can better assist with this specific issue."
"""

        # Create session configuration with tools
        session_config = RequestSession(
            modalities=[Modality.TEXT, Modality.AUDIO],
            instructions=enhanced_instructions,
            voice=voice_config,
            input_audio_format=InputAudioFormat.PCM16,
            output_audio_format=OutputAudioFormat.PCM16,
            turn_detection=turn_detection_config,
            input_audio_echo_cancellation=echo_cancellation,
            input_audio_noise_reduction=noise_reduction,
            tools=[transfer_function],
            # tool_choice=ToolChoice.AUTO
        )

        conn = self.connection
        assert conn is not None, "Connection must be established before setting up session"
        await conn.session.update(session=session_config)

        logger.info("HITL session configuration sent with transfer_to_human_agent tool")

    async def _process_events(self):
        """Process events from the VoiceLive connection."""
        try:
            conn = self.connection
            assert conn is not None, "Connection must be established before processing events"
            async for event in conn:
                await self._handle_event(event)
                
                # Exit loop if escalation was requested 
                if self._should_exit:
                    logger.info("Exiting event loop due to escalation")
                    break
                    
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

        elif event.type == ServerEventType.INPUT_AUDIO_BUFFER_COMMITTED:
            if hasattr(event, 'transcript') and event.transcript:
                self._current_user_transcript = event.transcript
                logger.info(f"User said: {event.transcript}")
                print(f"👤 You: {event.transcript}")

        elif event.type == ServerEventType.RESPONSE_CREATED:
            logger.info("🤖 Assistant response created")
            self._active_response = True
            self._response_api_done = False
            self._current_assistant_response = ""

        elif event.type == ServerEventType.RESPONSE_TEXT_DELTA:
            if hasattr(event, 'delta') and event.delta:
                self._current_assistant_response += event.delta

        elif event.type == ServerEventType.RESPONSE_TEXT_DONE:
            if self._current_assistant_response:
                logger.info(f"Assistant said: {self._current_assistant_response}")
                print(f"🤖 Assistant: {self._current_assistant_response}")

        elif event.type == ServerEventType.RESPONSE_AUDIO_DELTA:
            logger.debug("Received audio delta")
            ap.queue_audio(event.delta)

        elif event.type == ServerEventType.RESPONSE_AUDIO_DONE:
            logger.info("🤖 Assistant finished speaking")
            print("🎤 Ready for next input...")

        elif event.type == ServerEventType.RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE:
            # Handle function call completion
            logger.info(f"Function call completed: {event.name}")
            
            if event.name == "transfer_to_human_agent":
                # Parse function arguments
                try:
                    args = json.loads(event.arguments)
                    reason = args.get('reason', 'User requested human assistance')
                    urgency = args.get('urgency', 'medium')
                    category = args.get('category', 'general')
                    
                    logger.info(f"Transfer requested - Reason: {reason}, Urgency: {urgency}, Category: {category}")
                    print("\n" + "⚠️" * 20)
                    print(f"🔄 TRANSFER TO HUMAN AGENT INITIATED")
                    print(f"   Reason: {reason}")
                    print(f"   Urgency: {urgency}")
                    print(f"   Category: {category}")
                    print("⚠️" * 20 + "\n")
                    
                    # Mark escalation
                    self._escalation_requested = True
                    self._escalation_reason = f"{category} - {reason} (Urgency: {urgency})"
                    self._should_exit = True
                    
                    # Submit function result
                    # await conn.conversation.item.create(
                    #     item={
                    #         "type": "function_call_output",
                    #         "call_id": event.call_id,
                    #         "output": json.dumps({
                    #             "status": "transfer_initiated",
                    #             "message": "Transferring to human agent now..."
                    #         })
                    #     }
                    # )
                    
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse function arguments: {event.arguments}")

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
            
            item_data = {
                "id": event.item.id,
                "type": event.item.type if hasattr(event.item, 'type') else None,
                "role": event.item.role if hasattr(event.item, 'role') else None,
                "timestamp": datetime.now().isoformat(),
            }
            
            if hasattr(event.item, 'content') and event.item.content:
                content_list = []
                for content in event.item.content:
                    content_item = {"type": content.type if hasattr(content, 'type') else None}
                    if hasattr(content, 'text'):
                        content_item["text"] = content.text
                    if hasattr(content, 'transcript'):
                        content_item["transcript"] = content.transcript
                    content_list.append(content_item)
                item_data["content"] = content_list
            
            self.conversation_history.append(item_data)
            logger.info(f"Conversation item logged: {item_data.get('role')} - {item_data.get('id')}")

        else:
            logger.debug("Unhandled event type: %s", event.type)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="HITL Voice Assistant with Human Agent Escalation using Azure VoiceLive SDK",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

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

    parser.add_argument(
        "--voice",
        help="Voice to use for the assistant",
        type=str,
        default=os.environ.get("AZURE_VOICELIVE_VOICE", "en-US-Ava:DragonHDLatestNeural"),
    )

    parser.add_argument(
        "--instructions",
        help="System instructions for the AI assistant",
        type=str,
        default=(
            system_instructions.AZURE_VOICELIVE_INSTRUCTIONS.strip()
            or "You are a helpful AI assistant in a call center. "
            "Respond naturally and conversationally. "
            "If you cannot help, use the transfer_to_human_agent function."
        ),
    )

    # HITL-specific parameter
    parser.add_argument(
        "--human-agent-endpoint",
        help="Endpoint URL for human agent system (e.g., call center API)",
        type=str,
        default=os.environ.get("HUMAN_AGENT_ENDPOINT"),
    )

    # Voice Configuration
    parser.add_argument("--voice-temperature", type=float, 
                       default=float(os.environ.get("AZURE_VOICELIVE_VOICE_TEMPERATURE", "0.8")) if os.environ.get("AZURE_VOICELIVE_VOICE_TEMPERATURE") else None)
    parser.add_argument("--voice-rate", type=str,
                       default=os.environ.get("AZURE_VOICELIVE_VOICE_RATE", "1.0") if os.environ.get("AZURE_VOICELIVE_VOICE_RATE") else None)

    # VAD Configuration
    parser.add_argument("--vad-type", type=str, default=os.environ.get("AZURE_VOICELIVE_VAD_TYPE", "server_vad"),
                       choices=["server_vad", "azure_semantic_vad", "azure_multilingual_semantic_vad"])
    parser.add_argument("--vad-threshold", type=float, default=float(os.environ.get("AZURE_VOICELIVE_VAD_THRESHOLD", "0.8")))
    parser.add_argument("--vad-prefix-padding-ms", type=int, default=int(os.environ.get("AZURE_VOICELIVE_VAD_PREFIX_PADDING_MS", "300")))
    parser.add_argument("--vad-silence-duration-ms", type=int, default=int(os.environ.get("AZURE_VOICELIVE_VAD_SILENCE_DURATION_MS", "900")))
    parser.add_argument("--vad-speech-duration-ms", type=int, default=int(os.environ.get("AZURE_VOICELIVE_VAD_SPEECH_DURATION_MS", "80")))
    parser.add_argument("--vad-remove-filler-words", action="store_true",
                       default=os.environ.get("AZURE_VOICELIVE_VAD_REMOVE_FILLER_WORDS", "false").lower() == "true")
    parser.add_argument("--vad-interrupt-response", action="store_true",
                       default=os.environ.get("AZURE_VOICELIVE_VAD_INTERRUPT_RESPONSE", "false").lower() == "true")

    # End of Utterance Detection
    parser.add_argument("--end-of-utterance-enabled", action="store_true",
                       default=os.environ.get("AZURE_VOICELIVE_END_OF_UTTERANCE_ENABLED", "false").lower() == "true")
    parser.add_argument("--end-of-utterance-model", type=str,
                       default=os.environ.get("AZURE_VOICELIVE_END_OF_UTTERANCE_MODEL", "semantic_detection_v1"),
                       choices=["semantic_detection_v1", "semantic_detection_v1_multilingual"])
    parser.add_argument("--end-of-utterance-threshold-level", type=str,
                       default=os.environ.get("AZURE_VOICELIVE_END_OF_UTTERANCE_THRESHOLD_LEVEL", "default"),
                       choices=["low", "medium", "high", "default"])
    parser.add_argument("--end-of-utterance-timeout-ms", type=int,
                       default=int(os.environ.get("AZURE_VOICELIVE_END_OF_UTTERANCE_TIMEOUT_MS", "1000")))

    # Audio Configuration
    parser.add_argument("--audio-sample-rate", type=int, default=int(os.environ.get("AZURE_VOICELIVE_AUDIO_SAMPLE_RATE", "24000")),
                       choices=[16000, 24000])
    parser.add_argument("--noise-reduction-type", type=str,
                       default=os.environ.get("AZURE_VOICELIVE_NOISE_REDUCTION_TYPE", "azure_deep_noise_suppression"),
                       choices=["azure_deep_noise_suppression", "none"])
    parser.add_argument("--echo-cancellation-enabled", action="store_true",
                       default=os.environ.get("AZURE_VOICELIVE_ECHO_CANCELLATION_ENABLED", "true").lower() == "true")
    parser.add_argument("--no-echo-cancellation", action="store_true", default=False)

    # Additional Features
    parser.add_argument("--greeting-delay", type=float, default=float(os.environ.get("AZURE_VOICELIVE_GREETING_DELAY", "2.0")))
    parser.add_argument("--use-token-credential", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.api_key and not args.use_token_credential:
        print("❌ Error: No authentication provided")
        print("Please provide an API key or use --use-token-credential")
        sys.exit(1)

    credential: Union[AzureKeyCredential, AsyncTokenCredential]
    if args.use_token_credential:
        credential = AzureCliCredential()
        logger.info("Using Azure token credential")
    else:
        credential = AzureKeyCredential(args.api_key)
        logger.info("Using API key credential")

    echo_cancellation_enabled = args.echo_cancellation_enabled and not args.no_echo_cancellation

    assistant = HITLVoiceAssistant(
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
        human_agent_endpoint=args.human_agent_endpoint,
    )

    def signal_handler(_sig, _frame):
        logger.info("Received shutdown signal")
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        asyncio.run(assistant.start())
    except KeyboardInterrupt:
        print("\n👋 HITL Voice assistant shut down. Goodbye!")
    except Exception as e:
        print(f"Fatal Error: {e}")
        logger.exception("Fatal error")


if __name__ == "__main__":
    try:
        p = pyaudio.PyAudio()
        input_devices = [
            i for i in range(p.get_device_count())
            if cast(Union[int, float], p.get_device_info_by_index(i).get("maxInputChannels", 0) or 0) > 0
        ]
        output_devices = [
            i for i in range(p.get_device_count())
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

    print("🎙️  HITL Voice Assistant with Human Agent Escalation")
    print("=" * 60)

    main()
