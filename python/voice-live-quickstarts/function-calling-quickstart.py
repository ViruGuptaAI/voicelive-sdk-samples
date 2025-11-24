# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
from __future__ import annotations
import os
import sys
import argparse
import asyncio
import json
import base64
from datetime import datetime
import logging
import queue
import signal
from typing import Union, Optional, Dict, Any, Mapping, Callable, TYPE_CHECKING, cast

from azure.core.credentials import AzureKeyCredential
from azure.core.credentials_async import AsyncTokenCredential
from azure.identity.aio import AzureCliCredential, DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential as SyncDefaultAzureCredential

from azure.ai.voicelive.aio import connect
from azure.ai.voicelive.models import (
    AudioEchoCancellation,
    AudioNoiseReduction,
    AzureSemanticVad,
    AzureStandardVoice,
    EouDetection,
    InputAudioFormat,
    ItemType,
    Modality,
    OutputAudioFormat,
    RequestSession,
    ServerEventType,
    ServerVad,
    FunctionTool,
    FunctionCallOutputItem,
    ToolChoiceLiteral,
    AudioInputTranscriptionOptions,
    Tool,
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

# Required environment variables:
# - AZURE_VOICELIVE_API_KEY: API key for VoiceLive service
# - AZURE_VOICELIVE_ENDPOINT: Endpoint URL for VoiceLive service
# - AZURE_VOICELIVE_MODEL: Model name (e.g., gpt-realtime)
# - AZURE_VOICELIVE_VOICE: Voice name (e.g., en-US-Ava:DragonHDLatestNeural)
# - FOUNDRY_ENDPOINT: Azure AI Foundry endpoint for child agent (optional)
# - FOUNDRY_API_KEY: API key for Azure AI Foundry (optional, uses DefaultAzureCredential if not provided)
# - CHILD_AGENT_ID: Agent ID for customer information agent (optional)
# Note: A new thread is automatically created for each session

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



class AsyncFunctionCallingClient:
    """Voice assistant with function calling capabilities using VoiceLive SDK patterns."""

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
        self.conversation_started = False
        self._active_response = False
        self._response_api_done = False
        self._pending_function_call: Optional[Dict[str, Any]] = None

        # Initialize Azure AI Project Client for child agent
        foundry_endpoint = os.environ.get("FOUNDRY_ENDPOINT")
        self.child_agent_id = os.environ.get("CHILD_AGENT_ID", "asst_xBlcCfPyv1v9yDV8VWmmiBWk")
        self.child_thread_id = None  # Will create a new thread when needed
        
        if foundry_endpoint:
            try:
                # Strip any whitespace from endpoint
                foundry_endpoint = foundry_endpoint.strip()
                logger.info(f"Initializing Azure AI Project Client with endpoint: {foundry_endpoint[:50]}...")
                
                # Use DefaultAzureCredential (works with az login)
                credential = SyncDefaultAzureCredential()
                logger.info("Using DefaultAzureCredential for child agent")
                
                self.project_client = AIProjectClient(
                    credential=credential,
                    endpoint=foundry_endpoint,
                )
                # Create a new thread for this session
                thread = self.project_client.agents.threads.create()
                self.child_thread_id = thread.id
                logger.info(f"Azure AI Project Client initialized with new thread: {self.child_thread_id}")
                # print(f"✅ Child agent initialized - Thread: {self.child_thread_id}, Agent: {self.child_agent_id}")
            except Exception as e:
                logger.error(f"Failed to initialize Azure AI Project Client: {e}")
                print(f"❌ Failed to initialize child agent: {e}")
                self.project_client = None
        else:
            logger.warning("FOUNDRY_ENDPOINT not set - child agent calls will use mock data")
            print("⚠️ FOUNDRY_ENDPOINT not set")
            self.project_client = None

        # Define available functions
        self.available_functions: Dict[str, Callable[[Union[str, Mapping[str, Any]]], Mapping[str, Any]]] = {
            # "get_current_time": self.get_current_time,
            # "get_current_weather": self.get_current_weather,
            "get_customer_information": self.get_customer_information,
        }

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
                self.connection = connection
                self.audio_processor = AudioProcessor(connection, sample_rate=self.audio_sample_rate)
                self.audio_processor.start_playback()

                # Set up session and process events
                await self._setup_session()

                logger.info("Voice assistant with function calling ready! Start speaking...")
                print("\n" + "=" * 60)
                print("🎤 VOICE ASSISTANT WITH FUNCTION CALLING READY")
                print("Press Ctrl+C to exit")
                print("=" * 60 + "\n")

                # Process events
                await self._process_events()
        finally:
            if self.audio_processor:
                self.audio_processor.shutdown()

    async def _setup_session(self):
        """Configure the VoiceLive session for audio conversation with function tools."""
        logger.info("Setting up voice conversation session with function tools...")

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
                vad_kwargs["eou_detection"] = EouDetection(
                    model=self.end_of_utterance_model,
                    threshold_level=self.end_of_utterance_threshold_level,
                    timeout_ms=self.end_of_utterance_timeout_ms,
                )
            
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

        # Define function tools
        function_tools: list[Tool] = [
            FunctionTool(
                name="get_current_time",
                description="Get the current time",
                parameters={
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "The timezone to get the current time for, e.g., 'UTC', 'local'",
                        }
                    },
                    "required": [],
                },
            ),
            FunctionTool(
                name="get_current_weather",
                description="Get the current weather in a given location",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g., 'San Francisco, CA'",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The unit of temperature to use (celsius or fahrenheit)",
                        },
                    },
                    "required": ["location"],
                },
            ),
            FunctionTool(
                name="get_customer_information",
                description="Get customer information from the Customer Information Agent. Use this to fetch customer profile, loan eligibility, or credit card offers.",
                parameters={
                    "type": "object",
                    "properties": {
                        "customer_id": {
                            "type": "string",
                            "description": "The customer ID to fetch information for. Default is for Viru.",
                        },
                        "query": {
                            "type": "string",
                            "description": "What information to fetch. Examples: 'Get complete customer profile', 'What loans is this customer eligible for?', 'Show available credit card offers', 'Get loan interest rates and tenure options'",
                        },
                    },
                    "required": ["query"],
                },
            ),
        ]

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

        # Create session configuration with function tools
        session_config = RequestSession(
            modalities=[Modality.TEXT, Modality.AUDIO],
            instructions=self.instructions,
            voice=voice_config,
            input_audio_format=InputAudioFormat.PCM16,
            output_audio_format=OutputAudioFormat.PCM16,
            turn_detection=turn_detection_config,
            input_audio_echo_cancellation=echo_cancellation,
            input_audio_noise_reduction=noise_reduction,
            tools=function_tools,
            tool_choice=ToolChoiceLiteral.AUTO,
            input_audio_transcription=AudioInputTranscriptionOptions(model="whisper-1"),
        )

        conn = self.connection
        assert conn is not None, "Connection must be established before setting up session"
        await conn.session.update(session=session_config)

        logger.info("Session configuration with function tools sent")

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

            # Proactive greeting
            if not self.conversation_started:
                self.conversation_started = True
                await asyncio.sleep(self.greeting_delay)
                print("Agent initiated conversation...")
                logger.info("Sending proactive greeting request")
                try:
                    await conn.response.create()
                except Exception:
                    logger.exception("Failed to send proactive greeting request")

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

            # Execute pending function call if arguments are ready
            if self._pending_function_call and "arguments" in self._pending_function_call:
                await self._execute_function_call(self._pending_function_call)
                self._pending_function_call = None

        elif event.type == ServerEventType.ERROR:
            msg = event.error.message
            if "Cancellation failed: no active response" in msg:
                logger.debug("Benign cancellation error: %s", msg)
            else:
                logger.error("❌ VoiceLive error: %s", msg)
                print(f"Error: {msg}")

        elif event.type == ServerEventType.CONVERSATION_ITEM_CREATED:
            logger.debug("Conversation item created: %s", event.item.id)

            if event.item.type == ItemType.FUNCTION_CALL:
                function_call_item = event.item
                self._pending_function_call = {
                    "name": function_call_item.name,
                    "call_id": function_call_item.call_id,
                    "previous_item_id": function_call_item.id
                }
                print(f"🔧 Calling function: {function_call_item.name}")
                logger.info(f"Function call detected: {function_call_item.name} with call_id: {function_call_item.call_id}")

        elif event.type == ServerEventType.RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE:
            if self._pending_function_call and event.call_id == self._pending_function_call["call_id"]:
                logger.info(f"Function arguments received: {event.arguments}")
                self._pending_function_call["arguments"] = event.arguments

    async def _execute_function_call(self, function_call_info):
        """Execute a function call and send the result back to the conversation."""
        conn = self.connection
        assert conn is not None, "Connection must be established"
        
        function_name = function_call_info["name"]
        call_id = function_call_info["call_id"]
        previous_item_id = function_call_info["previous_item_id"]
        arguments = function_call_info["arguments"]

        try:
            if function_name in self.available_functions:
                logger.info(f"Executing function: {function_name}")
                result = self.available_functions[function_name](arguments)

                function_output = FunctionCallOutputItem(call_id=call_id, output=json.dumps(result))

                # Send result back to conversation
                await conn.conversation.item.create(previous_item_id=previous_item_id, item=function_output)
                logger.info(f"Function result sent: {result}")
                print(f"✅ Function {function_name} completed")

                # Request new response to process the function result
                await conn.response.create()
                logger.info("Requested new response with function result")

            else:
                logger.error(f"Unknown function: {function_name}")

        except Exception as e:
            logger.error(f"Error executing function {function_name}: {e}")

    def get_current_time(self, arguments: Optional[Union[str, Mapping[str, Any]]] = None) -> Dict[str, Any]:
        """Get the current time."""
        from datetime import datetime, timezone
        
        if isinstance(arguments, str):
            try:
                args = json.loads(arguments)
            except json.JSONDecodeError:
                args = {}
        else:
            args = arguments if isinstance(arguments, dict) else {}

        timezone_arg = args.get("timezone", "local")
        now = datetime.now()

        if timezone_arg.lower() == "utc":
            now = datetime.now(timezone.utc)
            timezone_name = "UTC"
        else:
            timezone_name = "local"

        formatted_time = now.strftime("%I:%M:%S %p")
        formatted_date = now.strftime("%A, %B %d, %Y")

        return {"time": formatted_time, "date": formatted_date, "timezone": timezone_name}

    def get_current_weather(self, arguments: Union[str, Mapping[str, Any]]):
        """Get the current weather for a location."""
        if isinstance(arguments, str):
            try:
                args = json.loads(arguments)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse weather arguments: {arguments}")
                return {"error": "Invalid arguments"}
        else:
            args = arguments if isinstance(arguments, dict) else {}

        location = args.get("location", "Unknown")
        unit = args.get("unit", "celsius")

        # Simulated weather response
        try:
            return {
                "location": location,
                "temperature": 22 if unit == "celsius" else 72,
                "unit": unit,
                "condition": "Partly Cloudy",
                "humidity": 65,
                "wind_speed": 10,
            }
        except Exception as e:
            logger.error(f"Error getting weather: {e}")
            return {"error": str(e)}
        
    def get_customer_information(self, arguments: Union[str, Mapping[str, Any]]) -> Dict[str, Any]:
        """
        Invokes the child agent to get customer information.
        The child agent has access to all customer data and returns structured information.
        """
        if isinstance(arguments, str):
            try:
                args = json.loads(arguments)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse customer information arguments: {arguments}")
                return {"error": "Invalid arguments"}
        else:
            args = arguments if isinstance(arguments, dict) else {}

        customer_id = args.get("customer_id", "viru")
        query = args.get("query", "")

        logger.info(f"Invoking child agent for customer_id: {customer_id}, query: {query}")
        print(f"📞 Fetching info from child agent: {query}")

        # Call the actual Azure AI Agent
        if not self.project_client:
            error_msg = "Project client not initialized. Check FOUNDRY_ENDPOINT and Azure credentials."
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            return {"error": error_msg}
        
        try:
            logger.info(f"Using thread: {self.child_thread_id}, agent: {self.child_agent_id}")
            
            # Create a message in the thread with the query
            message = self.project_client.agents.messages.create(
                thread_id=self.child_thread_id,
                role="user",
                content=query
            )
            logger.info(f"Message created: {message.id}")

            # Run the agent and process
            logger.info("Running child agent...")
            run = self.project_client.agents.runs.create_and_process(
                thread_id=self.child_thread_id,
                agent_id=self.child_agent_id
            )
            logger.info(f"Run completed with status: {run.status}")

            if run.status == "failed":
                error_msg = f"Agent run failed: {run.last_error}"
                logger.error(error_msg)
                print(f"❌ {error_msg}")
                return {"error": error_msg}
            
            # Get the latest assistant message
            messages = self.project_client.agents.messages.list(
                thread_id=self.child_thread_id, 
                order="desc"
            )
            
            latest_assistant_text: Optional[str] = None
            for msg in messages:
                if msg.role == "assistant" and msg.text_messages:
                    latest_assistant_text = msg.text_messages[-1].text.value
                    break
            
            if latest_assistant_text:
                logger.info(f"Child agent response received")
                print(f"✅ Response: {latest_assistant_text[:100]}...")
                return {"response": latest_assistant_text}
            else:
                error_msg = "No response from child agent"
                logger.warning(error_msg)
                print(f"⚠️ {error_msg}")
                return {"error": error_msg}

        except Exception as e:
            error_msg = f"Error calling child agent: {str(e)}"
            logger.error(error_msg, exc_info=True)
            print(f"❌ {error_msg}")
            return {"error": error_msg}


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Voice Assistant with Function Calling using Azure VoiceLive SDK",
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

    parser.add_argument(
        "--instructions",
        help="System instructions for the AI assistant",
        type=str,
        default=(
            system_instructions.Azure_Function_calling_instructions.strip()
            or "You are a helpful AI assistant. Respond naturally and conversationally. "
            "Keep your responses concise but engaging."
    ),
    )

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
        credential = AzureCliCredential()
        logger.info("Using Azure token credential")
    else:
        credential = AzureKeyCredential(args.api_key)
        logger.info("Using API key credential")

    # Handle echo cancellation flag override
    echo_cancellation_enabled = args.echo_cancellation_enabled and not args.no_echo_cancellation

    # Create and start voice assistant with function calling
    client = AsyncFunctionCallingClient(
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

    # Signal handlers for graceful shutdown
    def signal_handler(_sig, _frame):
        logger.info("Received shutdown signal")
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        asyncio.run(client.start())
    except KeyboardInterrupt:
        print("\n👋 Voice assistant shut down. Goodbye!")
    except Exception as e:
        logger.exception("Fatal error")
        print(f"Fatal Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Check for required dependencies
    dependencies = {
        "pyaudio": "Audio processing",
        "azure.ai.voicelive": "Azure VoiceLive SDK",
        "azure.core": "Azure Core libraries",
        "azure.ai.projects": "Azure AI Projects SDK (for child agent)",
    }

    missing_deps = []
    for dep, description in dependencies.items():
        try:
            __import__(dep.replace("-", "_"))
        except ImportError:
            missing_deps.append(f"{dep} ({description})")

    if missing_deps:
        print("❌ Missing required dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nInstall with: pip install azure-ai-voicelive azure-ai-projects pyaudio python-dotenv")
        sys.exit(1)

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

    print("🎙️  Voice Assistant with Function Calling - Azure VoiceLive SDK")
    print("=" * 65)

    # Run the assistant
    main()
