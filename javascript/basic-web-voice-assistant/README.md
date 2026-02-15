# Basic Voice Live Web Assistant

A web-based voice assistant demonstrating **Azure Voice Live SDK** integration in a browser application. This sample shows how to build real-time, bi-directional voice conversations with AI using TypeScript and the Web Audio API.

## Features Demonstrated

### Voice Live SDK Integration
- **Client/Session Architecture**: `VoiceLiveClient` creates `VoiceLiveSession` for managed connections
- **Handler-Based Events**: Type-safe event handling following Azure SDK patterns (EventHub, Service Bus)
- **Real-Time Audio Streaming**: Bi-directional PCM16 audio with the Voice Live service

### Real-Time Conversation
- **Live Transcription**: User speech transcribed in real-time
- **Streaming Responses**: Assistant text appears character-by-character as it's generated
- **Audio Playback**: Assistant voice plays as audio streams arrive
- **Barge-In Support**: Natural conversation interruption - speak to stop the assistant

### Audio Processing
- **Web Audio API**: Browser-native microphone capture and speaker output
- **PCM16 Conversion**: Automatic format conversion for Voice Live compatibility
- **Level Monitoring**: Real-time audio level visualization
- **Sequential Audio Queue**: Prevents overlapping audio chunks

### Voice Options
- **OpenAI Voices**: alloy, echo, shimmer, ash, ballad, coral, sage, verse
- **Azure Neural Voices**: Full Azure Speech Service voice catalog
- **Custom Voices**: Support for Azure custom voice deployments

## Quick Start

### Prerequisites
- **Node.js 18+** with npm
- **Azure AI Foundry** resource with Voice Live enabled
- Modern browser (Chrome 66+, Firefox 60+, Safari 11.1+, Edge 79+)

### 1. Install Dependencies
```bash
cd javascript/basic-web-voice-assistant
npm install
```

### 2. Start Development Server
```bash
npm run dev
```
Opens at **http://localhost:3000**

### 3. Configure & Connect
1. Enter your **Voice Live endpoint** (e.g., `https://your-resource.services.ai.azure.com/`)
2. Enter your **API key**
3. Choose a **voice** (OpenAI or Azure)
4. Customize **instructions** (optional)
5. Click **"Connect"**

### 4. Start Conversation
1. Click **"Start Conversation"**
2. **Allow microphone access** when prompted
3. **Start speaking** - the assistant responds in real-time!

## Configuration

### Voice Options
The SDK supports multiple voice providers:

| Type | Voice Names | Example |
|------|-------------|---------|
| OpenAI | alloy, echo, shimmer, ash, ballad, coral, sage, verse | `alloy` |
| Azure Standard | Azure Neural voice names | `en-US-Ava:DragonHDLatestNeural` |
| Azure Custom | Your custom voice deployment | Requires `endpointId` |

### Sample UI Settings

**For OpenAI Voice:**
- **Model**: `gpt-realtime`
- **Voice**: `alloy`
- **Instructions**: "You are a helpful AI assistant. Be conversational and engaging."

**For Azure Neural Voice:**
- **Model**: `gpt-realtime`
- **Voice**: `en-US-Ava:DragonHDLatestNeural`
- **Instructions**: "You are a professional AI assistant. Provide clear, concise responses."

## User Interface

### Configuration Panel
- **Endpoint**: Voice Live service URL
- **API Key**: Your Voice Live API credentials  
- **Voice Selection**: Choose from OpenAI or Azure Neural voices
- **Instructions**: Custom system prompt for the assistant

### Control Panel
- **Connection Status**: Real-time connection state indicators
- **Assistant Status**: Current state (idle, listening, thinking, speaking)
- **Audio Level Meter**: Live microphone input visualization

### Conversation Panel
- **Real-time Transcript**: Conversation history with timestamps
- **Role Indicators**: Clear distinction between user, assistant, and system messages
- **Auto-scroll**: Automatically follows the conversation

### Events Panel (Developer Features)
- **Live Event Stream**: All Voice Live SDK events in real-time
- **Event Filtering**: Toggle important events only
- **JSON Details**: Full event data for debugging

## Architecture

### SDK Pattern
```typescript
// Create client with credentials
const client = new VoiceLiveClient(endpoint, credential);

// Start a session
const session = await client.startSession({
  model: 'gpt-realtime',
  instructions: 'You are a helpful assistant',
  voice: { type: 'openai', name: 'alloy' }
});

// Subscribe to events with type-safe handlers
const subscription = session.subscribe({
  onResponseTextDelta: async (event, context) => {
    updateStreamingMessage(event.delta);
  },
  onResponseAudioDelta: async (event, context) => {
    await playAudioChunk(event.delta);
  },
  onInputAudioBufferSpeechStarted: async (event, context) => {
    clearAudioQueue(); // Barge-in support
  }
});

// Send audio from microphone
await session.sendAudio(pcm16Data);

// Cleanup
await subscription.close();
await session.dispose();
```

### Web Audio Integration
```typescript
const mediaStream = await navigator.mediaDevices.getUserMedia({
  audio: { sampleRate: 24000, channelCount: 1 }
});

const pcm16Data = convertToPCM16(floatAudioData);
await session.sendAudio(pcm16Data);
```

## Development

### Commands
```bash
npm run dev        # Development server with hot reload
npm run build      # Production build
npm run preview    # Preview production build
npm run type-check # TypeScript type checking
```

### Browser Requirements
| Browser | Minimum Version |
|---------|-----------------|
| Chrome  | 66+ (recommended) |
| Firefox | 60+ |
| Safari  | 11.1+ |
| Edge    | 79+ |

**Required browser features:**
- HTTPS connection (for microphone access)
- Web Audio API
- ES2020 module support
- WebSocket

## Troubleshooting

### "Microphone not accessible"
- Ensure you're using **HTTPS** (required for microphone access)
- Check browser permissions for microphone access
- Try refreshing the page and re-allowing permissions

### "Connection failed"
- Verify your **API key** and **endpoint** are correct
- Check browser console for detailed error messages  
- Ensure your Voice Live service is accessible

### "No audio playback"
- Check browser audio permissions and system volume
- Verify speakers/headphones are working
- Check browser console for audio errors

### "Events not showing"
- Click **"Show Events"** to display the events panel
- Toggle **"Filter Important Events Only"** to see key events

For advanced debugging (WebSocket inspection, SDK source debugging, performance monitoring), see [DEBUG_GUIDE.md](DEBUG_GUIDE.md).

## Extending the Sample

This sample provides a solid foundation for building production voice applications:

### **Add Avatar Support**
The avatar management system is ready for integration with Three.js, Babylon.js, or other 3D rendering libraries.

### **Enhanced Audio Processing**  
Extend with noise reduction, echo cancellation, or other DSP features using the AudioProcessor.

### **Conversation Persistence**
Add database integration to store and retrieve conversation history.

### **Multi-Language Support**
Implement dynamic language and voice switching with locale-specific instructions.

### **Custom UI Components**
Replace the basic HTML/CSS with React, Vue, Angular, or other modern UI frameworks.

## 📝 **License**

This sample is licensed under the MIT License. See the main SDK LICENSE file for details.
