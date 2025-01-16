# Insurance Claims Assistant API

A FastAPI-based service that provides AI-powered assistance for insurance claims processing using Google's Vertex AI and Gemini models.

## Features

- 🤖 **AI-Powered Chat**: Multi-turn conversations with context awareness using Gemini
- 📝 **Claim Summarization**: Automated claim analysis and summary generation
- 🎙️ **Audio Transcription**: Convert audio recordings to text with speaker identification
- 🖼️ **Image Analysis**: Process and analyze claim-related images
- 📊 **Damage Assessment**: Visual damage evaluation from submitted photos
- 📧 **Communications Generator**: Automated email and SMS message generation
- 📄 **Notes Generation**: Structured note-taking system for claims
- 🔍 **RAG Integration**: Context-aware responses using Retrieval Augmented Generation

## Technology Stack

- Python 3.10
- FastAPI
- Google Vertex AI
- Gemini 1.5 Flash
- Docker
- pydub (Audio Processing)
- PIL (Image Processing)
- SpeechRecognition

## Prerequisites

- Google Cloud Project with Vertex AI enabled
- Python 3.10+
- Docker (optional)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Google Cloud credentials:
- Ensure you have a Google Cloud project
- Enable Vertex AI API
- Set up authentication credentials

## Environment Variables

- `PORT`: Server port (default: 8080)
- Google Cloud credentials should be properly configured in your environment

## Running the Application

### Local Development
```bash
uvicorn app:app --host 0.0.0.0 --port 8080
```

### Using Docker
```bash
docker build -t insurance-claims-assistant .
docker run -p 8080:8080 insurance-claims-assistant
```

## API Endpoints

### Chat (`/chat`)
- Method: POST
- Handles multi-turn conversations with context awareness
- Maintains session state for continuous dialogue

### Summarize (`/summarize`)
- Method: POST
- Generates concise claim summaries
- Provides key insights and next steps

### Transcribe (`/transcribe`)
- Method: POST
- Converts audio to text
- Identifies speakers in conversations

### Image Analysis (`/image`)
- Method: POST
- Analyzes uploaded images
- Provides detailed descriptions

### Bounding Box Analysis (`/bounding-box`)
- Method: POST
- Processes specific regions in images
- Returns targeted analysis

### Communications (`/comms`)
- Method: POST
- Generates professional emails and SMS messages
- Customized based on claim context

### Notes (`/notes`)
- Method: POST
- Creates structured claim notes
- Incorporates audio transcripts and claim data

## Project Structure

```
.
├── app.py                 # Main application file
├── Dockerfile            # Docker configuration
├── utils/
│   ├── audio.py         # Audio processing utilities
│   ├── image.py         # Image processing utilities
│   ├── rag.py           # RAG system implementation
│   └── templates.py     # Prompt templates
└── requirements.txt     # Project dependencies
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
