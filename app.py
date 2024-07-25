import base64
import os
from pydub import AudioSegment
from io import BytesIO
from PIL import Image
from uuid import uuid4
import json

import vertexai
from vertexai.generative_models import GenerativeModel, ChatSession, Part, Content

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from typing import List, Optional

from utils.audio import save_base64_to_file
from utils.image import gemini_image_description
from utils.rag import generate_rag_response
from utils.templates import get_prompt_template

import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Vertex AI once for the application
vertexai.init(project="amfam-claims", location="us-central1")

# In-memory store for chat sessions
global chat_sessions
chat_sessions = {}

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    session_id: Optional[str] = ""
    claim: dict
    image_description: Optional[str] = None
    messages: List[Message]

class ChatResponse(BaseModel):
    session_id: str
    role: str
    content: str

@app.get("/")
async def main(request: Request):
    return JSONResponse(content={"Available APIs": ["/chat", "/summarize", "/transcribe", "/bounding-box", "/image", "/comms", "/notes"]}, status_code=200)

# Helper function for the Chat endpoint
def get_chat_session(session_id: str, claim_str: str) -> ChatSession:

    logger.info(f"CHAT SESSIONS: {chat_sessions}")
    #logger.info(f"CONTEXT: {context}")

    if session_id not in chat_sessions:
        # Set the context for each Chat session
        # Fetch the system instruction template
        context_template = get_prompt_template("context")
        context_prompt = context_template.format(claim=claim_str)

        # Initialize the model, create a ChatSession object, and add the session to our in-memory store
        model = GenerativeModel("gemini-1.5-flash-001", system_instruction=[context_prompt])
        chat = model.start_chat()
        chat_sessions[session_id] = chat

        logger.info(f"CHAT SESSION ADDED: {session_id}")

    return chat_sessions[session_id]

@app.post("/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest):
    """
        POST body
        {
            "session_id": "" OR "11111111-1111-1111-1111-111111111111",
            "claim": {"id": 123, "loss_description": "AAAAAAA", ...},
            "image_description": "BBBBBBB",
            "messages": [
                {
                    "role": "user",
                    "content": "XXXXXXX"
                },
                {
                    "role": "model",
                    "content": "YYYYYYY"
                },
                {
                    "role": "user",
                    "content": "ZZZZZZZ"
                }
            ]
        }
    """

    # Gather the variables from the POST request
    claim = chat_request.claim
    loss_description = claim.get("loss_description")
    image_description = chat_request.image_description
    messages = chat_request.messages
    # Convert the claim dictionary to a formatted string for the context
    claim_str = json.dumps(claim, indent=2)

    # Grab the session_id (if sent in the POST body) OR generate a new one
    session_id = chat_request.session_id if chat_request.session_id else str(uuid4())

    # Most recent message in the array is the user's current prompt
    message = messages[-1].content
    logger.info(f"MESSAGE: {message}")

    # Prompt to send to Vertex AI / LlamaIndex for RAG
    rag_template = get_prompt_template("rag")
    rag_prompt = rag_template.format(loss_description=loss_description, message=message)

    try:
        # Parse documents, including the customer's policy, to ground the answers
        rag_response = generate_rag_response(prompt=rag_prompt)
        logger.info(f"RAG RESPONSE: {rag_response}")
    except Exception as e:
        logger.error(f"Error generating RAG response: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

    # Fetch the template
    chat_template = get_prompt_template("chat")
    chat_prompt = chat_template.format(rag_response=rag_response, image_description=image_description, message=message)

    # Get or create chat session
    chat = get_chat_session(session_id, claim_str)
    #logger.info(f"CHAT HISTORY: {chat.history}")
    #logger.info(f"SESSION ID: {session_id}")

    try:
        parameters = {
            "max_output_tokens": 8192,
            "temperature": 0.8,
            "top_p": 0.8,
            "top_k": 40,
            "candidate_count": 1,
        }

        model_response = chat.send_message(chat_prompt, generation_config=parameters)

        response = ChatResponse(
            session_id=session_id,
            role="model",
            content=model_response.text
        )

        return JSONResponse(content=response.model_dump(), status_code=200)
    except Exception as e:
        logger.error(f"Error in chat response: {e}")
        raise HTTPException(status_code=500, detail="Request to Gemini's chat feature failed.")

@app.post("/summarize")
async def summarize_claim(request: Request):

    request_json = await request.json()

    claim = request_json.get("claim")

    # Fetch the template
    summary_template = get_prompt_template("summary")
    prompt = summary_template.format(claim=claim)

    # initialize the model, set the parameters
    model = GenerativeModel("gemini-1.5-flash-001")
    parameters = {
        "max_output_tokens": 8192,
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 40,
        "candidate_count": 1,
    }

    try:
        model_response = model.generate_content(
        prompt,
        generation_config=parameters,
        )

        return PlainTextResponse(content=model_response.text, status_code=200)
    except Exception as e:
        print("Error:", model_response.text)
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/transcribe")
async def transcribe(request: Request):

    request_json = await request.json()

    if "audio" in request_json:
        audio_data = request_json.get("audio")

    temp_file = "temp_audio"
    save_result = save_base64_to_file(audio_data, temp_file)
    logger.info(save_result)

    if "Error" in save_result:
        return save_result

    try:
        audio = AudioSegment.from_file(temp_file)
        temp_wav_file = "temp_audio.wav"
        audio.export(temp_wav_file, format="wav")
    except Exception as e:
        return f"Error creating audio segment: {e}"

    try:
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_wav_file) as source:
            audio_data = recognizer.record(source)

        text = recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        return "Google Web Speech API could not understand the audio"
    except sr.RequestError as e:
        return f"Could not request results from Google Web Speech API; {e}"
    except Exception as e:
        return f"Unexpected error during transcription: {e}"
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        if os.path.exists(temp_wav_file):
            os.remove(temp_wav_file)

    # Fetch the template
    transcription_template = get_prompt_template("transcription")
    prompt = transcription_template.format(audio_transcript=text)

    # initialize the model, set the parameters
    model = GenerativeModel("gemini-1.5-flash-001")
    parameters = {
        "max_output_tokens": 8192,
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 40,
        "candidate_count": 1,
    }

    try:
        model_response = model.generate_content(
        prompt,
        generation_config=parameters,
        )

        return PlainTextResponse(content=model_response.text, status_code=200)
    except Exception as e:
        print("Error:", model_response.text)
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/bounding-box")
async def process_bounding_box_image(request: Request):

    request_json = await request.json()

    if "image" in request_json:
        base64_image = request_json['image']
    if "bounding_boxes" in request_json:
        bounding_boxes = request_json['bounding_boxes']

    image_data = base64.b64decode(base64_image)
    image = Image.open(BytesIO(image_data))

    inference_results = []

    for box in bounding_boxes:
        x = box['x']
        y = box['y']
        w = box['width']
        h = box['height']
        label = box.get('label', 'undefined')

        cropped_image = image.crop((x, y, x + w, y + h))

        if cropped_image.mode == 'RGBA':
            cropped_image = cropped_image.convert('RGB')

        buffered = BytesIO()
        cropped_image.save(buffered, format="JPEG")
        cropped_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        inference = gemini_image_description(cropped_image_base64)
        inference_results.append({
            'bounding_box': {'x': x, 'y': y, 'width': w, 'height': h, 'label': label},
            'results': inference
        })

    payload = {'inference_results': inference_results}
    return JSONResponse(content=payload, status_code=200)

@app.post("/image")
async def process_image(request: Request):

    request_json = await request.json()

    if "image" in request_json:
        base64_image = request_json['image']

    inference = gemini_image_description(base64_image)

    payload = {'inference_results': inference.get("description")}
    return JSONResponse(content=payload, status_code=200)

@app.post("/comms")
async def generate_comms(request: Request):

    request_json = await request.json()

    email_flag = request_json.get('email', False)
    sms_flag = request_json.get('sms', False)
    claim = request_json.get("claim")

    if email_flag:
        # Fetch the template
        email_template = get_prompt_template("email")
        prompt = email_template.format(claim=claim)
    elif sms_flag:
        # Fetch the template
        sms_template = get_prompt_template("sms")
        prompt = sms_template.format(claim=claim)

    # initialize the model, set the parameters
    model = GenerativeModel("gemini-1.5-flash-001")
    parameters = {
        "max_output_tokens": 8192,
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 40,
        "candidate_count": 1,
    }

    try:
        model_response = model.generate_content(
        prompt,
        generation_config=parameters,
        )

        return PlainTextResponse(content=model_response.text, status_code=200)
    except Exception as e:
        print("Error:", model_response.text)
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/notes")
async def generate_notes(request: Request):

    request_json = await request.json()

    claim = request_json.get('claim')
    audio_transcript = request_json.get('audio_transcript', '')

    # Fetch the template
    notes_template = get_prompt_template("notes")
    prompt = notes_template.format(claim=claim, audio_transcript=audio_transcript)
    #logger.info(f"NOTES PROMPT: {prompt}")

    # initialize the model, set the parameters
    model = GenerativeModel("gemini-1.5-flash-001")
    parameters = {
        "max_output_tokens": 8192,
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 40,
        "candidate_count": 1,
    }

    try:
        model_response = model.generate_content(
        prompt,
        generation_config=parameters,
        )

        return PlainTextResponse(content=model_response.text, status_code=200)
    except Exception as e:
        print("Error:", model_response.text)
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
  port = int(os.environ.get('PORT', 8080))
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=port)
