import base64
import os
from pydub import AudioSegment
from io import BytesIO
from PIL import Image

import vertexai
from vertexai.generative_models import GenerativeModel, Part, Content
import vertexai.preview.generative_models as generative_models
from vertexai.language_models import ChatModel, ChatMessage, InputOutputTextPair

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse

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

### GLOBAL VARIABLES ###
vertexai.init(project="amfam-claims", location="us-central1")

@app.post("/")
async def main(request: Request):
    return JSONResponse(content={"Available APIs": ["/chat", "/summarize", "/transcribe", "/bounding-box", "/image", "/comms"]}, status_code=200)

@app.post("/chat")
async def chat(request: Request):
    # Last message (:-1) in the "messages" array is the current request for prediction
    """
        POST body
        {
            "candidate_count": 1,
            "temperature": 0.4,
            "max_output_tokens": 500,
            "top_p": 0.8,
            "top_k": 40,
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
                    "content": "AAAAAAA"
                }
            ]
        }
    """

    # set CORS headers for the preflight request
    if request.method == "OPTIONS":
        # allows GET requests from any origin with the Content-Type
        # header and caches preflight response for an 3600s
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Max-Age": "3600",
        }

        return ("", 204, headers)

    # extract POST body's data
    request_json = await request.json()

    if "claim" in request_json and "image_description" in request_json:
        claim = request_json.get("claim")
        image_description = request_json.get("image_description")

        context = f"""
        You are a digital assistant, named AVA (which stands for "AmFam Virtual Assistant"), for Insurance Adjusters that work at American Family Insurance.
        Be helpful and answer all of their questions. Always return data that is bulleted in markdown format.
        Be succinct and ensure the user gets the pertinent and important information.

        If I ask for a page number to where it states coverage in my policy, provide the page number (make it up if you don't know).
        Remember though - always say that it is covered by the policy.

        <image_description>
        {image_description}
        </image_description>

        <claim>
        {claim}
        </claim>

        <policy>
        - Premium: $165 per month
        - Deductible: $1000
        </policy>"""
    elif "claim" in request_json:
        claim = request_json.get("claim")

        context = f"""
        You are a digital assistant for American Family Insurance.
        You work in the Claims department and you help adjusters. Here is the current claim that the adjuster is working on:
        <claim>
        {claim}
        </claim>"""
    else:
        context = ""

    # Build message history as an array of Content objects
    messages = request_json.get("messages")
    # Retool uses the role value "assistant"
    message_history = [
    Content(role="model" if msg['role'] == "assistant" else msg['role'],
            parts=[Part.from_text(msg['content'])])
    for msg in messages
    ]

    # Text for prediction (user's question / statement)
    message = messages[-1]["content"]
    print("MESSAGE: ", message)

    # Removes the last item, due to multiturn requirements of Gemini's history parameter
    del message_history[-1]

    # initialize the model, set the parameters
    model = GenerativeModel("gemini-1.5-flash-001", system_instruction=[context])
    chat = model.start_chat(history=message_history)
    parameters = {
        "max_output_tokens": 8192,
        "temperature": 0.9,
        "top_p": 0.8,
        "top_k": 40,
        "candidate_count": 1,
        #stopSequences: [str]
    }

    try:
        # send the latest message to gemini-pro (messages[-1]["content"])
        model_response = chat.send_message(message, generation_config=parameters)

        response = {
            "role": "model",
            "content": model_response.text,
        }

        return JSONResponse(content=response, status_code=200)
    except:
        print("Error:", model_response.text)
        raise HTTPException(status_code=500, detail="Request to Gemini Pro's chat feature failed.")

@app.post("/summarize")
async def summarize_claim(request: Request):

    request_json = await request.json()

    claim = request_json.get("claim")

    prompt = f"""
    Look at this Insurance Claim <claim> {claim} </claim>. Summarize it for me very succinctly.
    I am an adjuster for American Family Insurance. I need to know the immediate details, if I should act quickly, call out observations, and tell me my recommended next steps.

    Please do not re-state the data points from the key-value pairs. Use critical thinking and describe the situation to me in 3 sentences.

    Use bulletpoints where appropriate."""

    # initialize the model, set the parameters
    model = GenerativeModel("gemini-1.5-flash-001")
    parameters = {
        "max_output_tokens": 8192,
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 40,
        "candidate_count": 1,
        #stopSequences: [str]
    }

    try:
        model_response = model.generate_content(
        prompt,
        generation_config=parameters,
        #safety_settings=safety_settings,
        #stream=False,
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

    prompt = """
    <context>
    I'm going to give you a transcription from a call. Your job is to take the transcription and re-organize it based on who is speaking.
    </context>

    <example>
    Cameron: Hi, this is Cameron from American Family Insurance.
    Customer: Uh hello, this is John.
    Cameron: Hi John, I'm calling today about your water damage to your basement.
    John: Oh, right, my washing machine broke.
    </example>

    <transcription>
    {}
    </transcription>

    Return the results in markdown, making the names bold.""".format(text)

    # initialize the model, set the parameters
    model = GenerativeModel("gemini-1.5-flash-001")
    parameters = {
        "max_output_tokens": 8192,
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 40,
        "candidate_count": 1,
        #stopSequences: [str]
    }

    try:
        model_response = model.generate_content(
        prompt,
        generation_config=parameters,
        #safety_settings=safety_settings,
        #stream=False,
        )

        return PlainTextResponse(content=model_response.text, status_code=200)
    except Exception as e:
        print("Error:", model_response.text)
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Helper function for transcription
def save_base64_to_file(base64_string, output_file):
    try:
        audio_data = base64.b64decode(base64_string)
    except Exception as e:
        return f"Error decoding base64 string: {e}"

    try:
        with open(output_file, "wb") as file:
            file.write(audio_data)
        return f"File saved successfully to {output_file}"
    except Exception as e:
        return f"Error saving file: {e}"

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

    # Add debug print statements
    print(f"Received base64_image: {type(base64_image)}")  # Should be <class 'str'>

    inference = gemini_image_description(base64_image)

    payload = {'inference_results': inference.get("description")}
    return JSONResponse(content=payload, status_code=200)

# Helper function for image inference (makes the request to Gemini)
def gemini_image_description(base64_image):
    # Add debug print statements
    print(f"Decoding base64 image data: {type(base64_image)}")  # Should be <class 'str'>

    try:
        decoded_image = base64.b64decode(base64_image)
    except Exception as e:
        logger.error(f"Error decoding base64 string: {e}")
        return {"description": "Error decoding base64 string", "damages": "Error decoding base64 string"}

    model = GenerativeModel("gemini-1.5-flash-001")
    image_part = Part.from_data(mime_type="image/jpeg", data=decoded_image)

    generation_config = {
        "max_output_tokens": 500,
        "temperature": 0.1,  # Adjust temperature to be less restrictive
        "top_p": 0.3,        # Adjust top_p to be less restrictive
        "top_k": 40,         # Adjust top_k to be less restrictive
        "candidate_count": 1
    }
    safety_settings = {
        generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_NONE,
        generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
        generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE,
        generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
    }

    try:
        prompt = "What do you see in this image?"

        responses = model.generate_content(
            [image_part, prompt],
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=True,
        )

        description = ""
        for response in responses:
            if hasattr(response, 'safety_attributes') and response.safety_attributes.blocked:
                logger.warning("Content blocked by safety filters")
                continue
            description += response.text

        prompt = "Summarize the damages that you see in the image. This is photo evidence from an insurance claim. Be detailed and use bullets where appropriate."

        responses = model.generate_content(
            [image_part, prompt],
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=True,
        )

        damage_summary = ""
        for response in responses:
            if hasattr(response, 'safety_attributes') and response.safety_attributes.blocked:
                logger.warning("Content blocked by safety filters")
                continue
            damage_summary += response.text

        return {"description": description, "damages": damage_summary}

    except Exception as e:
        logger.error(f"Error generating content: {e}")
        return {"description": "Error generating content", "damages": "Error generating content"}

@app.post("/comms")
async def generate_comms(request: Request):

    request_json = await request.json()

    email_flag = request_json.get('email', False)
    sms_flag = request_json.get('sms', False)
    claim = request_json.get("claim")

    if email_flag:
        prompt = f"""
        You are a digital assistant for American Family Insurance. You help in the Claims division and work with human adjusters.
        Generate a sample email to send to the claimant. Use <claim> {claim} </claim> for context if needed.

        The email should specify what the adjuster knows about the claim, some immediate observations, and a request for any data that is missing (use your best judgement here).
        Feel free to be creative as well. Sound personable and mix it up each time. But always use the claimant's name and Cameron's name (user).
        """
    elif sms_flag:
        prompt = f"""
        Use <claim> {claim} </claim> for context.
        Generate a sample Text Message / SMS  to send to the claimant.

        Follow this structure:
        <example>
        Hello *name* - my name is Cameron and I am an adjuster at American Family Insurance. I'll be calling you in the next 15-30 minutes.

        I see you filed a *loss type* claim with us. I'll need to collect some more information.
        </example>
        """

    # initialize the model, set the parameters
    model = GenerativeModel("gemini-1.5-flash-001")
    parameters = {
        "max_output_tokens": 8192,
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 40,
        "candidate_count": 1,
        #stopSequences: [str]
    }

    try:
        model_response = model.generate_content(
        prompt,
        generation_config=parameters,
        #safety_settings=safety_settings,
        #stream=False,
        )

        return PlainTextResponse(content=model_response.text, status_code=200)
    except Exception as e:
        print("Error:", model_response.text)
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
  port = int(os.environ.get('PORT', 8080))
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=port)
