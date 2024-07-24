import base64

import vertexai
from vertexai.generative_models import GenerativeModel, Part, Content
import vertexai.preview.generative_models as generative_models

import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

vertexai.init(project="amfam-claims", location="us-central1")

# Helper function for image inference (makes the request to Gemini)
def gemini_image_description(base64_image):

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
