import base64

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