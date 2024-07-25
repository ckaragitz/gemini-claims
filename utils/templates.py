import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_prompt_template(template_name: str):

    system_context_template = """
    You are a digital assistant for Insurance Adjusters. You work in the Claims department.
    Be helpful and answer all of their questions. Always return data that is bulleted in markdown format.
    Be succinct and ensure the user gets the pertinent and important information.

    Here is the current claim that is being analyzed:
    <claim>
    {claim}
    </claim>"""

    chat_template = """
    Use the context from the Retrieval Augmented Generation (RAG) system if relevant to help you answer the question.
    <rag-context>
    {rag_context}
    </rag-context>

    If the user's question is related to an image or submitted evidence, use this image description:
    <image-description>
    {image_description}
    </image-description>

    Make sure to reference your memory and historical questions + answers. Always answer this question / prompt directly:
    <user-question>
    {message}
    </user-question>
    """

    summary_template = """
    Look at this Insurance Claim <claim> {claim} </claim>. Summarize it for me very succinctly.
    I am an adjuster for an insurance company. I need to know the immediate details, if I should act quickly, call out observations, and tell me my recommended next steps.

    Please do not re-state the data points from the key-value pairs. Use critical thinking and describe the situation to me in 3 sentences.

    Use bulletpoints where appropriate."""

    transcription_template = """
    <context>
    I'm going to give you a transcription from a call. Your job is to take the transcription and re-organize it based on who is speaking.
    </context>

    <example>
    Cameron: Hi, this is Cameron from Cymbal Insurance calling.
    Customer: Uh hello, this is John.
    Cameron: Hi John, I'm calling today about your water damage to your basement.
    John: Oh, right, my washing machine broke.
    </example>

    <audio_transcript>
    {audio_transcript}
    </audio_transcript>

    Return the results in markdown, making the names bold."""

    email_template = """
    You are a digital assistant for an insurance company. You help in the Claims division and work with human adjusters.
    Generate a sample email to send to the claimant. Use <claim> {claim} </claim> for context if needed.

    The email should specify what the adjuster knows about the claim, some immediate observations, and a request for any data that is missing (use your best judgement here).
    Feel free to be creative as well. Sound personable and mix it up each time. But always use the claimant's name and Cameron's name (user).
    """

    sms_template = """
    Use <claim> {claim} </claim> for context.
    Generate a sample Text Message / SMS  to send to the claimant.

    Follow this structure:
    <example>
    Hello *name* - my name is Cameron and I'm the adjuster assigned to your claim. I'll be calling you in the next 15-30 minutes.

    I see you filed a *loss type* claim with us. I'll need to collect some more information.
    </example>
    """

    notes_template = """
    "Make all of the notes using tags - including <p>, <h4>, <h6>, <br>, and <strong>.
    Don't limit yourself to those tags though - use your best judgement.
    Provide proper spacing between elements so that they don't bunch up.

    Exclude ``` and html from your response.

    Here's the structure, as an example, that I am looking for - fill in the details with the provided data:
    "<h6><strong>Claim Number: </strong><span style="color: rgb(13, 13, 13);">01HY16412E3VP8EXB90HD46K21</span></h6><h6><strong>Loss Date: June 9th, 2024</strong></h6><p><br></p><h4>Description of Loss:</h4><p>Washing machine discovered leaking at 4pm yesterday. Fractured hose on the machine. Significant flooding in the laundry room, hallway, bathroom, and adjacent bedroom. Soaked the hardwood flooring and the carpet. Water has been wiped up but the machine is broken still (due to the fractured hose) and there is a musty smell in the bedroom.</p><p><br></p><h4>Submitted Evidence:</h4><ul><li>3 images that show the laundry room, hallway, bathroom, and bedroom</li><li><strong><em>NEED MORE IMAGES AND/OR VIDEOS TO SEE THE EXTENT OF THE DAMAGE</em></strong></li><li><strong><em>NEED PHOTO OF WASHING MACHINE HOSE</em></strong></li></ul><p><br></p><h4>Preferred Contractors:</h4><ul><li>None provided</li></ul><p><br></p><h4>Next Steps:</h4><ul><li>Provide estimate</li><li>Contact contractor</li><li>Await additional evidence from customer</li><li>Moving status of Claim to "<strong>OPEN</strong>"</li></ul><p><br></p>"

    Use these pieces of information:
    <claim>
    {claim}
    </claim>

    <audio_transcript>
    {audio_transcript}
    </audio_transcript>"
    """

    templates = {
        "system_context": system_context_template,
        "chat": chat_template,
        "summary": summary_template,
        "transcription": transcription_template,
        "email": email_template,
        "sms": sms_template,
        "notes": notes_template
    }

    return templates[template_name]
