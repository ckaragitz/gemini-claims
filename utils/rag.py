from vertexai.preview import rag
from vertexai.preview.generative_models import GenerativeModel, Tool
from vertexai.preview.rag.utils.resources import EmbeddingModelConfig, RagResource
import vertexai

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
project_id = "amfam-claims"
location = "us-central1"
paths = ["gs://ck-claims-data/policy", "gs://ck-claims-data/misc"]

# Initialize Vertex AI API once per session
vertexai.init(project=project_id, location=location)

def create_corpus(display_name):
    try:
        # Create RagCorpus
        embedding_model_config = EmbeddingModelConfig(
            publisher_model="publishers/google/models/text-embedding-004"
        )

        rag_corpus = rag.create_corpus(
            display_name=display_name,
            embedding_model_config=embedding_model_config,
        )

        logger.info(f"RagCorpus created with name: {rag_corpus.name}")

        return "Successfully created corpus named: " + str(rag_corpus.name)
    except Exception as e:
        logger.error(f"Error creating corpus: {e}")
        return str(e)

def ingest_files(paths=paths):

    rag_corpus_name = get_first_corpus()

    # Import files to the RagCorpus
    file_import_response = rag.import_files(
        rag_corpus_name,
        paths,
        chunk_size=512,  # Optional
        chunk_overlap=100,  # Optional
        max_embedding_requests_per_min=900,  # Optional
    )

    return file_import_response

def list_entire_corpora():
    try:
        corpora = rag.list_corpora()

        rag_corpora_list = list(corpora)
        length = len(rag_corpora_list)

        return {"count": length, "corpora": corpora}
    except Exception as e:
        logger.error(f"Error listing corpora: {e}")
        return str(e)

def get_first_corpus():
    # Grabs the first corpus's name - assuming you only have one
    try:
        corpora = rag.list_corpora()

        for corpus in corpora:
            corpus_name = corpus.name
            break

        return corpus_name
    except Exception as e:
        logger.error(f"Error listing corpora: {e}")
        return str(e)

def list_files_in_corpus():

    rag_corpus_name = get_first_corpus()

    files = rag.list_files(corpus_name=rag_corpus_name)
    files_array = []

    for file in files:
        files_array.append(file)

    return files_array

def delete_corpus(rag_corpus_name):

    try:
        rag.delete_corpus(name=rag_corpus_name)
        return "Successfully deleted corpus named: " + str(rag_corpus_name)
    except Exception as e:
        logger.error(f"Error deleting corpus: {e}")
        return str(e)

def generate_rag_context(prompt=""):

    rag_corpus_name = get_first_corpus()

    try:
        # Direct context retrieval
        response = rag.retrieval_query(
        rag_resources=[
            rag.RagResource(
                rag_corpus=rag_corpus_name,
                # Supply IDs from `rag.list_files()`.
                # rag_file_ids=["rag-file-1", "rag-file-2", ...],
            )
        ],
        text=prompt,
        similarity_top_k=1,  # Optional
        vector_distance_threshold=0.4,  # Optional
        )

        # Access the text from each context in the response
        if response:
            contextual_text = response.contexts.contexts[0].text
            source = response.contexts.contexts[0].source_uri
        else:
            contextual_text = "No relevant documents"
            source = "No relevant documents"

        return {"context": contextual_text, "source_uri": source}

    except Exception as e:
        logger.error(f"Error generating RAG context: {e}")
        return str(e)