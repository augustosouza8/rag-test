import os
import logging
import pickle
import re
from sentence_transformers import SentenceTransformer

logger = logging.getLogger('RAG_Project')


def load_embedding_model(model_name="all-MiniLM-L6-v2"):
    """
    Load a pre-trained SentenceTransformer model from Hugging Face.

    Parameters:
        model_name (str): Name of the model to load. Default is "all-MiniLM-L6-v2".

    Returns:
        SentenceTransformer: The loaded embedding model.
    """
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    logger.info("Embedding model loaded successfully.")
    return model


def generate_embeddings(chunks_mapping, model):
    """
    Generate embeddings for the provided document chunks.

    This function iterates over the mapping of document names to their corresponding text chunks
    and computes an embedding for each chunk.

    Parameters:
        chunks_mapping (dict): Mapping of document file names to list of text chunks.
        model (SentenceTransformer): Pre-trained embedding model.

    Returns:
        dict: Mapping of (document_name, chunk_index) to the embedding vector.
    """
    embeddings = {}
    for doc, chunks in chunks_mapping.items():
        logger.info(f"Generating embeddings for document: {doc}")
        for idx, chunk in enumerate(chunks):
            try:
                # Generate the embedding for the chunk.
                embedding = model.encode(chunk)
                embeddings[(doc, idx)] = embedding
                logger.debug(f"Generated embedding for {doc} chunk {idx}")
            except Exception as e:
                logger.error(f"Error generating embedding for {doc} chunk {idx}: {e}")
    return embeddings


def save_embeddings(embeddings, file_path='data/embeddings.pkl'):
    """
    Save the generated embeddings to a pickle file for persistence.

    Parameters:
        embeddings (dict): Mapping of embeddings data.
        file_path (str): The output file path where the embeddings will be stored.
    """
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(embeddings, f)
        logger.info(f"Embeddings saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving embeddings to {file_path}: {e}")


def load_chunks_mapping(chunks_folder='data/chunks'):
    """
    Load all chunk files from a given folder and create a mapping of document names to a list of chunks.

    Assumes each file in the chunks_folder is a text file containing chunks separated by a header delimiter
    in the format: '--- Chunk <number> ---'.

    Parameters:
        chunks_folder (str): Path to the folder containing chunk files.

    Returns:
        dict: Mapping where keys are document file names and values are lists of text chunks.
    """
    mapping = {}

    try:
        for file in os.listdir(chunks_folder):
            file_path = os.path.join(chunks_folder, file)
            if os.path.isfile(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    # Split content on the header that indicates the start of a new chunk
                    chunks = re.split(r"--- Chunk \d+ ---\n", content)
                    # Remove any empty results and trim whitespace
                    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
                    mapping[file] = chunks
                    logger.debug(f"Loaded {len(chunks)} chunks from {file_path}")
                except Exception as e:
                    logger.error(f"Error loading chunks from {file_path}: {e}")
    except Exception as e:
        logger.error(f"Error accessing folder {chunks_folder}: {e}")

    return mapping


if __name__ == '__main__':
    # Compute the absolute path to the chunks folder relative to this file's location.
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming your folder structure remains: [project root]/src/data/chunks.
    # We move one level up from src to get to project root and then into src/data/chunks.
    chunks_folder_path = os.path.join(current_file_dir, "data", "chunks")
    logger.info(f"Looking for chunks in: {chunks_folder_path}")

    # Load the chunks mapping from your real chunked files.
    chunks_mapping = load_chunks_mapping(chunks_folder=chunks_folder_path)
    logger.info(f"Loaded chunks for {len(chunks_mapping)} document(s).")

    # Load the embedding model.
    model = load_embedding_model()

    # Generate embeddings using the actual chunks.
    embeddings = generate_embeddings(chunks_mapping, model)

    # Save the embeddings to disk.
    save_embeddings(embeddings)

    print(f"Generated embeddings for {len(embeddings)} chunk(s). Check app.log and 'data/embeddings.pkl' for details.")