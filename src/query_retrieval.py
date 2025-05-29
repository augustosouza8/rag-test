import os
import logging
import pickle
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger('RAG_Project_Query')


def load_chunks_mapping(chunks_folder):
    """
    Load all chunk files from the given folder and create a mapping.

    Each file is expected to contain chunks separated by headers in the format:
      --- Chunk <number> ---

    Parameters:
        chunks_folder (str): Path to the folder with chunk files.

    Returns:
        dict: Mapping where each key is the file name and value is a list of text chunks.
    """
    mapping = {}
    try:
        for file in os.listdir(chunks_folder):
            file_path = os.path.join(chunks_folder, file)
            if os.path.isfile(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    # Split based on our inserted header during chunking.
                    chunks = re.split(r"--- Chunk \d+ ---\n", content)
                    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
                    mapping[file] = chunks
                    logger.debug(f"Loaded {len(chunks)} chunks from {file_path}")
                except Exception as e:
                    logger.error(f"Error reading {file_path}: {e}")
    except Exception as e:
        logger.error(f"Error accessing folder {chunks_folder}: {e}")
    return mapping


def load_embeddings(embeddings_path):
    """
    Load embeddings from a pickle file.

    Parameters:
        embeddings_path (str): Path to the embeddings pickle file.

    Returns:
        dict: Mapping of (document_name, chunk_index) to embedding vector.
    """
    try:
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        logger.info(f"Loaded embeddings from {embeddings_path}")
        return embeddings
    except Exception as e:
        logger.error(f"Error loading embeddings from {embeddings_path}: {e}")
        return None


def load_faiss_index(index_path):
    """
    Load a FAISS index from disk.

    Parameters:
        index_path (str): Path to the stored FAISS index.

    Returns:
        faiss.IndexFlatL2: The loaded FAISS index.
    """
    try:
        index = faiss.read_index(index_path)
        logger.info(f"Loaded FAISS index from {index_path}")
        return index
    except Exception as e:
        logger.error(f"Error loading FAISS index from {index_path}: {e}")
        return None


if __name__ == '__main__':
    # Set up a basic logging configuration.
    logging.basicConfig(level=logging.INFO)

    # Determine absolute paths relative to this file.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    embeddings_path = os.path.join(current_dir, "data", "embeddings.pkl")
    index_path = os.path.join(current_dir, "data", "faiss.index")
    chunks_folder = os.path.join(current_dir, "data", "chunks")

    # Load the FAISS index and embeddings dictionary.
    index = load_faiss_index(index_path)
    embeddings = load_embeddings(embeddings_path)
    if index is None or embeddings is None:
        logger.error("Could not load FAISS index or embeddings. Exiting.")
        exit(1)

    # Recreate the keys mapping from the embeddings dictionary.
    # Keys are tuples: (chunk_file, chunk_index)
    keys = list(embeddings.keys())

    # Load the chunks mapping and flatten it to map (file, index) -> chunk text.
    chunks_mapping = load_chunks_mapping(chunks_folder)
    flat_chunks = {}
    for file, chunks in chunks_mapping.items():
        for idx, chunk in enumerate(chunks):
            flat_chunks[(file, idx)] = chunk

    # Prompt the user for a query.
    query = input("Enter your query: ").strip()
    if not query:
        logger.error("Empty query provided. Exiting.")
        exit(1)

    # Ask user for the number of results to retrieve, with default k=5.
    k_input = input("Enter number of results to retrieve (default 5): ").strip()
    try:
        k = int(k_input) if k_input else 5
    except ValueError:
        logger.warning("Invalid input for number of results. Using default value 5.")
        k = 5

    # Load the embedding model to encode the query.
    logger.info("Loading embedding model for query...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode(query).reshape(1, -1).astype("float32")

    # Search the FAISS index.
    distances, indices = index.search(query_embedding, k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(keys):
            continue
        key = keys[idx]  # key is a tuple: (file, chunk_index)
        chunk_text = flat_chunks.get(key, "Chunk text not found.")
        results.append((key, dist, chunk_text))

    # Display the retrieval results.
    print(f"\nTop {k} results for your query:")
    for i, (key, dist, chunk_text) in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"  Document & Chunk: {key}")
        print(f"  Distance: {dist}")
        print(f"  Chunk Text: {chunk_text}")
