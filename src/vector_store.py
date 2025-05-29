import os
import logging
import pickle
import numpy as np
import faiss

logger = logging.getLogger('RAG_Project')


def load_embeddings(embeddings_path):
    """
    Load embeddings from a pickle file.

    Parameters:
        embeddings_path (str): The file path for the embeddings pickle file.

    Returns:
        dict: Embeddings mapping of (document_name, chunk_index) to embedding vector.
    """
    try:
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        logger.info(f"Loaded embeddings from {embeddings_path}")
        return embeddings
    except Exception as e:
        logger.error(f"Error loading embeddings from {embeddings_path}: {e}")
        return None


def create_faiss_index(embeddings):
    """
    Create a FAISS index from the embeddings dictionary.

    Parameters:
        embeddings (dict): Mapping of (document_name, chunk_index) to embedding vectors.

    Returns:
        tuple: A tuple containing:
            - index (faiss.IndexFlatL2): The populated FAISS index.
            - key_mapping (list): Ordered list of keys corresponding to index positions.
    """
    keys = list(embeddings.keys())
    if not keys:
        logger.error("Empty embeddings dictionary provided.")
        return None, None
    # Convert embeddings to a numpy array of type float32.
    vectors = [embeddings[key] for key in keys]
    vectors = np.array(vectors).astype('float32')
    dim = vectors.shape[1]

    # Create a flat (brute-force) index using L2 (Euclidean) distance.
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    logger.info(f"Created FAISS index with {index.ntotal} vectors of dimension {dim}.")
    return index, keys


def save_faiss_index(index, index_path):
    """
    Save the FAISS index to disk.

    Parameters:
        index (faiss.IndexFlatL2): The FAISS index to save.
        index_path (str): The file path where the index will be stored.
    """
    try:
        faiss.write_index(index, index_path)
        logger.info(f"FAISS index saved to {index_path}")
    except Exception as e:
        logger.error(f"Error saving FAISS index to {index_path}: {e}")


def load_faiss_index(index_path):
    """
    Load a FAISS index from disk.

    Parameters:
        index_path (str): The file path where the index is stored.

    Returns:
        faiss.IndexFlatL2: The loaded FAISS index, or None if loading fails.
    """
    try:
        index = faiss.read_index(index_path)
        logger.info(f"FAISS index loaded from {index_path}")
        return index
    except Exception as e:
        logger.error(f"Error loading FAISS index from {index_path}: {e}")
        return None


if __name__ == '__main__':
    # Compute the absolute paths relative to this file.
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    embeddings_path = os.path.join(current_file_dir, "data", "embeddings.pkl")
    index_path = os.path.join(current_file_dir, "data", "faiss.index")

    # Load embeddings from the embeddings pickle file.
    embeddings = load_embeddings(embeddings_path)
    if embeddings:
        # Create a FAISS index from the loaded embeddings.
        index, key_mapping = create_faiss_index(embeddings)
        if index is not None:
            # Save the FAISS index to disk.
            save_faiss_index(index, index_path)
            print(f"FAISS index created and saved with {index.ntotal} embeddings.")
        else:
            print("Failed to create FAISS index.")
    else:
        print("No embeddings loaded.")
