import logging

logger = logging.getLogger('RAG_Project')


def create_structured_prompt(query, retrieval_results, context_header="Relevant Context", query_header="User Query"):
    """
    Create a structured prompt for the LLM by combining the original query with retrieved context.

    The structured prompt includes:
      - A header for the query.
      - The original user query.
      - A header for the context.
      - The text of each retrieved chunk, formatted in order.

    Parameters:
        query (str): The original user query.
        retrieval_results (list of tuples): Each tuple should be in the form (identifier, distance, chunk_text),
            where 'identifier' is a tuple (e.g., (file_name, chunk_index)), 'distance' is the similarity score,
            and 'chunk_text' is the text content of the chunk.
        context_header (str): Header text to label the context section. Defaults to "Relevant Context".
        query_header (str): Header text to label the query section. Defaults to "User Query".

    Returns:
        str: A formatted prompt string suitable for sending to an LLM.

    Example:
        prompt = create_structured_prompt(
            "What are cat facts?",
            [
                (("cat_facts.txt.processed.txt.chunks.txt", 0), 0.45, "Cats are fascinating creatures."),
                (("cat_facts.txt.processed.txt.chunks.txt", 1), 0.50, "They are known for their agility and independence.")
            ]
        )
    """
    prompt_parts = []
    prompt_parts.append(f"{query_header}:\n{query}\n")
    prompt_parts.append(f"{context_header}:\n")

    # Add each retrieved chunk with its identifier for context reference.
    for idx, (identifier, distance, chunk_text) in enumerate(retrieval_results, start=1):
        prompt_parts.append(f"Chunk {idx} ({identifier}):\n{chunk_text}\n")

    # Optionally, you can add further instructions or delimiters.
    prompt_parts.append("Please use the above context to provide a detailed answer to the query.")

    structured_prompt = "\n".join(prompt_parts)
    logger.info("Structured prompt created successfully.")
    return structured_prompt


if __name__ == '__main__':
    # Example test: Define a sample query and retrieval results.
    sample_query = "What are some interesting cat facts?"
    sample_retrieval_results = [
        (("cat_facts.txt.processed.txt.chunks.txt", 0), 0.45, "Cats have flexible bodies and quick reflexes."),
        (("cat_facts.txt.processed.txt.chunks.txt", 1), 0.50, "They can jump up to six times their body length.")
    ]
    prompt = create_structured_prompt(sample_query, sample_retrieval_results)
    print("Generated Structured Prompt:\n")
    print(prompt)
