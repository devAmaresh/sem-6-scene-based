from sentence_transformers import SentenceTransformer, util

# Load the model
model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')


def get_best_match(object_descriptions, required_description):
    """
    Finds the best matching description from the list of object descriptions
    based on cosine similarity to the required description.

    Args:
    - object_descriptions (list of str): List of descriptions for objects.
    - required_description (str): Description to find the best match for.

    Returns:
    - str: Best matching description from `object_descriptions`.
    """

    # Precompute embeddings for object descriptions
    object_embeddings = [model.encode(desc, convert_to_tensor=True) for desc in object_descriptions]

    # Compute the embedding for the required description
    required_embedding = model.encode(required_description, convert_to_tensor=True)

    # Calculate cosine similarity scores
    cosine_scores = [util.cos_sim(required_embedding, emb)[0].item() for emb in object_embeddings]

    # Define the threshold for a match
    threshold = 0.1

    # Find the best match index and score
    best_match_index = None
    best_score = 0.0

    for i, score in enumerate(cosine_scores):
        if score > best_score and score >= threshold:
            best_match_index = i
            best_score = score

    # Return the best match if found, otherwise None
    return object_descriptions[best_match_index] if best_match_index is not None else None
