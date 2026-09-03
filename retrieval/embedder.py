
def get_local_embeddings():
    """
    Free local embedding model using HuggingFace.
    Runs on your CPU — no API key, no cost.
    Good enough for most RAG projects.
    
    Model: all-MiniLM-L6-v2
    Dimensions: 384
    Speed: fast on CPU
    Quality: good for English text
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings
    
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
