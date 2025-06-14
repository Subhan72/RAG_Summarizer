from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch

class DocumentRetriever:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the retriever with a sentence transformer model.
        Uses GPU if available, otherwise CPU.
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name, device=device)
        self.index = None
        self.chunks = []

    def embed_chunks(self, chunks):
        """
        Generate embeddings for the given list of chunks.
        Normalize embeddings for cosine similarity.
        """
        self.chunks = chunks
        embeddings = self.model.encode(chunks, show_progress_bar=True)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings

    def build_index(self, embeddings):
        """
        Build a FAISS index using Inner Product for cosine similarity.
        """
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  
        self.index.add(embeddings)

    def retrieve(self, query, k=5):
        """
        Retrieve the top-k chunks most similar to the query using cosine similarity.
        Returns the chunks and their cosine similarity scores.
        """
        if self.index is None:
            raise ValueError("Index is not built. Call build_index first.")
        query_embedding = self.model.encode([query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        similarities, indices = self.index.search(query_embedding, k)
        retrieved_chunks = [self.chunks[idx] for idx in indices[0]]
        return retrieved_chunks, similarities[0]