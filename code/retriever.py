import faiss
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
import pickle

child_path = '/home/vis5055/Documents/Medical-Assitant/vectorization/child_docs.index'
metadata_path = '/home/vis5055/Documents/Medical-Assitant/vectorization/child_docs_metadata.pkl'

faiss_index = faiss.read_index(child_path)

with open(metadata_path, "rb") as metadata_file:
    metadata = pickle.load(metadata_file)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

class customParentRetriever:
    def __init__(self, vectorstore, metadata, embedding_model, k =5):
        self.vectorstore = vectorstore
        self.metadata = metadata
        # self.parent_docstore = parent_docstore
        self.embedding_model = embedding_model
        self.k = k

    def get_relevant_documents(self, query):
        query_vector = self.embedding_model.embed_query(query)
        distances, indices = self.vectorstore.search(np.array([query_vector]), k =self.k)

        relevant_child_chunks = []
        for i in indices[0]:  
            relevant_child_chunks.append(self.metadata[i])  # Append the metadata dictionary to the results
        
        return relevant_child_chunks, distances
    
retriever = customParentRetriever(vectorstore=faiss_index, metadata=metadata, embedding_model=embeddings, k=5)
# query = "What is aging?"

# relevant_child_chunks, distances = retriever.get_relevant_documents(query)

# print("Relevant Child Chunks:")
# for chunk in relevant_child_chunks:
#     print(chunk)

