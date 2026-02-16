from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings
from chunking import child_docs
import faiss 
import numpy as np

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

index = faiss.IndexFlatL2(384)  

vectors = np.array([embeddings.embed_query(doc.page_content) for doc in child_docs])

index.add(vectors)
faiss.write_index(index, "child_docs.index")

print("Index created and saved successfully.")