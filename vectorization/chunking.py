import hashlib
from document_loader import load_markdown_files
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os, pickle

class LocalParentDocstore:
    def __init__(self, path = './parent_docstore'):
        self.path = path
        os.makedirs(self.path, exist_ok=True)

    def _get_file_path(self, key):
        return os.path.join(self.path, f"{key}.pkl")
    
    def mset(self, key_doc_pairs):
        for key, doc in key_doc_pairs:
            with open(self._get_file_path(key), 'wb') as f:
                pickle.dump(doc, f)

    def mget(self, keys):
        docs = []
        for key in keys:
            file_path = self._get_file_path(key)
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    docs.append(pickle.load(f))
            else:
                docs.append(None)
        return docs


def generate_deterministic_id(content, source_name):
    hash_input = f"{content}_{source_name}"
    return hashlib.sha256(hash_input.encode('utf-8')).hexdigest()


def create_parent_child_docs(document):
    parent_docs = []
    child_docs = []

    for doc in documents:
        parents = parent_splitter.split_text(doc.page_content)

        for parent in parents:
            parent_id = generate_deterministic_id(parent.page_content, doc.metadata['source'])
            parent.metadata.update({"parent_id": parent_id, "source": doc.metadata['source']
                                    })
            parent_docs.append(parent)
        
            children = child_splitter.split_text(parent.page_content)
            for child in children:

                if isinstance(child, str):
                    child = Document(page_content=child, metadata={})
                child.metadata.update({"parent_id": parent_id, "source": doc.metadata['source']})
                child_docs.append(child)
            break

    return parent_docs, child_docs

folder_path = '/home/vis5055/Documents/Medical-Assitant/DATA/WHO_Fact_Sheet'
documents = load_markdown_files(folder_path)

parent_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[('#', "h1"), ('##', "h2"), ('###', "h3")], strip_headers=True)

child_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)

parent_docs, child_docs = create_parent_child_docs(documents)

store = LocalParentDocstore("./parent_docstore")

store.mset([(parent.metadata['parent_id'], parent) for parent in parent_docs])
