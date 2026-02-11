import os
import pandas as pd
from langchain_community.document_loaders import DirectoryLoader, TextLoader

def load_markdown_files(folder_path):

    '''
    This function loads markdown files from a specified folder using langchain's DirectoryLoader. 
    It processes each document to extract the filename and assigns a unique parent_id for potential future chunking.
    
    Input: folder_path (str): The path to the folder containing markdown files.

    '''
    loader = DirectoryLoader(folder_path, glob="**/*.md", show_progress=True, silent_errors=True, loader_cls=TextLoader)
    documents = loader.load()

    return documents






