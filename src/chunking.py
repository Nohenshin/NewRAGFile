from typing import List, Tuple, Dict
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import hashlib

class TextSplitter:
    def __init__(self):
        # Parent splitter: large chunks for context
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        # Child splitter: small chunks for retrieval
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""]
        )
    
    def split_documents(self, docs: List[Document]) -> Tuple[List[Document], Dict[str, str]]:
        """
        Split documents into child chunks and return parent map.
        Returns:
            - child_chunks: List[Document] (small, for retrieval)
            - parent_map: Dict[parent_id, parent_text]
        """
        all_child_chunks = []
        parent_map = {}
        
        for doc in docs:
            parent_chunks = self.parent_splitter.split_text(doc.page_content)
            for parent_idx, parent_text in enumerate(parent_chunks):
                parent_id = hashlib.md5(parent_text.encode()).hexdigest()
                parent_map[parent_id] = parent_text
                
                child_chunks = self.child_splitter.split_text(parent_text)
                for child_text in child_chunks:
                    child_doc = Document(
                        page_content=child_text,
                        metadata={
                            **doc.metadata,
                            "parent_id": parent_id,
                            "parent_idx": parent_idx
                        }
                    )
                    all_child_chunks.append(child_doc)
        
        return all_child_chunks, parent_map