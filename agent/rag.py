import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

class AutoStreamRAG:
    """
    Manages the RAG pipeline: document loading, splitting, FAISS database construction, and retrieval.
    """
    def __init__(self, kb_path: str, embeddings: Embeddings):
        self.kb_path = kb_path
        self.embeddings = embeddings
        self.vector_store = None
        self.retriever = None
        self._initialize()

    def _initialize(self):
        """
        Loads the knowledge base markdown, splits it, and builds a FAISS vector store.
        """
        if not os.path.exists(self.kb_path):
            raise FileNotFoundError(f"Knowledge base file not found at: {self.kb_path}")
        
        try:
            # Load the markdown file
            loader = TextLoader(self.kb_path, encoding="utf-8")
            documents = loader.load()
            
            # Split the document into clean chunks for dense retrieval
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=300,
                chunk_overlap=50,
                separators=["\n\n", "\n", " ", ""]
            )
            split_docs = text_splitter.split_documents(documents)
            
            # Build index in-memory using FAISS and embeddings
            self.vector_store = FAISS.from_documents(split_docs, self.embeddings)
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 2})
            
        except Exception as e:
            print(f"[Error Initializing RAG]: {e}")
            raise e

    def retrieve_context(self, query: str) -> str:
        """
        Retrieves relevant context matching the query from the vector store.
        """
        if not self.retriever:
            return ""
        
        try:
            docs = self.retriever.invoke(query)
            context = "\n\n".join([doc.page_content for doc in docs])
            return context
        except Exception as e:
            print(f"[Error Retrieving Context]: {e}")
            return ""
