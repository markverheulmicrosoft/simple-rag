import os
from typing import List
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
)

# Load environment variables
load_dotenv()

class SimpleRAGApplication:
    def __init__(self):
        try:
            # Initialize Azure Search clients
            self.search_credential = AzureKeyCredential(os.getenv("AZURE_SEARCH_ADMIN_KEY"))
            
            endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
            if not endpoint.endswith('/'):
                endpoint += '/'
                
            self.index_client = SearchIndexClient(
                endpoint=endpoint,
                credential=self.search_credential
            )
            
            # Initialize Azure OpenAI client
            self.openai_client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version="2024-02-15-preview"
            )
            
            # Create index if it doesn't exist
            self.create_search_index()
            
            self.search_client = SearchClient(
                endpoint=endpoint,
                index_name=os.getenv("AZURE_SEARCH_INDEX_NAME_SIMPLE"),
                credential=self.search_credential
            )
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            raise

    def create_search_index(self):
        """Create basic search index"""
        try:
            index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
            
            # Check if index exists
            try:
                existing_index = self.index_client.get_index(index_name)
                print(f"Index {index_name} already exists")
                return
            except Exception:
                print(f"Creating new index {index_name}")

            # Define fields for basic search
            fields = [
                SearchField(name="id", type=SearchFieldDataType.String, key=True),
                SearchField(
                    name="content", 
                    type=SearchFieldDataType.String, 
                    searchable=True,
                    analyzer_name="en.microsoft"
                ),
                SearchField(name="file_name", type=SearchFieldDataType.String)
            ]

            # Create index
            index = SearchIndex(
                name=index_name,
                fields=fields
            )
            
            self.index_client.create_index(index)
            print(f"Created index {index_name}")
        except Exception as e:
            print(f"Error creating index: {str(e)}")
            raise

    def load_document(self, file_path: str) -> List[dict]:
        """Load and chunk document into sections"""
        chunks = []
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            paragraphs = text.split('\n\n')
            for i, para in enumerate(paragraphs):
                if para.strip():
                    chunks.append({
                        "id": str(i),
                        "content": para.strip(),
                        "file_name": os.path.basename(file_path)
                    })
        return chunks

    def index_documents(self, documents: List[dict]):
        """Index documents in Azure Search"""
        try:
            self.search_client.upload_documents(documents=documents)
            print(f"Indexed {len(documents)} documents successfully")
        except Exception as e:
            print(f"Error indexing documents: {e}")

    def search_documents(self, query: str, top: int = 3) -> List[dict]:
        """Search for relevant documents using keyword search"""
        try:
            results = self.search_client.search(
                search_text=query,
                select=["content", "file_name"],
                top=top
            )
            return [dict(result) for result in results]
        except Exception as e:
            print(f"Error searching documents: {e}")
            return []

    def generate_response(self, query: str, context: List[dict]) -> str:
        """Generate response using Azure OpenAI"""
        context_text = "\n".join([doc["content"] for doc in context])
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer questions."},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
        ]

        response = self.openai_client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )

        return response.choices[0].message.content

def main():
    # Initialize RAG application
    rag_app = SimpleRAGApplication()

    # Load and index documents
    documents = []
    data_folder = "Data"
    # Get all .txt files from the Data folder
    data_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.txt')]
    print(f"Found {len(data_files)} documents to process: {[os.path.basename(f) for f in data_files]}")
    
    for file_path in data_files:
        documents.extend(rag_app.load_document(file_path))
    rag_app.index_documents(documents)

    # Example queries
    queries = [
        "What are some environmental impacts of rising carbon dioxide levels?",
        "How is private industry changing space accessibility?",
        "What developments are happening in computational technology?",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 50)
        results = rag_app.search_documents(query)
        response = rag_app.generate_response(query, results)
        print("Response:", response)

if __name__ == "__main__":
    main() 