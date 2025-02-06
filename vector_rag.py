import os
from typing import List
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from azure.search.documents.indexes import SearchIndexClient

# Load environment variables
load_dotenv()

class VectorRAGApplication:
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
                index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
                credential=self.search_credential
            )
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            raise

    def get_embeddings(self, text: str) -> List[float]:
        """Generate embeddings using Azure OpenAI"""
        response = self.openai_client.embeddings.create(
            model=os.getenv("AZURE_OPENAI_EMB_DEPLOYMENT"),
            input=text
        )
        return response.data[0].embedding

    def create_search_index(self):
        """Create search index with vector search capability"""
        try:
            index_name = os.getenv("AZURE_SEARCH_INDEX_NAME_VECTOR")
            
            # Check if index exists
            try:
                existing_index = self.index_client.get_index(index_name)
                print(f"Index {index_name} already exists")
                return
            except Exception:
                print(f"Creating new index {index_name}")

            # Create the raw index definition
            index_definition = {
                "name": index_name,
                "fields": [
                    {
                        "name": "id",
                        "type": "Edm.String",
                        "key": True,
                        "filterable": True
                    },
                    {
                        "name": "content",
                        "type": "Edm.String",
                        "searchable": True,
                        "analyzer": "en.microsoft"
                    },
                    {
                        "name": "file_name",
                        "type": "Edm.String",
                        "filterable": True
                    },
                    {
                        "name": "content_vector",
                        "type": "Collection(Edm.Single)",
                        "searchable": True,
                        "dimensions": 1536,
                        "vectorSearchConfiguration": "default"
                    }
                ],
                "vectorSearch": {
                    "algorithmConfigurations": [
                        {
                            "name": "default",
                            "kind": "hnsw",
                            "parameters": {
                                "m": 4,
                                "efConstruction": 400,
                                "efSearch": 500,
                                "metric": "cosine"
                            }
                        }
                    ]
                }
            }

            # Create the index using the raw definition
            self.index_client._client.indexes.create(index_definition)
            print(f"Created index {index_name} with vector search")
        except Exception as e:
            print(f"Error creating index: {str(e)}")
            raise

    def load_document(self, file_path: str) -> List[dict]:
        """Load and chunk document into sections with embeddings"""
        chunks = []
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            paragraphs = text.split('\n\n')
            for i, para in enumerate(paragraphs):
                if para.strip():
                    content = para.strip()
                    chunks.append({
                        "id": str(i),
                        "content": content,
                        "file_name": os.path.basename(file_path),
                        "content_vector": self.get_embeddings(content)
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
        """Search for relevant documents using hybrid search"""
        try:
            # Get query embedding
            query_vector = self.get_embeddings(query)
            
            # Perform hybrid search (combines vector and keyword search)
            results = self.search_client.search(
                search_text=query,
                select=["content", "file_name"],
                vector_queries=[{
                    "vector": query_vector,
                    "fields": "content_vector",
                    "k": top
                }],
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
    rag_app = VectorRAGApplication()

    # Load and index documents
    documents = []
    data_folder = "Data"
    # Get all .txt files from the Data folder
    data_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.txt')]
    print(f"Found {len(data_files)} documents to process: {[os.path.basename(f) for f in data_files]}")
    
    for file_path in data_files:
        documents.extend(rag_app.load_document(file_path))
    rag_app.index_documents(documents)

    # Example queries to demonstrate semantic search
    queries = [
        # Semantic similarity without exact keyword matches
        "What are the risks to ocean life from pollution?",  # Should match ocean acidification content
        "How is the space industry becoming more accessible?",  # Should match SpaceX content
        "What's new in AI and computing?",  # Should match both AI and quantum computing content
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 50)
        results = rag_app.search_documents(query)
        response = rag_app.generate_response(query, results)
        print("Response:", response)

if __name__ == "__main__":
    main() 