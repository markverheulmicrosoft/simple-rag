import os
import logging
from typing import List
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.core.exceptions import HttpResponseError
from openai import AzureOpenAI  # new client-based interface

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Deployment names from environment variables
AZURE_OPENAI_EMB_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMB_DEPLOYMENT")  # e.g. "text-embedding-ada-002"
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")            # e.g. "gpt-35-turbo"

class VectorRAGApplication:
    def __init__(self):
        try:
            # Initialize Azure Search clients
            self.search_credential = AzureKeyCredential(os.getenv("AZURE_SEARCH_ADMIN_KEY"))
            
            endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
            # Ensure endpoint does not end with a slash
            if endpoint.endswith('/'):
                endpoint = endpoint[:-1]
                
            self.index_client = SearchIndexClient(
                endpoint=endpoint,
                credential=self.search_credential
            )
            
            # Create index if it doesn't exist
            self.create_search_index()
            
            self.search_client = SearchClient(
                endpoint=endpoint,
                index_name=os.getenv("AZURE_SEARCH_INDEX_NAME_VECTOR"),
                credential=self.search_credential
            )
            
            # Instantiate the AzureOpenAI client using the new client-based interface
            self.ai_client = AzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2024-02-01"  # Adjust as needed
            )
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            raise

    def create_search_index(self):
        """Create search index with vector search capability"""
        try:
            index_name = os.getenv("AZURE_SEARCH_INDEX_NAME_VECTOR")
            
            # Check if index exists
            try:
                existing_index = self.index_client.get_index(index_name)
                print(f"Index {index_name} already exists")
                return
            except HttpResponseError as e:
                if e.status_code == 404:
                    print(f"Creating new index {index_name}")
                else:
                    raise

            # Updated index definition using "vectorSearchProfile" for the vector field
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
                        "vectorSearchProfile": "default"  # updated property name
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
            self.index_client.create_index(index_definition)
            print(f"Created index {index_name} with vector search")
        except HttpResponseError as e:
            if e.status_code == 403:
                print("Error creating index: Forbidden. Please check your credentials and permissions.")
            else:
                print(f"Error creating index: {e}")
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
                    embedding = self.get_embeddings(content)
                    logging.info(f"Vectorizing content from {file_path}: {content[:100]}... -> {embedding[:5]}...")
                    chunks.append({
                        "id": f"{os.path.basename(file_path)}_{i}",  # Composite unique ID
                        "content": content,
                        "file_name": os.path.basename(file_path),
                        "content_vector": embedding
                    })
        return chunks


    def index_documents(self, documents: List[dict]):
        """Index documents in Azure Search"""
        try:
            self.search_client.upload_documents(documents=documents)
            logging.info(f"Indexed {len(documents)} documents successfully")
        except Exception as e:
            logging.error(f"Error indexing documents: {e}")

    def search_documents(self, query: str, top: int = 3) -> List[dict]:
        """Search for relevant documents using hybrid search that includes vector similarity"""
        try:
            # Get query embedding using the new client
            query_vector = self.get_embeddings(query)
            logging.info(f"Query vector: {query_vector[:5]}...")

            # Perform hybrid search using the correct parameter "vector_queries"
            results = self.search_client.search(
                search_text=query,
                select=["content", "file_name"],
                vector_queries=[{
                    "vector": query_vector,
                    "fields": "content_vector",
                    "k": top,
                    "similarityFunction": "cosine"
                }],
                top=top
            )
            return [dict(result) for result in results]
        except Exception as e:
            logging.error(f"Error searching documents: {e}")
            return []

    def get_existing_documents(self) -> set:
        """Get a set of file names that have already been indexed"""
        try:
            results = self.search_client.search(search_text="*", select=["file_name"], top=1000)
            existing_files = {result["file_name"] for result in results}
            logging.info(f"Existing documents in the index: {existing_files}")
            return existing_files
        except Exception as e:
            logging.error(f"Error retrieving existing documents: {e}")
            return set()

    def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings for a given text using the new AzureOpenAI client"""
        response = self.ai_client.embeddings.create(
            input=[text],
            model=AZURE_OPENAI_EMB_DEPLOYMENT
        )
        return response.data[0].embedding

    def generate_response(self, query: str, documents: List[dict]) -> str:
        """Generate a response based on the query and retrieved documents using the new AzureOpenAI client"""
        context = "\n".join([doc["content"] for doc in documents])
        response = self.ai_client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are an AI assistant."},
                {"role": "user", "content": query},
                {"role": "assistant", "content": context}
            ]
        )
        return response.choices[0].message.content

def main():
    # Initialize RAG application
    rag_app = VectorRAGApplication()

    # Get existing documents to avoid re-indexing
    existing_documents = rag_app.get_existing_documents()
    
    # Load and index documents
    documents = []
    data_folder = "Data"
    data_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.txt')]
    print(f"Found {len(data_files)} documents to process: {[os.path.basename(f) for f in data_files]}")
    
    for file_path in data_files:
        file_name = os.path.basename(file_path)
        if file_name not in existing_documents:
            documents.extend(rag_app.load_document(file_path))
        else:
            logging.info(f"Skipping already processed file: {file_name}")
    
    if documents:
        rag_app.index_documents(documents)
    else:
        logging.info("No new documents to index")

    # Prompt user for queries
    while True:
        query = input("\nEnter your query (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        print("-" * 50)
        results = rag_app.search_documents(query)
        response = rag_app.generate_response(query, results)
        print("Response:", response)

if __name__ == "__main__":
    main()
