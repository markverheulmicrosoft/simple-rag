import os
import logging
import base64  # for encoding keys
from typing import List
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from openai import AzureOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Additional imports for building the index with vector fields and profiles
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    HnswParameters,
    VectorSearchProfile
)
from azure.search.documents.models import VectorizedQuery  # for query construction

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

            self.index_client = SearchIndexClient(
                endpoint=endpoint,
                credential=self.search_credential
            )
            
            # Create index if it doesn't exist using the new SearchIndex object
            index_name=os.getenv("AZURE_SEARCH_INDEX_NAME_VECTOR")
            self.create_search_index(index_name)
            
            self.search_client = SearchClient(
                endpoint=endpoint,
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

    def create_search_index(self, index_name):
        """Create search index with vector search capability using a SearchIndex model"""
        try:
            # Check if the index already exists
            try:
                self.index_client.get_index(index_name)
                print(f"Index {index_name} already exists")
                return
            except HttpResponseError as e:
                if e.status_code == 404:
                    print(f"Creating new index {index_name}")
                else:
                    raise

            # Define fields using the new model classes:
            fields = [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
                SearchableField(name="content", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
                SimpleField(name="file_name", type=SearchFieldDataType.String, filterable=True),
                # Define the vector field with retrievable=True so it can be returned in queries.
                SearchField(
                    name="content_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    retrievable=True,
                    vector_search_dimensions=1536,
                    vector_search_profile_name="default"  # This must match a profile in vector_search.profiles
                )
            ]
            
            # Define the vector search configuration with both an algorithm configuration and a profile.
            vector_search_config = VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="default",
                        kind="hnsw",
                        parameters=HnswParameters(
                            m=4,
                            ef_construction=400,
                            ef_search=500,
                            metric="cosine"
                        )
                    )
                ],
                profiles=[
                    VectorSearchProfile(
                        name="default",
                        algorithm_configuration_name="default"
                    )
                ]
            )
            
            # Create the index object
            index_obj = SearchIndex(
                name=index_name,
                fields=fields,
                vector_search=vector_search_config
            )
            
            # Create the index
            self.index_client.create_index(index_obj)
            print(f"Created index {index_name} with vector search")
        except HttpResponseError as e:
            if e.status_code == 403:
                print("Error creating index: Forbidden. Please check your credentials and permissions.")
            else:
                print(f"Error creating index: {e}")
            raise

    def load_document(self, file_path: str) -> List[dict]:
        """Load and chunk a document using a recursive character text splitter."""
        chunks = []
        # Create a splitter with a chunk size of 1000 characters and an overlap of 200 characters.
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Use the splitter to get a list of text chunks.
        chunk_texts = splitter.split_text(text)
        
        for i, chunk in enumerate(chunk_texts):
            if chunk.strip():
                embedding = self.get_embeddings(chunk)
                logging.info(f"Vectorizing content from {file_path}: {chunk[:100]}... -> {embedding[:5]}...")
                # Generate a raw key and then encode it to be URL-safe
                raw_key = f"{os.path.basename(file_path)}_{i}"
                encoded_key = base64.urlsafe_b64encode(raw_key.encode("utf-8")).decode("ascii")
                chunks.append({
                    "id": encoded_key,
                    "content": chunk,
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
        """Search for relevant documents using hybrid search that includes vector similarity.
           Uses the VectorizedQuery object to set the query parameters.
        """
        try:
            query_vector = self.get_embeddings(query)
            logging.info(f"Query vector: {query_vector[:5]}...")
            
            # Use VectorizedQuery from the SDK instead of a dict
            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=top,
                fields="content_vector"
            )
            results = self.search_client.search(
                search_text=query,
                select=["content", "file_name"],
                vector_queries=[vector_query],
                top=top
            )
            return [dict(result) for result in results]
        except Exception as e:
            logging.error(f"Error searching documents: {e}")
            return []

    def get_existing_documents(self) -> set:
        """Retrieve a set of file names that are already indexed."""
        try:
            results = self.search_client.search(search_text="*", select=["file_name"], top=1000)
            existing_files = {result["file_name"] for result in results}
            logging.info(f"Existing documents in the index: {existing_files}")
            return existing_files
        except Exception as e:
            logging.error(f"Error retrieving existing documents: {e}")
            return set()

    def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings for a given text using the AzureOpenAI client."""
        response = self.ai_client.embeddings.create(
            input=[text],
            model=AZURE_OPENAI_EMB_DEPLOYMENT
        )
        return response.data[0].embedding

    def generate_response(self, query: str, documents: List[dict]) -> str:
        # Create a system prompt that instructs the model to use only the context
        system_prompt = (
            "You are an assistent helping to give as much information about funds based solely on the provided document context. "
            "If the context does not contain the answer or you are not able to answer, please elaborate why you can't answer."
            ""
        )
        context = "\n".join([doc["content"] for doc in documents])
        response = self.ai_client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
                {"role": "assistant", "content": context}
            ]
        )
        return response.choices[0].message.content


def main():
    # Initialize the application
    rag_app = VectorRAGApplication()
    
    # Retrieve existing file names to avoid re-indexing
    existing_documents = rag_app.get_existing_documents()
    
    # Load and index documents from the "Data" folder
    documents = []
    data_folder = "Data"
    data_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.txt')]
    print(f"Found {len(data_files)} documents to process: {[os.path.basename(f) for f in data_files]}")
    
    for file_path in data_files:
        file_name = os.path.basename(file_path)
        if file_name in existing_documents:
            logging.info(f"Skipping already processed file: {file_name}")
            continue
        documents.extend(rag_app.load_document(file_path))
    
    if documents:
        rag_app.index_documents(documents)
    else:
        logging.info("No new documents to index")
    
    # Query loop
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
