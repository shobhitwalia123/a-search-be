"""
AI Search Engine API - Core Application

This module contains the main FastAPI application for the AI Search Engine.
It provides semantic search capabilities using OpenAI embeddings and Pinecone vector database.

Features:
- Semantic search with OpenAI embeddings
- AI-powered answers using GPT-4
- Vector database integration with Pinecone
- Content processing and HTML cleaning
- External API integration

Author: AI Search Engine Team
Version: 1.0.0
"""

# Standard library imports
import os
import time
import hashlib
from typing import List, Any

# Third-party imports
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "pclocal")
PINECONE_HOST = os.environ.get("PINECONE_HOST", "http://localhost:5080")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT", "us-east-1")

# Initialize OpenAI
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

# We'll use OpenAI client directly for embeddings

client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
except Exception as e:
    print(f"Pinecone initialization failed: {e}")
    pc = None

index = None

def chunk_text_for_list(docs: List[str], max_chunk_size: int = 1000) -> List[List[str]]:
    """
    Break down each text in a list of texts into chunks of a maximum size, attempting to preserve whole paragraphs.
    """
    def chunk_text(text: str, max_chunk_size: int) -> List[str]:
        if not text.endswith("\n\n"):
            text += "\n\n"
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) + 2 > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            current_chunk += paragraph.strip() + "\n\n"
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    return [chunk_text(doc, max_chunk_size) for doc in docs]

def generate_embeddings(docs: List[Any]) -> List[List[float]]:
    """
    Generate embeddings for a list of documents (where each doc is a list of text chunks).
    """
    result = []
    for doc in docs:
        doc_embeddings = []
        for text in doc:
            response = client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            doc_embeddings.append(response.data[0].embedding)
        result.append(doc_embeddings)
    return result

def generate_short_id(content: str) -> str:
    """Generate a short ID based on the content using SHA-256 hash."""
    hash_obj = hashlib.sha256()
    hash_obj.update(content.encode("utf-8"))
    return hash_obj.hexdigest()

def combine_vector_and_text(docs: List[Any], doc_embeddings: List[List[float]]) -> List[dict]:
    """
    Combine text chunks and their embeddings into a structure suitable for Pinecone upsert.
    """
    data_with_metadata = []
    for doc_text, embedding in zip(docs, doc_embeddings):
        for chunk, vector in zip(doc_text, embedding):
            doc_id = generate_short_id(chunk)
            data_item = {
                "id": doc_id,
                "values": vector,
                "metadata": {"text": chunk}
            }
            data_with_metadata.append(data_item)
    return data_with_metadata

def upsert_data_to_pinecone(data_with_metadata: List[dict]) -> None:
    """Upsert data with metadata into the Pinecone index."""
    global index
    index.upsert(vectors=data_with_metadata)

def get_query_embeddings(query: str) -> List[float]:
    """Get embeddings for the given query string."""
    response = client.embeddings.create(
        input=query,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def query_pinecone_index(query_embeddings: List[float], top_k: int = 2) -> dict:
    """Query the Pinecone index for nearest neighbors."""
    global index
    query_response = index.query(
        vector=query_embeddings,
        top_k=top_k,
        include_metadata=True
    )
    return query_response

def better_query_response(prompt: str) -> str:
    """Use the LLM (OpenAI) to get a better, summarized response."""

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4o",
    )
    response_text = chat_completion.choices[0].message.content
    return response_text

def remove_html_tags_bs4(text: str) -> str:
    """
    Removes all HTML tags from the given string using BeautifulSoup.
    
    Args:
        text (str): The input string that may contain HTML tags.
    
    Returns:
        str: A string with all HTML tags removed.
    """
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

app = FastAPI(
    title="AI Search Engine API",
    description="""
    A powerful AI-powered search engine API that provides semantic search capabilities using OpenAI embeddings and Pinecone vector database.
    
    ## Features
    
    * **Semantic Search** - Uses OpenAI embeddings for intelligent search
    * **AI-Powered Answers** - GPT-4 generates contextual responses
    * **Vector Database** - Pinecone integration for efficient similarity search
    * **Content Processing** - Automatic text chunking and HTML cleaning
    * **External Integration** - Fetches content from external APIs
    
    ## Workflow
    
    1. Create a Pinecone index using `/create-index/{index_name}`
    2. Ingest documents using `/upsert-data`
    3. Search using `/search` endpoint
    4. Get AI-powered answers based on your indexed content
    
    ## Authentication
    
    This API requires OpenAI API key to be set in environment variables.
    """,
    version="1.0.0",
    contact={
        "name": "AI Search Engine Support",
        "email": "support@example.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    servers=[
        {
            "url": os.environ.get("API_BASE_URL", "http://localhost:8000"),
            "description": "API Server"
        }
    ]
)

# Add CORS middleware to allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for Vercel
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DocumentList(BaseModel):
    """List of documents to be processed and indexed"""
    documents: List[str]

class SearchRequest(BaseModel):
    """Search query request"""
    query: str

class SearchResult(BaseModel):
    """Individual search result"""
    id: str
    title: str
    description: str
    url: str
    score: float

class SearchResponse(BaseModel):
    """Search response containing results"""
    query: str
    results: List[SearchResult]
    total: int
    timestamp: str

class IndexResponse(BaseModel):
    """Index operation response"""
    message: str
    index_stats: str = None

class ErrorResponse(BaseModel):
    """Error response"""
    error: str

# 1) Create Index
@app.post(
    "/create-index/{index_name}",
    response_model=IndexResponse,
    responses={
        200: {"description": "Index created successfully"},
        400: {"model": ErrorResponse, "description": "Index already exists or invalid request"}
    },
    tags=["Index Management"],
    summary="Create a new Pinecone index",
    description="Creates a new Pinecone index with the specified name. The index will be ready for vector operations after creation."
)
def create_pinecone_index(index_name: str):
    """
    Create a Pinecone index with the specified name.
    
    - **index_name**: Name of the index to create
    - **dimension**: Fixed at 1536 (OpenAI embedding dimension)
    - **metric**: Cosine similarity
    - **spec**: AWS serverless specification
    
    Returns index statistics once the index is ready.
    """
    global index
    if pc is None:
        return {"error": "Pinecone not initialized. Check your PINECONE_API_KEY and PINECONE_HOST environment variables."}
    
     # Check if index already exists
    existing_indexes = pc.list_indexes()
    if index_name in existing_indexes.names():
        return {
            "error": f"Index '{index_name}' already exists. "
                     f"Use the load-index endpoint or choose a different name."
        }
    # Create the index
    pc.create_index(
        name=index_name,
        dimension=1536,  # Must match embedding dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT)
    )

    # Wait for the index to be ready
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

    # Instantiate and store the index object in the global variable
    index = pc.Index(index_name)

    return {
        "message": f"Index '{index_name}' created successfully and is ready.",
        "index_stats": index.describe_index_stats().to_str()
    }

# 2) Upsert Data
@app.post(
    "/upsert-data",
    tags=["Data Operations"],
    summary="Ingest and index documents",
    description="Fetches content from external APIs, combines with provided documents, chunks the text, generates embeddings, and upserts to Pinecone index."
)
def upsert_data(
    doc_list: DocumentList, 
    extra_param: str = Query(None, description="Any optional extra parameter to pass?")
):
    """
    1) Call /getPageList to retrieve a list of pages.
    2) For each page, call /getContent?pageId=<PAGE_ID> to get the content.
    3) Combine fetched content with any user-provided doc_list.documents.
    4) Chunk, embed, and upsert into Pinecone.
    """
    global index
    if index is None:
        return {"error": "Index not created or not initialized. Call /create-index OR /load-index first."}

    # 1) Call the getPageList endpoint
    page_list_url = "https://fd10-2405-201-4018-1852-45e9-d3b8-df3b-f8ed.ngrok-free.app/getPageList"
    try:
        page_list_response = requests.get(page_list_url)
        page_list_response.raise_for_status()  # Raise an exception if 4xx/5xx
        page_list_json = page_list_response.json()
    except Exception as e:
        return {"error": f"Failed to call getPageList: {e}"}

    if "results" not in page_list_json:
        return {"error": "No 'results' field found in /getPageList response."}

    # 2) For each page in results, call getContent
    #    We'll collect their content in a list
    combined_documents = []

    for page_info in page_list_json["results"]:
        page_id = page_info.get("id")
        if not page_id:
            # Skip this item if no ID
            continue

        # Build URL for second API call
        get_content_url = f"https://fd10-2405-201-4018-1852-45e9-d3b8-df3b-f8ed.ngrok-free.app/getContent?pageId={page_id}"

        # If you have an additional parameter to pass (extra_param), you could do:
        # get_content_url += f"&extraParam={extra_param}"
        # or handle logic differently if needed.

        try:
            content_response = requests.get(get_content_url)
            content_response.raise_for_status()
            content_json = content_response.json()
            print(f"Processed Id : {page_id}")
        except Exception as e:
            # If one page fails, log/collect the error but continue
            print(f"Error fetching pageId {page_id}: {e}")
            continue

        # Suppose content_json has a field "content" with the actual text
        page_content = content_json.get("value")
        if page_content:
            # Append the retrieved content
            combined_documents.append(remove_html_tags_bs4(page_content))
        else:
            # If the entire JSON is relevant, you could store the whole dict as a string
            # combined_documents.append(str(content_json))
            pass

    # 3) Also append any user-provided docs from the request body
    if doc_list.documents:
        combined_documents.extend(doc_list.documents)

    # 4) Now chunk the combined documents
    chunked_docs = chunk_text_for_list(combined_documents)

    # 5) Embed them
    doc_embeddings = generate_embeddings(chunked_docs)

    # 6) Convert to Pinecone's upsert format
    data_with_meta_data = combine_vector_and_text(chunked_docs, doc_embeddings)

    # 7) Upsert to Pinecone
    upsert_data_to_pinecone(data_with_meta_data)

    return {
        "message": "Documents chunked, embedded, and upserted successfully.",
        "upserted_records_count": len(data_with_meta_data),
        "additional_info": f"Called /getPageList, then /getContent for each pageId. extra_param={extra_param}"
    }

# 3) Search / Query - Updated to match frontend expectations
@app.post(
    "/search",
    response_model=SearchResponse,
    responses={
        200: {"description": "Search completed successfully"},
        400: {"model": ErrorResponse, "description": "Index not initialized or invalid request"}
    },
    tags=["Search"],
    summary="Search for relevant content",
    description="Performs semantic search using OpenAI embeddings and returns AI-powered answers based on retrieved context."
)
def search(search_request: SearchRequest):
    """
    Query the Pinecone index with a query string.
    Retrieve the matching chunks and pass them to the LLM for a summarized answer.
    Returns results in format expected by the frontend.
    """
    global index
    if index is None:
        return {"error": "Index not created or not initialized. Call /create-index first."}

    query = search_request.query

    # Get query embeddings
    q_embeddings = get_query_embeddings(query)

    # Query Pinecone
    answers = query_pinecone_index(q_embeddings, top_k=5)

    # Create results in format expected by frontend
    results = []
    for i, match in enumerate(answers["matches"]):
        result = {
            "id": str(i + 1),
            "title": f"Search Result {i + 1}",
            "description": match["metadata"]["text"][:200] + "..." if len(match["metadata"]["text"]) > 200 else match["metadata"]["text"],
            "url": f"https://example.com/result/{i + 1}",
            "score": match["score"]
        }
        results.append(result)

    # Get LLM response for the first result
    if results:
        text_answer = " ".join([doc["metadata"]["text"] for doc in answers["matches"]])
        prompt = f"{text_answer}\nOnly use the context provided. Do not use any prior information or training data. Now using the provided information only, give me a better and summarized answer to the query: '{query}'"
        
        try:
            final_answer = better_query_response(prompt)
            # Update the first result with the LLM response
            results[0]["description"] = final_answer
            results[0]["title"] = f"AI Answer: {query}"
        except Exception as e:
            print(f"Error getting LLM response: {e}")

    return {
        "query": query,
        "results": results,
        "total": len(results),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    }

# Optionally: Endpoint to delete the index if needed
@app.delete(
    "/delete-index/{index_name}",
    tags=["Index Management"],
    summary="Delete a Pinecone index",
    description="Permanently deletes the specified Pinecone index. This action cannot be undone."
)
def delete_index(index_name: str):
    """
    Delete the Pinecone index if you want to reset everything.
    
    - **index_name**: Name of the index to delete
    
    ⚠️ **Warning**: This action is irreversible!
    """
    pc.delete_index(index_name)
    return {"message": f"Index '{index_name}' deleted successfully."}

@app.post(
    "/load-index/{index_name}",
    response_model=IndexResponse,
    responses={
        200: {"description": "Index loaded successfully"},
        400: {"model": ErrorResponse, "description": "Index does not exist"}
    },
    tags=["Index Management"],
    summary="Load an existing Pinecone index",
    description="Loads an existing Pinecone index into memory for search operations."
)
def load_pinecone_index(index_name: str):
    """
    Load an existing Pinecone index into the global 'index' variable.
    
    - **index_name**: Name of the existing index to load
    
    Returns index statistics if successful.
    """
    global index
    
    # Check if index_name exists in your Pinecone project
    # (Optional) If you want to do a strict check:
    existing_indexes = pc.list_indexes()
    if index_name not in existing_indexes.names():
        return {
            "error": f"Index '{index_name}' does not exist. "
                     f"Available indexes: {existing_indexes}"
        }

    # Index exists, so load it
    index = pc.Index(index_name)
    
    return {
        "message": f"Index '{index_name}' loaded successfully.",
        "index_stats": index.describe_index_stats().to_str()
    }

# Health check endpoint
@app.get(
    "/health",
    tags=["System"],
    summary="Health check",
    description="Check if the API server is running and healthy."
)
def health_check():
    """
    Simple health check endpoint to verify the API is running.
    
    Returns basic status information.
    """
    return {"status": "healthy", "message": "AI Search Engine API is running"}

# Vercel compatibility
@app.get("/")
def root():
    return {"message": "AI Search Engine API", "docs": "/docs"}

# Export the app for Vercel
handler = app
app = app  # Also export as app for Vercel compatibility
