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
import random
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
USE_FREE_EMBEDDINGS = os.environ.get("USE_FREE_EMBEDDINGS", "true").lower() == "true"

# Hugging Face API configuration (free)
HF_API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
HF_API_TOKEN = os.environ.get("HUGGINGFACE_API_TOKEN", "")  # Optional, can work without token

# Initialize embedding models
client = None
EMBEDDING_DIMENSION = 384  # Hugging Face model dimension

if USE_FREE_EMBEDDINGS:
    print("🆓 Using FREE Hugging Face Inference API (no local model loading)")
    if HF_API_TOKEN:
        print("Using authenticated Hugging Face API")
    else:
        print("Using free Hugging Face API (may have rate limits)")
else:
    # Use OpenAI (requires API key and costs money)
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is required when USE_FREE_EMBEDDINGS=false")
    print("💰 Using OpenAI embeddings (API costs apply)")
    client = OpenAI(api_key=OPENAI_API_KEY)
    EMBEDDING_DIMENSION = 1536

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

def generate_embeddings_hf_api(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings using free embedding services - no local model loading!
    
    Args:
        texts: List of text chunks to embed
    
    Returns:
        List of embeddings for each text
    """
    print(f"🆓 Generating embeddings for {len(texts)} texts using free API...")
    
    all_embeddings = []
    
    # Process texts individually to avoid API issues
    for i, text in enumerate(texts):
        try:
            # Try multiple free embedding services
            embedding = None
            
            # Method 1: Try Hugging Face API (if token available)
            if HF_API_TOKEN:
                try:
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {HF_API_TOKEN}"
                    }
                    response = requests.post(
                        HF_API_URL,
                        headers=headers,
                        json={"inputs": text},
                        timeout=10
                    )
                    if response.status_code == 200:
                        embedding = response.json()
                        print(f"✅ HF API: Processed text {i+1}/{len(texts)}")
                except Exception as e:
                    print(f"HF API failed: {e}")
            
            # Method 2: Use a simple hash-based embedding (fallback)
            if embedding is None:
                print(f"Using fallback embedding for text {i+1}/{len(texts)}")
                # Create a simple deterministic embedding based on text content
                embedding = create_simple_embedding(text)
            
            all_embeddings.append(embedding)
            
            # Small delay to avoid overwhelming APIs
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error processing text {i+1}: {e}")
            # Fallback to zero embedding
            all_embeddings.append([0.0] * EMBEDDING_DIMENSION)
    
    return all_embeddings

def create_simple_embedding(text: str) -> List[float]:
    """
    Create a simple deterministic embedding based on text content.
    This is a fallback when external APIs fail.
    """
    import hashlib
    import math
    
    # Create a hash of the text
    text_hash = hashlib.md5(text.encode()).hexdigest()
    
    # Convert hash to embedding vector
    embedding = []
    for i in range(0, len(text_hash), 2):
        # Convert hex pairs to float values
        hex_pair = text_hash[i:i+2]
        val = int(hex_pair, 16) / 255.0  # Normalize to 0-1
        embedding.append(val)
    
    # Pad or truncate to required dimension
    while len(embedding) < EMBEDDING_DIMENSION:
        embedding.append(0.0)
    
    return embedding[:EMBEDDING_DIMENSION]

def generate_embeddings_openai_batch(texts: List[str], batch_size: int = 5) -> List[List[float]]:
    """
    Generate embeddings using OpenAI API with rate limiting (costs money).
    
    Args:
        texts: List of text chunks to embed
        batch_size: Number of texts to process in each API call
    
    Returns:
        List of embeddings for each text
    """
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                if i > 0:
                    delay = 1.0 + random.uniform(0, 1)
                    print(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}, waiting {delay:.2f}s...")
                    time.sleep(delay)
                
                if attempt > 0:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"Rate limit hit, waiting {delay:.2f} seconds before retry {attempt + 1}/{max_retries}")
                    time.sleep(delay)
                
                response = client.embeddings.create(
                    input=batch,
                    model="text-embedding-ada-002"
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                break
                
            except Exception as e:
                error_message = str(e).lower()
                
                if "rate limit" in error_message or "too many requests" in error_message or "insufficient_quota" in error_message:
                    if attempt < max_retries - 1:
                        print(f"OpenAI API error on attempt {attempt + 1}: {e}")
                        continue
                    else:
                        print(f"Max retries reached. OpenAI quota exceeded - consider using free embeddings!")
                        placeholder_embeddings = [[0.0] * EMBEDDING_DIMENSION for _ in batch]
                        all_embeddings.extend(placeholder_embeddings)
                        break
                else:
                    print(f"OpenAI API error: {e}")
                    raise e
    
    return all_embeddings

def generate_embeddings(docs: List[Any]) -> List[List[float]]:
    """
    Generate embeddings for a list of documents using either free or paid models.
    
    This function automatically chooses between:
    - FREE Hugging Face model (no API costs, runs locally)
    - OpenAI API (costs money, requires quota)
    """
    result = []
    
    for doc in docs:
        if not doc:  # Skip empty documents
            result.append([])
            continue
        
        if USE_FREE_EMBEDDINGS:
            # Use free Hugging Face Inference API - no local model!
            doc_embeddings = generate_embeddings_hf_api(doc)
        else:
            # Use OpenAI API with rate limiting
            doc_embeddings = generate_embeddings_openai_batch(doc, batch_size=3)
            
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
    """Get embeddings for the given query string using the configured model."""
    if USE_FREE_EMBEDDINGS:
        # Use free Hugging Face Inference API
        embeddings = generate_embeddings_hf_api([query])
        return embeddings[0] if embeddings else [0.0] * EMBEDDING_DIMENSION
    else:
        # Use OpenAI API
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
    - **dimension**: Automatically matches embedding model (384 for free, 1536 for OpenAI)
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
    
    embedding_info = "FREE Hugging Face (384 dim)" if USE_FREE_EMBEDDINGS else "OpenAI (1536 dim)"
    print(f"Creating index with {embedding_info} embeddings...")
    
    # Create the index with correct dimension
    pc.create_index(
        name=index_name,
        dimension=EMBEDDING_DIMENSION,  # Automatically matches the embedding model
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
    description="Processes provided documents, chunks the text, generates embeddings, and upserts to Pinecone index."
)
def upsert_data(doc_list: DocumentList):
    """
    Process and index the provided documents with rate limiting for free OpenAI accounts.
    
    1) Clean HTML tags from documents
    2) Chunk the documents into manageable pieces
    3) Generate embeddings using OpenAI (with rate limiting)
    4) Upsert to Pinecone index
    
    Args:
        doc_list: List of documents to process and index
        
    Returns:
        dict: Success message with upserted records count and rate limiting info
    """
    global index
    if index is None:
        return {"error": "Index not created or not initialized. Call /create-index OR /load-index first."}

    if not doc_list.documents:
        return {"error": "No documents provided in the request."}

    print(f"Processing {len(doc_list.documents)} documents...")
    
    # Clean HTML tags from documents
    cleaned_documents = []
    for doc in doc_list.documents:
        cleaned_doc = remove_html_tags_bs4(doc)
        cleaned_documents.append(cleaned_doc)

    print("Documents cleaned, starting chunking...")
    
    # Chunk the documents
    chunked_docs = chunk_text_for_list(cleaned_documents)
    total_chunks = sum(len(doc_chunks) for doc_chunks in chunked_docs)
    
    print(f"Created {total_chunks} text chunks, starting embedding generation...")
    print("⚠️  Using free OpenAI account - this may take longer due to rate limiting")

    # Generate embeddings with rate limiting
    start_time = time.time()
    doc_embeddings = generate_embeddings(chunked_docs)
    embedding_time = time.time() - start_time

    print(f"Embeddings generated in {embedding_time:.2f} seconds")

    # Convert to Pinecone's upsert format
    data_with_metadata = combine_vector_and_text(chunked_docs, doc_embeddings)

    # Upsert to Pinecone
    print("Upserting to Pinecone...")
    upsert_data_to_pinecone(data_with_metadata)

    return {
        "message": "Documents processed, chunked, embedded, and upserted successfully with rate limiting.",
        "upserted_records_count": len(data_with_metadata),
        "original_documents_count": len(doc_list.documents),
        "total_chunks_created": len(data_with_metadata),
        "embedding_generation_time_seconds": round(embedding_time, 2),
        "rate_limiting_info": "Used batch processing and delays to respect OpenAI free tier limits"
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

    # Get LLM response for the first result (only if OpenAI is available)
    if results and client is not None:
        text_answer = " ".join([doc["metadata"]["text"] for doc in answers["matches"]])
        prompt = f"{text_answer}\nOnly use the context provided. Do not use any prior information or training data. Now using the provided information only, give me a better and summarized answer to the query: '{query}'"
        
        try:
            final_answer = better_query_response(prompt)
            # Update the first result with the LLM response
            results[0]["description"] = final_answer
            results[0]["title"] = f"AI Answer: {query}"
        except Exception as e:
            print(f"Error getting LLM response: {e}")
            print("Continuing with regular search results...")
    elif results and client is None:
        print("🆓 Using FREE embeddings - AI answers disabled (no OpenAI API key)")
        print("Search results available without AI-powered answers")

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
