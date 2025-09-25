# AI Search Engine Backend

A Python FastAPI backend that provides AI-powered search capabilities using Pinecone vector database and OpenAI embeddings.

## Features

- 🔍 **Vector Search** - Semantic search using OpenAI embeddings
- 🧠 **AI-Powered Answers** - GPT-4 powered responses based on retrieved context
- 📊 **Pinecone Integration** - Vector database for efficient similarity search
- 🌐 **RESTful API** - Clean API endpoints for all operations
- 🔄 **Data Ingestion** - Automatic content fetching and processing
- 📝 **Text Chunking** - Intelligent text segmentation for better search

## Setup Instructions

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Environment Configuration

Copy the example environment file and configure your API keys:

```bash
cp env.example .env
```

Edit `.env` and add your configuration:

**For Local Pinecone (Default):**
```env
OPENAI_API_KEY=your_actual_openai_api_key_here
PINECONE_API_KEY=pclocal
PINECONE_HOST=http://localhost:5080
PINECONE_ENVIRONMENT=us-east-1
```

**For Cloud Pinecone:**
```env
OPENAI_API_KEY=your_actual_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_HOST=https://your-index-name-your-project.svc.pinecone.io
PINECONE_ENVIRONMENT=your_pinecone_environment_here
```

### 3. Start Pinecone (Local)

Make sure you have Pinecone running locally on port 5080. If you don't have it set up, you can use the cloud version by updating the configuration in `main.py`.

### 4. Run the API Server

```bash
python main.py
```

The API will be available at `http://localhost:8000`

## API Documentation

The API includes comprehensive Swagger/OpenAPI documentation:

- **Swagger UI**: `http://localhost:8000/docs` - Interactive API documentation
- **ReDoc**: `http://localhost:8000/redoc` - Alternative documentation format
- **OpenAPI Schema**: `http://localhost:8000/openapi.json` - Raw OpenAPI specification

### Swagger Features:
- 📚 **Complete Documentation** - All endpoints with detailed descriptions
- 🧪 **Interactive Testing** - Try API calls directly from the browser
- 📋 **Request/Response Examples** - See expected data formats
- 🏷️ **Organized by Tags** - Endpoints grouped by functionality:
  - **Index Management** - Create, load, delete indexes
  - **Data Operations** - Ingest and process documents
  - **Search** - Semantic search functionality
  - **System** - Health checks and system status

## API Endpoints

### Health Check
- **GET** `/health` - Check if the API is running

### Index Management
- **POST** `/create-index/{index_name}` - Create a new Pinecone index
- **POST** `/load-index/{index_name}` - Load an existing index
- **DELETE** `/delete-index/{index_name}` - Delete an index

### Data Operations
- **POST** `/upsert-data` - Ingest and index documents
- **POST** `/search` - Search for relevant content

### Search Endpoint

**Request:**
```json
POST /search
{
  "query": "your search query"
}
```

**Response:**
```json
{
  "query": "your search query",
  "results": [
    {
      "id": "1",
      "title": "AI Answer: your search query",
      "description": "AI-generated answer based on retrieved context...",
      "url": "https://example.com/result/1",
      "score": 0.95
    }
  ],
  "total": 1,
  "timestamp": "2024-01-01T00:00:00.000000Z"
}
```

## Workflow

1. **Create Index**: First, create a Pinecone index to store your vectors
2. **Ingest Data**: Use `/upsert-data` to fetch content from external APIs and index it
3. **Search**: Use `/search` to query your indexed content and get AI-powered answers

## External API Integration

The backend automatically fetches content from:
- `https://fd10-2405-201-4018-1852-45e9-d3b8-df3b-f8ed.ngrok-free.app/getPageList`
- `https://fd10-2405-201-4018-1852-45e9-d3b8-df3b-f8ed.ngrok-free.app/getContent?pageId={pageId}`

You can modify these URLs in `main.py` to point to your own content sources.

## Configuration

### Pinecone Setup
- **Local**: Uses `http://localhost:5080` (modify in `main.py`)
- **Cloud**: Update the PineconeGRPC configuration to use your cloud instance

### OpenAI Models
- **Embeddings**: Uses `text-embedding-ada-002` (via LangChain)
- **LLM**: Uses `gpt-4o` for generating answers

### Text Processing
- **Chunk Size**: 1000 characters (configurable in `chunk_text_for_list`)
- **Top K Results**: 5 results returned (configurable in `query_pinecone_index`)

## Development

### Running in Development Mode

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### API Documentation

Once running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Integration with Frontend

The Next.js frontend is configured to connect to this API at `http://localhost:8000`. Make sure both servers are running:

1. **Backend**: `python main.py` (port 8000)
2. **Frontend**: `npm run dev` (port 3000)

## Troubleshooting

### Common Issues

1. **OpenAI API Key**: Make sure your API key is set in the `.env` file
2. **Pinecone Connection**: Ensure Pinecone is running locally or update to cloud configuration
3. **CORS Issues**: The API includes CORS middleware for `http://localhost:3000`
4. **External API**: Check if the external content APIs are accessible

### Logs

The API logs important events including:
- Index creation/loading status
- Document processing progress
- Error messages for failed operations

## License

MIT License
