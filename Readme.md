# AI-Powered Video UI Retrieval Engine

An AI-powered semantic video search platform that extracts frames from videos, generates multimodal embeddings using SigLIP-2, and enables fast similarity search through PostgreSQL pgvector.

## Features

- Semantic video search using SigLIP-2 embeddings
- Automatic frame extraction with OpenCV
- Fast vector similarity search using PostgreSQL pgvector
- HNSW indexing for optimized retrieval
- REST APIs built with FastAPI
- Modern React.js user interface
- Docker-based deployment

## Tech Stack

### Backend
- FastAPI
- Python
- PostgreSQL
- pgvector

### AI & Computer Vision
- SigLIP-2
- OpenCV
- Transformers
- PyTorch

### Frontend
- React.js

### Deployment
- Docker
- Docker Compose

## Project Architecture

```
Video Upload
      │
      ▼
Frame Extraction (OpenCV)
      │
      ▼
SigLIP-2 Embedding Generation
      │
      ▼
Store Embeddings (PostgreSQL + pgvector)
      │
      ▼
Semantic Search API (FastAPI)
      │
      ▼
React Frontend
```

## Current Development Roadmap

### Phase 1
- FastAPI backend setup
- React frontend
- OpenCV frame extraction
- SigLIP-2 integration

### Phase 2
- PostgreSQL pgvector setup
- Embedding storage
- Vector search engine

### Phase 3
- HNSW indexing
- Search optimization
- Performance tuning

### Phase 4
- UI improvements
- Backend security
- Mobile responsiveness

### Phase 5
- Benchmarking
- Docker deployment
- Final demo

## Future Enhancements

- OCR-based text search
- Audio transcription search
- Multi-video collections
- Real-time indexing
- Cloud deployment
- User authentication

## Status

🚧 Currently under active development.
