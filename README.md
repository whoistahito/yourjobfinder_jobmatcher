# Novel Job Matching System (Dockerized)

This project builds upon the Bachelor thesis of **Taha Amirhosseini**: [Novel-Job-Matching](https://github.com/whoistahito/Novel-Job-Matching).
It provides a production-ready, dockerized version of the original job matching system.

## About

This is a job matching system that uses **LLMs (Large Language Models)** to extract structured requirements from job descriptions and **Embeddings** to match them against user profiles efficiently.

## Deployment

The project is packaged using `uv` and includes a secure, optimized Docker setup ready for deployment on platforms like Coolify.

### Run with Docker Compose

```bash
docker-compose up --build
```

### Environment Variables

Ensure you set the following environment variables:
- `API_ACCESS_TOKEN`: Token to secure your API.
- `EXTERNAL_LLM_API_KEY`: API Key for the external LLM provider.
