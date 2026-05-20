# Interactive Transformer Playground

A complete full-stack AI application that visualizes and demonstrates how transformers and LLMs work internally using a custom mini GPT-style model built in PyTorch.

## Features

- **Custom PyTorch Mini-GPT**: Small-scale character-level transformer trained to predict the next token.
- **FastAPI Backend**: Serves the model, tokenizes input, and returns next-token probabilities along with internal representations (embeddings, attention matrices).
- **React + Tailwind Frontend**: A futuristic AI engineering dashboard built with Vite, Framer Motion, and Recharts.
- **Visualizations**:
  - Interactive Transformer Pipeline diagram
  - Next-Token Probabilities bar chart
  - Self-Attention Heatmap (visualizes the $Q \times K^T$ attention weights)
  - Token Embeddings 2D Scatter plot (visualizes the combined word and positional embeddings)

## Getting Started

The easiest way to run the application is using Docker.

### Prerequisites

- Docker & Docker Compose

### Quick Start

1. Clone or navigate to this directory.
2. Run the following command:
   ```bash
   docker-compose up --build
   ```
3. The first time the backend starts, it will automatically train a tiny transformer model on a small dataset (Shakespeare text) and save `model.pth` and `vocab.json`.
4. Once both containers are running, open your browser and navigate to:
   - **Frontend**: http://localhost:3000
   - **Backend API**: http://localhost:8000/docs (for FastAPI Swagger UI)

## Architecture

- `backend/`: Contains the PyTorch `model.py`, the `train.py` script, and the FastAPI `main.py` application.
- `frontend/`: Contains the Vite + React application with all the UI components in `src/components/`.
