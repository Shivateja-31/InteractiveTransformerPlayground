from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn.functional as F
import json
import os
import re
from model import MiniGPT

app = FastAPI(title="Interactive Transformer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
stoi = {}
itos = {}
vocab_size = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'
block_size = 8  # Matched with train.py

class PredictRequest(BaseModel):
    text: str
    temperature: float = 1.0

def tokenize(text):
    return re.findall(r'\b\w+\b|[^\w\s]', text)

def load_model_and_vocab():
    global model, stoi, itos, vocab_size
    vocab_path = os.path.join(os.path.dirname(__file__), "vocab.json")
    model_path = os.path.join(os.path.dirname(__file__), "model.pth")
    
    if not os.path.exists(vocab_path) or not os.path.exists(model_path):
        import train
        train.train()
        
    with open(vocab_path, "r") as f:
        data = json.load(f)
        stoi = data["stoi"]
        itos = {int(k): v for k, v in data["itos"].items()}
        vocab_size = len(stoi)
        
    model = MiniGPT(
        vocab_size=vocab_size,
        max_length=block_size,
        embed_size=32,
        num_layers=1,
        heads=2,
        forward_expansion=4,
        dropout=0.0
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except RuntimeError:
        # If there's a shape mismatch from the old char-level model, retrain automatically
        print("Model shape mismatch detected. Retraining word-level model...")
        import train
        train.train()
        
        # Reload vocab and model after retraining
        with open(vocab_path, "r") as f:
            data = json.load(f)
            stoi = data["stoi"]
            itos = {int(k): v for k, v in data["itos"].items()}
            vocab_size = len(stoi)
            
        model = MiniGPT(
            vocab_size=vocab_size,
            max_length=block_size,
            embed_size=32,
            num_layers=1,
            heads=2,
            forward_expansion=4,
            dropout=0.0
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()

@app.on_event("startup")
async def startup_event():
    load_model_and_vocab()

@app.post("/predict")
async def predict(req: PredictRequest):
    if not req.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
        
    words = tokenize(req.text)
    
    # Map words to indices, using <UNK> (index 1) for unknown words
    unk_id = stoi.get("<UNK>", 1)
    context = [stoi.get(w, unk_id) for w in words]
    
    if len(context) == 0:
        context = [stoi.get("<PAD>", 0)]
        
    if len(context) > block_size:
        context = context[-block_size:]
        
    x = torch.tensor([context], dtype=torch.long).to(device)
    
    with torch.no_grad():
        logits, states = model(x, return_states=True)
        next_token_logits = logits[0, -1, :]
        
        if req.temperature == 0:
            req.temperature = 0.001
        scaled_logits = next_token_logits / req.temperature
        
        probs = F.softmax(scaled_logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, 5)
        
        top_tokens = []
        for i in range(5):
            idx = top_indices[i].item()
            prob = top_probs[i].item()
            word = itos.get(idx, "<UNK>")
            top_tokens.append({"token": word, "probability": prob})
            
        next_token_idx = top_indices[0].item()
        next_token_word = itos.get(next_token_idx, "<UNK>")
        
        # Output token format updated to use 'text'
        input_tokens = [{"id": t, "text": itos.get(t, "<UNK>")} for t in context]
        att_matrix = states["attention_matrices"][0][0, 0, :, :].cpu().numpy().tolist()
        embeddings = states["combined_embeddings"][0].cpu().numpy().tolist()
        
    return {
        "predicted_token": next_token_word,
        "top_tokens": top_tokens,
        "input_tokens": input_tokens,
        "attention_matrix": att_matrix,
        "embeddings": embeddings
    }

@app.get("/api/health")
async def health():
    return {"status": "ok"}
