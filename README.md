# InvestMatch Pro 🚀
### AI-Powered Startup-Investor Matchmaking Platform

---

## Project Structure

```
investmatch_pro/
├── requirements.txt
├── data/
│   └── dummy_data.py          # All synthetic investor + startup data
├── models/
│   ├── compatibility_model.py # Keras DNN — match probability
│   ├── traction_model.py      # Keras regression — traction score 0–100
│   ├── sector_model.py        # Node2Vec + SVD — sector similarity
│   ├── history_model.py       # Bidirectional LSTM — next-sector prediction
│   └── suggestion_engine.py   # Autoencoder + KMeans — novel suggestions
├── api/
│   └── main.py                # FastAPI backend (all endpoints)
└── frontend/
    └── index.html             # Full dashboard UI
```

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the server (from project root)
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# 3. Open browser
http://localhost:8000
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/investors` | List all investors |
| GET | `/api/startups` | List all startups with traction scores |
| GET | `/api/sectors` | List sectors and stages |
| POST | `/api/predict_compatibility` | Keras DNN match probability |
| POST | `/api/predict_traction` | Keras regression traction score |
| POST | `/api/sector_similarity` | Node2Vec sector similarity |
| POST | `/api/predict_next_sector` | BiLSTM next-sector prediction |
| POST | `/api/suggest` | Autoencoder + KMeans suggestions |
| GET | `/api/dashboard` | Platform-wide stats |

---

## Models

### 1. CompatibilityModel — `Keras DNN`
- **Input**: 19 features (sector match, stage match, check fit, growth, runway, risk, MRR, etc.)
- **Architecture**: Dense(128) → BatchNorm → Dropout → Dense(64) → Dense(32) → Sigmoid
- **Loss**: BinaryCrossentropy | **Metric**: AUC-ROC
- **Output**: match probability 0.0 – 1.0

### 2. TractionModel — `Keras Regression`
- **Input**: 10 engineered features (burn multiple, capital efficiency, growth, retention, etc.)
- **Architecture**: Dense(64) → LeakyReLU → Dense(32) → Dense(16) → Linear
- **Loss**: Huber | **Metric**: MAE
- **Output**: traction score 0 – 100

### 3. SectorCompatibilityModel — `Node2Vec + SVD`
- **Input**: Investor co-preference graph (sectors as nodes, co-investments as edges)
- **Method**: Node2Vec random walks → Word2Vec embeddings (falls back to SVD if node2vec not installed)
- **Output**: cosine similarity between any two sectors

### 4. InvestmentHistoryModel — `Bidirectional LSTM`
- **Input**: padded sequence of past sector investments (length 12)
- **Architecture**: Embedding(32) → BiLSTM(64) → Dropout → Dense(64) → Softmax(10)
- **Loss**: SparseCategoricalCrossentropy
- **Output**: probability distribution over next sectors

### 5. SuggestionEngine — `Autoencoder + KMeans`
- **Input**: TF-IDF text features + one-hot sector/stage + numeric profile
- **Architecture**: Encoder Dense(64→32→16) → Decoder Dense(32→64→input)
- **Clustering**: KMeans(k=6) on latent vectors
- **Output**: ranked startups blending cosine similarity + cross-cluster novelty

---

## Example API calls

```python
import requests

# Compatibility
r = requests.post("http://localhost:8000/api/predict_compatibility",
                  json={"investor_id": "INV_000", "startup_id": "STP_005"})
print(r.json())  # {"compatibility_score": 0.82, "label": "Strong Match", ...}

# Traction
r = requests.post("http://localhost:8000/api/predict_traction",
                  json={"startup_id": "STP_012"})
print(r.json())  # {"traction_score": 73.4, "growth_score": 89.0, ...}

# Sector similarity
r = requests.post("http://localhost:8000/api/sector_similarity",
                  json={"sector_a": "FinTech", "sector_b": "AI/ML"})
print(r.json())  # {"similarity_score": 0.74, ...}

# Next sector (LSTM)
r = requests.post("http://localhost:8000/api/predict_next_sector",
                  json={"history": ["FinTech", "AI/ML", "SaaS"], "top_k": 3})
print(r.json())  # {"predictions": [{"sector": "Cybersecurity", "probability": 0.21}, ...]}

# Suggestions
r = requests.post("http://localhost:8000/api/suggest",
                  json={"investor_id": "INV_003", "top_k": 5, "novelty_weight": 0.3})
print(r.json())  # {"suggestions": [...]}
```
