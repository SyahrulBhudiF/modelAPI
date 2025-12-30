# AI Model Serving API

FastAPI-based stateless inference service untuk serving multiple AI models.

## Features

- **Multi-Model Support** - Register multiple models dengan handler berbeda
- **Stateless Design** - Tidak ada database, session, atau server-side state
- **Bearer Token Auth** - Simple static token authentication
- **Video Processing** - Support preprocessing video untuk model berbasis micro-expression

## Quick Start

### 1. Install Dependencies

```bash
# Fedora/RHEL (untuk compile dlib)
sudo dnf install python3-devel cmake gcc-c++

# Ubuntu/Debian
sudo apt install python3-dev cmake g++
```

```bash
uv sync
```

### 2. Setup Environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
AI_API_TOKEN=your_secret_token_here

# Format: model_name:handler_key:model_dir
# Multiple models dipisah dengan koma
MODEL_REGISTRY=tabr:tabr:tabr_experiment/before_flatten

# CORS settings (multiple values dipisah dengan koma)
CORS_ORIGINS=http://localhost:3000,https://example.com
CORS_METHODS=GET,POST
```

### 3. Run Server

```bash
uvicorn app.main:app --reload
```

## API Endpoints

### Health Check

```
GET /api/v1/health
```

### List Models

```
GET /api/v1/models
Authorization: Bearer <token>
```

### Inference

```
POST /api/v1/infer/{model_name}
Authorization: Bearer <token>
Content-Type: multipart/form-data

files: file(s) untuk inference
```

### Model Status

```
GET /api/v1/models/{model_name}/status
Authorization: Bearer <token>
```

## Example Usage

### cURL

```bash
curl -X POST \
  -H "Authorization: Bearer your_token" \
  -F "files=@input.mp4" \
  http://localhost:8000/api/v1/infer/tabr
```

### Python

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/infer/tabr",
    headers={"Authorization": "Bearer your_token"},
    files={"files": open("input.mp4", "rb")}
)
print(response.json())
```

## Adding New Model Handler

1. Buat handler baru di `app/services/handlers/`:

```python
from app.services.base import BaseHandler

class MyHandler(BaseHandler):
    def load(self) -> None:
        # Load model artifacts
        self._is_loaded = True

    def preprocess(self, input_data):
        # Preprocess input
        return preprocessed_data

    def predict(self, preprocessed_data) -> dict:
        # Run inference
        return {"prediction": result}
```

2. Register di `app/services/model_registry.py`:

```python
HANDLER_TYPES = {
    "tabr": TabRHandler,
    "my_model": MyHandler,
}
```

3. Update `MODEL_REGISTRY` di `.env`:

```env
MODEL_REGISTRY=tabr:tabr:path/to/tabr,my_model:my_model:path/to/my_model
```

## Project Structure

```
modelAPI/
├── app/
│   ├── main.py
│   ├── api/routes.py
│   ├── core/
│   │   ├── config.py
│   │   └── security.py
│   ├── models/                    # Model implementations
│   ├── preprocessing/             # Preprocessing modules
│   └── services/
│       ├── base.py                # BaseHandler abstract class
│       ├── model_registry.py      # Handler registry
│       └── handlers/              # Model handlers
├── .env
└── pyproject.toml
```

## Model Registry Format

```
MODEL_REGISTRY=name:handler:path[,name:handler:path,...]
```

- `name` - API-facing identifier (used in `/infer/{name}`)
- `handler` - Handler key registered in `HANDLER_TYPES`
- `path` - Directory containing model artifacts

## TabR Handler

Handler untuk model TabR (micro-expression anxiety detection).

**Pipeline:**
```
Video → Frames → ROI (dlib) → POC-ABS → Feature Vector → TabR → Prediction
```

**Required Artifacts:**
```
{model_dir}/
├── shape_predictor_68_face_landmarks.dat
├── tabr_model.ckpt
├── imputer.joblib
├── scaler.joblib
├── feature_cols.joblib
└── context.joblib
```

**Download dlib predictor:**
```bash
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```

**Response:**
```json
{
  "prediction": 1,
  "prediction_label": "anxiety",
  "confidence": 0.85,
  "avg_probability": 0.72,
  "total_frames": 150,
  "anxiety_frame_ratio": 0.85,
  "non_anxiety_frame_ratio": 0.15,
  "frame_predictions": [1, 1, 0, 1, ...],
  "frame_probabilities": [0.78, 0.82, 0.45, ...]
}
```
