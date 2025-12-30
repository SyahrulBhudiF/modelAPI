# AGENT.md
# AI Model Serving Agent (FastAPI + Astral UV)

## Purpose
This application serves one or more AI models using FastAPI in a fully stateless manner.
The system is designed from the beginning to be secure, fast, and scalable, without
requiring architectural refactors as load increases.

Each model owns its own loading and preprocessing logic (model-specific).
The service performs inference only and does not depend on databases, sessions,
or server-side state.

Primary use case:
- 1 question = 1 video (≤ ~10 seconds)
- Full-FPS video processing (micro-expression sensitive)
- Preprocessing: ROI → Optical Flow → POC-ABS
- Output: single feature vector → tabular model (TabR)

---

## High-Level Architecture
- Framework: FastAPI
- Execution model: Stateless
- Authentication: Static Bearer Token
- Configuration: Environment variables (.env)
- Model management: Multi-model, handler-based
- Scope: Inference only
- Processing profile:
  - CPU-bound preprocessing
  - Optional GPU inference
- Scalability: Worker-based with queue-ready design

---

## Project Structure
```
.
├── app
│   ├── main.py
│   ├── core
│   │   ├── config.py
│   │   └── security.py
│   ├── api
│   │   └── routes.py
│   └── services
│       ├── base.py
│       ├── model_registry.py
│       └── handlers
│           └── tabr_handler.py
├── .env
├── pyproject.toml
└── agent.md
```
---

## Dependency Management (Astral UV)

This project uses **Astral UV** for dependency resolution and environment management.

### Core Dependencies
Declared in `pyproject.toml` and installed via `uv`:

- fastapi
- uvicorn[standard]
- pydantic-settings
- python-dotenv
- numpy
- pandas
- opencv-python
- dlib
- torch
- pytorch-lightning
- joblib

Optional (for scaling later):
- redis
- celery or rq

No runtime dependency relies on pip-specific behavior.

---

## Environment Configuration (.env)

AI_API_TOKEN=your_super_secret_token

# Format:
# MODEL_REGISTRY=model_name:handler_key:model_dir
MODEL_REGISTRY=tabr:tabr:tabr_experiment/before_flatten

Definitions:
- model_name: API-facing identifier
- handler_key: handler implementation selector
- model_dir: directory containing model artifacts

---

## Authentication Model
- All inference endpoints require a Bearer token
- Token is static and shared between frontend and backend
- Token is loaded from environment variables
- Validation occurs per request
- Invalid token → HTTP 401
- No sessions, cookies, or persistent auth state

---

## Model Handler Contract
Each model must implement its own handler.

A handler is responsible for:
- Loading model checkpoints
- Loading preprocessing artifacts (imputer, scaler, context, etc.)
- Performing input preprocessing
- Running inference

There is no global preprocessing logic.
Handlers are isolated, self-contained, and portable.

---

## Model Lifecycle
1. Application startup
2. MODEL_REGISTRY is parsed
3. Handlers are instantiated
4. Models are loaded once (startup or lazy)
5. Models remain resident in memory for inference

---

## Input & Preprocessing Policy
- One request corresponds to one inference sample
- For the TabR use case:
  - Input: one video per request
  - Maximum duration: ~10 seconds
  - FPS: full (no downsampling)
  - Preprocessing performed entirely inside the handler:
    - ROI extraction
    - Optical flow computation
    - POC-ABS feature extraction
    - Feature vector generation
- API routes do not transform or validate model-specific payloads

---

## Request Flow (Default Execution)
1. Client sends POST /infer/{model_name}
2. Bearer token is validated
3. Model handler is resolved from registry
4. Handler.preprocess(input) is executed
5. Handler.predict(feature_vector) is executed
6. JSON response is returned

This flow is synchronous and blocking by design.

---

## Scalability Strategy (Built-In, Optional)
The system is designed to scale without API or handler refactors.

### Default Mode
- Synchronous execution
- Multi-worker deployment (e.g., Gunicorn)
- One request occupies one worker
- Suitable for:
  - Research environments
  - Quiz-based systems
  - Moderate concurrent load

### Known Constraints
- Preprocessing is CPU-bound
- Async endpoints do not reduce computation time
- Throughput is bounded by worker count

---

## Queue-Based Execution (Optional Upgrade)
When load increases, execution can transition to a queue-based model
without changing API contracts or handler implementations.

- FastAPI handles request intake and authentication
- Requests are enqueued as jobs
- Dedicated workers execute:
  - preprocessing
  - inference
- Clients poll or receive callbacks for results

Example stack:
- Queue: Celery or RQ
- Broker: Redis
- Workers: CPU-bound preprocessing workers

Handlers remain unchanged.

---

## Model Registry Rules
- Registry maps model_name to handler instances
- Registry is unaware of model internals
- Registry logic is identical in sync and queue-based modes

---

## Best Practices
- Dependency injection via FastAPI
- Single-load model initialization
- Strict handler isolation
- Minimal error surface
- No exposure of filesystem paths or internal details

---

## Explicit Non-Goals
- No model training
- No database
- No user management
- No dynamic model registration
- No role-based authorization

---

## Security Notes
- Do not commit .env files
- Rotate tokens manually if compromised

---

## Status
Final design.
Production-safe, scalable by construction, and aligned with research-grade workloads.
