# PID V2 - PID Intelligent Tuning System

## Architecture

```
                    ┌──────────────────────────────┐
                    │      React + Ant Design Pro   │
                    │         Frontend (SPA)        │
                    └──────────────┬───────────────┘
                                   │ REST + SSE
                    ┌──────────────▼───────────────┐
                    │      FastAPI Backend          │
                    │                               │
                    │  ┌─────────────────────────┐  │
                    │  │   Deterministic Pipeline │  │
                    │  │                         │  │
                    │  │  1. load_data()         │  │
                    │  │  2. fit_model()         │  │
                    │  │  3. tune_pid()          │  │
                    │  │  4. evaluate_pid()      │  │
                    │  │                         │  │
                    │  │  (pure Python, fast,    │  │
                    │  │   deterministic)        │  │
                    │  └────────────┬────────────┘  │
                    │               │ results       │
                    │  ┌────────────▼────────────┐  │
                    │  │   LLM Consultant Agent  │  │
                    │  │                         │  │
                    │  │  - Review results       │  │
                    │  │  - Explain decisions    │  │
                    │  │  - Iterative tuning     │  │
                    │  │  - Experience retrieval  │  │
                    │  │                         │  │
                    │  │  Tools:                 │  │
                    │  │  - run_identification() │  │
                    │  │  - run_tuning()         │  │
                    │  │  - run_evaluation()     │  │
                    │  │  - search_experience()  │  │
                    │  │  - get_data_overview()  │  │
                    │  └─────────────────────────┘  │
                    │                               │
                    │  ┌─────────────────────────┐  │
                    │  │   External Integrations  │  │
                    │  │  - History Data API     │  │
                    │  │  - Knowledge Graph API  │  │
                    │  │  - Experience Store     │  │
                    │  └─────────────────────────┘  │
                    └───────────────────────────────┘
```

## Directory Structure

```
pid_v2/
├── backend/
│   ├── api/                    # FastAPI routes
│   │   ├── __init__.py
│   │   ├── app.py              # FastAPI app entry
│   │   ├── tuning_routes.py    # Tuning workflow endpoints
│   │   ├── data_routes.py      # Data inspection endpoints
│   │   ├── experience_routes.py# Experience management
│   │   └── config_routes.py    # System configuration
│   │
│   ├── core/                   # Core business logic
│   │   ├── algorithms/         # Pure computation (no LLM)
│   │   │   ├── __init__.py
│   │   │   ├── data_analysis.py    # CSV loading, cleaning, window detection
│   │   │   ├── system_id.py        # Model fitting (FO/FOPDT/SOPDT/IPDT)
│   │   │   ├── pid_tuning.py       # Tuning rules (IMC/Lambda/ZN/CHR)
│   │   │   ├── pid_evaluation.py   # Simulation and quality assessment
│   │   │   └── signal_processing.py# Denoise, detrend, alignment
│   │   │
│   │   ├── pipeline/           # Deterministic workflow
│   │   │   ├── __init__.py
│   │   │   ├── runner.py           # Sequential pipeline orchestration
│   │   │   └── events.py          # SSE event definitions
│   │   │
│   │   └── agent/              # LLM consultant (no framework)
│   │       ├── __init__.py
│   │       ├── consultant.py       # Tool-calling loop (~60 lines)
│   │       ├── tools.py            # Tool definitions for LLM
│   │       ├── prompts.py          # System prompt for PID consultant
│   │       └── stream.py          # SSE streaming helpers
│   │
│   ├── services/               # External integrations
│   │   ├── __init__.py
│   │   ├── history_data.py     # History data API client
│   │   ├── knowledge_graph.py  # Knowledge graph API client
│   │   └── experience_store.py # Experience persistence
│   │
│   ├── models/                 # Pydantic data models
│   │   ├── __init__.py
│   │   ├── tuning.py           # Request/response models
│   │   ├── process_model.py    # K/T/L model definitions
│   │   └── evaluation.py       # Evaluation result models
│   │
│   ├── config/                 # Configuration
│   │   ├── __init__.py
│   │   └── settings.py         # Pydantic Settings (env-based)
│   │
│   ├── tests/                  # Tests
│   │   ├── __init__.py
│   │   ├── test_system_id.py
│   │   ├── test_pid_tuning.py
│   │   └── test_pipeline.py
│   │
│   ├── main.py                 # Entry point
│   ├── requirements.txt
│   └── .env.example
│
├── frontend/                   # React + Ant Design Pro
│   ├── package.json
│   ├── src/
│   │   ├── pages/
│   │   ├── components/
│   │   ├── services/
│   │   └── ...
│   └── ...
│
├── CLAUDE.md
└── README.md
```

## Key Design Decisions

1. **Deterministic pipeline first** - All computation (data analysis, model fitting,
   PID tuning, evaluation) runs as pure Python without LLM involvement.

2. **LLM consultant is optional** - The system produces correct results without LLM.
   The consultant adds: result interpretation, iterative refinement via natural
   language, and experience-based recommendations.

3. **No agent framework** - The consultant is a simple tool-calling loop (~60 lines)
   using the OpenAI-compatible API (Qwen via DashScope). ReAct pattern is native
   to the function-calling protocol.

4. **Clean separation** - `core/algorithms/` has zero dependencies on LLM, API, or
   framework code. It can be tested and used independently.
