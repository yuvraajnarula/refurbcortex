# RefurbCortex
## High level System Architecture
```mermaid
graph TD
    subgraph "Client Layer"
        A[Browser / Mobile] --> B[Nginx Reverse Proxy]
        C[API Clients / Dealers] --> B
    end

    subgraph "Application Layer"
        B --> D[FastAPI Backend:8000]
        B --> E[Streamlit Analytics:8501]
        B --> F[Streamlit EV Dashboard:8502]
        B --> G[Streamlit Voice:8503]
        B --> H[Gradio Vision Demo:7860]
    end

    subgraph "Core Services"
        D --> I[Qdrant Vector DB:6333]
        D --> J[Redis Cache:6379]
        D --> K[Prometheus:9090]
        K --> L[Grafana:3000]
    end

    subgraph "Storage Layer"
        I --> M[(qdrant-data)]
        J --> N[(redis-data)]
        K --> O[(prom-data)]
        L --> P[(grafana-data)]
        D --> Q[(./data feedback.db)]
        D --> R[(./models yolov8s.onnx)]
    end

    style A fill:#e1f5fe
    style D fill:#e8f5e9,stroke:#2e7d32
    style I fill:#fff3e0,stroke:#ef6c00
    style M fill:#f5f5f5,stroke:#9e9e9e,stroke-dasharray:5
```
## Dataflow pipeline
```mermaid
flowchart LR
    A[Image Upload] --> B[Auth + Idempotency Check]
    B --> C{Cache Hit?}
    C -->|Yes| D[Return Cached Response]
    C -->|No| E[System 1: Quantized ONNX Vision]
    
    E --> F[Metacognitive Router]
    F --> G{Confidence >= 75%?}
    
    G -->|Yes| H[System 2: Agentic Trade-Off]
    G -->|No| I[Route to Human + Safety Margin]
    
    H --> J[SHAP Attribution + Cost Breakdown]
    I --> K[Log to Vector Memory]
    J --> K
    
    K --> L[Response + Heatmap + JSON]
    L --> M[Async: Feedback Loop + Drift Detection]
    
    M --> N[Prometheus Metrics]
    M --> O[ MLflow Model Registry]
    
    style A fill:#e1f5fe
    style E fill:#e8f5e9
    style F fill:#fff3e0
    style H fill:#e3f2fd
    style K fill:#f3e5f5
    style N fill:#ffebee
```
## Deterministic Principles layer
```mermaid
graph LR
    subgraph "6 Deterministic Principles"
        P1[Bounded Probabilism] --> V1[Pydantic Validation + Rule Fallbacks]
        P2[Traceability] --> V2[model_sha + prompt_hash + cost_table_version]
        P3[Idempotency] --> V3[X-Idempotency-Key + Redis Cache]
        P4[Graceful Degradation] --> V4[Circuit Breakers + Local Fallbacks]
        P5[Human Sovereignty] --> V5[HITL Override + High-Weight Feedback]
        P6[Data Minimization] --> V6[72h TTL + PII Hashing + Metadata Retention]
    end

    subgraph "Enforcement Points"
        V1 --> E1[API Schema Validation]
        V2 --> E2[Response Metadata + MLflow Tags]
        V3 --> E3[Idempotency Middleware]
        V4 --> E4[Tenacity Retries + Fallback Logic]
        V5 --> E5[/override Endpoint + Feedback Loop]
        V6 --> E6[Privacy Cleanup Cron + Hashing Utils]
    end

    style P1 fill:#e8f5e9,stroke:#2e7d32
    style P2 fill:#e3f2fd,stroke:#1565c0
    style P3 fill:#fff3e0,stroke:#ef6c00
    style P4 fill:#f3e5f5,stroke:#7b1fa2
    style P5 fill:#ffebee,stroke:#c62828
    style P6 fill:#e0f2f1,stroke:#00695c
```

## Deployment Topology

```mermaid
graph TB
    subgraph "Host Machine"
        subgraph "Docker Network: refurb-net (172.28.0.0/16)"
            subgraph "Compute Services"
                API[api:8000<br/>4 CPU / 6GB RAM]
                GRADIO[gradio-demo:7860<br/>2 CPU / 4GB RAM]
            end

            subgraph "UI Services"
                ANALYTICS[streamlit-analytics:8501<br/>2 CPU / 2GB RAM]
                EV[streamlit-ev:8502<br/>1.5 CPU / 1.5GB RAM]
                VOICE[streamlit-voice:8503<br/>1 CPU / 1GB RAM]
            end

            subgraph "Infrastructure"
                QDRANT[qdrant:6333<br/>1 CPU / 512MB RAM]
                REDIS[redis:6379<br/>0.5 CPU / 256MB RAM]
                PROM[prometheus:9090<br/>1 CPU / 512MB RAM]
                GRAF[grafana:3000<br/>0.5 CPU / 256MB RAM]
                NGINX[nginx:80<br/>0.5 CPU / 256MB RAM]
            end
        end

        subgraph "Persistent Volumes"
            QDATA[(qdrant-data)]
            RDATA[(redis-data)]
            PDATA[(prom-data)]
            GDATA[(grafana-data)]
            APPDATA[(./data ./models)]
        end
    end

    NGINX --> API
    NGINX --> ANALYTICS
    NGINX --> EV
    NGINX --> VOICE
    NGINX --> GRADIO

    API --> QDRANT
    API --> REDIS
    API --> PROM
    API --> APPDATA

    style API fill:#e8f5e9
    style GRADIO fill:#e3f2fd
    style ANALYTICS fill:#fff3e0
    style QDRANT fill:#f5f5f5,stroke:#616161
    style NGINX fill:#e0f2f1,stroke:#00695c
```
