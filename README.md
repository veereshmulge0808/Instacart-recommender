# 🛒 CartPulse: Real-Time Grocery Basket Recommender Microservice

[![CI Pipeline](https://github.com/veereshmulge0808/Instacart-recommender/actions/workflows/ci.yml/badge.svg)](https://github.com/veereshmulge0808/Instacart-recommender/actions)
[![Live Demo](https://img.shields.io/badge/Demo-Live%20on%20Render-brightgreen?style=flat-square&logo=render)](https://Instacart-recommender.onrender.com)
[![Swagger Docs](https://img.shields.io/badge/API-Swagger%20UI-009688?style=flat-square&logo=fastapi)](https://Instacart-recommender.onrender.com/docs)
[![Docker](https://img.shields.io/badge/Container-Docker-2496ED?style=flat-square&logo=docker)](https://hub.docker.com)

An end-to-end machine learning system trained on 32+ million grocery transactions from the Instacart dataset. Framed as a supervised classification and candidate ranking problem, CartPulse predicts and ranks catalog items that a customer is most likely to repurchase in their next order.

---

## System Architecture

```mermaid
flowchart LR
    A[Web UI / API Client] -->|user_id, context| B[FastAPI Service]
    B --> C[(In-Memory Parquet Store)]
    C -->|Candidate Vectors| D[LightGBM Booster]
    D -->|Probabilities| E[Threshold Filter t=0.22]
    E -->|Ranked Top-N JSON| B
    B --> A
```

---

## Key Engineering Highlights

* **Data Engineering at Scale:** Merged 6 relational tables across 32M+ interaction logs using custom numeric downcasting to cut memory overhead by **~70%**.
* **Leak-Free Validation:** Partitioned 131,000+ users via an 80/20 customer-level split rather than random row shuffling to prevent cross-order data leakage.
* **Metric Optimization:** Tuned classification decision thresholds ($0.12 \le t \le 0.36$) under extreme class imbalance (~9.8% positive rate), achieving a peak **validation $F_1$-score of 0.4425**.
* **Low-Latency Serving:** Decoupled offline feature engineering from online inference using Snappy-compressed columnar Parquet stores and an asynchronous FastAPI backend (**<25ms inference latency**).
* **Production CI/CD & Containerization:** Packaged in a slim Debian Docker container with OpenMP multi-threading, automated `pytest` verification via GitHub Actions, and live hosting on Render.

---

## Quickstart

### Run with Docker
```bash
docker build -t Instacart-recommender:v1 .
docker run -p 8000:8000 Instacart-recommender:v1
```
Open `http://localhost:8000` for the interactive dashboard or `http://localhost:8000/docs` for OpenAPI Swagger documentation.
