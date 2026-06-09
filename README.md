# 🛡 RCSA Intelligence Platform

> **AI-powered Risk & Control Self-Assessment automation for banking — real RAG over real banking control documents, with full source attribution.**

[![Made with Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)
[![GPT-4o](https://img.shields.io/badge/LLM-GPT--4o-412991.svg)](https://platform.openai.com/)
[![ChromaDB](https://img.shields.io/badge/Vector_DB-ChromaDB-FF6B35.svg)](https://www.trychroma.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## The Problem

Banks spend **87 days per RCSA cycle** — risk analysts spend three weeks per business unit drafting questions, scheduling control owner interviews, chasing documents. By the time the Board sees the assessment, the data is 60-90 days old.

In 2024 alone, **$4.3 billion in regulatory penalties** were paid by banks for control failures — TD Bank ($3.09B), Citi ($136M), Barclays (£39.3M). The cost of getting RCSA wrong has never been higher.

## The Solution

A conversational AI agent that:

- **Conducts adaptive risk interviews** with control owners — asks the right follow-up questions based on what it hears
- **Retrieves relevant control documentation** semantically from a vector knowledge base in real time
- **Flags risks with regulatory citations** — GENIUS Act, FCRA, FinCEN CDD, OCC, SR 11-7
- **Generates structured RCSA output** — 5×5 risk matrix, control effectiveness ratings, prioritised action plan, full source attribution

**Cycle time:** 87 days → effectively zero (continuous risk intelligence).

---

## 🎬 Live Demo

🔗 **Try it live:** [your-vercel-url.vercel.app](https://your-vercel-url.vercel.app)

📹 **90-second video walkthrough:** [Loom link]

---

## 🏗 Architecture

```
┌──────────────────────────────────────────────────────────┐
│  FRONTEND (Vanilla JS · Vercel)                          │
│  Real-time chat UI · Live retrieval visualisation         │
└──────────────────────┬───────────────────────────────────┘
                       │ HTTPS · JSON
                       ▼
┌──────────────────────────────────────────────────────────┐
│  BACKEND (FastAPI · Railway)                              │
│  • Document ingestion pipeline                            │
│  • Semantic retrieval orchestration                       │
│  • GPT-4o agent reasoning                                 │
└────────┬──────────────────────────────┬──────────────────┘
         │                              │
         ▼                              ▼
┌────────────────────┐         ┌────────────────────┐
│  ChromaDB          │         │  OpenAI API        │
│  Vector store      │         │  • GPT-4o (chat)   │
│  208 banking docs  │         │  • text-embed-3    │
│  1,000+ chunks     │         │    (embeddings)    │
└────────────────────┘         └────────────────────┘
```

### The RAG Pipeline

1. **User question** arrives at backend
2. **Embedding** generated via OpenAI `text-embedding-3-small`
3. **Vector search** over ChromaDB returns top-4 semantically similar chunks
4. **Context injection** — retrieved chunks added to GPT-4o system message
5. **LLM reasoning** produces response with regulatory citations
6. **Frontend renders** response + retrieved chunks with similarity scores

---

## 🧠 What Makes This Different

Most AI portfolio projects are ChatGPT wrappers. This is a **real RAG system** over realistic enterprise content.

| Capability | This Project | Typical Demo |
|---|---|---|
| Real vector retrieval | ✅ ChromaDB with OpenAI embeddings | ❌ Prompt-engineered context |
| Domain-specific documents | ✅ 208 banking control documents across 4 divisions | ❌ Wikipedia / generic data |
| Source attribution | ✅ Every response cites chunks + similarity scores | ❌ Black-box LLM output |
| Regulatory grounding | ✅ GENIUS Act, FCRA, FinCEN CDD, OCC, SR 11-7 | ❌ Generic AI conversation |
| Production-ready code | ✅ FastAPI, error handling, type hints, async | ❌ Notebook scripts |
| Cloud deployment | ✅ Railway + Vercel free-tier | ❌ Local-only |

---

## 📚 The Knowledge Base

The platform ships with **208 realistic banking control documents** spanning **42 controls across 4 business divisions** — a genuinely enterprise-scale RCSA knowledge base. Each control includes five documents: Control Narrative, Control Specification, Control Owner Interview, Performance Evaluation, and Design Testing (Excel).

### Retail Banking — 12 controls (RB-CTRL-01 to 12)
Exception Reporting, HITL Liveness (GENIUS Act gap), Data Reconciliation, KYC Review (FinCEN CDD breach), Mortgage Affordability, Overdraft Monitoring, Complaint Handling, Card Fraud Detection, Vulnerable Customer ID, Sanctions Screening, Branch Cash Handling, Mortgage Valuation.

### Corporate & Commercial Lending — 10 controls (CL-CTRL-01 to 10)
Large Exposure Monitoring, Loan Covenant Monitoring, Syndicated Loan Admin, Trade Finance Documentary Exam, Credit Risk Grading, Collateral Valuation, Problem Loan/Watchlist, Corporate KYB Onboarding, Concentration Risk, Facility Drawdown/CP.

### Markets & Trading — 10 controls (MT-CTRL-01 to 10)
Market Risk Limits/VaR, Trade Surveillance/Market Abuse, Independent Price Verification, Trade Booking/Confirmation, Counterparty Credit/Margin, Best Execution, Trader Mandate/Position Limits, Collateral/Settlement Risk, Model Risk Management, P&L Reconciliation/Attribution.

### Wealth & Private Banking — 10 controls (WM-CTRL-01 to 10)
Suitability Assessment, Investment Mandate Compliance, Private Client KYC, Conflicts of Interest, CASS Client Asset Protection, Fees/Cost Transparency, AML Transaction Monitoring, Cross-Border Advice, Vulnerable Client (Wealth), Product Governance.

### Deliberate gaps for the AI agent to find
The knowledge base embeds realistic control weaknesses across divisions — stale models, SLA breaches, alert backlogs, covenant tracking gaps, source-of-wealth deficiencies — each with regulatory citations (GENIUS Act, FinCEN CDD, FCA Consumer Duty, MAR, CASS, CRR large exposures, SR 11-7, EMIR, MiFID II). A mix of Strong / Adequate / Weak ratings makes the agent's reasoning realistic.

> All content is entirely synthetic — fictional "Global Retail Bank" / "Global Bank Group", no real customer data, no proprietary IP.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- OpenAI API key (~$10 free credit covers months of demo usage)

### Run the backend locally

```bash
git clone https://github.com/yourusername/rcsa-ai-platform.git
cd rcsa-ai-platform/backend

pip install -r requirements.txt
export OPENAI_API_KEY="sk-your-key-here"

uvicorn main:app --reload
```

Backend runs on `http://localhost:8000`.

### Open the frontend

Just open `frontend/index.html` in your browser. Enter `http://localhost:8000` as the backend URL and click Connect.

### Deploy to the cloud (free tier)

**Backend → Railway** (free $5/month credit):
1. Push this repo to GitHub
2. Create new project on [railway.app](https://railway.app)
3. Connect your repo, select the `/backend` folder
4. Add environment variable `OPENAI_API_KEY`
5. Deploy — Railway gives you a public URL

**Frontend → Vercel** (free forever):
1. Import repo on [vercel.com](https://vercel.com)
2. Set output directory to `/frontend`
3. Deploy

Total cost: **~$10/month** for OpenAI usage at moderate demo volumes.

---

## 🧪 Try These Prompts

Once connected, see the agent retrieve real chunks for these:

- *"What are our biggest operational risks in retail loan origination?"*
- *"Tell me about the GENIUS Act compliance gap in our liveness model"*
- *"What controls relate to KYC and FinCEN CDD compliance?"*
- *"Give me a full risk assessment summary with scores"*

Watch the **Knowledge Base panel** at the top — the relevant document tiles will highlight as they're retrieved.

---

## 🛠 Tech Stack

| Layer | Technology | Why |
|---|---|---|
| LLM | OpenAI GPT-4o | Best-in-class reasoning, 128K context |
| Embeddings | OpenAI text-embedding-3-small | Fast, cheap ($0.02/1M tokens), accurate |
| Vector DB | ChromaDB | Embedded, no external service needed |
| Backend | FastAPI | Async, type-safe, OpenAPI docs built-in |
| Document parsing | python-docx | Handles Word docs natively including tables |
| Frontend | Vanilla JS | Zero dependencies, ships fast |
| Backend hosting | Railway | Free tier, auto-deploys from GitHub |
| Frontend hosting | Vercel | Free tier, edge CDN |

---

## 📈 What's Next

This MVP is the first step in a multi-stage build:

- **Phase 1 (this repo):** Personal MVP — real RAG, public demo ✅
- **Phase 2:** Migration to Azure AI Foundry — enterprise-grade hosting, Teams integration via Copilot Studio
- **Phase 3:** Power Automate workflows — scheduling, reminders, GRC system write-back
- **Phase 4:** Production deployment with MetricStream / ServiceNow GRC connectors

---

## 🤝 Connect

Built by [Your Name] — Senior Consultant, Non-Financial Risk Management

- LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

Open to discussing AI in risk management, banking RCSA modernisation, and consulting opportunities.

---

## 📄 License

MIT — use this code, learn from it, build on it.

> All banking content is synthetic. No real customer, client, or proprietary information is included in this repository.
