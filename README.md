# PromptKitPro

**PromptKitPro** is a production-ready digital product for delivering a professionally engineered prompt library for large language models (LLMs).

It is intentionally **not** a SaaS, dashboard, or subscription platform.

Instead, it is designed as a **deterministic, auditable delivery pipeline** that transforms a version-controlled prompt corpus into customer-ready artefacts following verified payment.

The system prioritises clarity, reproducibility, and operational robustness over feature breadth.

---

## What Problem This Solves

Most public prompt libraries fail in predictable ways:

- Prompts are brittle or model-specific  
- Content relies on informal “prompt hacks”  
- Updates are hard to audit or reproduce  
- Delivery relies on static downloads or client-side trust  

PromptKitPro treats prompts as **engineering artefacts**, not casual content.

Every stage — from source management to payment verification to fulfilment — is explicitly controlled.

---

## Product Overview

The product is a library of **250 professionally structured prompts** designed for people who already use AI tools regularly for real work.

Each prompt is:

- Explicitly structured  
- Model-aware but model-agnostic  
- Designed to reduce output variance  
- Reusable across workflows and contexts  

### Delivered Formats

All customer artefacts are generated from a **single source of truth** and delivered as a secure ZIP containing:

- PDF (primary browsing and reference format)  
- CSV (editable source)  
- JSON (programmatic use)  
- Markdown (human-readable reuse)  
- README (usage guidance)  

---

## System Architecture (Conceptual)

PromptKitPro is best understood as a **five-stage pipeline**:

1. **Prompt Source Management**  
   - One CSV file is the sole runtime input  
   - Version-controlled and immutable per release  

2. **Artefact Generation**  
   - All deliverables generated server-side on demand  
   - No long-lived artefacts stored  

3. **Payment & Verification**  
   - Stripe Checkout as the trust boundary  
   - Payment confirmed via signed webhooks  

4. **Fulfilment & Access Control**  
   - Unique, non-guessable download tokens  
   - Artefacts generated only after verified payment  

5. **Deployment & Operations**  
   - Single FastAPI service  
   - Minimal persistent state (orders + tokens only)  

---

## Design Philosophy

The system is intentionally designed to be:

- **Minimal** — only essential components  
- **Deterministic** — identical outputs from identical inputs  
- **Auditable** — all changes traceable via version control  
- **Low-maintenance** — no background jobs or queues  
- **Easy to hand over** — explicit documentation, no tribal knowledge  

PromptKitPro is explicitly **not**:

- A multi-tenant SaaS  
- A continuously mutating content platform  
- A subscription system  
- A dashboard-heavy application  

This aligns closely with ISO-9001 principles around controlled change, reproducibility, and process clarity.

---

## Technical Stack

- **Language:** Python 3.11+  
- **Framework:** FastAPI  
- **PDF Generation:** ReportLab  
- **Payments:** Stripe Checkout  
- **Database:** SQLite  
- **Hosting:** Render  
- **Frontend:** Static HTML + CSS  
- **OS Tested:** Windows 11 x64  

---

## Data & Privacy

Only minimal customer data is stored:

- Email address  
- Stripe session ID  
- Payment status  
- Download token  

No additional personal data is collected or retained.

---

## Current System State

PromptKitPro is currently:

- Fully deployed and live  
- Taking real payments  
- Delivering artefacts from a version-controlled prompt corpus  
- Operating with verified end-to-end payment and fulfilment flows  

The system has been validated through:

- Local artefact generation  
- Live Stripe checkout testing  
- Token-gated download verification  
- Controlled deployment and rollback  

---

## Summary

PromptKitPro is a commercially active digital product implemented as a **controlled pipeline rather than a conventional web application**.

Its architecture emphasises engineering discipline, operational clarity, and auditability, making it suitable for long-term maintenance, handover, and incremental evolution without architectural drift.
