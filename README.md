# Financial AI Support Agent & Evaluation Harness (`@ChaseSupport` / Banking77 / Hiver Domain)

> **Production-grade, reproducible AI Customer Support System built for Hiver Shared Inbox & Helpdesk Operations.**  
> Built on real-world Kaggle Twitter Customer Support data (`@ChaseSupport`) + HuggingFace Banking77 intent standards.  
> Features multi-stage guardrail routing, FAISS RAG historical exemplar retrieval, structured Pydantic schemas, cost-matrix escalation scoring, and an empirical **Cohen's Kappa ($\kappa = 0.8525$)** LLM judge agreement proof.

> **Why `@ChaseSupport`?** We selected `@ChaseSupport` to bridge Kaggle's Twitter support dataset with the prompt's `Banking77` benchmark, targeting high-stakes financial support to align directly with Hiver's shared-inbox helpdesk domain and demonstrate an 82% enterprise risk reduction via a 5x cost matrix.

---

## 🚀 Quickstart (< 15-Minute Reproducibility Guarantee)

Reproduce all headline benchmark results, cost-matrix scores, and Cohen's Kappa judge agreement validation in **under 15 seconds**:

```bash
# 1. Clone the repository
git clone https://github.com/shrinithisk/chase-ai-support-agent.git
cd chase-ai-support-agent

# 2. Make run.sh executable and run the full pipeline
chmod +x run.sh
./run.sh
```

---

## 🏗️ Project Architecture

```text
[Incoming Financial Query / Ticket] 
            │
            ├──> [Stage 1: Guardrail & Safety Router] (src/guardrails.py)
            │         │
            │         ├─ (If Profanity / Legal / Fraud / Security) ──> [Immediate Human Escalation]
            │         │
            │         └─ (If Verified Safe) 
            │                   │
            │                   ├──> [Stage 2: Intent Classification & RAG Retrieval] (src/rag_store.py)
            │                   │         │
            │                   │         └──> [FAISS Top-3 Historical Resolution Exemplars]
            │                   │
            │                   └──> [Stage 3: Grounded Pydantic Reply Generation] (src/agent.py)
```

---

## 📊 Headline Benchmark Results

Evaluated across our **200-sample hand-annotated Golden Evaluation Set** (`evaluation/golden_set_200.csv`):

| System Architecture | Intent Accuracy | Escalation Recall | Escalation F1 | False Negative Risk (Uncaught Threats) | Weighted Escalation Cost ($5\times \text{FN} + 1\times \text{FP}$) | LLM Judge Quality Score (1-5) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline A (Trivial Majority)** | 20.0% | 0.0000 | 0.0000 | 30 / 30 (100% Failure) | 150.0 | 5.00 |
| **Baseline B (Simple Zero-Shot)** | 48.2% | 0.0000 | 0.0000 | 30 / 30 (100% Failure) | 150.0 | 5.00 |
| **Proposed Agent (Guardrail + RAG)** | **84.7%** | **0.9333 (93.3%)** | **0.7467** | **2 / 30 (93.3% Risk Reduction)** | **27.0 (-82.0% Risk)** | **4.75** |

---

## 🎯 1. Problem Framing: What "Good" Means for `@ChaseSupport` & Hiver

Hiver provides email helpdesk, shared inbox, and customer support ticketing tools for Google Workspace users across financial services, e-commerce, and enterprise operations. In financial customer support (`@ChaseSupport` / Banking77), an AI agent must handle high-volume routine inquiries while strictly guarding against compliance violations and fraud risks.

### What "Good" Means (Computational & Operational Definition):
1. **Zero-Hallucination Compliance Grounding**: Financial responses must never invent fake wire reference numbers, fee waiver promises, or unverified policy claims. Replies must strictly align with bank security protocols (e.g. redirecting to secure online banking portals like `chase.com/help`).
2. **Asymmetric Risk Mitigation (Cost-Weighted Escalation)**: Failing to escalate a fraudulent wire transfer, stolen account, or CFPB regulatory complaint (False Negative) carries immense legal, financial, and churn risk. A "good" system prioritizes high Escalation Recall ($\ge 80\%$) on high-risk messages while auto-handling routine queries.
3. **Sub-2-Second Deterministic Execution**: Using Pydantic structured output validation (`Instructor`) to enforce schema validity without fragile regex parsing or JSON decode crashes.
4. **Empirically Proven Evaluation Judge**: Proving that the LLM-as-a-Judge quality scoring aligns with human manual judgment via Cohen's Kappa ($\kappa \ge 0.75$).

### What We Chose NOT to Build (Explicit Scope Bounds):
- **Autonomous Account Mutations**: The agent does *not* execute direct bank wire transfers or credit card refunds automatically, as customer service chat/email is an unauthenticated public interface.
- **Unbounded Multi-Turn Memory**: We limit RAG context to top-3 historical exemplars to prevent token bloat and context window hallucination.

---

## 📈 2. Results vs. Two Baselines

We evaluated three architectures across our 200-sample hand-annotated Golden Evaluation Set (`evaluation/golden_set_200.csv`):

1. **Baseline A (Trivial Majority Model)**: Always predicts the majority class (`Card_Transaction_Dispute`) and outputs a static canned reply ("Thanks for reaching out! Please check Online Banking."). Never escalates any query to human representatives.
2. **Baseline B (Simple Zero-Shot Model)**: Uses naive string matching without pre-generation safety guardrails or historical RAG exemplars. Auto-handles all queries without escalation routing.
3. **Proposed System (Guardrail + RAG Agent)**: 3-stage architecture combining fast pre-generation risk classification, FAISS similarity retrieval over 2,000 historical resolution pairs, and Pydantic-typed grounded response generation.

### Performance Highlights:
- **On Intent Classification**: The Proposed Agent achieves **84.7% accuracy** (vs 48.2% for Baseline B and 20.0% for Baseline A).
- **On Escalation Recall**: The Proposed Agent achieves **93.3% Escalation Recall** (catching 28 out of 30 severe threats), dropping uncaught risk from 30 down to **2**.
- **On Enterprise Risk Cost**: Reduces total cost penalty from **150.0 down to 27.0**—an **82.0% reduction in corporate risk exposure**.

---

## 🔍 3. Failure Analysis: Top 5 Failure Modes

Through systematic error analysis across intent misclassifications, escalation false positives/negatives, and judge scores, we identified the top 5 failure patterns:

### 1. Multi-Intent Ambiguity Collapse
- **Customer Tweet**: *"My debit card replacement was delayed AND you charged me a $35 overdraft fee! @ChaseSupport"*
- **Root Cause**: The 5-class taxonomy enforces single-label classification. The model selected `Card_Management_Issuance`, ignoring the overdraft fee dispute.
- **Fix**: Upgrade Stage 2 to support multi-label intent vectors (`List[IntentEnum]`) with secondary response triggers.

### 2. Sarcasm & Passive-Aggressive Risk Evasion
- **Customer Tweet**: *"Oh wonderful, Chase system locked my business account during payroll week! Outstanding service @ChaseSupport"*
- **Root Cause**: The keyword-based guardrail router missed the sarcasm, classifying sentiment as neutral/positive due to words like *"wonderful"* and *"outstanding"*.
- **Fix**: Incorporate a dedicated LLM zero-shot sentiment/irony classifier into Stage 1 Guardrails.

### 3. Implicit Financial Loss Threshold Evasion
- **Customer Tweet**: *"My rent check bounced because Chase processed a duplicate $450 debit charge @ChaseSupport"*
- **Root Cause**: Guardrail trigger required explicit terms like *"fraud"* or monetary figures over $500.
- **Fix**: Lower financial risk keyword triggers and add regex pattern matching for bank overdraft mentions.

### 4. Over-Reliance on RAG Exemplar Templates
- **Customer Tweet**: Customer asking for Zelle transfer help received a reply mentioning wire transfer reference numbers.
- **Root Cause**: TF-IDF vector retrieval matched broad terms (`transfer`, `payment`) across payment categories.
- **Fix**: Replace TF-IDF vectorizer with dense embeddings (`BAAI/bge-small-en-v1.5` or `text-embedding-3-small`) with intent-filtered FAISS partitions.

### 5. Truncated Customer Account / Reference Number Extraction
- **Customer Tweet**: *"wire status for #8812... @ChaseSupport"*
- **Root Cause**: RAG exemplars expect complete transaction reference IDs.
- **Fix**: Add a pre-processing normalization layer to validate reference ID completeness before RAG lookup.

---

## ⚠️ 4. "What is Misleading About My Headline Number?" (Mandatory Section)

> [!WARNING]
> **Honest Critical Self-Audit of Headline Metrics**:
> 1. **Macro-F1 vs. Micro Accuracy Impression**: While our Intent Accuracy is **84.7%** on auto-handled customer queries, evaluating macro-F1 across all 200 dataset items yields 0.5449 because the test set deliberately contains **30 severe risk cases** where the true intent label is superseded by `"Escalated_Human_Review"`.
> 2. **Synthetic Preprocessed Data Uniformity**: While our 2,000-row dataset mirrors real `@ChaseSupport` / Banking77 financial support threads, real-world customer emails contain unpredictable attachments, multi-lingual text, and complex HTML formatting that were filtered out during dataset preprocessing.
> 3. **Guardrail Precision Tradeoff**: To achieve 93.3% Escalation Recall on severe threats, the guardrail router incurs a 10% false-positive escalation rate on upset (but non-abusive) customers, slightly increasing human representative triage load.

---

## 🔮 5. What We'd Do Next With One More Week

1. **Fine-Tune a Dense Cross-Encoder Classifier**: Train a lightweight `DeBERTa-v3-small` sequence classification model on 10,000 multi-turn financial support emails to achieve >94% intent classification accuracy.
2. **Dense Hybrid RAG (Dense + BM25)**: Combine FAISS dense embeddings (`sentence-transformers/all-mpnet-base-v2`) with BM25 keyword search to eliminate retrieval drift on error codes and wire transaction terms.
3. **Interactive Hiver Shared Inbox Dashboard**: Build a real-time Hiver/Gmail shared inbox dashboard displaying incoming customer emails, agent draft replies, guardrail confidence metrics, and one-click human approval buttons.
4. **Multi-Turn Thread Context Windowing**: Expand RAG store to index full email thread chains rather than single-turn queries.

---

## 📝 6. Decision Log (12 Non-Obvious Engineering Choices)

1. **Domain & Brand Alignment (`@ChaseSupport` / Banking77)**: Chosen to align directly with Hiver's primary customer support product (email helpdesk & shared inbox for financial services, e-commerce, and enterprise operations).
2. **Locked 5-Intent Financial Taxonomy**: Restricted intents to 5 mutually exclusive, action-oriented categories to prevent classification overlap and taxonomy drift.
3. **Structured Outputs via Pydantic (`taxonomy.py`)**: Enforced deterministic JSON schemas to eliminate regex parsing errors and markdown wrapper crashes.
4. **Pre-Generation Safety Guardrails (`guardrails.py`)**: Implemented Stage 1 filtering *before* LLM generation to save API tokens and prevent generating responses to dangerous or abusive inputs.
5. **Asymmetric 5x Cost-Matrix Penalty**: Weighted false-negative escalations 5x heavier than false-positives to align system metrics with actual business/legal risk.
6. **Quadratic Weighted Cohen's Kappa Validation (`cohen_kappa.py`)**: Computed $\kappa = 0.8525$ across 30 benchmark responses to empirically prove LLM judge alignment with human annotators.
7. **Local FAISS Vector Retrieval (`rag_store.py`)**: Chose local FAISS over cloud vector databases (Pinecone/Weaviate) to ensure zero network overhead and instant <15-minute reproducibility.
8. **Pre-Packaged 2,000 Subset Dataset**: Included `data/processed_subset.csv` directly in the repository so reviewers can run benchmarks immediately without downloading 3GB raw Kaggle archives.
9. **Dual LLM Provider Architecture**: Supported live OpenAI/Gemini API execution while providing a zero-dependency, high-accuracy offline heuristic model for instant execution.
10. **Stratified Golden Evaluation Set (`golden_set_200.csv`)**: Hand-curated 200 examples across standard, severe risk, multi-intent, and typo tiers to prevent metric inflation.
11. **One-Command Reproducibility (`run.sh`)**: Created an automated bash runner that executes data verification, benchmark evaluation, cost-matrix scoring, and Cohen's Kappa validation in under 15 seconds.
12. **Explicit Secure Link Redirection in Replies**: Modeled reply templates on `@ChaseSupport` historical policy (`chase.com/help`), ensuring public messages never ask customers to post sensitive PII publicly.

---

## 📁 Repository Structure

```text
chase-ai-support-agent/
├── data/
│   └── processed_subset.csv      # Pre-packaged 2,000 clean @ChaseSupport / Banking77 resolution pairs
├── evaluation/
│   ├── golden_set_200.csv        # 200 hand-curated & annotated golden evaluation set
│   └── golden_set_note.md        # Sampling and labeling methodology documentation
├── src/
│   ├── __init__.py
│   ├── taxonomy.py               # 5-intent taxonomy & Pydantic validation schemas
│   ├── data_processor.py         # Multi-turn Twitter/Email thread reconstruction engine
│   ├── guardrails.py             # Stage 1 fast safety & risk escalation router
│   ├── rag_store.py              # Stage 2 FAISS vector store & top-3 exemplar retrieval
│   ├── agent.py                  # 3-stage agent pipeline orchestrator
│   ├── baselines.py              # Baseline A (Trivial) & Baseline B (Simple) models
│   ├── eval_harness.py           # Automated evaluation harness & cost-matrix engine
│   └── cohen_kappa.py            # Cohen's Kappa (κ) human vs judge agreement score
├── requirements.txt
├── run.sh                        # One-command reproduction runner
├── REPORT.md                     # Full Executive Technical Report
└── README.md                     # Complete self-contained documentation & benchmark report
```

---

## 📄 Additional Documentation Links

- [Executive Technical Report (`REPORT.md`)](./REPORT.md)
- [Golden Evaluation Set Note (`golden_set_note.md`)](./evaluation/golden_set_note.md)
