# Konkan Vani — Comprehensive System Implementation Document

> **Document Version:** 1.0.0  
> **System Name:** Konkan Vani (कोंकण वाणी / Konkan Vani)  
> **Sub-title:** A Bilingual Konkani Idiom and Proverb Meaning Retrieval System  
> **Target Audience:** Evaluators, Developers, NLP Engineers, and System Architects  
> **Last Updated:** October 2026  

---

## Table of Contents

1. [Executive Summary & System Purpose](#1-executive-summary--system-purpose)
2. [High-Level Architecture & Repository Blueprint](#2-high-level-architecture--repository-blueprint)
3. [Data Assets, Schema & Provenance Engineering](#3-data-assets-schema--provenance-engineering)
4. [The 4-Stage Natural Language Processing Pipeline](#4-the-4-stage-natural-language-processing-pipeline)
   - [4.1 Phonetic Normalization Engine](#41-phonetic-normalization-engine)
   - [4.2 Stage 1: Lexical & Phonetic Fuzzy Matcher](#42-stage-1-lexical--phonetic-fuzzy-matcher)
   - [4.3 Stage 2: Dense Semantic Vector Retrieval with Overlap Filter](#43-stage-2-dense-semantic-vector-retrieval-with-overlap-filter)
   - [4.4 Stage 3: Dynamic AI Fallback (Optional)](#44-stage-3-dynamic-ai-fallback-optional)
   - [4.5 Stage 4: Compositional Word-by-Word Fallback Reading](#45-stage-4-compositional-word-by-word-fallback-reading)
   - [4.6 Word-Level Morphological Analysis](#46-word-level-morphological-analysis)
5. [Neural Model Fine-Tuning & Representation Learning](#5-neural-model-fine-tuning--representation-learning)
   - [5.1 Base Model Architecture](#51-base-model-architecture)
   - [5.2 Multiple Negatives Ranking Loss (MNRL) Formulation](#52-multiple-negatives-ranking-loss-mnrl-formulation)
   - [5.3 Training Data Construction & Optimization](#53-training-data-construction--optimization)
   - [5.4 Model Registry & In-Memory Singleton](#54-model-registry--in-memory-singleton)
6. [Backend Service & REST API Specification](#6-backend-service--rest-api-specification)
   - [6.1 Application Structure & Middleware](#61-application-structure--middleware)
   - [6.2 In-Memory Store & FAISS Integration (`data_loader.py`)](#62-in-memory-store--faiss-integration-data_loaderpy)
   - [6.3 Complete API Endpoint Catalog](#63-complete-api-endpoint-catalog)
7. [Frontend Web Application Architecture](#7-frontend-web-application-architecture)
   - [7.1 Tech Stack & Design System](#71-tech-stack--design-system)
   - [7.2 Routing & Page Architecture](#72-routing--page-architecture)
   - [7.3 Specialized Component Specifications](#73-specialized-component-specifications)
   - [7.4 Client-Side API Integration & Type Definitions](#74-client-side-api-integration--type-definitions)
8. [Empirical Evaluation & Measured Benchmark Results](#8-empirical-evaluation--measured-benchmark-results)
   - [8.1 Effect of Encoder Fine-Tuning](#81-effect-of-encoder-fine-tuning)
   - [8.2 Stage-Wise Top-1 Retrieval Accuracy](#82-stage-wise-top-1-retrieval-accuracy)
   - [8.3 End-to-End Pipeline Accuracy & Out-of-Domain Rejection](#83-end-to-end-pipeline-accuracy--out-of-domain-rejection)
   - [8.4 Threshold Sensitivity & Score Separation](#84-threshold-sensitivity--score-separation)
   - [8.5 System Latency Benchmarks](#85-system-latency-benchmarks)
9. [Operational Guide: Deployment & Reproduction](#9-operational-guide-deployment--reproduction)
   - [9.1 Prerequisites & System Dependencies](#91-prerequisites--system-dependencies)
   - [9.2 Step-by-Step Installation & Service Startup](#92-step-by-step-installation--service-startup)
   - [9.3 Reproducing Embeddings, Indices, and Benchmarks](#93-reproducing-embeddings-indices-and-benchmarks)
10. [Known Limitations & Future Technical Roadmap](#10-known-limitations--future-technical-roadmap)

---

## 1. Executive Summary & System Purpose

**Konkan Vani** is a specialized Natural Language Processing (NLP) retrieval and semantic comprehension platform engineered for Konkani idioms and proverbs (*mhonni*). 

### The Problem
Konkani is an Indo-Aryan language spoken along India's western coast (Goa, coastal Karnataka, Maharashtra, and parts of Kerala). It presents distinct challenges for standard NLP architectures:
1. **Multi-Script Orthography:** Konkani has no single universal script. It is actively written across five distinct scripts: **Devanagari** (official in Goa), **Romi / Latin script** (prevalent in coastal Goa and diaspora), **Kannada script** (Karnataka Konkani), **Malayalam script** (Cochin/Kerala), and historical **Perso-Arabic**.
2. **Extreme Transliteration Noise:** On modern digital devices and messaging platforms, users predominantly communicate in informal Romanized Konkani with non-standard, phonetic, and dialectal spellings (e.g., `"Haat dakhvun"`, `"Hat dakwun"`, `"Hat dakhavn"`).
3. **Idiomatic Non-Compositionality:** General-purpose translation models (such as Google Translate) fail catastrophically on idioms because they attempt literal word-by-word substitution, converting rich figurative wisdom into nonsensical phrasing.
4. **Scarcity of Attested Corpora:** Konkani is a low-resource language with minimal digital linguistic tools, lack of formal wordnets, and absence of standardized sentence-embedding baselines.

### The Solution
Konkan Vani addresses these issues through a multi-tiered architecture combining:
- **Rule-based phonetic normalization** to collapse character and vowel transliteration variants.
- **Order-insensitive token fuzzy matching** (RapidFuzz) for rapid lexical resolution.
- **Domain-adapted multilingual dense embeddings** (a fine-tuned LaBSE transformer model: `konkan-vani-encoder-v1`) powering exact inner-product vector search in **Meta FAISS**.
- **Lexical overlap gating** to eliminate topical false positives.
- **A multi-layered compositional fallback engine** backed by a 3,000+ entry Konkani word lexicon that provides word-by-word glossing and thematic inference when an uncataloged proverb is queried, eliminating the traditional "Not Found" dead end.

---

## 2. High-Level Architecture & Repository Blueprint

Konkan Vani is structured as a decoupled monorepo comprising a modern Next.js client, a Python FastAPI NLP engine, local offline model artifacts, precomputed vector indices, and reproducible training/benchmarking scripts.

### High-Level System Architecture Diagram

```
                        +----------------------------------------+
                        |           Next.js Web Client           |
                        | (React 18, App Router, TypeScript, UI) |
                        +----------------------------------------+
                                            |
                                  REST API (HTTP / JSON)
                                            v
                        +----------------------------------------+
                        |          FastAPI NLP Service           |
                        |      (Python 3.12, Uvicorn, CORS)      |
                        +----------------------------------------+
                                            |
                      +---------------------+---------------------+
                      |                                           |
                      v                                           v
           [ In-Memory Idiom Store ]                  [ Core NLP Pipeline ]
           - 5,200 Gold Idioms (CSV)                  - Unicode & Phonetic Normalizer
           - Precomputed Phonetic Keys                - RapidFuzz Matcher (Stage 1)
           - FAISS IndexFlatIP (768-d)                - Fine-tuned LaBSE Encoder (Stage 2)
           - Unit Embeddings (float32)                - Lexical Overlap Gate
           - 3,000+ Word Lexicon                      - Compositional Engine (Stage 4)
                                                      - Word Analysis Module
```

### Repository Directory Blueprint

```
Konkan-Vani/
├── client/                               # Next.js Frontend Application
│   ├── app/                              # Next.js App Router Pages
│   │   ├── contribute/page.tsx           # Community proverb submission page
│   │   ├── explore/page.tsx              # Paginated corpus browser with filters
│   │   ├── results/page.tsx              # Main query result & resolution screen
│   │   ├── globals.css                   # Global styling and Tailwind directives
│   │   ├── layout.tsx                    # Root layout wrapper with navigation
│   │   └── page.tsx                      # Landing search interface
│   ├── components/                       # Reusable React UI Components
│   │   ├── ConfidenceBadge.tsx           # Match type & score pill renderer
│   │   ├── IdiomCard.tsx                 # Detailed card displaying verified idioms
│   │   ├── ProvisionalMeaning.tsx        # Word-by-word compositional fallback UI
│   │   ├── ScriptToggle.tsx              # Multi-script switcher (Devanagari/Roman)
│   │   └── SideNavBar.tsx                # Application navigation drawer
│   ├── lib/                              # Client-side utility and library code
│   │   ├── api.ts                        # Type-safe fetch client with auto-fallback
│   │   └── types.ts                      # TypeScript schemas matching backend DTOs
│   ├── package.json                      # Node.js dependencies
│   └── tsconfig.json                     # TypeScript compiler configuration
│
├── server-nlp/                           # Production FastAPI NLP Backend
│   ├── app/                              # Application Package
│   │   ├── core/                         # Core Algorithmic & NLP Engines
│   │   │   ├── compose.py                # Compositional fallback meaning assembler
│   │   │   ├── dynamic_fallback.py       # Opt-in Gemini LLM fallback (disabled by default)
│   │   │   ├── lexicon.py                # Multi-tier Konkani word dictionary lookup
│   │   │   ├── model_registry.py         # Thread-safe singleton for transformer encoder
│   │   │   ├── phonetic_normalize.py     # Rule-based vowel/consonant normalizer
│   │   │   ├── query_pipeline.py         # Staged 4-tier query resolver & search logic
│   │   │   └── word_analysis.py          # Morphological token decomposition
│   │   ├── routers/                      # FastAPI Endpoint Handlers
│   │   │   ├── health.py                 # System health and memory monitor
│   │   │   ├── resolve.py                # POST /api/resolve (Primary resolver endpoint)
│   │   │   └── search.py                 # GET /api/search, /api/browse, /api/stats
│   │   ├── data_loader.py                # In-memory IdiomStore & FAISS index loader
│   │   └── main.py                       # FastAPI application factory & CORS setup
│   ├── scripts/                          # Backend Utility & Training Scripts
│   │   ├── build_embeddings.py           # Precomputes 768-d vectors and FAISS index
│   │   ├── diagnose_matches.py           # Analysis script for score separation
│   │   ├── smoke_test.py                 # Quick verification of server endpoints
│   │   ├── test_compositional.py         # Unit tests for compositional fallback
│   │   ├── test_live_api.py              # End-to-end HTTP integration tests
│   │   ├── test_pipeline.py              # Pipeline accuracy & test cases
│   │   └── train_model.py                # Fine-tunes LaBSE with MNRL loss
│   ├── Dockerfile                        # Container definition for NLP service
│   ├── requirements.txt                  # Python runtime dependencies
│   └── run_server.py                     # Entrypoint script for Uvicorn runner
│
├── data/                                 # Data Assets & Processed Corpora
│   ├── lexicon/                          # Word-level vocabulary assets
│   │   └── konkani_lexicon.csv           # 3,000+ words with POS, tags, Devanagari/Roman
│   ├── processed/                        # Gold & Index Corpora
│   │   ├── index/                        # Persisted vector index files
│   │   │   ├── konkani.index             # Meta FAISS IndexFlatIP binary
│   │   │   └── konkani_meta.json         # Metadata mappings for indexed vectors
│   │   ├── idioms_with_phonetic_keys.csv # Complete 5,200 idiom gold corpus
│   │   └── idioms_with_phonetic_keys_3772_backup.csv
│   └── idiom_embeddings.npy              # Serialized (5200, 768) float32 NumPy array
│
├── models/                               # Local Model Checkpoints
│   └── konkan-vani-encoder-v1/           # Fine-tuned LaBSE model artifacts (PyTorch)
│
├── src/                                  # Repository Root Benchmarking & Build Scripts
│   ├── build_index.py                    # Rebuilds the FAISS vector index
│   ├── build_lexicon.py                  # Compiles word lexicon from corpora
│   ├── calibrate_lexicon.py              # Calibrates cosine thresholds for word lookup
│   ├── rebuild_dataset.py                # Verifies and rebuilds the 5,200 idiom CSV
│   ├── run_full_benchmark.py             # Automated reproducible test suite
│   └── make_results_assets.py            # Generates Matplotlib evaluation charts
│
├── results/                              # Benchmark Outputs & TeX Tables
│   ├── chapter5_results.tex              # Results chapter LaTeX source
│   ├── tables.tex                        # Formatted LaTeX data tables
│   └── tables_composition.tex            # Fallback composition performance tables
│
├── figures/                              # Generated Charts and Visualizations
│   ├── fig_accuracy.png                  # Retrieval accuracy chart
│   ├── fig_dataset.png                   # Category and provenance breakdown
│   ├── fig_embedding_heatmap.png         # Vector activation slice heatmap
│   ├── fig_finetune.png                  # Fine-tuning baseline vs custom chart
│   ├── fig_latency.png                   # Measured response latency chart
│   ├── fig_latency_cdf.png               # Cumulative latency distribution (CDF)
│   ├── fig_similarity_hist.png           # In-domain vs out-of-domain histogram
│   └── fig_threshold.png                 # Threshold sensitivity curve
│
├── konkan_vani_report.tex                # Formal Academic Mini-Project Report (LaTeX)
├── ARCHITECTURE.md                       # High-level architecture documentation
├── README.md                             # Repository setup and overview
└── task.md                               # Project backlog and milestone tracking
```

---

## 3. Data Assets, Schema & Provenance Engineering

### 3.1 The 5,200 Idiom Gold Corpus
The primary dataset is stored at `data/processed/idioms_with_phonetic_keys.csv` containing exactly **5,200 validated records**. Every record is strictly verified with zero missing values across all structural fields.

#### Schema Fields per Record:
| Column Name | Data Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `id` | String | Unique record identifier | `"mhonni_0042"` |
| `konkani_text` | String | Canonical expression in Devanagari script | `"हाताचे काकणाक आरसो कित्याक"` |
| `romanized_text`| String | Canonical Romanized / Romi Konkani transcription | `"Hatache kaknak aanso kityak"` |
| `script` | String | Primary native script | `"Devanagari"` |
| `literal_meaning` | String | Direct, word-for-word translation | `"Why does one need a mirror to see the bangles on one's own hand?"` |
| `figurative_meaning`| String | Cultural semantic meaning / contextual wisdom | `"What is obvious and directly visible requires no further proof."` |
| `english_meaning` | String | Concise English explanatory definition | `"The obvious needs no proof; self-evident truth."` |
| `marathi_meaning` | String | Concise Marathi explanatory definition | `"प्रत्यक्ष दिसणाऱ्या गोष्टीला पुराव्याची गरज नसते."` |
| `example_sentence`| String | Contextual illustrative sentence in Devanagari | `"तो चोरी करताना पकडला गेला, आता हाताचे काकणाक आरसो कित्याक?"` |
| `category` | String | 1 of 15 thematic taxonomy categories | `"wisdom"` |
| `cultural_context`| String | Historical or cultural context of usage | `"Commonly spoken in coastal household disputes when evidence is clear."` |
| `phonetic_key` | String | Precomputed normalized phonetic search key | `"hatace kaknak aanso kityak"` |
| `source` | String | Provenance tracker (`curated` vs `pattern_v1`) | `"curated_folklore_v1"` |

### 3.2 Provenance & Dataset Split
To maintain scientific integrity and academic transparency, Konkan Vani explicitly tracks provenance in the `source` attribute:
- **Curated Authentic Entries (105 records, 2.0%):** Hand-transcribed and verified idioms and proverbs sourced from Konkani literature, community elders, and folk collections.
- **Pattern-Generated Proverb Frames (5,095 records, 98.0%):** Deterministically constructed entries following validated Konkani noun-verb-consequence syntactic templates (`X does Y means Z`). These synthetic frames provide the large lexical and structural contrast necessary for:
  1. Fine-tuning the multilingual transformer using contrastive in-batch negatives.
  2. Indexing a rich vocabulary across diverse grammatical declensions.
  3. Stress-testing retrieval throughput and vector search scalability at production volume.

### 3.3 Thematic Taxonomy
The corpus spans **15 distinct thematic categories**:
`wisdom`, `character`, `nature`, `food`, `animals`, `family`, `social life`, `wealth`, `destiny`, `deception`, `diligence`, `caution`, `relationships`, `spirituality`, and `irony`.

### 3.4 Precomputed Vector Store & The Word Lexicon
- **`data/idiom_embeddings.npy`:** A serialized NumPy array of shape `(5200, 768)` in `float32` precision. Each row represents the $L_2$-normalized composite embedding of an idiom generated by `konkan-vani-encoder-v1`.
- **`data/lexicon/konkani_lexicon.csv`:** A dedicated lexical resource comprising over **3,000 Konkani root words**, mapped with:
  - Devanagari headword (`शब्द`)
  - Romanized transliteration (`shabd`)
  - Phonetic key
  - Part of Speech (`noun`, `verb`, `adjective`, `connective`, `negation`, etc.)
  - Semantic tags (`water`, `food`, `animal`, `body`, `motion`, `speech`, etc.)
  - English and Marathi glosses.

---

## 4. The 4-Stage Natural Language Processing Pipeline

When a user submits a query $Q$ through the web client or REST API, it is resolved through a deterministic, staged cascade designed to balance millisecond execution speed with deep semantic comprehension.

```
Incoming Query Q
       |
       v
[ Unicode & Phonetic Normalization ]
       |
       v
===================================================================
STAGE 1: Lexical & Phonetic Fuzzy Match (RapidFuzz, token_sort_ratio)
===================================================================
       |
       +---> Score >= 75? ----(YES)----> [ Return Record (Type: "phonetic", <3ms) ]
       |
      (NO)
       v
===================================================================
STAGE 2: Dense Semantic Vector Retrieval (Fine-tuned LaBSE + FAISS)
===================================================================
       |
       +---> Cosine >= 0.63 AND Overlap >= 0.34?
                 |
               (YES) ------------------> [ Return Record (Type: "semantic", ~25ms) ]
                 |
               (NO)
                 v
===================================================================
STAGE 3: Dynamic AI Fallback (Google Gemini, disabled by default)
===================================================================
       |
       +---> Enabled & Valid? ---------> [ Return AI Fallback Record ]
       |
      (NO)
       v
===================================================================
STAGE 4: Compositional Word-by-Word Reading (Lexicon + Theme Inference)
===================================================================
       |
       v
[ Return Provisional Reading with Word Glosses & Closest Known Idioms ]
```

---

### 4.1 Phonetic Normalization Engine
Implemented in `server-nlp/app/core/phonetic_normalize.py`:

Informal Romanized Konkani contains heavy spelling inconsistencies due to inconsistent vowel length representations and consonant substitutions. The `phonetic_normalize(text)` function converts strings into a unified phonetic key through the following deterministic pipeline:
1. **Unicode NFC Decomposition & Lowercasing:** Strips leading/trailing whitespace, converts input to Unicode standard form, and folds all characters to lowercase.
2. **Vowel-Length Folding:**
   - `"aa"` $\to$ `"a"` (e.g., `"haat"` $\to$ `"hat"`)
   - `"ee"` $\to$ `"i"` (e.g., `"teek"` $\to$ `"tik"`)
   - `"oo"` $\to$ `"u"` (e.g., `"doodh"` $\to$ `"dudh"`)
3. **Consonant Normalization:**
   - `"ph"` $\to$ `"f"` (e.g., `"phal"` $\to$ `"fal"`)
   - `"v"` $\to$ `"w"` (e.g., `"dakhvun"` $\to$ `"dakhwun"`)
   - `"zh"` $\to$ `"j"` (e.g., `"zhadd"` $\to$ `"jadd"`)
   - `"x"` $\to$ `"sh"` (Portuguese-influenced Goan Romi transliteration: `"xet"` $\to$ `"shet"`)
   - `"c"` $\to$ `"k"` (when non-aspirated: `"corat"` $\to$ `"korat"`)
4. **Punctuation Stripping:** Removes all characters outside `[a-z\s]` and standardizes multiple internal spaces to a single space.

---

### 4.2 Stage 1: Lexical & Phonetic Fuzzy Matcher
Implemented in `server-nlp/app/core/query_pipeline.py` (`fuzzy_match()`):

Stage 1 resolves exact matches, minor typos, and surface spelling differences without invoking GPU neural inference.

1. **Script Detection:** Uses regex `[\u0900-\u097F]` to detect if the query is in Devanagari script.
   - If **Devanagari**: The normalized query is compared directly against `store.df["konkani_text"]`.
   - If **Romanized**: The phonetic key of the query is compared against all precomputed `stored_keys` in the dataset.
2. **Metric: RapidFuzz `token_sort_ratio`:**
   $$\mathrm{score}(q, k) = \left( 1 - \frac{\mathrm{Lev}(q_{\mathrm{sorted}}, k_{\mathrm{sorted}})}{|q_{\mathrm{sorted}}| + |k_{\mathrm{sorted}}|} \right) \times 100$$
   *Mechanism:* Words in both query $q$ and candidate $k$ are tokenized, alphabetically sorted, re-joined, and evaluated via normalized Levenshtein distance. This ensures that word order permutations (e.g., `"kityak hatache kaknak"` vs `"hatache kaknak kityak"`) achieve a 100% similarity score.
3. **Cutoff Decision:**
   $$\mathrm{Accept} \iff \mathrm{score}(q, k^*) \ge \tau_{\mathrm{fuzzy}}, \quad \text{where } \tau_{\mathrm{fuzzy}} = 75$$
   If accepted, Stage 1 immediately returns the idiom record with `confidence = score / 100.0` and `match_type = "phonetic"`. Mean execution latency: **2.94 ms**.

---

### 4.3 Stage 2: Dense Semantic Vector Retrieval with Overlap Filter
Implemented in `server-nlp/app/core/query_pipeline.py` (`semantic_match()`):

When lexical fuzzy matching fails (e.g., the user submits a conceptual description in English, a Marathi paraphrase, or a severely corrupted query), Stage 2 performs multilingual semantic vector retrieval.

1. **Dense Query Encoding:**
   The raw query string $Q$ is passed to the fine-tuned model `konkan-vani-encoder-v1` via `model_registry.encode([Q])`, yielding a 768-dimensional float32 vector $\vec{e}_Q$.
2. **$L_2$ Normalization:**
   $$\vec{q} = \frac{\vec{e}_Q}{\|\vec{e}_Q\|_2} \in \mathbb{R}^{768}$$
3. **Exhaustive FAISS Search (`IndexFlatIP`):**
   Because all indexed vectors $\vec{d}_i$ are pre-normalized to unit length ($\|\vec{d}_i\|_2 = 1.0$), their inner product computed by FAISS SIMD kernels is mathematically identical to cosine similarity:
   $$\cos(\vec{q}, \vec{d}_i) = \vec{q} \cdot \vec{d}_i$$
   FAISS retrieves the top-$k$ nearest neighbors ($k=3$).
4. **The Semantic Threshold:**
   $$\text{Candidate accepted if } \cos(\vec{q}, \vec{d}^*) \ge \theta_{\mathrm{semantic}}, \quad \text{where } \theta_{\mathrm{semantic}} = 0.63$$
5. **The Lexical Overlap Gate (`SEMANTIC_MIN_OVERLAP = 0.34`):**
   *Why this is critical:* A proverb is a fixed linguistic expression, not merely a topical subject. During development diagnostics, it was observed that an unseen proverb such as *"तांदूळ पोटांत शित कुशीत"* scored 0.725 against *"भाताच्या पोळीत तूप"* because both idioms contain agricultural and food vocabulary, despite having completely distinct figurative meanings.
   To prevent topical false positives, Stage 2 enforces a lexical overlap check:
   $$\mathrm{overlap}(Q, \text{Candidate}) = \frac{|K_Q \cap K_C|}{|K_Q|} \ge 0.34$$
   where $K_Q$ and $K_C$ are the sets of phonetic-normalized content words (excluding stopwords such as `"mhanje"`, `"ani"`, `"na"`). Candidates failing this gate are rejected.

---

### 4.4 Stage 3: Dynamic AI Fallback (Optional)
Implemented in `server-nlp/app/core/dynamic_fallback.py`:
- Enabled only when `DYNAMIC_FALLBACK_ENABLED=1` is configured in the environment alongside a valid `GEMINI_API_KEY`.
- Queries the Google Gemini 1.5 Flash API to generate a provisional explanation.
- **Default State:** Kept strictly **disabled by default** in production to prevent LLM hallucinations, ensuring all returned idiom definitions stem from verified corpus records or rule-based lexical analyses.

---

### 4.5 Stage 4: Compositional Word-by-Word Fallback Reading
Implemented in `server-nlp/app/core/compose.py` and `lexicon.py`:

In traditional search engines, an uncataloged query returns an empty `"404 / Not Found"` response. Konkan Vani replaces this failure mode with an intelligent, transparent **compositional reading** that breaks down unknown proverbs word-by-word.

#### The 6-Tier Lexicon Lookup Hierarchy (`lexicon.py`):
For every token in the query, the engine attempts lookup in the 3,000-word lexicon through 6 successive strategies, recording the exact resolution method:
1. **`exact`:** Token matches a dictionary headword in Devanagari, Roman, or phonetic key.
2. **`suffix`:** Token is an inflected Konkani form. The engine strips standard nominal and verbal case endings:
   - Locative: `-ant` (`-ांत`), `-cher` (`-चेर`)
   - Genitive: `-che` (`-चे`), `-cho` (`-चो`), `-chi` (`-ची`)
   - Dative: `-k` (`-क`), `-ak` (`-ाक`)
   - Perfective participles: `-un` (`-ून`)
   and re-queries the base stem.
3. **`phonetic`:** Matches after rule-based phonetic normalization (`v/w`, `aa/a`, etc.).
4. **`fuzzy`:** RapidFuzz token match against Romanized headwords (strict cutoff: 84).
5. **`embedding`:** For uncataloged Devanagari words, computes the cosine similarity between the word's LaBSE vector and the English gloss embeddings of neighboring lexicon entries (acceptance threshold: $\ge 0.70$).
6. **`none`:** If no match satisfies criteria, the word is retained verbatim and marked `known: false`.

#### Thematic & Imagery Deduction (`compose.py`):
- Filters out grammatical function words (`connective`, `auxiliary`, `negation`, `pronoun`).
- Identifies compound verbs (e.g., `"धुवून घेवप"` $\to$ `"wash for oneself"`).
- Aggregates semantic tags across nouns and verbs, mapping them to prioritized cultural imagery: `water`, `food`, `animal`, `body`, `ritual`, `person`, `speech`, `nature`.
- Composes a provisional literal explanation (e.g., *"Provisional reading based on recognized words: water, flow and things that cannot be held back"*), reports exact word coverage (e.g., *"3 of 4 words recognized"*), and attaches the closest known proverbs from the FAISS index for user reference.
- **Transparency Guarantee:** Every compositional result is strictly flagged with `"provisional": true` and `"match_type": "compositional"`, ensuring it is never mistaken for verified folklore.

---

### 4.6 Word-Level Morphological Analysis
Implemented in `server-nlp/app/core/word_analysis.py`:
- Executed on **every response** across all stages.
- Deconstructs the query into individual tokens and pairs each word with its transliteration, grammatical role hint, and top lexicon candidates.
- Sent to the frontend inside the `word_analysis` array, allowing users to inspect word meanings interactively.

---

## 5. Neural Model Fine-Tuning & Representation Learning

### 5.1 Base Model Architecture
The semantic foundation of Konkan Vani is Google's **LaBSE** (`sentence-transformers/LaBSE`):
- **Architecture:** 12-layer BERT transformer supporting 109+ languages.
- **Output:** 768-dimensional dense sentence embeddings.
- **Pretrained Limitation:** While LaBSE supports Devanagari script (primarily through Hindi and Marathi corpora), its baseline representation of Konkani idioms is poor: baseline Recall@1 on Konkani idiom retrieval was only **51.35%**, and average cosine similarity between positive Konkani pairs was only **0.4368**.

---

### 5.2 Multiple Negatives Ranking Loss (MNRL) Formulation
To adapt LaBSE to the Konkani idiomatic domain, the model was fine-tuned using contrastive representation learning with `MultipleNegativesRankingLoss`:

$$\mathcal{L}_{\mathrm{MNRL}} = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{\exp\left(\frac{\vec{u}_i \cdot \vec{v}_i}{\tau}\right)}{\sum_{j=1}^{B} \exp\left(\frac{\vec{u}_i \cdot \vec{v}_j}{\tau}\right)}$$

where:
- $B$ is the training mini-batch size ($B = 16$).
- $\vec{u}_i \in \mathbb{R}^{768}$ is the $L_2$-normalized embedding of the $i$-th anchor query sequence (e.g., a transliterated or colloquial phrasing).
- $\vec{v}_i \in \mathbb{R}^{768}$ is the $L_2$-normalized embedding of the corresponding true positive target (e.g., canonical Devanagari idiom or its English/Marathi meaning text).
- $\vec{v}_j \in \mathbb{R}^{768}$ ($j \neq i$) represents the embeddings of other positive sequences in the same batch, which act as **implicit hard negative examples** for anchor $\vec{u}_i$.
- $\tau$ is the softmax temperature scaling parameter ($\tau > 0$).
- $\vec{u}_i \cdot \vec{v}_j$ represents the cosine similarity between unit vectors.

The loss penalizes proximity between dissimilar idioms while pulling positive cross-lingual and cross-script pairs into tight spatial clusters.

---

### 5.3 Training Data Construction & Optimization
Implemented in `server-nlp/scripts/train_model.py`:
- **Training Corpus:** **8,320 positive pair tuples** generated from the idiom dataset:
  - `(Devanagari text, Romanized text)`
  - `(Romanized text, English meaning)`
  - `(Devanagari text, Marathi meaning)`
  - `(Corrupted/noisy spelling, Canonical Devanagari text)`
- **Validation Corpus:** **1,040 held-out pairs** used strictly for evaluation during training.
- **Hyperparameters:**
  - Optimizer: AdamW (weight decay 0.01)
  - Learning Rate: $2 \times 10^{-5}$ with warm-up
  - Batch Size: 16
  - Epochs: 3
  - Hardware: CUDA-enabled GPU (training time: approximately 2.1 hours).
- **Produced Checkpoint:** Saved locally at `models/konkan-vani-encoder-v1/`.

---

### 5.4 Model Registry & In-Memory Singleton
Implemented in `server-nlp/app/core/model_registry.py`:
- Employs a thread-safe double-checked locking singleton (`get_model()`).
- On server boot, checks if `models/konkan-vani-encoder-v1/` exists locally:
  - If **present**: Loads the custom fine-tuned weights into GPU/CPU memory.
  - If **absent**: Automatically falls back to downloading `sentence-transformers/LaBSE` from Hugging Face, ensuring a fresh repository clone boots without crashing.
- Shares the loaded model instance across `semantic_match`, `word_analysis`, and `compose` modules, preventing redundant VRAM allocation.

---

## 6. Backend Service & REST API Specification

### 6.1 Application Structure & Middleware
Implemented in `server-nlp/app/main.py`:
- Framework: **FastAPI** (Python 3.12+).
- Web Server: **Uvicorn** running ASGI event loops.
- CORS Configuration: Fully enabled for Next.js development and production domains (`localhost:3000`, `127.0.0.1:3000`, and wildcards).

---

### 6.2 In-Memory Store & FAISS Integration (`data_loader.py`)
At startup, `IdiomStore` loads and validates all primary data assets:
1. Loads `data/processed/idioms_with_phonetic_keys.csv` into a Pandas DataFrame.
2. Loads `data/idiom_embeddings.npy` into memory.
3. **Integrity Assertion:** Executes `assert len(self.df) == len(self.embeddings)`, instantly throwing an error if the CSV row count and vector matrix count diverge (preventing index misalignment).
4. Loads or builds `faiss.IndexFlatIP(768)` containing all 5,200 unit-normalized vectors.
5. Caches lowercase Roman strings and raw Devanagari lists to accelerate substring filters during multi-result search without Pandas overhead.

---

### 6.3 Complete API Endpoint Catalog

#### 1. Primary Idiom Resolution Endpoint
```http
POST /api/resolve
Content-Type: application/json
```
**Request Body:**
```json
{
  "text": "Hatache kaknak aanso kityak"
}
```
**Response Body (Verified Match):**
```json
{
  "matched": true,
  "match_type": "phonetic",
  "confidence": 1.0,
  "provisional": false,
  "konkani_text": "हाताचे काकणाक आरसो कित्याक",
  "romanized_text": "Hatache kaknak aanso kityak",
  "literal_meaning": "Why does one need a mirror to see the bangles on one's own hand?",
  "figurative_meaning": "What is obvious and directly visible requires no further proof.",
  "english_meaning": "The obvious needs no proof; self-evident truth.",
  "marathi_meaning": "प्रत्यक्ष दिसणाऱ्या गोष्टीला पुराव्याची गरज नसते.",
  "example_sentence": "तो चोरी करताना पकडला गेला, आता हाताचे काकणाक आरसो कित्याक?",
  "category": "wisdom",
  "cultural_context": "Commonly spoken in coastal household disputes when evidence is clear.",
  "word_analysis": [
    {
      "word": "hatache",
      "role_hint": "noun",
      "best_matches": [{"devanagari": "हात", "english": "hand", "method": "suffix"}]
    }
  ]
}
```

**Response Body (Stage 4 Compositional Fallback):**
```json
{
  "matched": false,
  "match_type": "compositional",
  "confidence": 0.52,
  "provisional": true,
  "literal_meaning": "Provisional reading: water, flow and things that cannot be held back",
  "figurative_meaning": "Derived word-by-word interpretation based on recognized Konkani roots.",
  "coverage": {
    "total_words": 4,
    "recognized_words": 3,
    "ratio": 0.75
  },
  "themes": ["water", "motion"],
  "word_glosses": [
    {
      "token": "udkant",
      "devanagari": "उदक",
      "english": "water",
      "marathi": "पाणी",
      "method": "suffix",
      "known": true
    }
  ],
  "closest_known": [
    {
      "konkani_text": "उदकांत चोरो सुक्यान मारो",
      "english_meaning": "Trouble in water, punishment on dry land.",
      "similarity": 0.584
    }
  ]
}
```

---

#### 2. Multi-Result Hybrid Search Endpoint
```http
GET /api/search?q={query}&category={cat}&script={script}&limit={limit}
```
- Performs a hybrid scan: retrieves candidates matching phonetic substring / fuzzy criteria, combines them with top FAISS semantic neighbors, deduplicates records, and returns an ordered list of `SearchResultOut` objects.

#### 3. Corpus Exploration Endpoint
```http
GET /api/browse?page=1&limit=20&category=wisdom&script=devanagari
```
- Paginated table browser supporting category filtering, script selection, and total record counting.

#### 4. Diagnostic & Health Endpoints
- `GET /health`: Returns server status, memory utilization, FAISS vector count, and model checkpoint name.
- `GET /api/stats`: Returns dataset totals, curated vs pattern count, and category distribution.
- `GET /api/categories`: Returns an array of the 15 available taxonomy categories.
- `GET /api/idioms/{id}`: Returns the full record for a specific idiom ID.
- `POST /api/contribute`: Validates and accepts community proverb submissions.

---

## 7. Frontend Web Application Architecture

### 7.1 Tech Stack & Design System
Located in `client/`:
- **Framework:** **Next.js 14+** (React 18, App Router architecture).
- **Language:** **TypeScript** with strict type-checking.
- **Styling:** **Tailwind CSS** with a warm Konkan-inspired earth/coastal color palette (`amber`, `emerald`, `slate`).
- **Icons:** `lucide-react`.

---

### 7.2 Routing & Page Architecture
- **`app/page.tsx` (Search Hub):**
  Features a hero search bar with auto-script detection, quick sample query chips (in Devanagari, Romi, and English), category shortcuts, and real-time query routing.
- **`app/results/page.tsx` (Resolution View):**
  Parses query parameters and renders either an `IdiomCard` (for verified matches) or a `ProvisionalMeaning` component (for uncataloged compositional readings).
- **`app/explore/page.tsx` (Corpus Browser):**
  A full catalog exploration interface with category filter pills, script toggles, and paginated proverb grids.
- **`app/contribute/page.tsx` (Community Portal):**
  Structured multi-field form allowing native speakers to contribute new proverbs, meanings, and cultural context.

---

### 7.3 Specialized Component Specifications

#### 1. `IdiomCard.tsx`
Renders verified idioms across all matching stages:
- Large canonical Devanagari heading alongside Romanized subtitle.
- **Confidence Badge:** Color-coded indicator displaying match methodology (`Phonetic Match`, `Semantic Match`, `Exact Match`) and percentage score.
- **Meaning Matrix:** Distinct, well-spaced visual blocks for:
  - *Literal Meaning* (word-for-word translation)
  - *Figurative Meaning* (underlying cultural wisdom)
  - *English Meaning* & *Marathi Meaning*
- **Usage Example:** Stylized quotation block showing contextual usage in Konkani.
- **Cultural Context & Provenance:** Explains when and why the proverb is spoken, alongside a source badge (`Curated` vs `Pattern-Generated`).

#### 2. `ProvisionalMeaning.tsx`
Renders Stage 4 compositional fallback results with complete scientific transparency:
- Displays a warning banner indicating that the proverb is not yet documented in the verified corpus.
- **Word Breakdown Grid:** Renders each token as a chip indicating its root word, part of speech, English/Marathi gloss, and how it was identified (`exact`, `suffix`, `phonetic`, `fuzzy`, or `embedding`).
- **Deducted Theme:** Displays inferred imagery (e.g., *"Themes: water, flow and things that cannot be held back"*).
- **Coverage Progress Bar:** Shows the percentage of words successfully recognized.
- **Closest Known Proverbs:** Presents mini-cards of structurally or thematically similar idioms from the dataset.

#### 3. `ScriptToggle.tsx`
A persistent toggle component enabling instant switching of card displays between **Devanagari**, **Romi (Latin)**, and **Kannada** orthographies.

---

### 7.4 Client-Side API Integration & Type Definitions
Implemented in `client/lib/api.ts` and `types.ts`:
- Type-safe async functions: `resolveIdiom()`, `searchIdioms()`, `browseIdioms()`, `getStats()`.
- **Fault-Tolerant Client Fallback:** If the Python NLP backend is offline or unreachable during development, `api.ts` gracefully degrades to a client-side mock search over local JSON fixtures, preventing application crashes.

---

## 8. Empirical Evaluation & Measured Benchmark Results

All figures reported in this section were generated through reproducible automated benchmarking scripts (`src/run_full_benchmark.py` and `server-nlp/scripts/train_model.py`) executed on a real CUDA machine (Windows 11, PyTorch, Python 3.12).

---

### 8.1 Effect of Encoder Fine-Tuning
Evaluated on **1,040 held-out test pairs** comparing base `sentence-transformers/LaBSE` against the fine-tuned `konkan-vani-encoder-v1`:

$$\mathrm{Recall@}K = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \mathbb{I}(\mathrm{rank}_i \le K)$$

$$\mathrm{MRR@}K = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \rho_i, \quad \text{where } \rho_i = \begin{cases} \frac{1}{\mathrm{rank}_i}, & \text{if } \mathrm{rank}_i \le K \\ 0, & \text{otherwise} \end{cases}$$

| Metric | Generic LaBSE Baseline | Fine-Tuned Encoder (`v1`) | Relative Gain |
| :--- | :---: | :---: | :---: |
| **Recall@1** | 51.35% | **99.81%** | **+94.4%** |
| **Recall@5** | 78.75% | **99.90%** | **+26.9%** |
| **Recall@10** | 86.35% | **99.90%** | **+15.7%** |
| **MRR@10** | 62.75% | **99.84%** | **+59.1%** |
| **Avg Positive-Pair Cosine** | 0.4368 | **0.9404** | **+115.3%** |

*Analysis:* Fine-tuning with Multiple Negatives Ranking Loss nearly doubled Recall@1 and boosted average positive-pair cosine similarity from 0.44 to 0.94, demonstrating an exceptionally compact, semantically aligned metric space.

---

### 8.2 Stage-Wise Top-1 Retrieval Accuracy
Evaluated over 200 sampled dataset entries under both clean conditions and deliberately corrupted spellings (doubled/shortened vowels, `v/w` swaps, `ph/f` substitutions):

| Stage & Input Condition | Top-1 Accuracy | Mean Similarity Score |
| :--- | :---: | :---: |
| **Fuzzy** — Exact Romanized | **100.0%** | 1.000 |
| **Fuzzy** — Corrupted Spelling | **100.0%** | 0.987 |
| **Semantic** — Exact Romanized | **100.0%** | 0.904 |
| **Semantic** — Corrupted Spelling | **99.0%** | 0.879 |

---

### 8.3 End-to-End Pipeline Accuracy & Out-of-Domain Rejection
Tested across 300 in-domain queries and 8 unrelated out-of-domain negative queries (English and Marathi text unrelated to idioms):

| Test Condition | Sample Size | Accuracy / Rejection Rate | False Positives |
| :--- | :---: | :---: | :---: |
| **Romanized Query Resolution** | 150 | **100.0%** (150/150) | 0 |
| **Devanagari Query Resolution** | 150 | **100.0%** (150/150) | 0 |
| **Out-of-Domain Rejection** | 8 | **100.0%** (8/8 rejected) | **0** |

---

### 8.4 Threshold Sensitivity & Score Separation
The semantic threshold $\theta = 0.63$ was chosen based on an empirical score distribution analysis:
- In-domain query cosine scores: range from **0.773 to 0.998**.
- Unrelated out-of-domain query cosine scores: never exceeded **0.513**.
- **Separation Gap:** $[0.513, 0.773]$. Setting $\theta = 0.63$ places the decision boundary directly in this gap, providing maximum tolerance against noise while ensuring 0 false positives.

| Cosine Threshold | In-Domain Accuracy | Out-of-Domain Rejection Rate |
| :---: | :---: | :---: |
| 0.50 | 100.0% | 87.5% |
| 0.55 | 100.0% | 100.0% |
| 0.60 | 100.0% | 100.0% |
| **0.63 (Configured)** | **100.0%** | **100.0%** |
| 0.70 | 100.0% | 100.0% |
| 0.80 | 100.0% | 100.0% |

---

### 8.5 System Latency Benchmarks
Measured on a standard workstation with CUDA acceleration ($n=60$ warm queries per operation):

| Operation | Mean Latency | Median Latency | 95th Percentile (p95) | 99th Percentile (p99) |
| :--- | :---: | :---: | :---: | :---: |
| **Phonetic Fuzzy Match (Stage 1)** | **2.94 ms** | 2.88 ms | 5.06 ms | 6.55 ms |
| **Semantic Match (Stage 2: Encode + FAISS)**| **25.64 ms** | 30.36 ms | 34.23 ms | 34.80 ms |
| **Full `resolve_idiom()` Pipeline** | **35.28 ms** | 39.32 ms | **46.98 ms** | 49.29 ms |
| **Multi-Result Search (Top-10 Rank)** | 170.52 ms | 165.59 ms | 205.54 ms | 252.83 ms |

*Finding:* Over 95% of full query resolutions complete in under **50 ms**, well within interactive web UI thresholds.

---

## 9. Operational Guide: Deployment & Reproduction

### 9.1 Prerequisites & System Dependencies
- **OS:** Windows 10/11, Ubuntu 20.04+, or macOS
- **Python:** Version 3.10, 3.11, or 3.12
- **Node.js:** Version 18.x or 20.x with `npm`
- **Hardware:** Modern multi-core CPU (CUDA-capable GPU strongly recommended for neural encoding and model retraining, though CPU inference is supported).

---

### 9.2 Step-by-Step Installation & Service Startup

#### Step 1: Environment Setup & Backend Installation
```bash
# 1. Clone the repository
git clone https://github.com/your-username/Konkan-Vani.git
cd Konkan-Vani

# 2. Create and activate a Python virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install Python dependencies
pip install --upgrade pip
pip install -r server-nlp/requirements.txt
```

#### Step 2: Start the FastAPI NLP Backend
```bash
cd server-nlp
python run_server.py
```
*The server will start at `http://localhost:8000`. You can verify by opening `http://localhost:8000/docs` in your browser to inspect interactive Swagger documentation.*

#### Step 3: Start the Next.js Web Client
Open a second terminal window:
```bash
cd Konkan-Vani/client
npm install
npm run dev
```
*The web client will compile and serve at `http://localhost:3000`.*

---

### 9.3 Reproducing Embeddings, Indices, and Benchmarks

#### Rebuilding Dataset and Validating Integrity:
```bash
python src/rebuild_dataset.py
```

#### Rebuilding the FAISS Vector Index:
```bash
python server-nlp/scripts/build_embeddings.py
```

#### Retraining the Custom Neural Encoder:
```bash
python server-nlp/scripts/train_model.py
```

#### Running the Full Automated Benchmark Suite:
```bash
python src/run_full_benchmark.py
```

---

## 10. Known Limitations & Future Technical Roadmap

### 10.1 Known Limitations
1. **Curated vs. Synthetic Imbalance:** While the total corpus contains 5,200 idioms, 5,095 (98%) are deterministic pattern-generated frames, and 105 (2%) are manually verified folk idioms. The high benchmark scores partly reflect the structured nature of the synthetic subspace.
2. **Negative Evaluation Set Size:** Out-of-domain rejection was benchmarked on a small suite of 8 handcrafted negative queries; larger statistical benchmarks across general conversational corpora are planned.
3. **Hardware Dependence:** Sub-50 ms latency metrics require GPU acceleration (CUDA). CPU-only environments experience inference times of approximately 120–180 ms for the neural encoding step.

### 10.2 Future Technical Roadmap
- **Community Curation Dashboard:** Expand the `/contribute` flow into a peer-reviewed approval workflow where fluent Konkani speakers can review, score, and authenticate synthetic idioms.
- **Audio Pronunciation (Text-to-Speech):** Integrate a Kokani TTS voice model to provide authentic native pronunciation audio for each idiom.
- **IndicBERT / MuRIL Comparative Benchmark:** Evaluate other Indic-focused transformer architectures (e.g., AI4Bharat IndicBERT, Google MuRIL) to compare cross-dialectal Kannada-script representations against LaBSE.
- **Hierarchical Graph Indexing (HNSW):** As the corpus scales beyond 50,000 idioms, transition FAISS from `IndexFlatIP` to `IndexHNSWFlat` to maintain sub-10 ms nearest-neighbor search times.

---
*End of Implementation Document.*
