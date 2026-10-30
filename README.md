# AGENT-KGRAPH: Self-Evolving Agentic Knowledge-Graph Curation with Hybrid KG–Vector Retrieval

AGENT-KGRAPH is a modular question-answering research project that combines Knowledge Graph (KG) retrieval, dense vector retrieval, agentic query routing, evidence-grounded answer generation, and self-evolving knowledge graph curation.

The system aims to improve knowledge-intensive question answering by combining structured relationships represented in a Knowledge Graph with semantically relevant textual evidence retrieved through vector search.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [System Architecture](#system-architecture)
4. [Project Roadmap](#project-roadmap)
5. [Technologies Used](#technologies-used)
6. [Prerequisites](#prerequisites)
7. [Installation and Setup](#installation-and-setup)
8. [Configuration](#configuration)
9. [Dataset Preparation](#dataset-preparation)
10. [Running the Project](#running-the-project)
11. [Running Tests](#running-tests)
12. [Evaluation and Results](#evaluation-and-results)
13. [Expected Outputs](#expected-outputs)
14. [Troubleshooting](#troubleshooting)
15. [Limitations and Future Scope](#limitations-and-future-scope)

---

## 1. Project Overview

Traditional question-answering systems often rely on either structured knowledge retrieval or semantic document retrieval. Knowledge Graphs represent entities and their relationships explicitly, making them useful for multi-hop reasoning. Vector retrieval identifies semantically relevant passages even when the wording differs from the query.

AGENT-KGRAPH integrates these complementary approaches into an agentic question-answering pipeline.

The system consists of the following major components:

- Dataset preprocessing and integration.
- Knowledge Graph construction using subject–relation–object triples.
- Extractor, Verifier, and Merger agents.
- Dense vector retrieval using passage embeddings.
- Knowledge Graph retrieval and multi-hop path exploration.
- Agentic routing between vector, KG, and hybrid retrieval.
- LLM-based answer generation with evidence provenance.
- Self-evolution of the Knowledge Graph through verified fact updates.
- Evaluation of answer quality, retrieval performance, and system efficiency.

## 2. Key Features

### 2.1 Knowledge Graph Construction

- Extracts subject–relation–object triples from available information.
- Verifies candidate facts against their source documents.
- Resolves duplicate entities and merges equivalent information.
- Constructs and maintains a structured Knowledge Graph.

### 2.2 Agent-Based Knowledge Curation

The system implements three major agents:

**Extractor Agent**
- Extracts candidate facts and relationships from source information.
- Converts extracted information into structured triples.

**Verifier Agent**
- Checks candidate facts against their supporting source documents.
- Evaluates whether extracted information is sufficiently supported.

**Merger Agent**
- Resolves duplicate or equivalent entities.
- Merges verified facts into the Knowledge Graph.

### 2.3 Hybrid Retrieval

AGENT-KGRAPH supports three retrieval strategies:

- **Vector Retrieval:** Retrieves semantically relevant passages using dense embeddings.
- **KG Retrieval:** Retrieves facts and relationships through graph traversal and multi-hop paths.
- **Hybrid Retrieval:** Combines graph-based and vector-based evidence.

### 2.4 Agentic Query Routing

The routing component determines which retrieval strategy should be used for a question.

Supported retrieval actions:

- `vector`
- `kg`
- `hybrid`

The router uses query-related features and its configured decision policy to select a retrieval action.

### 2.5 Evidence-Based Answer Generation

- Uses a configured Large Language Model for answer synthesis.
- Generates answers using retrieved evidence.
- Supports passage identifiers and Knowledge Graph path provenance.
- Provides an integrated question-answering pipeline.

### 2.6 Self-Evolving Knowledge Graph

The self-evolution module supports:

- Candidate fact extraction.
- Confidence-based filtering.
- Fact verification.
- Entity resolution and merging.
- Knowledge Graph updates.
- Graph decay and pruning operations.

New facts are processed through verification before being accepted into the graph.

## 3. System Architecture

The overall AGENT-KGRAPH workflow is:

1. Dataset preprocessing.
2. Document and passage preparation.
3. Candidate triple extraction.
4. Triple verification.
5. Entity resolution and merging.
6. Knowledge Graph construction.
7. Vector embedding generation and indexing.
8. Query processing and agentic routing.
9. Vector, KG, or hybrid retrieval.
10. Evidence-based answer synthesis.
11. Provenance generation.
12. Self-evolution and Knowledge Graph maintenance.

### High-Level Workflow

```text
             Input Dataset
                   |
                   v
          Dataset Preprocessing
                   |
          +--------+--------+
          |                 |
          v                 v
    Document Corpus    Triple Extraction
          |                 |
          v                 v
    Vector Embeddings  Triple Verification
          |                 |
          v                 v
    Vector Index      Entity Resolution
          |                 |
          |                 v
          |           Knowledge Graph
          |                 |
          +--------+--------+
                   |
                   v
             User Query
                   |
                   v
             Query Router
                   |
          +--------+--------+
          |        |        |
          v        v        v
        Vector    KG     Hybrid
       Retrieval Retrieval Retrieval
          |        |        |
          +--------+--------+
                   |
                   v
            Evidence Fusion
                   |
                   v
            LLM Answer Generation
                   |
                   v
           Answer + Provenance
                   |
                   v
          Self-Evolution Workflow
                   |
                   v
          Updated Knowledge Graph
```

## 4. Project Roadmap

The project was developed and organized into nine implementation batches.

| Batch | Description | Status |
|---|---|---|
| Batch 1 | Foundation and project infrastructure | Completed |
| Batch 2 | Knowledge Graph construction | Completed |
| Batch 3 | Retrieval system | Completed |
| Batch 4 | Agentic QA and query routing | Completed |
| Batch 5 | Self-Evolution workflow | Completed |
| Batch 6 | Dataset Integration | Completed |
| Batch 7 | Evaluation and Metrics | Completed |
| Batch 8 | Real-Model Integration and Experiments | Completed |
| Batch 9 | Final Review and Project Consolidation | Completed |

All nine batches have been implemented as part of the AGENT-KGRAPH project. The experimental results and limitations are discussed in the project report.

## 5. Technologies Used

The project uses the following technologies and libraries. Exact versions should be taken from the repository's dependency files.

| Category | Technology |
|---|---|
| Programming Language | Python |
| Knowledge Graph | NetworkX |
| Vector Embeddings | BGE-small |
| Language Models | Qwen-family models |
| Retrieval | Vector, KG, and Hybrid Retrieval |
| Agent Architecture | Extractor, Verifier, Merger, and Router |
| Testing | pytest |
| Data Processing | Python data-processing libraries |

Additional libraries and dependencies are listed in the project's dependency manifest.

## 6. Prerequisites

Before running the project, ensure that the following are installed:

- Git
- Python version compatible with the project dependencies
- pip
- Python virtual environment support
- Sufficient storage for datasets, model weights, graph files, and vector indexes
- Internet connectivity if model weights or permitted datasets need to be downloaded

A compatible GPU may improve model inference performance, depending on the configured model and hardware requirements.

## 7. Installation and Setup

### Step 1: Clone the Repository

Replace the placeholder with the actual repository URL.

```bash
git clone <https://github.com/Devanshu90/AGENT-KGRAPH>
cd AGENT-KGRAPH
```

### Step 2: Create a Virtual Environment

**Windows PowerShell**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Upgrade pip

```bash
python -m pip install --upgrade pip
```

### Step 4: Install Dependencies

If the repository contains a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

If the repository uses `pyproject.toml`, follow its installation instructions instead.

### Step 5: Verify Installation

```bash
python --version
pip --version
```

Ensure that the virtual environment is activated before running the project.

## 8. Configuration

Before execution, verify the project configuration.

The configuration may include:

- Dataset input and output paths.
- Processed document and passage directories.
- Knowledge Graph storage location.
- Vector index location.
- Embedding model identifier.
- Language model identifier.
- Retrieval mode.
- Number of retrieved passages.
- Graph traversal depth.
- Confidence and verification thresholds.
- Model device and precision settings.
- Evaluation output directory.

Use the configuration files already provided in the repository.

Do not commit API keys, passwords, authentication tokens, or other confidential information.

## 9. Dataset Preparation

### MetaQA

MetaQA is used for the project's question-answering benchmark.

The existing preprocessing and Knowledge Graph import artifacts are used by the pipeline.

The dataset preparation workflow includes:

- Question and answer preparation.
- Passage and identifier mapping.
- Knowledge Graph data preparation.
- Question-hop categorization.
- Dataset validation.

The recorded benchmark uses 102 questions:

- 34 one-hop questions.
- 34 two-hop questions.
- 34 three-hop questions.

This represents a sampled evaluation rather than a full-dataset evaluation.

### HotpotQA and WebQSP

Where supported by the repository implementation, these datasets are handled through dataset-specific preprocessing and mapping.

The integration workflow must preserve consistent question, answer, passage, and entity identifiers.

Follow the official dataset access and licensing requirements before downloading or redistributing dataset files.

## 10. Running the Project

The exact execution commands depend on the entry-point scripts available in the repository.

First inspect the repository structure:

```bash
ls
```

On Windows PowerShell:

```powershell
Get-ChildItem
```

Identify the actual scripts for preprocessing, Knowledge Graph construction, question answering, API execution, and evaluation.

### 10.1 Dataset Preprocessing

Use the preprocessing entry point provided in the repository.

```bash
python <preprocessing_script>.py --help
```

Refer to the script's supported arguments before executing it.

### 10.2 Knowledge Graph Construction

Run the Knowledge Graph construction or import script available in the repository.

```bash
python <kg_script>.py --help
```

### 10.3 Question Answering

Run the integrated question-answering entry point.

```bash
python <qa_entry_point>.py --help
```

The configured retrieval strategy can use:

- Vector retrieval.
- Knowledge Graph retrieval.
- Hybrid retrieval.

The agentic router can select the retrieval action according to its implementation and configuration.

### 10.4 API Execution

If the repository includes a FastAPI application with an ASGI object named `app` in `api_server.py`, the typical local startup command is:

```bash
uvicorn api_server:app --host 127.0.0.1 --port 8000
```

Verify the actual module and application name in the repository before using this command.

### 10.5 Running Evaluation

Use the evaluation script provided by the repository.

```bash
python <evaluation_script>.py --help
```

The evaluation workflow should use consistent test questions, model settings, answer normalization, and latency definitions when comparing retrieval methods.

## 11. Running Tests

The project report records a successful automated test run with:

**71 tests passed.**

If the repository uses pytest, execute:

```bash
pytest
```

For detailed output:

```bash
pytest -v
```

A passing test suite verifies the tested software functionality. It should be distinguished from experimental question-answering accuracy and comparative benchmark results.

## 12. Evaluation and Results

AGENT-KGRAPH was evaluated using a sampled MetaQA benchmark with a real language model for answer synthesis.

### Recorded Overall Benchmark Results

| Metric | Result |
|---|---:|
| Total Questions | 102 |
| 1-Hop Questions | 34 |
| 2-Hop Questions | 34 |
| 3-Hop Questions | 34 |
| Exact Match (EM) | 74.51% |
| Token F1 | 83.54% |
| Mean Latency | 7,684.04 ms/query |
| Questions Routed to KG Retrieval | 102 |

### Evaluation Metrics

The project evaluation considers:

- Exact Match (EM).
- Token-level F1.
- Hop-wise answer performance.
- Retrieval performance.
- Knowledge Graph evolution.
- Evidence provenance.
- Response latency.
- System resource utilization.

Only metrics supported by actual experimental records should be reported as measured results.

**Note:** The benchmark above represents the recorded KG-routed MetaQA sample. It should not be interpreted as proof that AGENT-KGRAPH outperforms vector-only, hybrid, or other baseline systems. Verify hop-wise metrics and latency calculations against the saved question-level outputs before publishing final numerical comparisons.

## 13. Expected Outputs

Depending on the enabled pipeline components, execution may generate:

- Processed dataset records.
- Prepared passage corpus.
- Knowledge Graph files and snapshots.
- Extracted and verified triples.
- Vector embeddings and indexes.
- Retrieved graph paths and passages.
- Generated answers with available provenance.
- Evaluation predictions and metric summaries.
- Test execution logs.
- Knowledge Graph evolution records.

The exact output locations are determined by the repository configuration.

## 14. Troubleshooting

### Dependency Installation Errors

- Verify that the correct Python version is installed.
- Activate the virtual environment.
- Install dependencies from the repository's dependency manifest.
- Check for package-version conflicts.

### Model Loading Errors

- Verify the model identifier.
- Check network connectivity and model access permissions.
- Ensure sufficient RAM, VRAM, and disk space.
- Check the configured device and precision settings.

### Dataset File Not Found

- Verify the working directory.
- Check configured dataset paths.
- Ensure that required raw and processed files exist.
- Confirm that the pipeline is using the correct dataset mapping.

### Slow Answer Generation

- Check the selected language model.
- Inspect per-question latency records.
- Identify slow fallback or generation operations.
- Compare latency only when measurement definitions are consistent.

### Different Results from the Report

- Confirm that the same test questions are being used.
- Verify the model configuration.
- Check retrieval settings and Knowledge Graph snapshots.
- Use the same answer-normalization procedure.
- Recalculate metrics from question-level predictions.

## 15. Limitations and Future Scope

Although AGENT-KGRAPH implements an integrated agentic Knowledge Graph and hybrid retrieval architecture, several research and engineering challenges remain relevant:

- Improving multi-hop reasoning accuracy.
- Reducing response latency for complex questions.
- Improving entity resolution and graph-path selection.
- Improving answer extraction and output normalization.
- Measuring evidence quality and provenance completeness.
- Conducting controlled static-versus-evolving Knowledge Graph comparisons.
- Expanding evaluation across datasets and model configurations.
- Improving resource efficiency and scalability.

The reported experimental results should be interpreted within the scope of the datasets, configurations, and evaluation procedures actually used.

---

## Academic Project

**Project:** AGENT-KGRAPH  
**Title:** Self-Evolving Agentic Knowledge-Graph Curation with Hybrid KG–Vector Retrieval

This project was developed as an academic research and implementation project. Please cite the relevant datasets, language models, embedding models, and research publications according to their respective citation and licensing requirements.