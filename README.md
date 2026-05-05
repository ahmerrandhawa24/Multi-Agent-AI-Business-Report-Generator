# Multi-Agent AI Business Report Generator

🌐 **Live Demo:** [Click here to open app](https://multi-agent-ai-business-report-generator-sh6zrfxkmen9ncpbiwh8p.streamlit.app/)


## Overview

This project is an **advanced multi-agent AI system** that generates complete, professional business reports from a simple user query.

It combines:

* Large Language Models (LLMs)
* Multi-agent collaboration
* Data Engineering (ETL)
* Memory systems

The system is built **100% using free tools**

---

## Project Phases

### Phase 1 — Foundations

* Concepts: Agents, Prompt Engineering, Tool Usage, Memory Types
* Setup using Python + Groq API

---

### Phase 2 — Basic Multi-Agent Pipeline

* Implemented core workflow:

  * Research Agent
  * Analyst Agent
  * Writer Agent
* Agents pass outputs sequentially

---

### Phase 3 — Manager Agent (Planner)

* Introduced dynamic orchestration
* Manager agent assigns tasks intelligently
* Removed hardcoded pipeline

---

### 🧰 Phase 4 — Tool Integration

Agents enhanced with real-world tools:

* Web Search (DuckDuckGo)
* Calculator
* File Reader (PDF + CSV using PyMuPDF)

---

### Phase 5 — Memory System

* Short-term memory (session-based)
* Long-term memory using vector database (FAISS)

---

### Phase 6 — Data Agent + ETL Automation

* Integrated structured data processing

* Load and clean datasets (CSV/Excel) using Pandas

* Data Agent generates:

  * Trends
  * Insights
  * Statistics

* Automated execution using Windows Task Scheduler

---

### Phase 7 — Streamlit UI

* Interactive web interface
* Features:

  * User input box
  * Live agent step tracking
  * Report generation
  * PDF download

---

### Phase 8 — Evaluation & Collaboration Loop

* Agents review each other’s outputs
* Weak results trigger re-processing
* Evaluation agent scores final report quality

---

### Phase 9 — Advanced Features

* Multi-modal input:

  * PDF
  * CSV
  * Images

* Auto-generated reports include:

  * Executive Summary
  * Charts
  * Recommendations

---

## How It Works

1. User enters a query
   Example:

   ```
   Analyze smartphone market in Pakistan
   ```

2. Manager Agent plans execution

3. System runs:

   * Research Agent → gathers data
   * Data Agent → analyzes dataset
   * Analyst Agent → extracts insights
   * Writer Agent → generates report

4. Evaluation Agent checks quality

5. Final output:
   Professional business report

---

## Tech Stack

* Python 3.11
* CrewAI (Multi-Agent Framework)
* Groq API (LLM)
* Streamlit (UI)
* Pandas (Data Processing / ETL)
* FAISS (Vector Database - local use)
* DuckDuckGo Search
* PyMuPDF

---

## Installation

```bash
# Create virtual environment
py -3.11 -m venv venv

# Activate
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---



* FAISS is used for **local development**
* In some cloud environments (e.g., Streamlit Cloud), FAISS may not be supported

---

## Final Output

A complete **Multi-Agent AI System** that:

* Understands user queries
* Uses real-world data
* Performs analysis
* Generates professional business reports

Built entirely using **free tools and frameworks**

---


## Highlights

* Multi-Agent AI Architecture
* LLM + Data Engineering Integration
* Real-world Business Intelligence System
* End-to-End AI Application
