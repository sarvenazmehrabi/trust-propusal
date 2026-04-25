# Modeling Trust in AI Agents (PhD Dissertation Research)

This repository contains the engineering framework for a PhD dissertation focused on modeling user trust in interactive intelligent systems (AI Chatbots).

## Project Structure
- `codes/`: Main source code directory.
  - `data_ingestion/`: Modules for fetching and storing raw data.
  - `ctenv/`: Python virtual environment.
- `data/`: (Local only) Storage for CSV and JSON outputs.

## Technology Stack
- **Language:** Python 3.10+
- **Data Scraping:** `google-play-scraper`
- **Data Analysis:** `pandas`, `numpy`
- **Environment Management:** `venv`
- **Version Control:** `Git` & `GitHub`

## Features
- **Human-like Ingestion:** Implements random delays and dynamic batching to prevent IP blacklisting.
- **Incremental Scraping:** Checkpoint-based system that resumes from the last captured date.
- **Data Normalization:** Splits timestamps into granular `Date` and `Time` features for temporal analysis.
- **Multi-Agent Support:** Capable of scraping data for ChatGPT, Gemini, Copilot, DeepSeek, and Grok simultaneously.

## How to Run
1. Activate the environment: `source ctenv/bin/activate` (Linux) or `ctenv\Scripts\activate` (Windows).
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the scraper: `python codes/data_ingestion/data_extraction.py`.