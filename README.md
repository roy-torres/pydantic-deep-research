# Deep Research Agent

A multi-agent research system built with **Pydantic AI**, **Gradio**, and **DuckDuckGo**.

Give it a free-text query or a stock ticker (for example `LTH` or `TSLA`). It resolves the topic, runs a discovery search, generates 3–4 research angles, fetches sources in parallel, extracts cited facts, and synthesizes a structured Markdown report.

## How it works

Four specialized Pydantic AI agents run in sequence:

1. **Query analyzer** — Detects a ticker vs a general query, resolves the company or topic, and builds a discovery search query.
2. **Angle generator** — Reads the first DuckDuckGo results and produces 3–4 non-overlapping follow-up searches (for example SWOT, last-12-month performance, competition, latest earnings). Financial and news angles prefer primary sources (`site:sec.gov`) and reputable outlets.
3. **Fact extractor** — Fetches page HTML where possible, extracts claims/numbers/risks, and keeps the source title and URL on each fact.
4. **Synthesizer** — Writes one report with an executive summary, findings per angle, evidence bullets with citations, risks/conflicts, and a “what to watch next” list.

Progress streams into the Gradio chat as each step finishes.

## Prerequisites

- Python 3.9+
- An OpenAI API key (the agents use `openai:gpt-5-mini`)

## Setup

1. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   ```

   macOS / Linux:

   ```bash
   source venv/bin/activate
   ```

   Windows:

   ```bash
   venv\Scripts\activate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**

   Put your key in `.env`:

   ```env
   OPENAI_API_KEY=your_actual_api_key_here
   PYDANTIC_AI_NO_BANNER=1
   ```

4. **Run the app**

   ```bash
   python main.py
   ```

5. **Open the UI**

   Use the local URL printed in the terminal (usually `http://127.0.0.1:7860`). Gradio will pick the next free port if `7860` is already in use.

   After you change `main.py`, stop the server with `Ctrl+C` and run `python main.py` again. A browser refresh alone does not reload Python code.

## Example queries

- `LTH`
- `TSLA`
- `What is the current state of Solid State Batteries?`
