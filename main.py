import os
import asyncio
import httpx
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from pydantic import BaseModel, Field
from pydantic_ai import Agent
import gradio as gr
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# --- Models ---
class QueryAnalysis(BaseModel):
    is_ticker: bool = Field(description="True if the input appears to be a stock ticker, False otherwise.")
    resolved_name: str = Field(description="The full company name if a ticker, otherwise the original query.")
    discover_query: str = Field(description="A good initial DuckDuckGo search query to get an overview.")

class AngleGeneration(BaseModel):
    angles: list[str] = Field(description="3 to 4 distinct, non-overlapping research angles (DuckDuckGo search queries).")

class Fact(BaseModel):
    claim: str = Field(description="A specific key fact, number, or claim.")
    source_url: str = Field(description="The URL where the fact was found.")
    source_title: str = Field(description="The title of the source page.")

class ExtractedFacts(BaseModel):
    facts: list[Fact]

# --- Agents ---
query_analyzer = Agent(
    'openai:gpt-5-mini',
    system_prompt="Analyze the user input. If it is a stock ticker, resolve it to the full company name and industry. Produce a good initial search query to get a general overview.",
    output_type=QueryAnalysis
)

angle_generator = Agent(
    'openai:gpt-5-mini',
    system_prompt=(
        "Based on the initial search results, generate 3-4 specific DuckDuckGo search queries (angles) to research further. "
        "Examples for a stock: 'SWOT analysis', 'recent financial results', 'market competition'. "
        "Make them non-overlapping and highly specific. "
        "QUALITY BAR: For financials or news, append operators like 'site:sec.gov' or target reputable news domains (e.g., 'site:bloomberg.com' or 'site:reuters.com')."
    ),
    output_type=AngleGeneration
)

fact_extractor = Agent(
    'openai:gpt-5-mini',
    system_prompt="Extract key facts, numbers, claims, and risks from the provided page content relevant to the research angle. Be concise but specific. Do not hallucinate.",
    output_type=ExtractedFacts
)

synthesizer = Agent(
    'openai:gpt-5-mini',
    system_prompt=(
        "You are an expert financial and general researcher. Synthesize the provided facts into a structured, detailed Markdown report.\n"
        "Must include:\n"
        "- Executive Summary\n"
        "- Key findings per section (angle)\n"
        "- Evidence-based bullets with clarifications (include [Source Title](URL) inline for each fact)\n"
        "- Risks/uncertainties and conflicting info\n"
        "- 'What to watch next' list\n"
        "Format beautifully in Markdown."
    )
)

# --- Helper Functions ---
def search_ddg(query: str, max_results: int = 5):
    try:
        results = DDGS().text(query, max_results=max_results)
        return list(results)
    except Exception as e:
        print(f"Search error for {query}: {e}")
        return []

async def fetch_page(url: str):
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            resp = await client.get(url)
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                return ""
            soup = BeautifulSoup(resp.text, 'html.parser')
            for script in soup(["script", "style"]):
                script.extract()
            text = soup.get_text(separator=' ', strip=True)
            return text[:15000]
    except Exception as e:
        print(f"Fetch error for {url}: {e}")
        return ""

async def extract_facts_from_url(angle: str, result: dict):
    url = result.get('href', '')
    title = result.get('title', '')
    snippet = result.get('body', '')
    
    content = await fetch_page(url)
    if len(content) < 100:
        content = snippet  # Fallback to snippet if fetch fails or is too short
        
    prompt = f"Research Angle: {angle}\nSource Title: {title}\nURL: {url}\n\nContent:\n{content}"
    
    try:
        res = await fact_extractor.run(prompt)
        facts = res.output.facts
        # Ensure url and title are strictly set
        for fact in facts:
            if not fact.source_url or fact.source_url == "string": fact.source_url = url
            if not fact.source_title or fact.source_title == "string": fact.source_title = title
        return facts
    except Exception as e:
        print(f"Extraction error for {url}: {e}")
        return []

async def process_angle(angle: str):
    search_results = search_ddg(angle, max_results=3)
    tasks = [extract_facts_from_url(angle, r) for r in search_results]
    facts_arrays = await asyncio.gather(*tasks)
    
    all_facts = []
    for f_array in facts_arrays:
        all_facts.extend(f_array)
        
    return {"angle": angle, "facts": all_facts}

# --- Main Chat Function ---
async def deep_research(message, history):
    progress = f"🔍 **Step 1: Analyzing query...**\n"
    yield progress
    
    try:
        analysis = await query_analyzer.run(f"User input: {message}")
        data = analysis.output
        progress += f"- Identified as: **{data.resolved_name}** (Ticker: {data.is_ticker})\n"
        progress += f"🔍 **Step 2: Discover Search...** (`{data.discover_query}`)\n"
        yield progress
        
        discover_results = search_ddg(data.discover_query, max_results=5)
        discover_text = "\n".join([f"- {r.get('title')}: {r.get('body')}" for r in discover_results])
        
        progress += f"🔍 **Step 3: Generating research angles...**\n"
        yield progress
        
        angles_res = await angle_generator.run(f"Query: {data.resolved_name}\n\nInitial Results:\n{discover_text}")
        angles = angles_res.output.angles
        for a in angles:
            progress += f"  - `{a}`\n"
        progress += f"🔍 **Step 4: Running parallel leap dives (fetching & extracting facts)...**\n"
        yield progress
        
        angle_tasks = [process_angle(angle) for angle in angles]
        angle_results = await asyncio.gather(*angle_tasks)
        
        progress += f"✅ **Information gathered. Synthesizing final report...**\n\n---\n\n"
        yield progress
        
        synthesis_input = f"Original Query: {message}\nResolved Entity: {data.resolved_name}\n\nExtracted Information by Angle:\n"
        for ar in angle_results:
            synthesis_input += f"\n### Angle: {ar['angle']}\n"
            for fact in ar['facts']:
                synthesis_input += f"- {fact.claim} (Source: [{fact.source_title}]({fact.source_url}))\n"
                
        final_report = await synthesizer.run(synthesis_input)
        
        report_text = final_report.output
        
        yield progress + report_text

    except Exception as e:
        yield progress + f"\n\n❌ **Error during research:** {str(e)}"

# --- Gradio Interface ---
demo = gr.ChatInterface(
    fn=deep_research,
    title="Deep Research Agent",
    description="Ask a general question or provide a stock ticker (e.g., LTH) for an in-depth, multi-step research report.",
    examples=[
        "LTH",
        "What is the current state of Solid State Batteries?",
        "TSLA"
    ]
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
