# Art 302 — We'll Take You to the Art

AI-powered local art discovery platform that uses **NVIDIA Nemotron** models to find murals, street art, sculptures, and local artists in any city — then plots them on an interactive map.

> HTTP 302 = Redirect. We redirect you to the art.

## How It Works

```
City + Art Types
    → Nemotron Nano 9B (Router Agent) plans search queries
    → DuckDuckGo + BeautifulSoup scrape the web
    → Nemotron Super 49B (Extractor Agent) extracts structured art data
    → ArcGIS geocodes addresses to coordinates
    → Folium renders an interactive map with color-coded markers
```

## Features

- **Art Map Discovery** — Search any US city for murals, street art, sculptures, galleries, installations
- **Multi-Agent Pipeline** — Nano 9B routes queries, Super 49B extracts structured data with JSON parsing and regex fallback
- **Interactive Folium Map** — Color-coded markers by art type with popups showing artist, type, description, and address
- **Live Web RAG** — No pre-built dataset needed; data comes from the web in real-time via DuckDuckGo search and scraping
- **Research Pipeline** — Ask any question and the agent searches the web, scrapes sources, and synthesizes an answer with citations
- **Comparison Pipeline** — Compare two topics side-by-side with structured analysis
- **Data Analysis** — Upload CSV/text files and get AI-powered analysis with the Data Analyst agent
- **Direct Chat** — Conversation with either Nemotron Nano 9B or Super 49B model
- **Image Search** — Finds relevant images for discovered art locations
- **Session Dashboard** — Real-time analytics: queries run, pages scraped, tokens used, response times, query history
- **Concurrent Scraping** — ThreadPoolExecutor for parallel web scraping across multiple sources

## Multi-Agent Architecture

| Agent | Model | Role |
|-------|-------|------|
| Router Agent | Nemotron Nano 9B | Plans search queries and coordinates the pipeline |
| Art Extractor Agent | Nemotron Super 49B | Extracts structured art data (name, artist, type, address) from scraped text |
| Synthesizer Agent | Nemotron Super 49B | Combines research from multiple sources into coherent answers |
| Data Analyst Agent | Nemotron Super 49B | Analyzes uploaded datasets and generates insights |
| Comparison Agent | Nemotron Super 49B | Structured side-by-side comparison of topics |

## Tech Stack

| Component | Technology |
|-----------|------------|
| Router Agent | `nvidia/nvidia-nemotron-nano-9b-v2` |
| Extractor Agent | `nvidia/llama-3.3-nemotron-super-49b-v1.5` |
| Web Search | DuckDuckGo |
| Scraping | BeautifulSoup4 |
| Maps | Folium + OpenStreetMap |
| Geocoding | ArcGIS |
| API | NVIDIA NIM (OpenAI-compatible) |
| Frontend | Gradio |

## Quick Start

```bash
# Clone
git clone https://github.com/ColinM-sys/art-302.git
cd art-302

# Install
pip install -r requirements.txt

# Set your NVIDIA API key
echo "NVIDIA_API_KEY=nvapi-your-key-here" > .env

# Run
python app.py
```

Then open http://localhost:7860

## UI Tabs

- **Art Map** — Enter a city and select art types, get an interactive map with discovered art
- **Research** — Ask any question, get a sourced answer from web research
- **Compare** — Side-by-side comparison of two topics
- **Data Analysis** — Upload files for AI analysis
- **Chat** — Direct conversation with Nemotron models
- **Dashboard** — Session metrics and architecture overview

## Example Queries

**Art Discovery:**
- Search "Austin" for murals, street art, sculptures
- Search "Portland" for galleries, installations

**Research:**
- "What is the history of street art in New York City?"
- "Best public art installations in the world"

**Comparison:**
- "Banksy vs Shepard Fairey"
- "Street art vs graffiti"

## Built For

**The Shortest Vibehack @ GTC 2026**
