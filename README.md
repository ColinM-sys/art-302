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

## Quick Start

```bash
# Clone
git clone https://github.com/cmcdo/art-302.git
cd art-302

# Install
pip install -r requirements.txt

# Set your NVIDIA API key
echo "NVIDIA_API_KEY=nvapi-your-key-here" > .env

# Run
python app.py
```

Then open http://localhost:7860

## Features

- **Art Map Discovery** — Search any US city for murals, street art, sculptures, galleries, installations
- **Multi-Agent Pipeline** — Nano 9B routes, Super 49B extracts structured data
- **Interactive Folium Map** — Color-coded markers with popups showing artist, type, description
- **Live Web RAG** — No pre-built dataset needed; data comes from the web in real-time
- **Chat** — Direct conversation with either Nemotron model
- **Dashboard** — Session analytics and architecture overview

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

## Built For

**The Shortest Hackathon @ GTC 2026**
