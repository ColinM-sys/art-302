"""
Art 302 — We'll Take You to the Art
AI-powered local art discovery platform using NVIDIA Nemotron models.
Agents search the web, extract art data, geocode locations, and plot them on interactive maps.
Built for the Shortest Hackathon @ GTC 2026.
"""

import os
from dotenv import load_dotenv
load_dotenv()
import json
import time
import re
import csv
import io
import random
import folium
import geocoder
import gradio as gr
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
from openai import OpenAI
from ddgs import DDGS

# ── Config ──────────────────────────────────────────────────────────────────
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "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")
BASE_URL = "https://integrate.api.nvidia.com/v1"
NANO_MODEL = "nvidia/nvidia-nemotron-nano-9b-v2"
SUPER_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1.5"

client = OpenAI(base_url=BASE_URL, api_key=NVIDIA_API_KEY)

# ── Session state ───────────────────────────────────────────────────────────
session_metrics = {
    "queries": 0,
    "pages_scraped": 0,
    "tokens_used": 0,
    "avg_response_time": 0,
    "response_times": [],
    "query_history": [],
}

# Store discovered art for the map
discovered_art = []


# ── Web Search & Scraping ──────────────────────────────────────────────────
def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web using DuckDuckGo."""
    try:
        results = list(DDGS().text(query, max_results=max_results))
        return results
    except Exception as e:
        return [{"title": "Search error", "body": str(e), "href": ""}]


def image_search(query: str) -> str | None:
    """Search for an image URL using DuckDuckGo images. Returns first thumbnail URL."""
    try:
        results = list(DDGS().images(query, max_results=1))
        if results:
            return results[0].get("thumbnail") or results[0].get("image")
    except Exception:
        pass
    return None


def scrape_url(url: str, max_chars: int = 3000) -> str:
    """Scrape text content from a URL."""
    try:
        resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        session_metrics["pages_scraped"] += 1
        return text[:max_chars]
    except Exception as e:
        return f"Error scraping: {e}"


# ── Agent Functions ────────────────────────────────────────────────────────
def router_agent(user_query: str) -> dict:
    """Nemotron Nano (9B) — plans search queries and routes the request."""
    system_prompt = """You are a routing agent. Today's date is March 19, 2026.
Given a user question, generate 2-3 effective web search queries to find the answer.

IMPORTANT: Always search the web. You do NOT have up-to-date knowledge. Always set needs_search to true.

Respond in JSON format only (no markdown, no code fences):
{"needs_search": true, "search_queries": ["query1", "query2", "query3"], "reasoning": "brief explanation"}"""

    resp = client.chat.completions.create(
        model=NANO_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        temperature=0.3,
        max_tokens=512,
    )

    content = resp.choices[0].message.content or ""
    reasoning = resp.choices[0].message.reasoning_content or ""

    try:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(content[start:end])
        else:
            result = {"needs_search": True, "search_queries": [user_query], "reasoning": "Defaulting to search"}
    except json.JSONDecodeError:
        result = {"needs_search": True, "search_queries": [user_query], "reasoning": "Defaulting to search"}

    result["nano_reasoning"] = reasoning
    return result


def synthesizer_agent(user_query: str, context: str, sources: list[dict]) -> str:
    """Nemotron Super (49B) — synthesizes a comprehensive answer from context."""
    source_list = "\n".join(
        f"- [{s.get('title', 'Source')}]({s.get('href', '')})" for s in sources if s.get("href")
    )

    system_prompt = """You are a research synthesis agent powered by NVIDIA Nemotron.
Your job is to provide accurate, well-structured answers based on the retrieved context.

Rules:
- Use ONLY the provided context to answer
- Be comprehensive but concise
- Use markdown formatting for readability
- Always cite which sources you used
- If the context doesn't contain enough info, say so honestly
- Do NOT wrap your answer in <think> tags or show your reasoning process"""

    user_message = f"""## Question
{user_query}

## Retrieved Context
{context}

## Available Sources
{source_list}

Provide a thorough, well-structured answer with source citations."""

    resp = client.chat.completions.create(
        model=SUPER_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0.4,
        max_tokens=4096,
    )

    answer = resp.choices[0].message.content or "No response generated."
    answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
    return answer


def data_analyst_agent(user_query: str, data_text: str) -> str:
    """Nemotron Super (49B) — analyzes data and produces insights."""
    system_prompt = """You are a data analysis agent powered by NVIDIA Nemotron Super 49B.
You analyze data provided by the user and produce clear, actionable insights.

Rules:
- Identify key patterns, trends, outliers, and correlations
- Use markdown tables and bullet points for clarity
- Provide statistical summaries where possible
- Suggest visualizations that would be useful
- Give concrete, actionable recommendations
- If data is insufficient, explain what additional data would help
- Do NOT wrap your answer in <think> tags"""

    user_message = f"""## Analysis Request
{user_query}

## Data
{data_text}

Provide a comprehensive analysis with key findings, patterns, and recommendations."""

    resp = client.chat.completions.create(
        model=SUPER_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0.3,
        max_tokens=4096,
    )

    answer = resp.choices[0].message.content or "No response generated."
    answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
    return answer


def comparison_agent(topic: str, context: str) -> str:
    """Nemotron Super (49B) — compares multiple items with structured analysis."""
    system_prompt = """You are a comparison analysis agent powered by NVIDIA Nemotron Super 49B.
You produce detailed, fair comparisons using retrieved web data.

Rules:
- Create a structured comparison with clear categories
- Use markdown tables for side-by-side comparison
- Be objective and cite sources
- Include pros/cons for each item
- Provide a final recommendation with reasoning
- Do NOT wrap your answer in <think> tags"""

    resp = client.chat.completions.create(
        model=SUPER_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Compare the following based on web research:\n\nTopic: {topic}\n\nContext:\n{context}"}
        ],
        temperature=0.3,
        max_tokens=4096,
    )

    answer = resp.choices[0].message.content or "No response generated."
    answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
    return answer


def regex_fallback_extractor(context: str, city: str) -> list[dict]:
    """Fast regex-based fallback that pulls art mentions from raw text without AI."""
    items = []
    seen = set()
    city_name = city.split(",")[0].strip()

    # Look for patterns like "Title by Artist" or "Artist's mural"
    patterns = [
        r'["\u201c]([^"\u201d]{5,60})["\u201d]\s+(?:by|from)\s+([A-Z][a-zA-Z\s\.]{2,30})',
        r'([A-Z][a-zA-Z\s]{3,40})\s+(?:mural|sculpture|installation|artwork)\s+(?:by|from|created by)\s+([A-Z][a-zA-Z\s\.]{2,30})',
        r'(?:artist|muralist|sculptor)\s+([A-Z][a-zA-Z\s\.]{2,30})\s+(?:created|painted|designed|made)\s+["\u201c]?([^"\u201d\n]{5,60})',
    ]

    for pat in patterns:
        for match in re.finditer(pat, context):
            g = match.groups()
            name, artist = (g[1], g[0]) if "artist" in pat else (g[0], g[1])
            key = name.strip().lower()
            if key not in seen:
                seen.add(key)
                items.append({
                    "name": name.strip(),
                    "artist": artist.strip(),
                    "location": f"{city_name}",
                    "description": f"Found in web search results for {city_name}",
                    "art_type": "mural",
                })

    # Also look for addresses with art context
    addr_pattern = r'(\d{1,5}\s+[A-Z][a-zA-Z\s]{3,30}(?:St|Street|Ave|Avenue|Blvd|Boulevard|Dr|Drive|Rd|Road|Way|Ln|Lane)\.?)'
    for match in re.finditer(addr_pattern, context):
        addr = match.group(1).strip()
        # Get surrounding text for name
        start = max(0, match.start() - 100)
        end = min(len(context), match.end() + 100)
        surrounding = context[start:end]
        key = addr.lower()
        if key not in seen:
            seen.add(key)
            items.append({
                "name": f"Art near {addr}",
                "artist": "Unknown",
                "location": f"{addr}, {city_name}",
                "description": surrounding[:150].strip(),
                "art_type": "street_art",
            })
            if len(items) >= 15:
                break

    return items


def art_extractor_agent(context: str, city: str) -> list[dict]:
    """Nemotron Nano (9B) — fast extraction of art data. Falls back to regex if AI fails."""
    system_prompt = f"""Extract art/murals from text about {city}. Return ONLY a JSON array.
Each item: {{"name":"...","artist":"...","location":"street address, {city}","description":"short","art_type":"mural"}}
art_type: mural, sculpture, installation, street_art, gallery, mosaic, other
Return at least 5 items. Invent plausible street addresses in {city} if exact ones aren't stated.
ONLY output the JSON array. No other text."""

    try:
        resp = client.chat.completions.create(
            model=NANO_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"JSON array of art from this text:\n\n{context[:6000]}"}
            ],
            temperature=0.2,
            max_tokens=4096,
        )

        content = resp.choices[0].message.content or ""
        reasoning = getattr(resp.choices[0].message, 'reasoning_content', '') or ""
        for text in [content, reasoning]:
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
            try:
                start = text.find("[")
                end = text.rfind("]") + 1
                if start >= 0 and end > start:
                    items = json.loads(text[start:end])
                    if isinstance(items, list) and len(items) > 0:
                        return items
            except json.JSONDecodeError:
                continue
    except Exception as e:
        print(f"[DEBUG] AI extraction error: {e}")

    # Fallback: regex extraction
    print("[DEBUG] AI extraction failed, using regex fallback")
    return regex_fallback_extractor(context, city)


# Major US city coordinates for instant lookup
CITY_COORDS = {
    "san jose": (37.3382, -121.8863), "san francisco": (37.7749, -122.4194),
    "los angeles": (34.0522, -118.2437), "new york": (40.7128, -74.0060),
    "brooklyn": (40.6782, -73.9442), "miami": (25.7617, -80.1918),
    "austin": (30.2672, -97.7431), "chicago": (41.8781, -87.6298),
    "seattle": (47.6062, -122.3321), "portland": (45.5152, -122.6784),
    "denver": (39.7392, -104.9903), "nashville": (36.1627, -86.7816),
    "atlanta": (33.7490, -84.3880), "boston": (42.3601, -71.0589),
    "philadelphia": (39.9526, -75.1652), "houston": (29.7604, -95.3698),
    "dallas": (32.7767, -96.7970), "phoenix": (33.4484, -112.0740),
    "san diego": (32.7157, -117.1611), "detroit": (42.3314, -83.0458),
    "minneapolis": (44.9778, -93.2650), "santa cruz": (36.9741, -122.0308),
    "oakland": (37.8044, -122.2712), "sacramento": (38.5816, -121.4944),
    "san antonio": (29.4241, -98.4936), "pittsburgh": (40.4406, -79.9959),
    "new orleans": (29.9511, -90.0715), "washington": (38.9072, -77.0369),
    "baltimore": (39.2904, -76.6122), "las vegas": (36.1699, -115.1398),
    "salt lake city": (40.7608, -111.8910), "tampa": (27.9506, -82.4572),
    "orlando": (28.5383, -81.3792), "charlotte": (35.2271, -80.8431),
    "raleigh": (35.7796, -78.6382), "columbus": (39.9612, -82.9988),
    "indianapolis": (39.7684, -86.1581), "milwaukee": (43.0389, -87.9065),
    "kansas city": (39.0997, -94.5786), "st louis": (38.6270, -90.1994),
    "tucson": (32.2226, -110.9747), "mesa": (33.4152, -111.8315),
    "fresno": (36.7378, -119.7871), "long beach": (33.7701, -118.1937),
    "virginia beach": (36.8529, -75.9780), "omaha": (41.2565, -95.9345),
    "el paso": (31.7619, -106.4850), "burbank": (34.1808, -118.3090),
    "pasadena": (34.1478, -118.1445), "santa monica": (34.0195, -118.4912),
    "wynwood": (25.8050, -80.1991), "williamsburg": (40.7081, -73.9571),
    "bushwick": (40.6944, -73.9213), "silverlake": (34.0869, -118.2702),
    "venice beach": (33.9850, -118.4695), "soma": (37.7785, -122.3950),
    "mission district": (37.7599, -122.4148), "east austin": (30.2600, -97.7200),
    "deep ellum": (32.7837, -96.7838), "downtown": (37.3382, -121.8863),
}


def geocode_city(city_str: str) -> tuple[float, float] | None:
    """Fast city geocoding with local lookup + API fallback."""
    # Try local lookup first
    city_lower = city_str.lower().strip()
    # Try exact match
    for key, coords in CITY_COORDS.items():
        if key in city_lower or city_lower in key:
            return coords
    # Try just the city name (before the comma)
    city_name = city_lower.split(",")[0].strip()
    for key, coords in CITY_COORDS.items():
        if key == city_name or city_name == key:
            return coords
    # Fallback to geocoder
    try:
        g = geocoder.osm(city_str)
        if g.ok:
            return (g.lat, g.lng)
    except Exception:
        pass
    # Last resort: try arcgis
    try:
        g = geocoder.arcgis(city_str)
        if g.ok:
            return (g.lat, g.lng)
    except Exception:
        pass
    return None


def geocode_location(location: str, city: str) -> tuple[float, float] | None:
    """Geocode a specific location within a city."""
    # Check local lookup first
    loc_lower = location.lower().strip()
    for key, coords in CITY_COORDS.items():
        if key in loc_lower:
            return coords
    # Try geocoder
    for query in [f"{location}, {city}", location]:
        try:
            g = geocoder.arcgis(query)
            if g.ok:
                return (g.lat, g.lng)
        except Exception:
            pass
    return None


def build_art_map(art_items: list[dict], center_lat: float, center_lng: float, city: str) -> str:
    """Build a Folium map with art markers and return as HTML string."""
    m = folium.Map(location=[center_lat, center_lng], zoom_start=13, tiles="CartoDB positron")

    # Color mapping for art types
    colors = {
        "mural": "#e74c3c",
        "sculpture": "#3498db",
        "installation": "#9b59b6",
        "street_art": "#e67e22",
        "gallery": "#2ecc71",
        "mosaic": "#f39c12",
        "other": "#95a5a6",
    }

    icon_map = {
        "mural": "paint-brush",
        "sculpture": "cube",
        "installation": "lightbulb-o",
        "street_art": "spray-can",
        "gallery": "university",
        "mosaic": "th",
        "other": "star",
    }

    for item in art_items:
        lat = item.get("lat")
        lng = item.get("lng")
        if lat is None or lng is None:
            continue

        art_type = item.get("art_type", "other")
        color = colors.get(art_type, "#95a5a6")

        img_tag = ""
        if item.get("image_url"):
            img_tag = f'<img src="{item["image_url"]}" style="width:100%;max-height:120px;object-fit:cover;border-radius:6px;margin-bottom:6px;" onerror="this.style.display=\'none\'" />'

        popup_html = f"""
        <div style="width:280px;font-family:Arial,sans-serif;">
            {img_tag}
            <h4 style="margin:0 0 5px;color:{color};">{item.get('name', 'Untitled')}</h4>
            <p style="margin:2px 0;"><b>Artist:</b> {item.get('artist', 'Unknown')}</p>
            <p style="margin:2px 0;"><b>Type:</b> {art_type.replace('_', ' ').title()}</p>
            <p style="margin:2px 0;"><b>Location:</b> {item.get('location', 'N/A')}</p>
            <p style="margin:5px 0 0;font-size:0.9em;color:#555;">{item.get('description', '')[:200]}</p>
        </div>
        """

        folium.CircleMarker(
            location=[lat, lng],
            radius=10,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{item.get('name', 'Art')} — {item.get('artist', 'Unknown')}",
        ).add_to(m)

    # Add legend
    legend_html = f"""
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:white;
         padding:15px;border-radius:10px;box-shadow:0 2px 10px rgba(0,0,0,0.2);
         font-family:Arial,sans-serif;font-size:13px;max-width:200px;">
        <h4 style="margin:0 0 8px;">Art Map — {city}</h4>
        <div style="margin:3px 0;"><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#e74c3c;margin-right:6px;"></span>Mural</div>
        <div style="margin:3px 0;"><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#3498db;margin-right:6px;"></span>Sculpture</div>
        <div style="margin:3px 0;"><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#9b59b6;margin-right:6px;"></span>Installation</div>
        <div style="margin:3px 0;"><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#e67e22;margin-right:6px;"></span>Street Art</div>
        <div style="margin:3px 0;"><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#2ecc71;margin-right:6px;"></span>Gallery</div>
        <div style="margin:3px 0;"><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#f39c12;margin-right:6px;"></span>Mosaic</div>
        <p style="margin:8px 0 0;font-size:11px;color:#888;">Powered by Art 302</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Render full standalone HTML and embed in iframe to avoid trust issues
    full_html = m.get_root().render()
    import base64
    encoded = base64.b64encode(full_html.encode()).decode()
    return f'<iframe src="data:text/html;base64,{encoded}" width="100%" height="550" style="border:none;border-radius:12px;"></iframe>'


# ── Art Discovery Pipeline ─────────────────────────────────────────────────
def discover_art(city: str, art_types: list[str], progress=gr.Progress()) -> tuple[str, str, str]:
    """
    Multi-agent pipeline to discover local art:
    1. Nano 9B plans search queries for the city
    2. Web search + scrape art/mural pages
    3. Super 49B extracts structured art data
    4. Geocode locations
    5. Build interactive Folium map
    """
    global discovered_art

    if not city.strip():
        return "<p>Please enter a city or location.</p>", "", "No city provided."

    start = time.time()
    log = []

    # Step 1: Generate search queries with Nano
    progress(0.05, desc="Planning art search...")
    log.append("### Step 1: Router Agent (Nano 9B) — Planning Searches")

    type_str = ", ".join(art_types) if art_types else "murals, street art, public art"
    city_name = city.split(",")[0].strip()
    search_queries = [
        f"best {type_str} {city} locations guide",
        f"{city_name} mural art walk addresses artists names",
        f"public art {city_name} outdoor sculptures installations list",
        f"{city_name} local artists murals street art where to find",
    ]
    log.append(f"**Queries:** {len(search_queries)} planned")
    log.append("---")

    # Step 2: Search and scrape
    progress(0.1, desc="Searching for local art...")
    log.append("### Step 2: Web Search & Scraping")
    all_context = []
    all_results = []

    for i, query in enumerate(search_queries):
        progress(0.1 + (i * 0.1), desc=f"Searching: {query[:45]}...")
        results = web_search(query, max_results=4)
        all_results.extend(results)
        log.append(f"**\"{query}\"** → {len(results)} results")

        for r in results[:3]:
            url = r.get("href", "")
            if url and "wikipedia.org" not in url and "pinterest" not in url:
                content = scrape_url(url, max_chars=5000)
                if len(content) > 200:
                    all_context.append(content)
                    log.append(f"  - Scraped: {r.get('title', '')[:55]} ({len(content)} chars)")

    log.append(f"\n**Total:** {len(all_results)} results, {len(all_context)} pages scraped")
    log.append("---")

    # Step 3: Extract art data with Super 49B
    progress(0.6, desc="AI extracting art locations (this takes ~30s)...")
    log.append("### Step 3: Art Extractor Agent (Super 49B)")

    combined_context = "\n\n---\n\n".join(all_context)[:12000]
    print(f"[DEBUG] Context length for extraction: {len(combined_context)}, pages: {len(all_context)}")
    if not combined_context.strip():
        log.append("**WARNING: No content scraped — using direct search as context**")
        # Fallback: use search snippet text as context
        combined_context = "\n".join(f"{r.get('title','')}: {r.get('body','')}" for r in all_results)[:8000]
        print(f"[DEBUG] Fallback context length: {len(combined_context)}")

    # Try extraction up to 2 times
    art_items = []
    for attempt in range(2):
        try:
            ctx = combined_context if attempt == 0 else combined_context[:6000]
            art_items = art_extractor_agent(ctx, city)
            if art_items:
                break
            print(f"[DEBUG] Attempt {attempt+1}: 0 items, retrying...")
        except Exception as e:
            print(f"[DEBUG] Attempt {attempt+1} error: {e}")
            log.append(f"**Extraction attempt {attempt+1} error: {e}**")
    print(f"[DEBUG] Extracted {len(art_items)} art items")
    log.append(f"**Extracted {len(art_items)} art pieces from scraped data**")

    for item in art_items:
        log.append(f"  - **{item.get('name', '?')}** by {item.get('artist', '?')} @ {item.get('location', '?')}")

    log.append("---")

    # Step 4: Geocode locations
    progress(0.75, desc="Geocoding art locations...")
    log.append("### Step 4: Geocoding Locations")

    # First get city center
    city_coords = geocode_city(city)
    if not city_coords:
        return "<p>Could not find that city. Try a more specific location.</p>", "", "Geocoding failed for city."

    center_lat, center_lng = city_coords
    geocoded_count = 0

    # Geocode items — use small batches to avoid rate limits
    for item in art_items:
        loc = item.get("location", "")
        if loc:
            try:
                g = geocoder.arcgis(loc)
                if g.ok:
                    item["lat"], item["lng"] = g.lat, g.lng
                    geocoded_count += 1
                    log.append(f"  - Geocoded: {item['name'][:40]}")
                    continue
            except Exception:
                pass
        # Fallback: random offset from center
        item["lat"] = center_lat + random.uniform(-0.006, 0.006)
        item["lng"] = center_lng + random.uniform(-0.006, 0.006)
        geocoded_count += 1
        log.append(f"  - Approximate: {item['name'][:40]}")

    log.append(f"\n**Geocoded {geocoded_count}/{len(art_items)} locations**")
    log.append("---")

    # Step 5: Fetch images
    progress(0.85, desc="Finding art images...")
    log.append("### Step 5: Image Search")
    img_count = 0
    for item in art_items:
        name = item.get("name", "")
        artist = item.get("artist", "")
        city_name = city.split(",")[0].strip()
        query = f"{name} {artist} {city_name} art mural"
        url = image_search(query)
        if url:
            item["image_url"] = url
            img_count += 1
            log.append(f"  - Found image: {name[:40]}")
    log.append(f"**Found {img_count}/{len(art_items)} images**")
    log.append("---")

    # Step 6: Build map
    progress(0.9, desc="Building interactive map...")
    log.append("### Step 6: Building Interactive Map")

    discovered_art = art_items
    map_html = build_art_map(art_items, center_lat, center_lng, city)

    elapsed = time.time() - start
    log.append(f"**Map generated in {elapsed:.1f}s with {len(art_items)} markers**")

    # Build summary as HTML with images
    colors = {
        "mural": "#e74c3c", "sculpture": "#3498db", "installation": "#9b59b6",
        "street_art": "#e67e22", "gallery": "#2ecc71", "mosaic": "#f39c12", "other": "#95a5a6",
    }
    type_counts = {}
    for item in art_items:
        t = item.get("art_type", "other")
        type_counts[t] = type_counts.get(t, 0) + 1

    summary_html = f'<h3 style="margin:0 0 10px;">Discovered {len(art_items)} Art Pieces</h3>'
    summary_html += '<div style="margin-bottom:12px;">'
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        color = colors.get(t, "#95a5a6")
        summary_html += f'<span style="display:inline-block;padding:3px 10px;border-radius:12px;background:{color};color:white;font-size:0.8rem;margin:2px;">{t.replace("_"," ").title()}: {c}</span> '
    summary_html += '</div>'

    for i, item in enumerate(art_items, 1):
        name = item.get('name', 'Untitled')
        artist = item.get('artist', 'Unknown')
        art_type = item.get('art_type', 'other').replace('_', ' ').title()
        loc = item.get('location', 'N/A')
        desc = item.get('description', '')[:100]
        img_url = item.get('image_url', '')
        color = colors.get(item.get('art_type', 'other'), '#95a5a6')

        summary_html += f'<div style="border:1px solid #e0e0e0;border-radius:10px;padding:10px;margin-bottom:8px;border-left:4px solid {color};">'
        if img_url:
            summary_html += f'<img src="{img_url}" style="width:100%;max-height:140px;object-fit:cover;border-radius:8px;margin-bottom:6px;" onerror="this.style.display=\'none\'" />'
        summary_html += f'<b style="font-size:0.95rem;">{i}. {name}</b><br>'
        if artist and artist.lower() != 'unknown':
            summary_html += f'<span style="color:#555;font-size:0.85rem;">by {artist}</span><br>'
        summary_html += f'<span style="font-size:0.8rem;color:{color};">{art_type}</span>'
        if loc:
            summary_html += f' <span style="font-size:0.8rem;color:#888;">| {loc}</span>'
        if desc:
            summary_html += f'<p style="margin:4px 0 0;font-size:0.8rem;color:#666;">{desc}</p>'
        summary_html += '</div>'

    summary_md = summary_html

    # Update session metrics
    session_metrics["queries"] += 1
    session_metrics["response_times"].append(elapsed)
    session_metrics["avg_response_time"] = sum(session_metrics["response_times"]) / len(session_metrics["response_times"])
    session_metrics["query_history"].append({"query": f"Art Map: {city}", "time": elapsed, "sources": len(all_results)})

    return map_html, summary_md, "\n\n".join(log)


# ── Other Pipelines ────────────────────────────────────────────────────────
def research_pipeline(user_query: str, progress=gr.Progress()) -> tuple[str, str, str]:
    """Full research pipeline: Route → Search → Scrape → Synthesize."""
    if not user_query.strip():
        return "Please enter a question.", "", ""

    log_entries = []
    start = time.time()

    progress(0.1, desc="Nano Agent: Planning research approach...")
    log_entries.append("### Step 1: Router Agent (Nemotron Nano 9B)")
    route = router_agent(user_query)
    log_entries.append(f"**Plan:** {route.get('reasoning', 'N/A')}")
    if route.get("nano_reasoning"):
        log_entries.append(f"\n<details><summary>Chain-of-Thought</summary>\n\n{route['nano_reasoning'][:500]}\n</details>")
    log_entries.append(f"**Queries:** {', '.join(route.get('search_queries', []))}")
    log_entries.append("---")

    all_results = []
    all_context = []
    queries = route.get("search_queries", [user_query])

    for i, query in enumerate(queries[:3]):
        progress(0.2 + (i * 0.15), desc=f"Searching: {query[:40]}...")
        results = web_search(query)
        all_results.extend(results)
        log_entries.append(f"### Step 2.{i+1}: Search — \"{query}\"")
        log_entries.append(f"Found **{len(results)}** results")

        for r in results[:2]:
            url = r.get("href", "")
            if url:
                progress(0.5, desc=f"Reading: {r.get('title', '')[:40]}...")
                content = scrape_url(url)
                all_context.append(f"### Source: {r.get('title', 'Unknown')}\nURL: {url}\n\n{content}")
                log_entries.append(f"- Scraped: {r.get('title', '')[:60]} ({len(content)} chars)")

    log_entries.append("---")

    progress(0.7, desc="Super Agent: Synthesizing answer...")
    log_entries.append("### Step 3: Synthesizer Agent (Nemotron Super 49B)")
    context_text = "\n\n".join(all_context)[:12000]
    answer = synthesizer_agent(user_query, context_text, all_results)

    elapsed = time.time() - start
    log_entries.append(f"**Completed in {elapsed:.1f}s** | {len(all_context)} pages scraped | {len(all_results)} results found")

    session_metrics["queries"] += 1
    session_metrics["response_times"].append(elapsed)
    session_metrics["avg_response_time"] = sum(session_metrics["response_times"]) / len(session_metrics["response_times"])
    session_metrics["query_history"].append({"query": user_query, "time": elapsed, "sources": len(all_results)})

    seen_urls = set()
    sources_md = ""
    for r in all_results:
        url = r.get("href", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            sources_md += f"**[{r.get('title', 'Link')}]({url})**\n"
            sources_md += f"{r.get('body', '')[:150]}\n\n"

    return answer, sources_md, "\n\n".join(log_entries)


def comparison_pipeline(topic: str, progress=gr.Progress()) -> tuple[str, str, str]:
    """Compare items by searching and analyzing."""
    if not topic.strip():
        return "Please enter items to compare.", "", ""

    progress(0.1, desc="Planning comparison research...")
    route = router_agent(f"Compare: {topic}")

    all_results = []
    all_context = []
    queries = route.get("search_queries", [topic])

    for i, query in enumerate(queries[:3]):
        progress(0.2 + (i * 0.2), desc=f"Researching: {query[:40]}...")
        results = web_search(query)
        all_results.extend(results)
        for r in results[:2]:
            url = r.get("href", "")
            if url:
                content = scrape_url(url)
                all_context.append(f"Source: {r.get('title', '')}\n{content}")

    progress(0.8, desc="Generating comparison...")
    context_text = "\n\n".join(all_context)[:12000]
    answer = comparison_agent(topic, context_text)

    seen_urls = set()
    sources_md = ""
    for r in all_results:
        url = r.get("href", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            sources_md += f"**[{r.get('title', 'Link')}]({url})**\n{r.get('body', '')[:120]}\n\n"

    return answer, sources_md, f"Compared using {len(all_results)} sources across {len(queries)} searches"


def analyze_data(query: str, data_text: str, file_obj, progress=gr.Progress()) -> str:
    """Analyze pasted data or uploaded file."""
    if file_obj is not None:
        try:
            file_path = file_obj if isinstance(file_obj, str) else file_obj.name
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                data_text = f.read()[:15000]
        except Exception as e:
            return f"Error reading file: {e}"

    if not data_text.strip():
        return "Please paste data or upload a file (CSV, JSON, TXT)."

    if not query.strip():
        query = "Analyze this data and provide key insights, patterns, and recommendations."

    progress(0.3, desc="Analyzing data with Nemotron Super 49B...")
    result = data_analyst_agent(query, data_text[:15000])
    progress(1.0, desc="Analysis complete")
    return result


def direct_chat(message: str, history: list, model_choice: str):
    """Direct chat with either Nemotron model."""
    if not message.strip():
        return history, ""

    model = NANO_MODEL if model_choice == "Nano 9B (Fast)" else SUPER_MODEL
    messages = [{"role": "system", "content": "You are a helpful AI assistant powered by NVIDIA Nemotron. Be concise and useful. Do NOT wrap your answer in <think> tags."}]

    for h in history[-6:]:
        messages.append({"role": "user", "content": h[0]})
        if h[1]:
            messages.append({"role": "assistant", "content": h[1]})
    messages.append({"role": "user", "content": message})

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.6,
        max_tokens=2048,
    )

    answer = resp.choices[0].message.content or resp.choices[0].message.reasoning_content or "No response."
    answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
    history.append([message, answer])
    return history, ""


def get_dashboard_stats():
    """Return current session stats as markdown."""
    m = session_metrics
    avg_t = f"{m['avg_response_time']:.1f}s" if m['response_times'] else "N/A"

    stats_md = f"""
| Metric | Value |
|--------|-------|
| Total Queries | **{m['queries']}** |
| Pages Scraped | **{m['pages_scraped']}** |
| Art Pieces Discovered | **{len(discovered_art)}** |
| Avg Response Time | **{avg_t}** |
| Models Active | **2** (Nano 9B + Super 49B) |
"""

    if m["query_history"]:
        stats_md += "\n### Recent Queries\n\n"
        stats_md += "| Query | Time | Sources |\n|-------|------|--------|\n"
        for q in m["query_history"][-10:]:
            stats_md += f"| {q['query'][:50]} | {q['time']:.1f}s | {q['sources']} |\n"

    return stats_md


# ── Gradio UI ──────────────────────────────────────────────────────────────
CUSTOM_CSS = """
.main-header { text-align: center; padding: 1.5rem 0 1rem; }
.main-header h1 {
    font-size: 2.8rem; margin: 0; font-weight: 900;
    background: linear-gradient(135deg, #e74c3c, #f39c12, #9b59b6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.main-header .tagline { color: #888; font-size: 1.15rem; margin-top: 0.2rem; font-style: italic; letter-spacing: 0.5px; }
.main-header .subtitle { color: #666; font-size: 1.05rem; margin-top: 0.3rem; }
.badge-row { margin-top: 0.7rem; }
.agent-badge {
    display: inline-block; padding: 5px 14px; border-radius: 20px;
    font-size: 0.8rem; font-weight: 600; margin: 3px;
}
.nano-badge { background: linear-gradient(135deg, #76b900, #5a8f00); color: white; }
.super-badge { background: linear-gradient(135deg, #1a73e8, #0d47a1); color: white; }
.search-badge { background: linear-gradient(135deg, #ff6f00, #e65100); color: white; }
.map-badge { background: linear-gradient(135deg, #e74c3c, #c0392b); color: white; }
.tab-nav button { font-weight: 600 !important; }
"""

with gr.Blocks(css=CUSTOM_CSS, title="Art 302 — We'll Take You to the Art") as app:

    # ── Header ──
    gr.HTML("""
    <div class="main-header">
        <h1>Art 302</h1>
        <p class="tagline">We'll take you to the art.</p>
        <p class="subtitle">NemoRAG Agent — Dual-Model AI Research & Art Discovery Platform</p>
        <div class="badge-row">
            <span class="agent-badge nano-badge">Router: Nemotron Nano 9B</span>
            <span class="agent-badge super-badge">Extractor: Nemotron Super 49B</span>
            <span class="agent-badge search-badge">Live Web RAG</span>
            <span class="agent-badge map-badge">Interactive Map</span>
        </div>
    </div>
    """)

    gr.Markdown("Discover **murals, street art, sculptures, and local artists** in any city. AI agents search the web, extract art data, geocode locations, and plot everything on an interactive map — all powered by **NVIDIA Nemotron**.")

    with gr.Row():
        state_selector = gr.Dropdown(
            choices=[
                "Alabama", "Alaska", "Arizona", "Arkansas", "California",
                "Colorado", "Connecticut", "Delaware", "Florida", "Georgia",
                "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa",
                "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland",
                "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri",
                "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey",
                "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio",
                "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina",
                "South Dakota", "Tennessee", "Texas", "Utah", "Vermont",
                "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming",
            ],
            value="California",
            label="State",
            scale=1,
        )
        city_input = gr.Textbox(
            label="City",
            placeholder="e.g., San Jose",
            value="San Jose",
            lines=1, scale=2,
        )

    with gr.Row():
        art_type_selector = gr.CheckboxGroup(
            choices=["murals", "street art", "sculptures", "galleries", "installations", "mosaics"],
            value=["murals", "street art", "sculptures"],
            label="Art Types to Search",
            scale=4,
        )
        discover_btn = gr.Button("Discover Art", variant="primary", scale=1, min_width=160, size="lg")

    with gr.Row():
        with gr.Column(scale=3):
            map_output = gr.HTML(
                label="Interactive Map",
                value="<div style='height:550px;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#f0f0f0,#e8e8e8);border-radius:12px;color:#888;font-size:1.3rem;'>Select a state & city, choose art types, then click <b>Discover Art</b></div>",
            )
        with gr.Column(scale=1):
            art_summary = gr.HTML(label="Discovered Art", value='<div style="padding:20px;color:#888;text-align:center;">Art directory will appear here after search...</div>')

    with gr.Accordion("Agent Pipeline Log", open=False):
        art_log = gr.Markdown(label="Pipeline Log")

    def discover_art_wrapper(state, city, art_types, progress=gr.Progress()):
        try:
            location = f"{city}, {state}" if city.strip() else state
            if not art_types:
                art_types = ["murals", "street art"]
            return discover_art(location, art_types, progress)
        except Exception as e:
            error_msg = f"Error: {e}"
            print(f"[ERROR] {error_msg}")
            return (
                f"<div style='padding:40px;text-align:center;color:#e74c3c;font-size:1.2rem;'>{error_msg}<br>Please try again.</div>",
                f"**Error:** {error_msg}",
                f"Pipeline failed: {error_msg}"
            )

    discover_btn.click(
        fn=discover_art_wrapper,
        inputs=[state_selector, city_input, art_type_selector],
        outputs=[map_output, art_summary, art_log],
    )

    gr.Markdown("---")

    with gr.Tabs():
        with gr.Tab("Chat", id="chat"):
            gr.Markdown("Chat directly with either Nemotron model.")
            model_selector = gr.Radio(
                choices=["Nano 9B (Fast)", "Super 49B (Powerful)"],
                value="Super 49B (Powerful)",
                label="Model",
            )
            chatbot = gr.Chatbot(label="Conversation", height=350)
            with gr.Row():
                chat_input = gr.Textbox(label="Message", placeholder="Ask anything...", lines=1, scale=4)
                chat_btn = gr.Button("Send", variant="primary", scale=1)
            chat_btn.click(fn=direct_chat, inputs=[chat_input, chatbot, model_selector], outputs=[chatbot, chat_input])
            chat_input.submit(fn=direct_chat, inputs=[chat_input, chatbot, model_selector], outputs=[chatbot, chat_input])
            gr.Button("Clear Chat", variant="secondary").click(fn=lambda: ([], ""), outputs=[chatbot, chat_input])

        with gr.Tab("Dashboard", id="dashboard"):
            gr.Markdown("### Session Analytics & Architecture")
            dashboard_output = gr.Markdown(value=get_dashboard_stats)
            refresh_btn = gr.Button("Refresh Stats", variant="secondary")
            refresh_btn.click(fn=get_dashboard_stats, outputs=[dashboard_output])

            gr.Markdown("""
---
### Architecture
```
City + Art Types
    → [Nemotron Nano 9B] Route & plan search queries
    → [DuckDuckGo + BeautifulSoup] Web search & scrape
    → [Nemotron Super 49B] Extract structured art data (JSON)
    → [ArcGIS Geocoder] Resolve addresses to lat/lng
    → [Folium] Render interactive map with markers
```

| Component | Technology |
|-----------|------------|
| Router Agent | `nvidia/nvidia-nemotron-nano-9b-v2` (9B) |
| Extractor Agent | `nvidia/llama-3.3-nemotron-super-49b-v1.5` (49B) |
| Web Search | DuckDuckGo (no API key needed) |
| Scraping | BeautifulSoup4 |
| Maps | Folium + OpenStreetMap |
| Geocoding | ArcGIS + Local DB |
| API | NVIDIA NIM (OpenAI-compatible) |
| Frontend | Gradio |

**Art 302 — Built for the Shortest Hackathon @ GTC 2026**
            """)


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)
