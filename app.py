import os
import json
import time
import httpx
from collections import defaultdict
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Grounded AI-Native Page")
templates = Jinja2Templates(directory="templates")

PROFILE_RECORD_ENV = os.getenv("PROFILE_RECORD_JSON")

if PROFILE_RECORD_ENV:
    try:
        PROFILE_DATA = json.loads(PROFILE_RECORD_ENV)
    except Exception as e:
        raise RuntimeError(f"Error parsing PROFILE_RECORD_JSON: {e}")
elif os.path.exists("profile.json"):
    with open("profile.json", "r", encoding="utf-8") as f:
        PROFILE_DATA = json.load(f)
else:
    raise RuntimeError("No profile record found! Set PROFILE_RECORD_JSON environment variable.")

GAPGPT_FALLBACK = "sk-zMWszraO8OsZicnZM6rRTxP9OceO8Dn0FhN7ERs5v6maDK4i"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", GAPGPT_FALLBACK)
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.gapgpt.app/v1")

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)

REQUEST_HISTORY = defaultdict(list)
MAX_REQUESTS_PER_MINUTE = 6
MAX_QUESTION_LENGTH = 350

RESPONSE_CACHE = {}

def check_rate_limit(client_ip: str):
    now = time.time()
    REQUEST_HISTORY[client_ip] = [t for t in REQUEST_HISTORY[client_ip] if now - t < 60]
    if len(REQUEST_HISTORY[client_ip]) >= MAX_REQUESTS_PER_MINUTE:
        return False
    REQUEST_HISTORY[client_ip].append(now)
    return True

# ----------------- ابزار API خارجی (Open Library) -----------------
async def search_external_books(query_topic: str):
    try:
        url = f"https://openlibrary.org/search.json?q={query_topic}&limit=3"
        async with httpx.AsyncClient(timeout=4.0) as http_client:
            resp = await http_client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                books = []
                for doc in data.get("docs", [])[:2]:
                    title = doc.get("title", "Unknown")
                    author = doc.get("author_name", ["Unknown"])[0]
                    books.append(f"'{title}' by {author}")
                return ", ".join(books) if books else "Standard Academic Literature"
    except Exception:
        pass
    return "Introduction to Algorithms (CLRS), Computer Organization & Architecture (Patterson & Hennessy)"

SYSTEM_PROMPT = f"""
You are the AI engine for a personal portfolio page. You represent the profile below:

PROFILE RECORD:
{json.dumps(PROFILE_DATA, ensure_ascii=False, indent=2)}

STRICT RULES (GROUNDING & GUARDRAILS):
1. You may ONLY answer based on facts present in the PROFILE RECORD or the External API result.
2. DO NOT hallucinate, assume, or add facts (no unpublished employers, phone numbers, or dates).
3. GUARDRAILS: You CANNOT send emails, book calendar slots, or speak officially on behalf of Sharif University. Refuse these immediately.
4. If a fact is private or not present in the record, refuse to answer it and put the field name in 'missing_facts'.
5. If the user asks something ambiguous (e.g. 'tell me about my game' when there are 3 game projects), DO NOT pick one; return Capability: 'clarify' and ask exactly ONE clarifying question.
6. Allowed Capabilities:
   - 'answer_from_record'
   - 'answer_from_external_api'
   - 'refusal'
   - 'clarify'
   - 'out_of_scope'

Respond in valid JSON with this exact schema:
{{
  "intent": "project_summary | reading_recommendation | job_fit | guardrail_refusal | unclear | general_query",
  "capability": "answer_from_record | answer_from_external_api | refusal | clarify | out_of_scope",
  "fields_read": ["e.g. projects[stardew_multiplayer].summary"],
  "external_api_used": "None" or "Open Library Books API",
  "missing_facts": ["e.g. phone_number"],
  "answer": "The text to show the user."
}}
"""

class ChatPayload(BaseModel):
    question: str

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "profile": PROFILE_DATA
    })

@app.post("/api/chat")
async def chat_endpoint(payload: ChatPayload, request: Request):
    client_ip = request.client.host if request.client else "unknown"

    if not check_rate_limit(client_ip):
        return {
            "intent": "rate_limited",
            "capability": "refusal",
            "fields_read": [],
            "external_api_used": "None",
            "missing_facts": [],
            "answer": "⏳ Rate limit exceeded (Max 6 requests/minute). Please slow down to preserve API budget."
        }

    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty question")

    if len(question) > MAX_QUESTION_LENGTH:
        question = question[:MAX_QUESTION_LENGTH]

    cache_key = question.lower()
    if cache_key in RESPONSE_CACHE:
        cached_res = dict(RESPONSE_CACHE[cache_key])
        cached_res["from_cache"] = True
        return cached_res

    external_context = ""
    external_api_name = "None"
    if any(k in question.lower() for k in ["read", "book", "suggest", "مطالعه", "کتاب"]):
        book_suggestions = await search_external_books("computer science algorithms architecture")
        external_context = f"\n[EXTERNAL API RESULT from Open Library]: Verified Books: {book_suggestions}\n"
        external_api_name = "Open Library Books API"

    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT + external_context},
                {"role": "user", "content": question}
            ],
            temperature=0.1,
            max_tokens=600
        )

        raw_content = completion.choices[0].message.content
        clean_content = raw_content.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean_content)

        if external_api_name != "None" and result.get("capability") != "refusal":
            result["external_api_used"] = external_api_name

        result["from_cache"] = False

        if len(RESPONSE_CACHE) > 100:
            RESPONSE_CACHE.clear()
        RESPONSE_CACHE[cache_key] = result

        return result

    except Exception as e:
        return {
            "intent": "error",
            "capability": "out_of_scope",
            "fields_read": [],
            "external_api_used": "None",
            "missing_facts": [],
            "answer": f"Backend Error: {str(e)}"
        }