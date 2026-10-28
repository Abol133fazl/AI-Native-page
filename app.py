import os
import json
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# خواندن فایل پروفایل
with open("profile.json", "r", encoding="utf-8") as f:
    PROFILE_DATA = json.load(f)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.gapgpt.app/v1")

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)

class QueryRequest(BaseModel):
    question: str

async def search_external_books(query_topic: str):
    """API خارجی: جستجوی کتاب از Open Library"""
    try:
        url = f"https://openlibrary.org/search.json?q={query_topic}&limit=3"
        async with httpx.AsyncClient(timeout=5.0) as http_client:
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
    return "Introduction to Algorithms (CLRS) and AI: A Modern Approach (Russell & Norvig)"

SYSTEM_PROMPT = f"""
You are the AI engine for a personal portfolio page. You represent the profile below:

PROFILE RECORD:
{json.dumps(PROFILE_DATA, ensure_ascii=False, indent=2)}

STRICT RULES (GROUNDING & GUARDRAILS):
1. You may ONLY answer based on the facts present in the PROFILE RECORD or the External API result.
2. DO NOT hallucinate or assume facts (no unpublished employers, grades, phone numbers, or dates).
3. GUARDRAILS: You CANNOT send emails, book meetings, or speak officially on behalf of Sharif University. Refuse these actions immediately.
4. If a fact is marked private or missing in the record, refuse to answer it and mention it in 'missing_facts'.
5. If the question is ambiguous (e.g., 'tell me about my project' when there are 3 projects), DO NOT pick one. Use capability 'clarify' and ask exactly ONE clarifying question.
6. When evaluating job fit (Intent: 'job_fit'), compare the job requirements against the candidate's skills and projects. Provide an estimated match_score (0 to 100) and list missing skills.

You must respond in valid JSON with this exact schema:
{{
  "intent": "project_summary | reading_recommendation | job_fit | guardrail_refusal | general_query | unclear",
  "capability": "answer_from_record | answer_from_external_api | refusal | clarify | out_of_scope",
  "fields_read": ["e.g. projects[compiler].summary", "skills.languages"],
  "external_api_used": "None" or "Open Library Books API",
  "missing_facts": ["e.g. phone_number", "graduation_date"],
  "job_match_score": 0,
  "answer": "The text to show the user."
}}
"""

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "profile": PROFILE_DATA
    })

@app.post("/api/ask")
async def ask_question(payload: QueryRequest):
    question = payload.question.strip()
    
    # بررسی نیاز به API خارجی برای مطالعه/کتاب
    external_context = ""
    external_api_name = "None"
    if any(k in question.lower() for k in ["read", "book", "suggest", "مطالعه", "کتاب"]):
        book_suggestions = await search_external_books("computer science algorithms ai")
        external_context = f"\n[EXTERNAL API RESULT from Open Library]: Relevant Books: {book_suggestions}\n"
        external_api_name = "Open Library Books API"

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT + external_context},
                {"role": "user", "content": question}
            ],
            temperature=0.1
        )
        raw_content = completion.choices[0].message.content
        clean_content = raw_content.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean_content)
        if external_api_name != "None" and result.get("capability") != "refusal":
            result["external_api_used"] = external_api_name
        return result
    except Exception as e:
        return {
            "intent": "error",
            "capability": "out_of_scope",
            "fields_read": [],
            "external_api_used": "None",
            "missing_facts": [],
            "answer": f"Error communicating with AI engine: {str(e)}"
        }