from __future__ import annotations

import io
import os
import csv
import json
import hmac
import time
import zipfile
import sqlite3
from dataclasses import dataclass
from typing import List, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import stripe

# PDF generation (ReportLab)
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib import colors

# Load environment variables from .env
load_dotenv()

# --- Stripe environment / mode switch ---
STRIPE_MODE = os.getenv("STRIPE_MODE", "test").strip().lower()

# Test (default) values
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_PRICE_ID = os.getenv("STRIPE_PRICE_ID", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")

# Live values (used when STRIPE_MODE=live)
STRIPE_LIVE_SECRET_KEY = os.getenv("STRIPE_LIVE_SECRET_KEY", "")
STRIPE_LIVE_PRICE_ID = os.getenv("STRIPE_LIVE_PRICE_ID", "")
STRIPE_WEBHOOK_SECRET_LIVE = os.getenv("STRIPE_WEBHOOK_SECRET_LIVE", "")

SITE_URL = os.getenv("SITE_URL", "http://127.0.0.1:8000")


def _active_stripe_secret_key() -> str:
    return STRIPE_LIVE_SECRET_KEY if STRIPE_MODE == "live" else STRIPE_SECRET_KEY


def _active_price_id() -> str:
    return STRIPE_LIVE_PRICE_ID if STRIPE_MODE == "live" else STRIPE_PRICE_ID


def _active_webhook_secret() -> str:
    return STRIPE_WEBHOOK_SECRET_LIVE if STRIPE_MODE == "live" else STRIPE_WEBHOOK_SECRET


stripe.api_key = _active_stripe_secret_key()

# --- App / product identity (used in templates + artefacts) ---
APP_NAME = "PromptKitPro"
APP_VERSION = "2.0"

PRODUCT_NAME = "Operations & Productivity Prompt Library"
PRODUCT_PRICE_GBP = "£25"
PRODUCT_PROMPT_COUNT = 250

# --- PDF theme (mirrors site.css) ---
PDF_BG = colors.HexColor("#0b0f19")
PDF_TEXT = colors.whitesmoke
PDF_MUTED = colors.Color(1, 1, 1, alpha=0.7)

PDF_ACCENT = colors.HexColor("#7c3aed")
PDF_ACCENT_2 = colors.HexColor("#22c55e")

PDF_DIVIDER = colors.Color(1, 1, 1, alpha=0.15)

app = FastAPI(title=APP_NAME)
templates = Jinja2Templates(directory="app/templates")

# Serve /static/* (site.css, etc.)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

DB_PATH = "app/orders.sqlite3"

# Source-of-truth prompt library file
PROMPTS_CSV_PATH = "app/data/prompt_library_v3_250.csv"


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_ts INTEGER NOT NULL,
            email TEXT NOT NULL,
            session_id TEXT NOT NULL UNIQUE,
            status TEXT NOT NULL,
            download_token TEXT NOT NULL UNIQUE
        )"""
    )
    conn.commit()
    return conn


def make_token(session_id: str) -> str:
    """
    Generate a download token. For MVP, we use an HMAC based on the active Stripe secret key.
    """
    secret = (_active_stripe_secret_key() or "dev-secret").encode("utf-8")
    msg = f"{session_id}:{int(time.time())}".encode("utf-8")
    return hmac.new(secret, msg, "sha256").hexdigest()


@dataclass
class LibraryConfig:
    name: str
    audience: str
    categories: List[str]


def load_prompt_library(csv_path: str = PROMPTS_CSV_PATH) -> List[Dict[str, str]]:
    """
    Load the prompt library from a controlled CSV file.
    Expected columns: id, category, prompt
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Prompt library not found: {csv_path}")

    prompts: List[Dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        required = {"id", "category", "prompt"}
        fieldnames = set(reader.fieldnames or [])

        if not required.issubset(fieldnames):
            raise ValueError(f"CSV must contain columns {sorted(required)}; got {reader.fieldnames}")

        for row in reader:
            prompts.append(
                {
                    "id": str(row.get("id", "")).strip(),
                    "category": str(row.get("category", "")).strip(),
                    "prompt": str(row.get("prompt", "")).strip(),
                }
            )

    # Basic quality checks
    if len(prompts) == 0:
        raise ValueError("Prompt library CSV loaded 0 prompts (empty file or parse error).")

    # Optional: ensure IDs exist
    for p in prompts[:5]:
        if not p["id"] or not p["category"] or not p["prompt"]:
            raise ValueError("CSV rows must include non-empty id, category, and prompt fields.")

    return prompts


def _draw_wrapped_text(
    c: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    max_width: float,
    line_height: float,
) -> float:
    """
    Draw wrapped text onto a reportlab canvas. Returns updated y position.
    """
    words = text.split()
    line = ""
    for w in words:
        candidate = (line + " " + w).strip()
        if c.stringWidth(candidate) <= max_width:
            line = candidate
        else:
            if line:
                c.drawString(x, y, line)
                y -= line_height
            line = w

    if line:
        c.drawString(x, y, line)
        y -= line_height

    return y


def draw_page_background(c: canvas.Canvas):
    page_w, page_h = A4
    c.setFillColor(PDF_BG)
    c.rect(0, 0, page_w, page_h, stroke=0, fill=1)


def _fit_title_lines(
    c: canvas.Canvas,
    title: str,
    max_width: float,
    font_name: str = "Helvetica-Bold",
    start_size: int = 20,
    min_size: int = 12,
    max_lines: int = 3,
) -> tuple[list[str], int]:
    """
    Fit a title into max_width by wrapping at natural breakpoints and/or shrinking font.
    Returns (lines, font_size).
    """
    # Prefer breaking at these separators (in order)
    separators = [" — ", " - ", " | ", ": "]

    def try_split(t: str) -> list[str]:
        for sep in separators:
            if sep in t:
                parts = t.split(sep)
                # Rebuild lines while preserving separator meaning
                lines = [parts[0]]
                for p in parts[1:]:
                    # Add separator back to previous line if it fits better as a continuation
                    lines.append(p)
                # Put the separator back visually between first two chunks if we used it
                if len(lines) >= 2 and sep.strip():
                    lines[0] = lines[0] + sep.strip()
                return [ln.strip() for ln in lines if ln.strip()]
        return [t.strip()]

    # First attempt: split by separators
    base_lines = try_split(title)

    # If still too long, do word-wrapping
    def word_wrap(t: str) -> list[str]:
        words = t.split()
        lines: list[str] = []
        cur = ""
        for w in words:
            cand = (cur + " " + w).strip()
            if c.stringWidth(cand, font_name, start_size) <= max_width:
                cur = cand
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines

    # Try decreasing font size until it fits within width and line count
    for size in range(start_size, min_size - 1, -1):
        c.setFont(font_name, size)

        # Start from separator split, then wrap each piece if needed
        lines: list[str] = []
        for chunk in base_lines:
            if c.stringWidth(chunk, font_name, size) <= max_width:
                lines.append(chunk)
            else:
                # wrap chunk by words at this font size
                words = chunk.split()
                cur = ""
                for w in words:
                    cand = (cur + " " + w).strip()
                    if c.stringWidth(cand, font_name, size) <= max_width:
                        cur = cand
                    else:
                        if cur:
                            lines.append(cur)
                        cur = w
                if cur:
                    lines.append(cur)

        if len(lines) <= max_lines and all(c.stringWidth(ln, font_name, size) <= max_width for ln in lines):
            return lines, size

    # Fall back: hard truncate (should rarely happen)
    c.setFont(font_name, min_size)
    t = title
    while c.stringWidth(t + "…", font_name, min_size) > max_width and len(t) > 10:
        t = t[:-1]
    return [t + "…"], min_size

def _render_prompts_into_canvas(
    c: canvas.Canvas,
    prompts: List[Dict[str, str]],
    title: str,
    include_title_and_contents: bool,
    category_ranges: Dict[str, tuple[int, int]] | None = None,
) -> Dict[str, int]:
    """
    Render prompts, optionally preceded by Title + Contents pages.
    Returns a dict mapping prompt id -> page number (1-based).
    """
    page_w, page_h = A4
    margin = 15 * mm
    x = margin
    max_width = page_w - 2 * margin
    line_height = 4.5 * mm

    c.setTitle(title)
    draw_page_background(c)

    # Track prompt start pages
    prompt_start_page: Dict[str, int] = {}

    def new_page():
        c.showPage()
        draw_page_background(c)

    if include_title_and_contents:
        # --- Title page ---
        y = page_h - margin
        
        c.setFillColor(PDF_TEXT)

        
        title_lines, title_size = _fit_title_lines(
            c=c,
            title=title,
            max_width=max_width,
            font_name="Helvetica-Bold",
            start_size=20,
            min_size=12,
            max_lines=3,
        )
        
        c.setFont("Helvetica-Bold", title_size)
        for line in title_lines:
            c.drawString(x, y, line)
            y -= (title_size * 0.9) * mm  # line spacing tied to font size

        # Accent bar (brand)
        c.setFillColor(PDF_ACCENT)
        c.rect(x, y - 4 * mm, 40 * mm, 2 * mm, stroke=0, fill=1)
        y -= 12 * mm

        c.setFillColor(PDF_MUTED)
        c.setFont("Helvetica", 11)
        c.drawString(x, y, f"Total prompts: {len(prompts)}")
        y -= 6 * mm
        c.setFont("Helvetica", 9)
        c.drawString(x, y, "Formats: PDF / Markdown / CSV / JSON")
        y -= 8 * mm

        new_page()

        # --- Contents page (Category ranges) ---
        y = page_h - margin
        c.setFillColor(PDF_TEXT)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(x, y, "Contents")
        y -= 10 * mm

        c.setFillColor(PDF_MUTED)
        c.setFont("Helvetica", 10)
        c.drawString(x, y, "Categories (page ranges show where each category begins and ends).")
        y -= 8 * mm

        if category_ranges is None:
            category_ranges = {}

        # Sort categories by their start page (then name)
        items = sorted(category_ranges.items(), key=lambda kv: (kv[1][0], kv[0].lower()))

        # Draw entries
        c.setFillColor(PDF_TEXT)
        c.setFont("Helvetica", 10)
        for cat, (start_p, end_p) in items:
            if y < margin + 20 * mm:
                new_page()
                y = page_h - margin
                c.setFillColor(PDF_TEXT)
                c.setFont("Helvetica-Bold", 16)
                c.drawString(x, y, "Contents (cont.)")
                y -= 10 * mm
                c.setFillColor(PDF_TEXT)
                c.setFont("Helvetica", 10)

            # Left: category
            c.drawString(x, y, cat)

            # Right: page range, aligned to the right margin
            range_text = f"{start_p}–{end_p}" if start_p != end_p else f"{start_p}"
            c.drawRightString(page_w - margin, y, range_text)

            y -= 6 * mm

        new_page()

    # --- Prompts section ---
    y = page_h - margin
    c.setFillColor(PDF_TEXT)
    c.setFont("Helvetica", 9)

    for p in prompts:
        # Record the page where this prompt starts
        prompt_start_page[p.get("id", "").strip()] = c.getPageNumber()

        # Page break if needed
        if y < margin + 40 * mm:
            new_page()
            y = page_h - margin
            c.setFillColor(PDF_TEXT)
            c.setFont("Helvetica", 9)

        pid = p.get("id", "").strip()
        cat = p.get("category", "").strip()
        body = p.get("prompt", "").strip()

        c.setFillColor(PDF_ACCENT_2)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(x, y, f"{pid} — {cat}")
        y -= 6 * mm

        c.setFillColor(PDF_TEXT)
        c.setFont("Helvetica", 9)

        for para in body.splitlines():
            if not para.strip():
                y -= line_height
                continue

            y = _draw_wrapped_text(c, para, x, y, max_width, line_height)

            if y < margin + 20 * mm:
                new_page()
                y = page_h - margin
                c.setFillColor(PDF_TEXT)
                c.setFont("Helvetica", 9)

        # Subtle divider between prompts (optional but tasteful)
        c.setStrokeColor(PDF_DIVIDER)
        c.line(x, y, x + max_width, y)
        y -= 6 * mm

        y -= 6 * mm

    return prompt_start_page


def _compute_category_page_ranges(
    prompts: List[Dict[str, str]],
    prompt_start_page: Dict[str, int],
) -> Dict[str, tuple[int, int]]:
    """
    Compute category -> (start_page, end_page) based on prompt start pages.
    End page is approximated as the page before the next category starts (or last prompt page).
    """
    per_cat_pages: Dict[str, List[int]] = {}
    for p in prompts:
        cat = (p.get("category") or "").strip()
        pid = (p.get("id") or "").strip()
        if not cat or not pid:
            continue
        page = prompt_start_page.get(pid)
        if page is None:
            continue
        per_cat_pages.setdefault(cat, []).append(page)

    cat_start = {cat: min(pages) for cat, pages in per_cat_pages.items()}
    sorted_cats = sorted(cat_start.items(), key=lambda kv: kv[1])
    if not sorted_cats:
        return {}

    last_prompt_page = max(prompt_start_page.values()) if prompt_start_page else 1

    ranges: Dict[str, tuple[int, int]] = {}
    for i, (cat, start_p) in enumerate(sorted_cats):
        if i < len(sorted_cats) - 1:
            next_start = sorted_cats[i + 1][1]
            end_p = max(start_p, next_start - 1)
        else:
            end_p = last_prompt_page
        ranges[cat] = (start_p, end_p)

    return ranges


def build_library_pdf(prompts: List[Dict[str, str]], title: str) -> bytes:
    """
    Create a PDF with:
    - Title page
    - Contents page listing categories with page ranges (B)
    - Prompts section
    Uses a two-pass approach to compute page ranges accurately.
    """
    # PASS 1: render prompts only (no title/contents) to learn prompt start pages
    buf1 = io.BytesIO()
    c1 = canvas.Canvas(buf1, pagesize=A4)
    start_pages_pass1 = _render_prompts_into_canvas(
        c=c1,
        prompts=prompts,
        title=title,
        include_title_and_contents=False,
    )
    c1.save()

    assumed_offset = 2
    category_ranges = {}

    for _ in range(3):
        shifted_start_pages = {pid: (pg + assumed_offset) for pid, pg in start_pages_pass1.items()}
        category_ranges = _compute_category_page_ranges(prompts, shifted_start_pages)

        # PASS 2 draft: render with title/contents and see where prompts actually start
        buf2 = io.BytesIO()
        c2 = canvas.Canvas(buf2, pagesize=A4)
        start_pages_pass2 = _render_prompts_into_canvas(
            c=c2,
            prompts=prompts,
            title=title,
            include_title_and_contents=True,
            category_ranges=category_ranges,
        )
        c2.save()

        first_id = prompts[0].get("id", "").strip()
        if first_id and first_id in start_pages_pass2 and first_id in start_pages_pass1:
            true_offset = start_pages_pass2[first_id] - start_pages_pass1[first_id]
        else:
            true_offset = assumed_offset

        if true_offset == assumed_offset:
            return buf2.getvalue()

        assumed_offset = true_offset

    return buf2.getvalue()


def build_library_zip(cfg: LibraryConfig) -> bytes:
    prompts = load_prompt_library()

    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as z:
        # 1) Markdown
        md = [f"# {cfg.name}\n", f"Audience: **{cfg.audience}**\n", "## Prompt Library\n"]
        for p in prompts:
            md.append(f"### {p['id']} — {p['category']}\n")
            md.append(p["prompt"] + "\n")
        z.writestr("prompt_library.md", "\n".join(md))

        # 2) CSV (as-is, source of truth)
        z.write(PROMPTS_CSV_PATH, arcname="prompt_library.csv")

        # 3) JSON
        z.writestr("prompt_library.json", json.dumps(prompts, indent=2, ensure_ascii=False))

        # 4) PDF
        pdf_bytes = build_library_pdf(prompts, title=cfg.name)
        z.writestr("prompt_library.pdf", pdf_bytes)

        # 5) README
        z.writestr(
            "README.txt",
            "Prompt Library\n"
            "Files included:\n"
            "- prompt_library.md\n"
            "- prompt_library.csv (source of truth)\n"
            "- prompt_library.json\n"
            "- prompt_library.pdf\n\n"
            "How to use:\n"
            "1) Pick a prompt\n"
            "2) Paste into your AI tool\n"
            "3) Fill the Inputs sections\n"
        )

    return mem.getvalue()


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    # --- Landing-page preview prompt (deterministic: prompt id "1") ---
    preview_prompt = {"id": "1", "category": "Meetings", "prompt": ""}
    preview_excerpt = ""

    try:
        prompts = load_prompt_library()
        found = None
        for p in prompts:
            if str(p.get("id", "")).strip() == "1":
                found = p
                break

        if found is None and prompts:
            found = prompts[0]

        if found:
            preview_prompt = found
            prompt_text = (found.get("prompt") or "").strip()
            preview_excerpt = prompt_text
    except Exception:
        # Intentionally do not fail the landing page if preview loading fails.
        # The rest of the system (fulfilment) will still fail loudly if the CSV is missing.
        preview_excerpt = ""

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "app_name": APP_NAME,
            "version": APP_VERSION,
            "product_name": PRODUCT_NAME,
            "price": PRODUCT_PRICE_GBP,
            "count": PRODUCT_PROMPT_COUNT,
            "preview_prompt": preview_prompt,
            "preview_excerpt": preview_excerpt,
        },
    )


@app.post("/buy")
def buy(email: str = Form(...)):
    active_key = _active_stripe_secret_key()
    active_price = _active_price_id()

    if not active_key or not active_price:
        raise HTTPException(
            status_code=500,
            detail="Missing Stripe configuration for current STRIPE_MODE (secret key / price id).",
        )

    # Ensure Stripe SDK uses the correct key (helps if env vars change between deploys)
    stripe.api_key = active_key

    session = stripe.checkout.Session.create(
        mode="payment",
        line_items=[{"price": active_price, "quantity": 1}],
        customer_email=email,
        success_url=f"{SITE_URL}/success?session_id={{CHECKOUT_SESSION_ID}}",
        cancel_url=f"{SITE_URL}/",
    )
    return RedirectResponse(url=session.url, status_code=303)


@app.get("/success", response_class=HTMLResponse)
def success(request: Request, session_id: str):
    return templates.TemplateResponse(
        "success.html",
        {
            "request": request,
            "session_id": session_id,
            "app_name": APP_NAME,
            "version": APP_VERSION,
        },
    )


@app.post("/stripe-webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    active_webhook_secret = _active_webhook_secret()
    if not active_webhook_secret:
        raise HTTPException(status_code=500, detail="Webhook secret not configured for current STRIPE_MODE.")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, active_webhook_secret)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Webhook error: {e}")

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        email = session.get("customer_email") or "unknown"
        session_id = session["id"]

        token = make_token(session_id)
        conn = db()
        conn.execute(
            "INSERT OR IGNORE INTO orders(created_ts, email, session_id, status, download_token) "
            "VALUES (?, ?, ?, ?, ?)",
            (int(time.time()), email, session_id, "paid", token),
        )
        conn.commit()
        conn.close()

    return {"ok": True}


@app.get("/download", response_class=HTMLResponse)
def download_page(request: Request, session_id: str):
    conn = db()
    row = conn.execute(
        "SELECT download_token FROM orders WHERE session_id=? AND status='paid'",
        (session_id,),
    ).fetchone()
    conn.close()

    token = row[0] if row else None
    return templates.TemplateResponse(
        "download.html",
        {
            "request": request,
            "token": token,
            "session_id": session_id,
            "app_name": APP_NAME,
            "version": APP_VERSION,
        },
    )


@app.get("/file")
def download_file(token: str):
    conn = db()
    row = conn.execute("SELECT status FROM orders WHERE download_token=?", (token,)).fetchone()
    conn.close()

    if not row or row[0] != "paid":
        raise HTTPException(status_code=403, detail="Invalid or unpaid token.")

    # Metadata used for the ZIP header/README/PDF title; prompts come from CSV.
    cfg = LibraryConfig(
        name=f"{APP_NAME} v{APP_VERSION} — {PRODUCT_NAME} ({PRODUCT_PROMPT_COUNT} prompts)",
        audience="Professional users (Operations / Management / Knowledge work)",
        categories=[],  # retained for compatibility; source-of-truth is the CSV
    )

    zip_bytes = build_library_zip(cfg)

    return StreamingResponse(
        io.BytesIO(zip_bytes),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="prompt_library.zip"'},
    )
