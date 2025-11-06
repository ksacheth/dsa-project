import os
import sys
import hashlib
import string
import random
import heapq
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse
from jinja2 import Environment, FileSystemLoader
from loguru import logger

# -------------------------------
# Setup
# -------------------------------
logger.add(sys.stdout, format="{time} {level} {message}", level="DEBUG")
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

package_dir = os.path.dirname(__file__)
static_dir = os.path.join(package_dir, "static")
fonts_dir = os.path.join(static_dir, "fonts/IBM_Plex")
app.mount("/static", StaticFiles(directory=static_dir), name="static")
app.mount("/fonts", StaticFiles(directory=fonts_dir), name="fonts")

templates_dir = os.path.join(os.path.dirname(__file__), "templates")
jinja_env = Environment(loader=FileSystemLoader(templates_dir))
templates = Jinja2Templates(directory="templates")
templates.env = jinja_env

# -------------------------------
# DSA helpers: Base62 + hashing
# -------------------------------
BASE62_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

def base62_encode(num: int) -> str:
    if num == 0:
        return BASE62_ALPHABET[0]
    arr = []
    base = len(BASE62_ALPHABET)
    while num:
        num, rem = divmod(num, base)
        arr.append(BASE62_ALPHABET[rem])
    arr.reverse()
    return "".join(arr)

def sha256_int(s: str) -> int:
    return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest(), "big")

def generate_deterministic_code(url: str, salt: str = "", counter: int = 0, length: int = 6) -> str:
    """Generate deterministic short code using SHA256 + Base62."""
    payload = f"{url}|{salt}|{counter}"
    n = sha256_int(payload)
    code = base62_encode(n)
    if len(code) < length:
        code = (code * ((length // len(code)) + 1))[:length]
    return code[:length]

# -------------------------------
# DSA structure: HashMap with separate chaining
# -------------------------------
class HashMap:
    def __init__(self, capacity=2048):
        self.capacity = capacity
        self.buckets = [[] for _ in range(capacity)]

    def _index(self, key: str) -> int:
        return sha256_int(key) % self.capacity

    def insert(self, key: str, value: Any):
        idx = self._index(key)
        bucket = self.buckets[idx]
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        bucket.append((key, value))

    def get(self, key: str) -> Optional[Any]:
        idx = self._index(key)
        for k, v in self.buckets[idx]:
            if k == key:
                return v
        return None

    def delete(self, key: str):
        idx = self._index(key)
        bucket = self.buckets[idx]
        for i, (k, v) in enumerate(bucket):
            if k == key:
                del bucket[i]
                return

# -------------------------------
# URL store + DSA-based expiration queue
# -------------------------------
url_store = HashMap()
CLICK_STATS = {}
expiry_heap: list[Tuple[float, str]] = []  # (timestamp, short_code)

def cleanup_expired():
    """Remove all expired links whose time <= now in O(log n) time using heap."""
    now = datetime.now().timestamp()
    while expiry_heap and expiry_heap[0][0] <= now:
        exp_time, code = heapq.heappop(expiry_heap)
        record = url_store.get(code)
        if record and record["expiry"] and record["expiry"].timestamp() <= now:
            logger.info(f"Auto-cleaning expired link {code}")
            url_store.delete(code)
            CLICK_STATS.pop(code, None)

# -------------------------------
# Routes
# -------------------------------
@app.get("/")
async def render_page(request: Request):
    return templates.TemplateResponse("page.html", {"request": request})

@app.post("/shorten")
async def shorten_url(url: str = Form(...), expiry: str = Form(...)):
    """Shorten URL deterministically, store with optional expiry."""
    cleanup_expired()

    counter = 0
    short_code = generate_deterministic_code(url, salt="v1", counter=counter)

    # Handle collisions
    while url_store.get(short_code) and url_store.get(short_code)["url"] != url:
        counter += 1
        short_code = generate_deterministic_code(url, salt="v1", counter=counter)

    # Expiry handling
    expiry_time = None
    if expiry != "never":
        expiry_time = datetime.now() + timedelta(seconds=int(expiry))
        heapq.heappush(expiry_heap, (expiry_time.timestamp(), short_code))

    url_store.insert(short_code, {"url": url, "expiry": expiry_time})
    CLICK_STATS[short_code] = 0

    short_full = f"http://localhost:8000/{short_code}"
    return JSONResponse(content={"short_url": short_full})

@app.get("/{short_code}")
async def redirect_to_original(request: Request, short_code: str):
    """Redirect to original URL; cleanup expired items first."""
    cleanup_expired()

    record = url_store.get(short_code)
    if not record:
        raise HTTPException(status_code=404, detail="Short URL not found")

    expiry = record.get("expiry")
    if expiry and datetime.now() > expiry:
        url_store.delete(short_code)
        return templates.TemplateResponse("expired.html", {"request": request})

    CLICK_STATS[short_code] = CLICK_STATS.get(short_code, 0) + 1
    return RedirectResponse(record["url"])

@app.get("/stats/{short_code}")
async def get_stats(request: Request, short_code: str):
    """Return stats as JSON or HTML page depending on client Accept header."""
    cleanup_expired()

    record = url_store.get(short_code)
    if not record:
        raise HTTPException(status_code=404, detail="Short URL not found")

    stats = {
        "short_code": short_code,
        "original_url": record["url"],
        "clicks": CLICK_STATS.get(short_code, 0),
        "expiry": record["expiry"].isoformat() if record["expiry"] else "never",
    }

    # Render HTML if browser requested it
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return templates.TemplateResponse("stats.html", {"request": request, "stats": stats})
    return JSONResponse(content=stats)

# -------------------------------
# Server entry point
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

