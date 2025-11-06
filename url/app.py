# import os
# import sys
# import hashlib
# import string
# import random
# import heapq
# from datetime import datetime, timedelta
# from typing import Optional, Dict, Any, Tuple
# from fastapi import FastAPI, HTTPException, Request, Form
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import RedirectResponse, JSONResponse
# from jinja2 import Environment, FileSystemLoader
# from loguru import logger

# # -------------------------------
# # Setup
# # -------------------------------
# logger.add(sys.stdout, format="{time} {level} {message}", level="DEBUG")
# app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["GET", "POST", "OPTIONS"],
#     allow_headers=["*"],
# )

# package_dir = os.path.dirname(__file__)
# static_dir = os.path.join(package_dir, "static")
# fonts_dir = os.path.join(static_dir, "fonts/IBM_Plex")
# app.mount("/static", StaticFiles(directory=static_dir), name="static")
# app.mount("/fonts", StaticFiles(directory=fonts_dir), name="fonts")

# templates_dir = os.path.join(os.path.dirname(__file__), "templates")
# jinja_env = Environment(loader=FileSystemLoader(templates_dir))
# templates = Jinja2Templates(directory="templates")
# templates.env = jinja_env

# # -------------------------------
# # DSA helpers: Base62 + hashing
# # -------------------------------
# BASE62_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

# def base62_encode(num: int) -> str:
#     if num == 0:
#         return BASE62_ALPHABET[0]
#     arr = []
#     base = len(BASE62_ALPHABET)
#     while num:
#         num, rem = divmod(num, base)
#         arr.append(BASE62_ALPHABET[rem])
#     arr.reverse()
#     return "".join(arr)

# def sha256_int(s: str) -> int:
#     return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest(), "big")

# def generate_deterministic_code(url: str, salt: str = "", counter: int = 0, length: int = 6) -> str:
#     """Generate deterministic short code using SHA256 + Base62."""
#     payload = f"{url}|{salt}|{counter}"
#     n = sha256_int(payload)
#     code = base62_encode(n)
#     if len(code) < length:
#         code = (code * ((length // len(code)) + 1))[:length]
#     return code[:length]

# # -------------------------------
# # DSA structure: HashMap with separate chaining
# # -------------------------------
# class HashMap:
#     def __init__(self, capacity=2048):
#         self.capacity = capacity
#         self.buckets = [[] for _ in range(capacity)]

#     def _index(self, key: str) -> int:
#         return sha256_int(key) % self.capacity

#     def insert(self, key: str, value: Any):
#         idx = self._index(key)
#         bucket = self.buckets[idx]
#         for i, (k, v) in enumerate(bucket):
#             if k == key:
#                 bucket[i] = (key, value)
#                 return
#         bucket.append((key, value))

#     def get(self, key: str) -> Optional[Any]:
#         idx = self._index(key)
#         for k, v in self.buckets[idx]:
#             if k == key:
#                 return v
#         return None

#     def delete(self, key: str):
#         idx = self._index(key)
#         bucket = self.buckets[idx]
#         for i, (k, v) in enumerate(bucket):
#             if k == key:
#                 del bucket[i]
#                 return

# # -------------------------------
# # URL store + DSA-based expiration queue
# # -------------------------------
# url_store = HashMap()
# CLICK_STATS = {}
# expiry_heap: list[Tuple[float, str]] = []  # (timestamp, short_code)

# def cleanup_expired():
#     """Remove all expired links whose time <= now in O(log n) time using heap."""
#     now = datetime.now().timestamp()
#     while expiry_heap and expiry_heap[0][0] <= now:
#         exp_time, code = heapq.heappop(expiry_heap)
#         record = url_store.get(code)
#         if record and record["expiry"] and record["expiry"].timestamp() <= now:
#             logger.info(f"Auto-cleaning expired link {code}")
#             url_store.delete(code)
#             CLICK_STATS.pop(code, None)

# # -------------------------------
# # Routes
# # -------------------------------
# @app.get("/")
# async def render_page(request: Request):
#     return templates.TemplateResponse("page.html", {"request": request})

# @app.post("/shorten")
# async def shorten_url(url: str = Form(...), expiry: str = Form(...)):
#     """Shorten URL deterministically, store with optional expiry."""
#     cleanup_expired()

#     counter = 0
#     short_code = generate_deterministic_code(url, salt="v1", counter=counter)

#     # Handle collisions
#     while url_store.get(short_code) and url_store.get(short_code)["url"] != url:
#         counter += 1
#         short_code = generate_deterministic_code(url, salt="v1", counter=counter)

#     # Expiry handling
#     expiry_time = None
#     if expiry != "never":
#         expiry_time = datetime.now() + timedelta(seconds=int(expiry))
#         heapq.heappush(expiry_heap, (expiry_time.timestamp(), short_code))

#     url_store.insert(short_code, {"url": url, "expiry": expiry_time})
#     CLICK_STATS[short_code] = 0

#     short_full = f"http://localhost:8000/{short_code}"
#     return JSONResponse(content={"short_url": short_full})

# @app.get("/{short_code}")
# async def redirect_to_original(request: Request, short_code: str):
#     """Redirect to original URL; cleanup expired items first."""
#     cleanup_expired()

#     record = url_store.get(short_code)
#     if not record:
#         raise HTTPException(status_code=404, detail="Short URL not found")

#     expiry = record.get("expiry")
#     if expiry and datetime.now() > expiry:
#         url_store.delete(short_code)
#         return templates.TemplateResponse("expired.html", {"request": request})

#     CLICK_STATS[short_code] = CLICK_STATS.get(short_code, 0) + 1
#     return RedirectResponse(record["url"])

# @app.get("/stats/{short_code}")
# async def get_stats(request: Request, short_code: str):
#     """Return stats as JSON or HTML page depending on client Accept header."""
#     cleanup_expired()

#     record = url_store.get(short_code)
#     if not record:
#         raise HTTPException(status_code=404, detail="Short URL not found")

#     stats = {
#         "short_code": short_code,
#         "original_url": record["url"],
#         "clicks": CLICK_STATS.get(short_code, 0),
#         "expiry": record["expiry"].isoformat() if record["expiry"] else "never",
#     }

#     # Render HTML if browser requested it
#     accept = request.headers.get("accept", "")
#     if "text/html" in accept:
#         return templates.TemplateResponse("stats.html", {"request": request, "stats": stats})
#     return JSONResponse(content=stats)

# # -------------------------------
# # Server entry point
# # -------------------------------
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

import os
import sys
import hashlib
import heapq
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Paths
package_dir = os.path.dirname(__file__)
static_dir = os.path.join(package_dir, "static")
fonts_dir = os.path.join(static_dir, "fonts/IBM_Plex")
templates_dir = os.path.join(package_dir, "templates")

# Mount static if present (avoids crash if folders not created yet)
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
if os.path.isdir(fonts_dir):
    app.mount("/fonts", StaticFiles(directory=fonts_dir), name="fonts")

# Jinja
jinja_env = Environment(loader=FileSystemLoader(templates_dir))
templates = Jinja2Templates(directory=templates_dir)
templates.env = jinja_env

# -------------------------------
# Base62 + two hash functions (md5 + sha1)
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

# Two hash fns (int outputs); decent collisions in short-prefix space
def hash_fn_md5(s: str) -> int:
    return int.from_bytes(hashlib.md5(s.encode("utf-8")).digest(), "big")

def hash_fn_sha1(s: str) -> int:
    return int.from_bytes(hashlib.sha1(s.encode("utf-8")).digest(), "big")

# Shortcode generator — includes timestamp so same URL can yield different codes
def generate_code_with_hash(
    url: str,
    *,
    salt: str = "",
    counter: int = 0,
    length: int = 6,
    use_sha1: bool = False,
    timestamp: Optional[datetime] = None,
) -> str:
    ts = (timestamp or datetime.now()).isoformat()
    payload = f"{url}|{salt}|{counter}|{ts}"
    n = hash_fn_sha1(payload) if use_sha1 else hash_fn_md5(payload)
    code = base62_encode(n)
    if len(code) < length:
        code = (code * ((length // len(code)) + 1))[:length]
    return code[:length]

# -------------------------------
# Hash store with 4 collision strategies
# -------------------------------
class HashMap:
    def __init__(self, capacity=2048, strategy="separate_chaining"):
        self.capacity = max(16, capacity)
        self.strategy = strategy
        if strategy == "separate_chaining":
            self.buckets = [[] for _ in range(self.capacity)]
        elif strategy in ("linear_probing", "double_hashing"):
            self.keys = [None] * self.capacity
            self.values = [None] * self.capacity
            self.tombstone = object()
        elif strategy == "cuckoo":
            self.table1_keys = [None] * self.capacity
            self.table1_vals = [None] * self.capacity
            self.table2_keys = [None] * self.capacity
            self.table2_vals = [None] * self.capacity
        else:
            raise ValueError("unknown strategy")

    # helper hashes
    def _h1(self, key: str) -> int:
        return hash_fn_md5(key) % self.capacity

    def _h2(self, key: str) -> int:
        # avoid zero step
        return (hash_fn_sha1(key) % (self.capacity - 1)) + 1

    def insert(self, key: str, value: Any):
        if self.strategy == "separate_chaining":
            idx = self._h1(key)
            bucket = self.buckets[idx]
            for i, (k, _) in enumerate(bucket):
                if k == key:
                    bucket[i] = (key, value)
                    return
            bucket.append((key, value))

        elif self.strategy == "linear_probing":
            idx = self._h1(key)
            for i in range(self.capacity):
                j = (idx + i) % self.capacity
                k = self.keys[j]
                if k is None or k is self.tombstone or k == key:
                    self.keys[j] = key
                    self.values[j] = value
                    return
            raise RuntimeError("HashMap full")

        elif self.strategy == "double_hashing":
            idx = self._h1(key)
            step = self._h2(key)
            for i in range(self.capacity):
                j = (idx + i * step) % self.capacity
                k = self.keys[j]
                if k is None or k is self.tombstone or k == key:
                    self.keys[j] = key
                    self.values[j] = value
                    return
            raise RuntimeError("HashMap full")

        elif self.strategy == "cuckoo":
            key_i, val_i = key, value
            for _ in range(500):
                idx1 = self._h1(key_i)
                if self.table1_keys[idx1] is None or self.table1_keys[idx1] == key_i:
                    self.table1_keys[idx1] = key_i
                    self.table1_vals[idx1] = val_i
                    return
                # kick from table1
                key_i, val_i, self.table1_keys[idx1], self.table1_vals[idx1] = (
                    self.table1_keys[idx1],
                    self.table1_vals[idx1],
                    key_i,
                    val_i,
                )
                idx2 = self._h2(key_i) % self.capacity
                if self.table2_keys[idx2] is None or self.table2_keys[idx2] == key_i:
                    self.table2_keys[idx2] = key_i
                    self.table2_vals[idx2] = val_i
                    return
                # kick from table2
                key_i, val_i, self.table2_keys[idx2], self.table2_vals[idx2] = (
                    self.table2_keys[idx2],
                    self.table2_vals[idx2],
                    key_i,
                    val_i,
                )
            # grow + reinsert
            self._grow()
            return self.insert(key, value)

    def get(self, key: str) -> Optional[Any]:
        if self.strategy == "separate_chaining":
            idx = self._h1(key)
            for k, v in self.buckets[idx]:
                if k == key:
                    return v
            return None

        elif self.strategy == "linear_probing":
            idx = self._h1(key)
            for i in range(self.capacity):
                j = (idx + i) % self.capacity
                k = self.keys[j]
                if k is None:
                    return None
                if k == key:
                    return self.values[j]
            return None

        elif self.strategy == "double_hashing":
            idx = self._h1(key)
            step = self._h2(key)
            for i in range(self.capacity):
                j = (idx + i * step) % self.capacity
                k = self.keys[j]
                if k is None:
                    return None
                if k == key:
                    return self.values[j]
            return None

        elif self.strategy == "cuckoo":
            idx1 = self._h1(key)
            if self.table1_keys[idx1] == key:
                return self.table1_vals[idx1]
            idx2 = self._h2(key) % self.capacity
            if self.table2_keys[idx2] == key:
                return self.table2_vals[idx2]
            return None

    def delete(self, key: str):
        if self.strategy == "separate_chaining":
            idx = self._h1(key)
            bucket = self.buckets[idx]
            for i, (k, _) in enumerate(bucket):
                if k == key:
                    del bucket[i]
                    return

        elif self.strategy in ("linear_probing", "double_hashing"):
            idx = self._h1(key)
            step = self._h2(key) if self.strategy == "double_hashing" else 1
            for i in range(self.capacity):
                j = (idx + i * step) % self.capacity
                k = self.keys[j]
                if k is None:
                    return
                if k == key:
                    self.keys[j] = self.tombstone
                    self.values[j] = None
                    return

        elif self.strategy == "cuckoo":
            idx1 = self._h1(key)
            if self.table1_keys[idx1] == key:
                self.table1_keys[idx1] = None
                self.table1_vals[idx1] = None
                return
            idx2 = self._h2(key) % self.capacity
            if self.table2_keys[idx2] == key:
                self.table2_keys[idx2] = None
                self.table2_vals[idx2] = None
                return

    def _grow(self):
        old = self.__dict__.copy()
        new_capacity = self.capacity * 2
        self.__init__(capacity=new_capacity, strategy=self.strategy)
        # reinsert
        if self.strategy == "separate_chaining":
            for bucket in old["buckets"]:
                for k, v in bucket:
                    self.insert(k, v)
        elif self.strategy in ("linear_probing", "double_hashing"):
            for k, v in zip(old["keys"], old["values"]):
                if k is not None and k is not old.get("tombstone", None):
                    self.insert(k, v)
        elif self.strategy == "cuckoo":
            for k, v in zip(old.get("table1_keys", []), old.get("table1_vals", [])):
                if k:
                    self.insert(k, v)
            for k, v in zip(old.get("table2_keys", []), old.get("table2_vals", [])):
                if k:
                    self.insert(k, v)

# -------------------------------
# Multi-store to support dropdown strategies
# -------------------------------
STRATEGIES = ["separate_chaining", "linear_probing", "double_hashing", "cuckoo"]
stores: Dict[str, HashMap] = {s: HashMap(capacity=2048, strategy=s) for s in STRATEGIES}
CLICK_STATS: Dict[str, Dict[str, int]] = {s: {} for s in STRATEGIES}
expiry_heaps: Dict[str, list] = {s: [] for s in STRATEGIES}  # list[(timestamp, code)]

# -------------------------------
# Expiry cleanup
# -------------------------------
def cleanup_expired(strategy: Optional[str] = None):
    now = datetime.now().timestamp()
    strategies = [strategy] if strategy else STRATEGIES
    for strat in strategies:
        heap = expiry_heaps[strat]
        store = stores[strat]
        while heap and heap[0][0] <= now:
            exp_time, code = heapq.heappop(heap)
            record = store.get(code)
            if record and record.get("expiry") and record["expiry"].timestamp() <= now:
                logger.info(f"Auto-cleaning expired link {code} (strategy={strat})")
                store.delete(code)
                CLICK_STATS[strat].pop(code, None)

# -------------------------------
# Routes
# -------------------------------
@app.get("/")
async def render_page(request: Request):
    # expects templates/page.html to exist
    return templates.TemplateResponse("page.html", {"request": request, "title": "Linky"})

@app.post("/shorten")
async def shorten_url(
    request: Request,
    url: str = Form(...),
    expiry: str = Form(...),
    strategy: str = Form("separate_chaining"),
):
    """Shorten URL; collision strategy comes from form dropdown."""
    if strategy not in STRATEGIES:
        raise HTTPException(status_code=400, detail="Unknown collision strategy")

    cleanup_expired(strategy)

    store = stores[strategy]
    stats = CLICK_STATS[strategy]
    heap = expiry_heaps[strategy]

    counter = 0
    # time-based generation so same URL can produce different shortcodes
    short_code = generate_code_with_hash(
        url, salt="v1", counter=counter, length=6, use_sha1=(strategy == "cuckoo")
    )

    # handle shortcode-level collisions by bumping counter + toggling hash
    while store.get(short_code) and store.get(short_code).get("url") != url:
        counter += 1
        short_code = generate_code_with_hash(
            url,
            salt="v1",
            counter=counter,
            length=6,
            use_sha1=(counter % 2 == 0),
        )

    expiry_time = None
    if expiry != "never":
        expiry_time = datetime.now() + timedelta(seconds=int(expiry))
        heapq.heappush(heap, (expiry_time.timestamp(), short_code))

    store.insert(short_code, {"url": url, "expiry": expiry_time})
    stats.setdefault(short_code, 0)

    # Build absolute short URL
    short_full = str(request.base_url).rstrip("/") + "/" + short_code

    response = {"short_url": short_full}
    if url.startswith("http://"):
        response["warning"] = "The URL uses http — not secure. Consider using https."
    return JSONResponse(content=response)

@app.get("/{short_code}")
async def redirect_to_original(request: Request, short_code: str):
    cleanup_expired()

    record = None
    used_strategy = None
    for strat, store in stores.items():
        rec = store.get(short_code)
        if rec:
            record = rec
            used_strategy = strat
            break

    if not record:
        raise HTTPException(status_code=404, detail="Short URL not found")

    expiry = record.get("expiry")
    if expiry and datetime.now() > expiry:
        stores[used_strategy].delete(short_code)
        return templates.TemplateResponse("expired.html", {"request": request})

    CLICK_STATS[used_strategy][short_code] = CLICK_STATS[used_strategy].get(short_code, 0) + 1
    return RedirectResponse(record["url"])

@app.get("/stats/{short_code}")
async def get_stats(request: Request, short_code: str):
    cleanup_expired()

    record = None
    used_strategy = None
    for strat, store in stores.items():
        rec = store.get(short_code)
        if rec:
            record = rec
            used_strategy = strat
            break

    if not record:
        raise HTTPException(status_code=404, detail="Short URL not found")

    stats = {
        "short_code": short_code,
        "original_url": record["url"],
        "clicks": CLICK_STATS[used_strategy].get(short_code, 0),
        "expiry": record["expiry"].isoformat() if record["expiry"] else "never",
        "strategy": used_strategy,
    }

    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return templates.TemplateResponse("stats.html", {"request": request, "stats": stats})
    return JSONResponse(content=stats)

# -------------------------------
# Entry
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)