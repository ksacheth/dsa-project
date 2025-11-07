import os
import sys
import hashlib
import heapq
import time
import random
import string
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from collections import defaultdict

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

# Multiple hash functions for benchmarking
def hash_fn_md5(s: str) -> int:
    return int.from_bytes(hashlib.md5(s.encode("utf-8")).digest(), "big")

def hash_fn_sha1(s: str) -> int:
    return int.from_bytes(hashlib.sha1(s.encode("utf-8")).digest(), "big")

def hash_fn_sha256(s: str) -> int:
    return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest(), "big")

def hash_fn_djb2(s: str) -> int:
    """DJB2 hash - simple multiplicative hash"""
    h = 5381
    for c in s:
        h = ((h << 5) + h) + ord(c)  # h * 33 + c
    return h & 0xFFFFFFFFFFFFFFFF

def hash_fn_murmur3(s: str) -> int:
    """Simplified MurmurHash3 implementation"""
    data = s.encode('utf-8')
    c1 = 0xcc9e2d51
    c2 = 0x1b873593
    h = 0x9747b28c  # seed

    for i in range(0, len(data) - 3, 4):
        k = int.from_bytes(data[i:i+4], 'little')
        k = (k * c1) & 0xFFFFFFFF
        k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
        k = (k * c2) & 0xFFFFFFFF

        h ^= k
        h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF
        h = (h * 5 + 0xe6546b64) & 0xFFFFFFFF

    # Handle remaining bytes
    remainder = len(data) % 4
    if remainder:
        k = 0
        for i in range(remainder):
            k |= data[len(data) - remainder + i] << (i * 8)
        k = (k * c1) & 0xFFFFFFFF
        k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
        k = (k * c2) & 0xFFFFFFFF
        h ^= k

    h ^= len(data)
    h ^= (h >> 16)
    h = (h * 0x85ebca6b) & 0xFFFFFFFF
    h ^= (h >> 13)
    h = (h * 0xc2b2ae35) & 0xFFFFFFFF
    h ^= (h >> 16)

    return h

def hash_fn_cityhash(s: str) -> int:
    """Simplified CityHash-like implementation"""
    data = s.encode('utf-8')
    if len(data) <= 16:
        return hash_fn_murmur3(s)

    # CityHash constants
    k0 = 0xc3a5c85c97cb3127
    k1 = 0xb492b66fbe98f273
    k2 = 0x9ae16a3b2f90404f

    h = len(data) * k2
    for i in range(0, len(data) - 7, 8):
        chunk = int.from_bytes(data[i:i+8], 'little')
        h ^= chunk
        h = (h * k0) & 0xFFFFFFFFFFFFFFFF
        h = ((h << 31) | (h >> 33)) & 0xFFFFFFFFFFFFFFFF
        h = (h * k1) & 0xFFFFFFFFFFFFFFFF

    return h

# Hash function registry
HASH_FUNCTIONS = {
    "md5": hash_fn_md5,
    "sha1": hash_fn_sha1,
    "sha256": hash_fn_sha256,
    "djb2": hash_fn_djb2,
    "murmur3": hash_fn_murmur3,
    "cityhash": hash_fn_cityhash,
}

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
# Instrumented HashMap for benchmarking
# -------------------------------
class InstrumentedHashMap:
    """HashMap with comprehensive metrics collection for benchmarking"""
    def __init__(self, capacity=2048, strategy="separate_chaining", hash_fn_name="md5"):
        self.capacity = max(16, capacity)
        self.initial_capacity = self.capacity
        self.strategy = strategy
        self.hash_fn_name = hash_fn_name
        self.hash_fn = HASH_FUNCTIONS.get(hash_fn_name, hash_fn_md5)
        self.hash_fn2 = hash_fn_sha1  # secondary hash for double hashing

        # Metrics
        self.metrics = {
            "collision_count": 0,
            "total_probe_length": 0,
            "max_probe_length": 0,
            "insertion_times": [],
            "lookup_times": [],
            "deletion_times": [],
            "rehash_count": 0,
            "element_count": 0,
            "hash_computation_times": [],
        }

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

    def _h1(self, key: str) -> int:
        start = time.perf_counter()
        result = self.hash_fn(key) % self.capacity
        self.metrics["hash_computation_times"].append(time.perf_counter() - start)
        return result

    def _h2(self, key: str) -> int:
        return (self.hash_fn2(key) % (self.capacity - 1)) + 1

    def insert(self, key: str, value: Any):
        start_time = time.perf_counter()
        probe_length = 0

        if self.strategy == "separate_chaining":
            idx = self._h1(key)
            bucket = self.buckets[idx]

            # Track collision if bucket not empty
            if len(bucket) > 0:
                self.metrics["collision_count"] += 1
                probe_length = len(bucket)

            for i, (k, _) in enumerate(bucket):
                if k == key:
                    bucket[i] = (key, value)
                    self.metrics["insertion_times"].append(time.perf_counter() - start_time)
                    return
            bucket.append((key, value))
            self.metrics["element_count"] += 1

        elif self.strategy == "linear_probing":
            idx = self._h1(key)
            for i in range(self.capacity):
                j = (idx + i) % self.capacity
                k = self.keys[j]
                probe_length = i + 1

                if i > 0:
                    self.metrics["collision_count"] += 1

                if k is None or k is self.tombstone or k == key:
                    if k != key and k is not None:
                        self.metrics["element_count"] += 1
                    elif k is None or k is self.tombstone:
                        self.metrics["element_count"] += 1
                    self.keys[j] = key
                    self.values[j] = value
                    break
            else:
                raise RuntimeError("HashMap full")

        elif self.strategy == "double_hashing":
            idx = self._h1(key)
            step = self._h2(key)
            for i in range(self.capacity):
                j = (idx + i * step) % self.capacity
                k = self.keys[j]
                probe_length = i + 1

                if i > 0:
                    self.metrics["collision_count"] += 1

                if k is None or k is self.tombstone or k == key:
                    if k != key and k is not None:
                        self.metrics["element_count"] += 1
                    elif k is None or k is self.tombstone:
                        self.metrics["element_count"] += 1
                    self.keys[j] = key
                    self.values[j] = value
                    break
            else:
                raise RuntimeError("HashMap full")

        elif self.strategy == "cuckoo":
            key_i, val_i = key, value
            for iteration in range(500):
                probe_length = iteration + 1
                if iteration > 0:
                    self.metrics["collision_count"] += 1

                idx1 = self._h1(key_i)
                if self.table1_keys[idx1] is None or self.table1_keys[idx1] == key_i:
                    if self.table1_keys[idx1] is None:
                        self.metrics["element_count"] += 1
                    self.table1_keys[idx1] = key_i
                    self.table1_vals[idx1] = val_i
                    break
                # kick from table1
                key_i, val_i, self.table1_keys[idx1], self.table1_vals[idx1] = (
                    self.table1_keys[idx1],
                    self.table1_vals[idx1],
                    key_i,
                    val_i,
                )
                idx2 = self._h2(key_i) % self.capacity
                if self.table2_keys[idx2] is None or self.table2_keys[idx2] == key_i:
                    if self.table2_keys[idx2] is None:
                        self.metrics["element_count"] += 1
                    self.table2_keys[idx2] = key_i
                    self.table2_vals[idx2] = val_i
                    break
                # kick from table2
                key_i, val_i, self.table2_keys[idx2], self.table2_vals[idx2] = (
                    self.table2_keys[idx2],
                    self.table2_vals[idx2],
                    key_i,
                    val_i,
                )
            else:
                # grow + reinsert
                self._grow()
                return self.insert(key, value)

        self.metrics["total_probe_length"] += probe_length
        self.metrics["max_probe_length"] = max(self.metrics["max_probe_length"], probe_length)
        self.metrics["insertion_times"].append(time.perf_counter() - start_time)

    def get(self, key: str) -> Optional[Any]:
        start_time = time.perf_counter()
        result = None

        if self.strategy == "separate_chaining":
            idx = self._h1(key)
            for k, v in self.buckets[idx]:
                if k == key:
                    result = v
                    break

        elif self.strategy == "linear_probing":
            idx = self._h1(key)
            for i in range(self.capacity):
                j = (idx + i) % self.capacity
                k = self.keys[j]
                if k is None:
                    break
                if k == key:
                    result = self.values[j]
                    break

        elif self.strategy == "double_hashing":
            idx = self._h1(key)
            step = self._h2(key)
            for i in range(self.capacity):
                j = (idx + i * step) % self.capacity
                k = self.keys[j]
                if k is None:
                    break
                if k == key:
                    result = self.values[j]
                    break

        elif self.strategy == "cuckoo":
            idx1 = self._h1(key)
            if self.table1_keys[idx1] == key:
                result = self.table1_vals[idx1]
            else:
                idx2 = self._h2(key) % self.capacity
                if self.table2_keys[idx2] == key:
                    result = self.table2_vals[idx2]

        self.metrics["lookup_times"].append(time.perf_counter() - start_time)
        return result

    def _grow(self):
        old = self.__dict__.copy()
        new_capacity = self.capacity * 2
        self.capacity = new_capacity
        self.metrics["rehash_count"] += 1

        # Reinitialize structures
        if self.strategy == "separate_chaining":
            self.buckets = [[] for _ in range(self.capacity)]
        elif self.strategy in ("linear_probing", "double_hashing"):
            self.keys = [None] * self.capacity
            self.values = [None] * self.capacity
        elif self.strategy == "cuckoo":
            self.table1_keys = [None] * self.capacity
            self.table1_vals = [None] * self.capacity
            self.table2_keys = [None] * self.capacity
            self.table2_vals = [None] * self.capacity

        # Reinsert all elements
        old_count = self.metrics["element_count"]
        self.metrics["element_count"] = 0

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

    def get_load_factor(self) -> float:
        return self.metrics["element_count"] / self.capacity if self.capacity > 0 else 0

    def get_metrics_summary(self) -> Dict:
        """Return computed metrics"""
        insertion_times = self.metrics["insertion_times"]
        lookup_times = self.metrics["lookup_times"]
        hash_times = self.metrics["hash_computation_times"]

        avg_probe = (self.metrics["total_probe_length"] / max(1, self.metrics["element_count"]))

        return {
            "collision_count": self.metrics["collision_count"],
            "avg_probe_length": round(avg_probe, 4),
            "max_probe_length": self.metrics["max_probe_length"],
            "load_factor": round(self.get_load_factor(), 4),
            "final_capacity": self.capacity,
            "initial_capacity": self.initial_capacity,
            "rehash_count": self.metrics["rehash_count"],
            "element_count": self.metrics["element_count"],
            "avg_insertion_time_us": round(sum(insertion_times) / max(1, len(insertion_times)) * 1_000_000, 2),
            "total_insertion_time_ms": round(sum(insertion_times) * 1000, 2),
            "avg_lookup_time_us": round(sum(lookup_times) / max(1, len(lookup_times)) * 1_000_000, 2) if lookup_times else 0,
            "avg_hash_time_us": round(sum(hash_times) / max(1, len(hash_times)) * 1_000_000, 4) if hash_times else 0,
        }

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
# URL Generator for Benchmarking
# -------------------------------
def generate_test_urls(count: int = 10000) -> List[str]:
    """Generate realistic URLs for testing"""
    urls = []

    # Popular domains
    domains = [
        "google.com", "youtube.com", "facebook.com", "twitter.com", "instagram.com",
        "linkedin.com", "reddit.com", "amazon.com", "netflix.com", "github.com",
        "stackoverflow.com", "medium.com", "wikipedia.org", "apple.com", "microsoft.com",
        "yahoo.com", "pinterest.com", "tumblr.com", "wordpress.com", "blogger.com"
    ]

    # Path components
    paths = [
        "search", "user", "profile", "post", "article", "video", "watch", "page",
        "category", "product", "item", "blog", "docs", "api", "dashboard", "settings"
    ]

    # Generate varied URLs
    for i in range(count):
        domain = random.choice(domains)

        # Mix of URL patterns
        pattern = random.randint(1, 10)

        if pattern <= 3:  # Simple domain
            url = f"https://{domain}"
        elif pattern <= 6:  # Single path
            path = random.choice(paths)
            url = f"https://{domain}/{path}"
        elif pattern <= 8:  # Multi-level path
            path1 = random.choice(paths)
            path2 = random.choice(paths)
            rand_id = random.randint(1000, 999999)
            url = f"https://{domain}/{path1}/{path2}/{rand_id}"
        else:  # With query parameters
            path = random.choice(paths)
            param_count = random.randint(1, 3)
            params = []
            for _ in range(param_count):
                key = random.choice(["q", "id", "page", "sort", "filter", "user", "tag"])
                value = ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(3, 10)))
                params.append(f"{key}={value}")
            url = f"https://{domain}/{path}?{'&'.join(params)}"

        # Add some similar URLs to test avalanche effect
        if i % 100 == 0 and i > 0:
            # Create similar URL to previous one
            prev_url = urls[-1]
            url = prev_url + str(random.randint(0, 9))

        urls.append(url)

    return urls

# -------------------------------
# Benchmark Runner
# -------------------------------
class BenchmarkRunner:
    """Runs comprehensive benchmarks on HashMap implementations"""

    @staticmethod
    def benchmark_collision_strategies(url_count: int = 10000, capacities: List[int] = None, hash_fn_name: str = "md5"):
        """Test all collision strategies with a specified hash function"""
        if capacities is None:
            capacities = [8192, 16384, 32768]

        strategies = ["separate_chaining", "linear_probing", "double_hashing", "cuckoo"]

        results = {}
        urls = generate_test_urls(url_count)

        for capacity in capacities:
            results[capacity] = {}

            for strategy in strategies:
                logger.info(f"Testing {strategy} with capacity {capacity} using {hash_fn_name}")

                hashmap = InstrumentedHashMap(
                    capacity=capacity,
                    strategy=strategy,
                    hash_fn_name=hash_fn_name
                )

                # Insert all URLs
                for url in urls:
                    try:
                        hashmap.insert(url, {"url": url})
                    except RuntimeError as e:
                        logger.warning(f"Failed to insert into {strategy}: {e}")
                        break

                # Perform lookups to test read performance
                sample_urls = random.sample(urls, min(1000, len(urls)))
                for url in sample_urls:
                    hashmap.get(url)

                # Get metrics
                metrics = hashmap.get_metrics_summary()
                metrics["strategy"] = strategy
                metrics["hash_function"] = hash_fn_name
                metrics["test_url_count"] = url_count

                results[capacity][strategy] = metrics

        return results, urls

    @staticmethod
    def benchmark_hash_functions(url_count: int = 10000, capacity: int = 16384, strategy: str = "separate_chaining"):
        """Test all hash functions with a specified collision strategy"""
        hash_functions = list(HASH_FUNCTIONS.keys())

        results = {}
        urls = generate_test_urls(url_count)

        for hash_fn in hash_functions:
            logger.info(f"Testing hash function: {hash_fn} with strategy {strategy}")

            hashmap = InstrumentedHashMap(
                capacity=capacity,
                strategy=strategy,
                hash_fn_name=hash_fn
            )

            # Track hash distribution
            bucket_sizes = defaultdict(int)

            # Insert all URLs
            for url in urls:
                hashmap.insert(url, {"url": url})
                # Track bucket distribution for separate chaining
                if strategy == "separate_chaining":
                    idx = hashmap._h1(url)
                    bucket_sizes[idx] = len(hashmap.buckets[idx])

            # Perform lookups
            sample_urls = random.sample(urls, min(1000, len(urls)))
            for url in sample_urls:
                hashmap.get(url)

            # Calculate distribution metrics
            if bucket_sizes:
                sizes = list(bucket_sizes.values())
                avg_bucket_size = sum(sizes) / len(sizes)
                # Standard deviation
                variance = sum((x - avg_bucket_size) ** 2 for x in sizes) / len(sizes)
                std_dev = variance ** 0.5

                # Chi-square test for uniformity
                expected = url_count / capacity
                chi_square = sum((size - expected) ** 2 / expected for size in sizes if expected > 0)
            else:
                std_dev = 0
                chi_square = 0

            # Get metrics
            metrics = hashmap.get_metrics_summary()
            metrics["hash_function"] = hash_fn
            metrics["strategy"] = strategy
            metrics["test_url_count"] = url_count
            metrics["distribution_std_dev"] = round(std_dev, 4)
            metrics["chi_square_statistic"] = round(chi_square, 2)

            results[hash_fn] = metrics

        return results, urls

# In-memory storage for benchmark results
benchmark_results_cache = {}

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

# -------------------------------
# Benchmark Endpoints (must come before /{short_code})
# -------------------------------
@app.get("/benchmark")
async def render_benchmark_page(request: Request):
    """Render the benchmark visualization page"""
    return templates.TemplateResponse("benchmark.html", {"request": request})

@app.post("/api/benchmark/collision-strategies")
async def run_collision_benchmark(
    url_count: int = 10000,
    capacity: int = 8192,
    hash_function: str = "md5"
):
    """Run benchmark comparing collision strategies"""
    try:
        capacities = [capacity]

        logger.info(f"Starting collision strategy benchmark with {url_count} URLs using {hash_function} with capacity {capacity}")
        results, urls = BenchmarkRunner.benchmark_collision_strategies(url_count, capacities, hash_function)

        # Cache results
        benchmark_results_cache["collision_strategies"] = results

        return JSONResponse(content={
            "success": True,
            "results": results,
            "test_params": {
                "url_count": url_count,
                "capacities": capacities,
                "hash_function": hash_function,
                "sample_urls": urls[:50]  # Return first 50 URLs as sample
            }
        })
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/benchmark/hash-functions")
async def run_hash_benchmark(
    url_count: int = 10000,
    capacity: int = 16384,
    strategy: str = "separate_chaining"
):
    """Run benchmark comparing hash functions"""
    try:
        logger.info(f"Starting hash function benchmark with {url_count} URLs using {strategy}")
        results, urls = BenchmarkRunner.benchmark_hash_functions(url_count, capacity, strategy)

        # Cache results
        benchmark_results_cache["hash_functions"] = results

        return JSONResponse(content={
            "success": True,
            "results": results,
            "test_params": {
                "url_count": url_count,
                "capacity": capacity,
                "collision_strategy": strategy,
                "sample_urls": urls[:50]  # Return first 50 URLs as sample
            }
        })
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/benchmark/results")
async def get_benchmark_results():
    """Get cached benchmark results"""
    return JSONResponse(content={
        "success": True,
        "results": benchmark_results_cache
    })

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

# This catch-all route must come LAST
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

# -------------------------------
# Entry
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)