import asyncio, time, re
from typing import Dict, List
from collections import OrderedDict
import httpx
from app.utils.logger import app_logger
from app.config import settings

MODE_SEARCHES = {"standard": 5, "reasoning": 12, "deep": 30}

_PRICE_RE = re.compile(
    r"(?:₹|Rs\.?|INR|rupees?)\s*(\d[\d,]*(?:\.\d+)?)|(\d[\d,]*(?:\.\d)?)\s*(?:INR|rupees?)",
    re.IGNORECASE,
)

class AsyncLRUCache:
    def __init__(self, maxsize=200, ttl=86400):
        self.maxsize, self.ttl, self.cache = maxsize, ttl, OrderedDict()
    def get(self, k):
        if k in self.cache:
            v, ts = self.cache[k]
            if time.time() - ts < self.ttl:
                self.cache.move_to_end(k); return v
            del self.cache[k]
        return None
    def put(self, k, v):
        if k in self.cache: self.cache.move_to_end(k)
        self.cache[k] = (v, time.time())
        if len(self.cache) > self.maxsize: self.cache.popitem(last=False)

def _extract_prices(text: str) -> List[float]:
    prices: List[float] = []
    for m in _PRICE_RE.finditer(text):
        val = m.group(1) or m.group(2)
        if not val:
            continue
        try:
            prices.append(float(val.replace(",", "")))
        except ValueError:
            continue
    return prices

def _normalize_result(brand: str, panel: str, category: str, snippets: List[str]) -> Dict:
    all_text = " ".join(snippets)
    prices = _extract_prices(all_text)
    prices = [p for p in prices if 500 <= p <= 500000]
    oem = after = labor = None
    if len(prices) >= 2:
        oem, after = sorted(prices)[-1], sorted(prices)[0]
    elif len(prices) == 1:
        oem = after = prices[0]
    recall = "NONE"
    for tok in ["recall", "safety", "defect", "campaign"]:
        if tok in all_text.lower():
            recall = "POSSIBLE"
            break
    return {
        "oem_part_price": round(oem, 2) if oem is not None else 12000,
        "aftermarket_price": round(after, 2) if after is not None else 6500,
        "labor_hr_rate": 450,
        "recall": recall,
        "source": "duckduckgo",
        "brand": brand,
        "panel": panel,
        "category": category,
    }

class AsyncWebEnricher:
    def __init__(self):
        self.cache = AsyncLRUCache()
        self.circuit_open, self.failures = False, 0
        self.client = httpx.AsyncClient(timeout=12.0, headers={"User-Agent": "Mozilla/5.0"})
        app_logger.info("✅ AsyncWebEnricher initialized (DuckDuckGo pipeline)")

    async def fetch(self, brand: str, panel: str, category: str) -> Dict:
        key = f"{brand}_{panel}_{category}"
        cached = self.cache.get(key)
        if cached: return cached

        if self.circuit_open: return self._fallback(brand, panel, category)

        mode = settings.WEB_MODE or "standard"
        n_searches = MODE_SEARCHES.get(mode, 5)
        queries = self._build_queries(brand, panel, category, n_searches)

        try:
            snippets = await self._search_parallel(queries)
            data = _normalize_result(brand, panel, category, snippets)
            self.cache.put(key, data)
            self.failures = 0
            return data
        except Exception as e:
            self.failures += 1
            app_logger.error(f"❌ Web agent failed: {e}")
            if self.failures >= 3:
                self.circuit_open = True
                asyncio.create_task(self._reset_circuit())
            return self._fallback(brand, panel, category)

    def _build_queries(self, brand: str, panel: str, category: str, n: int) -> List[str]:
        base = [
            f"{brand} {panel} {category} repair cost India",
            f"{brand} {panel} replacement price OEM aftermarket",
            f"{brand} {panel} {category} spare part price",
            f"{brand} {panel} dent scratch repair cost",
            f"{brand} vehicle {panel} body shop estimate",
        ]
        return base[:n]

    async def _search_parallel(self, queries: List[str]) -> List[str]:
        sem = asyncio.Semaphore(8)

        async def one(q: str) -> str:
            async with sem:
                try:
                    resp = await self.client.post(
                        settings.WEB_SEARCH_BASE_URL,
                        data={"q": q},
                    )
                    resp.raise_for_status()
                    return self._parse_snippets(resp.text)
                except Exception:
                    return ""

        results = await asyncio.gather(*[one(q) for q in queries])
        flat: List[str] = []
        for r in results:
            flat.extend(r)
        return flat

    def _parse_snippets(self, html: str) -> List[str]:
        snippets: List[str] = []
        for m in re.finditer(r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>', html, re.S | re.I):
            text = re.sub(r"<[^>]+>", " ", m.group(1))
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                snippets.append(text)
        return snippets[:6]

    def _fallback(self, b: str, p: str, c: str) -> Dict:
        return {"oem_part_price": 10000, "aftermarket_price": 5000, "labor_hr_rate": 400, "recall": "UNKNOWN", "source": "fallback", "brand": b, "panel": p, "category": c}

    async def _reset_circuit(self):
        await asyncio.sleep(30); self.circuit_open, self.failures = False, 0
