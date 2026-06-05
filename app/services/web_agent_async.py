import asyncio, time
from typing import Dict
from collections import OrderedDict
from app.utils.logger import app_logger

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

class AsyncWebEnricher:
    def __init__(self):
        self.cache = AsyncLRUCache()
        self.circuit_open, self.failures = False, 0
        app_logger.info("✅ AsyncWebEnricher initialized")

    async def fetch(self, brand: str, panel: str, category: str) -> Dict:
        key = f"{brand}_{panel}_{category}"
        cached = self.cache.get(key)
        if cached: return cached

        if self.circuit_open: return self._fallback()
        try:
            # Replace with aiohttp call to Tavily/Serper/OEM API in P2
            await asyncio.sleep(0.05) 
            data = self._mock_data(brand, panel, category)
            self.cache.put(key, data)
            self.failures = 0
            return data
        except Exception as e:
            self.failures += 1
            app_logger.error(f"❌ Web agent failed: {e}")
            if self.failures >= 3:
                self.circuit_open = True
                asyncio.create_task(self._reset_circuit())
            return self._fallback()

    def _mock_data(self, b, p, c):
        return {"oem_part_price": 12000, "aftermarket_price": 6500, "labor_hr_rate": 450, "recall": "NONE", "source": "web_agent"}
    def _fallback(self):
        return {"oem_part_price": 10000, "aftermarket_price": 5000, "labor_hr_rate": 400, "recall": "UNKNOWN", "source": "fallback"}
    async def _reset_circuit(self):
        await asyncio.sleep(30); self.circuit_open, self.failures = False, 0