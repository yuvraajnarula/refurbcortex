import httpx, asyncio, time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import BaseModel, ValidationError
from typing import Dict, Optional, List
from app.utils.logger import app_logger
from collections import OrderedDict

class RecallResponse(BaseModel):
    recall_id: str
    component: str
    summary: str
    date: str

class PartsPriceResponse(BaseModel):
    oem_price_inr: float
    aftermarket_price_inr: float
    availability: str
    last_updated: str

class AsyncExternalAPIManager:
    def __init__(self, cache_ttl: int = 3600, max_cache: int = 500, timeout: float = 5.0):
        self.cache = OrderedDict()
        self.cache_ttl = cache_ttl
        self.max_cache = max_cache
        self.timeout = timeout
        self.circuit_open = False
        self.failure_count = 0
        self.client = httpx.AsyncClient(timeout=self.timeout, limits=httpx.Limits(max_connections=20))
        app_logger.info("✅ ExternalAPIManager initialized (httpx + tenacity + LRU cache)")

    async def _get_cached(self, key: str) -> Optional[Dict]:
        if key in self.cache:
            val, ts = self.cache[key]
            if time.time() - ts < self.cache_ttl:
                self.cache.move_to_end(key)
                return val
            del self.cache[key]
        return None

    def _set_cached(self, key: str, val: Dict):
        if key in self.cache: self.cache.move_to_end(key)
        self.cache[key] = (val, time.time())
        if len(self.cache) > self.max_cache: self.cache.popitem(last=False)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError))
    )
    async def fetch_recalls(self, year: int, make: str, model: str) -> List[RecallResponse]:
        key = f"recall_{year}_{make}_{model}"
        cached = await self._get_cached(key)
        if cached: return cached

        if self.circuit_open:
            app_logger.warning("⚡ Circuit breaker open. Skipping NHTSA call.")
            return []

        try:
            url = f"https://api.nhtsa.gov/vehicles/{year}/{make}/{model}/recalls?format=json"
            resp = await self.client.get(url)
            resp.raise_for_status()
            data = resp.json()
            recalls = [
                RecallResponse(
                    recall_id=r.get("NHTSACampaignNumber", "UNK"),
                    component=r.get("Component", "Unknown"),
                    summary=r.get("Summary", "")[:150],
                    date=r.get("ReportReceivedDate", "")[:10]
                ) for r in data.get("results", [])
            ]
            self._set_cached(key, [r.model_dump() for r in recalls])
            self.failure_count = 0
            return recalls
        except Exception as e:
            self.failure_count += 1
            if self.failure_count >= 5:
                self.circuit_open = True
                asyncio.create_task(self._reset_circuit())
            app_logger.error(f"❌ NHTSA API failed: {e}")
            return []

    async def fetch_parts_price(self, panel: str, category: str, city: str) -> PartsPriceResponse:
        key = f"parts_{panel}_{category}_{city}"
        cached = await self._get_cached(key)
        if cached: return PartsPriceResponse(**cached)

        # Real-world constraint: OEM parts APIs are proprietary.
        # Production pattern: Use structured open-market aggregation + caching + fallback.
        # Here: Deterministic pricing curve based on panel/category + regional labor multiplier.
        base_map = {
            ("front_bumper", "dent"): (8500, 4200), ("hood", "corrosion"): (14500, 7800),
            ("side_mirror", "scratch"): (2200, 950), ("windshield", "glass"): (18000, 12500),
            ("battery_pack", "ev_degradation"): (45000, 28000)
        }
        oem, after = base_map.get((panel, category), (6000, 3000))
        city_mult = {"tier_1": 1.3, "tier_2": 1.1, "tier_3": 1.0}.get(city.lower(), 1.15)
        
        result = PartsPriceResponse(
            oem_price_inr=round(oem * city_mult),
            aftermarket_price_inr=round(after * city_mult),
            availability="IN_STOCK" if oem < 15000 else "BACKORDER",
            last_updated=time.strftime("%Y-%m-%d")
        )
        self._set_cached(key, result.model_dump())
        return result

    async def _reset_circuit(self):
        await asyncio.sleep(60)
        self.circuit_open = False
        self.failure_count = 0

    async def close(self):
        await self.client.aclose()