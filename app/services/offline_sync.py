import sqlite3, asyncio, time, os, json
from typing import Dict, List
from app.utils.logger import app_logger

class OfflineSyncQueue:
    def __init__(self, db_path="./data/offline_queue.db"):
        self.db = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS pending_syncs (
                id TEXT PRIMARY KEY, payload TEXT, retries INTEGER DEFAULT 0, created_at REAL)""")

    async def enqueue(self, inspection_id: str, payload: Dict):
        with sqlite3.connect(self.db) as conn:
            conn.execute("INSERT OR REPLACE INTO pending_syncs VALUES (?,?,0,?)",
                         (inspection_id, json.dumps(payload), time.time()))
        app_logger.info(f"📦 Offline enqueue: {inspection_id}")

    async def flush_and_sync(self, sync_fn):
        """Attempts to send pending payloads to cloud. Returns successful IDs."""
        synced = []
        with sqlite3.connect(self.db) as conn:
            cursor = conn.execute("SELECT id, payload FROM pending_syncs WHERE retries < 3")
            for row in cursor.fetchall():
                rid, pdata = row
                try:
                    await sync_fn(json.loads(pdata))
                    conn.execute("DELETE FROM pending_syncs WHERE id=?", (rid,))
                    synced.append(rid)
                except Exception as e:
                    conn.execute("UPDATE pending_syncs SET retries=retries+1 WHERE id=?", (rid,))
                    app_logger.warning(f"Sync retry failed for {rid}: {e}")
        return synced