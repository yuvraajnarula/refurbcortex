# app/services/vector_memory.py
import os
import chromadb
from typing import Dict, List, Optional, Any
from datetime import datetime
from app.config import settings
from app.utils.logger import app_logger

class VectorMemoryService:
    def __init__(self, db_path: Optional[str] = None, collection_name: str = "inspection_memory"):
        self.db_path = db_path or settings.CHROMA_DB_PATH
        self.collection_name = collection_name
        os.makedirs(self.db_path, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self._init_collection()
        app_logger.info(f"VectorMemory initialized: {self.db_path} | Collection: {self.collection_name}")

    def _init_collection(self) -> chromadb.Collection:
        try:
            return self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine", "description": "Historical inspection errors & routing context"}
            )
        except Exception as e:
            app_logger.error(f"Failed to init ChromaDB collection: {e}")
            raise

    def add_memory(self, case_id: str, context: str, error_value: float, metadata: Dict[str, Any] = None) -> bool:
        """Log a new inspection case with error magnitude for future calibration."""
        try:
            meta = metadata or {}
            meta.update({
                "case_id": case_id, 
                "error_inr": error_value, 
                "timestamp": datetime.now().isoformat()
            })
            # Chroma auto-embeds text using default all-MiniLM-L6-v2
            self.collection.add(
                documents=[context],
                metadatas=[meta],
                ids=[f"{case_id}_{context[:32]}"]
            )
            app_logger.info(f"VectorMemory.add | {case_id} | Error: ₹{error_value}")
            return True
        except Exception as e:
            app_logger.error(f"VectorMemory.add failed: {e}")
            return False

    def query_similar(self, context: str, n_results: int = 3, filters: Optional[Dict] = None) -> List[Dict]:
        """Find historically similar inspections to compute safety margins."""
        try:
            query_kwargs = {
                "query_texts": [context],
                "n_results": n_results,
                "include": ["documents", "metadatas", "distances"]
            }
            if filters:
                query_kwargs["where"] = filters

            results = self.collection.query(**query_kwargs)
            if not results["ids"] or not results["ids"][0]:
                return []

            similar_cases = []
            for i in range(len(results["ids"][0])):
                similar_cases.append({
                    "id": results["ids"][0][i],
                    "context": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i]
                })
            return similar_cases
        except Exception as e:
            app_logger.warning(f"VectorMemory.query failed: {e}")
            return []

    def update_memory(self, case_id: str, new_error_value: float, updated_context: Optional[str] = None) -> bool:
        """Update an existing case with actual error from feedback loop."""
        try:
            existing = self.collection.get(ids=[case_id])
            if not existing["documents"]:
                app_logger.warning(f"Case {case_id} not found in memory. Adding as new.")
                return self.add_memory(case_id, updated_context or case_id, new_error_value)

            meta = existing["metadatas"][0] if existing["metadatas"] else {}
            meta.update({"error_inr": new_error_value, "updated": True, "updated_at": datetime.now().isoformat()})

            self.collection.update(
                ids=[case_id],
                documents=[updated_context or existing["documents"][0]],
                metadatas=[meta]
            )
            app_logger.info(f"VectorMemory.update | {case_id} | New Error: ₹{new_error_value}")
            return True
        except Exception as e:
            app_logger.error(f"VectorMemory.update failed: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Return collection metrics for dashboard/monitoring."""
        try:
            return {
                "total_records": self.collection.count(),
                "collection_name": self.collection_name,
                "db_path": self.db_path
            }
        except Exception as e:
            app_logger.error(f"VectorMemory.stats failed: {e}")
            return {"total_records": 0}

    def clear_collection(self) -> bool:
        """Utility for testing/reset. Use with caution in prod."""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self._init_collection()
            app_logger.info("VectorMemory cleared & re-initialized")
            return True
        except Exception as e:
            app_logger.error(f"VectorMemory.clear failed: {e}")
            return False