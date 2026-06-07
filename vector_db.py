from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


class QdrantStorage:
    def __init__(self, url="http://localhost:6333", collection="docs", dim=3072):
        self.client = QdrantClient(url=url, timeout=30)
        self.collection = collection
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def upsert(self, ids, vectors, payloads):
        points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
        self.client.upsert(self.collection, points=points)

    def search(self, query_vector, top_k: int = 5):
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            with_payload=True,
            limit=top_k,
        )
        chunks = []
        sources = set()

        for r in response.points:
            payload = getattr(r, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source", "")
            if text:
                chunks.append({"text": text, "source": source})
                sources.add(source)

        return {
            "chunks": chunks,
            "contexts": [c["text"] for c in chunks],
            "sources": list(sources),
        }

    def list_sources(self) -> list[str]:
        """Get all unique source filenames in the collection."""
        try:
            scroll_result = self.client.scroll(
                collection_name=self.collection,
                limit=10000,
                with_payload=True,
                with_vectors=False,
            )
            sources = set()
            for point in scroll_result[0]:
                payload = getattr(point, "payload", None) or {}
                source = payload.get("source", "")
                if source:
                    sources.add(source)
            return sorted(sources)
        except Exception:
            return []

    def delete_by_source(self, source: str) -> int:
        """Delete all chunks from a specific source document."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        try:
            result = self.client.delete(
                collection_name=self.collection,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="source",
                            match=MatchValue(value=source)
                        )
                    ]
                )
            )
            return getattr(result, "operation_id", 0)
        except Exception:
            return 0

    def clear_all(self) -> bool:
        """Delete all documents from the collection."""
        try:
            self.client.delete_collection(self.collection)
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
            )
            return True
        except Exception:
            return False
