import logging
from datetime import datetime, timezone
from fastapi import FastAPI
import inngest
import inngest.fast_api
from dotenv import load_dotenv
import uuid
from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from models import RAGUpsertResult, RAGChunkAndSrc
from rag_agent import run_agent_query

load_dotenv()

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
    # throttle=inngest.Throttle(
    #     count=2, period=datetime.timedelta(minutes=1)
    # ),
    # rate_limit=inngest.RateLimit(
    #     limit=1,
    #     period=datetime.timedelta(hours=4),
    #     key="event.data.source_id",
    # ),
)

async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        content_hash = ctx.event.data.get("content_hash")
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id, content_hash=content_hash)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        content_hash = chunks_and_src.content_hash
        updated_at = datetime.now(timezone.utc).isoformat()

        store = QdrantStorage()
        store.delete_by_source(source_id)

        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [
            {
                "source": source_id,
                "text": chunks[i],
                "chunk_index": i,
                "content_hash": content_hash,
                "updated_at": updated_at,
            }
            for i in range(len(chunks))
        ]
        store.upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)
    return ingested.model_dump()


@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))
    thread_id = ctx.event.data.get("thread_id", "default")

    return await ctx.step.run(
        "react-agent-query",
        lambda: run_agent_query(question, top_k, thread_id),
    )


app = FastAPI()
inngest.fast_api.serve(app, inngest_client,[rag_ingest_pdf,rag_query_pdf_ai])