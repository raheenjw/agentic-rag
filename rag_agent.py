import re

from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from tools import retrieve_documents, list_indexed_documents

AGENT_PROMPT = (
    "You are a helpful assistant that answers questions about PDFs in the knowledge base. "
    "For questions about document content, use retrieve_documents before answering. "
    "If the user asks what documents are available or indexed, use list_indexed_documents. "
    "Ground answers in retrieved context and cite sources when possible. "
    "If retrieval finds nothing useful, say you could not find relevant information."
)

TOOLS = [retrieve_documents, list_indexed_documents]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

checkpointer = MemorySaver()
agent_graph = create_react_agent(
    model=llm,
    tools=TOOLS,
    prompt=AGENT_PROMPT,
    checkpointer=checkpointer
)


def _extract_answer(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return ""


def _extract_sources(messages: list) -> list[str]:
    sources: set[str] = set()
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.content:
            for match in re.finditer(r"\[Source: ([^\]]+)\]", str(msg.content)):
                sources.add(match.group(1))
    return sorted(sources)


def _extract_chunks(messages: list) -> list[dict]:
    chunks = []
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.content:
            content = str(msg.content)
            # Split by the separator used in tools.py
            for block in content.split("\n\n---\n\n"):
                if block.strip():
                    # Extract source and text
                    match = re.match(r"\[Source: ([^\]]+)\]\n(.+)", block, re.DOTALL)
                    if match:
                        chunks.append({
                            "source": match.group(1),
                            "text": match.group(2).strip()
                        })
    return chunks


def run_agent_query(question: str, top_k: int = 5, thread_id: str = "default") -> dict:
    user_content = question
    if top_k != 5:
        user_content = f"{question}\n\n(Use top_k={top_k} when calling retrieve_documents.)"

    config = {"configurable": {"thread_id": thread_id}}
    result = agent_graph.invoke({"messages": [("user", user_content)]}, config=config)
    messages = result["messages"]
    num_contexts = 0
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.content:
            num_contexts += len(re.findall(r"\[Source:", str(msg.content)))

    return {
        "answer": _extract_answer(messages),
        "sources": _extract_sources(messages),
        "chunks": _extract_chunks(messages),
        "num_contexts": num_contexts,
    }
