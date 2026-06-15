import os
import json
import tempfile
from typing import List, TypedDict
from pathlib import Path

import chainlit as cl
from langgraph.graph import StateGraph, START, END
from groq_llm import get_groq_llm

from tools.ppc_rag_pipeline import rag_context_collector as get_legal_context
from tools.user_files_rag import add_user_files_to_chroma, get_user_files_context
from tools.web_search import run_web_search

# ----------------------------------------------------------------------
#                           LangGraph State & Nodes
# ----------------------------------------------------------------------
class LegalSchema(TypedDict):
    query: str
    tools: List[str]
    get_legal_context: str
    get_web_context: str
    get_user_files_context: str
    final_answer: str

llm = get_groq_llm()

def route_query(state: dict) -> dict:
    query = state["query"]
    prompt = f"""
You are a legal AI routing engine. Return ONLY valid JSON with key "tools".
Schema: {{"tools": ["get_legal_context", "get_user_files_context", "get_web_context"]}}
Rules:
- Output ONLY JSON
- No markdown, no explanation
User Query: {query}
"""
    try:
        response = llm.invoke(prompt)
        return json.loads(response.content)
    except Exception as e:
        return {"tools": ["get_legal_context", "get_user_files_context"]}

def get_legal_context_node(state: LegalSchema) -> LegalSchema:
    context = get_legal_context(state["query"])
    return {"get_legal_context": context}

def get_web_context_node(state: LegalSchema) -> LegalSchema:
    try:
        results = run_web_search(state["query"])
        return {"get_web_context": results}
    except Exception as e:
        return {"get_web_context": f"Web search failed: {str(e)}"}

def get_user_files_context_node(state: LegalSchema) -> LegalSchema:
    session_id = cl.user_session.get("id") if cl.context.session else "unknown"
    context = get_user_files_context(state["query"], session_id)
    return {"get_user_files_context": context}

def merge_contexts(state: LegalSchema) -> LegalSchema:
    return {
        "get_legal_context": state["get_legal_context"],
        "get_web_context": state["get_web_context"],
        "get_user_files_context": state["get_user_files_context"]
    }

def generate_answer(state: LegalSchema) -> LegalSchema:
    context_dict = {
        "legal_context": state["get_legal_context"],
        "web_context": state["get_web_context"],
        "user_files_context": state["get_user_files_context"]
    }
    prompt = f"""
Here is the collected context from various tools:
{json.dumps(context_dict, indent=2)}

You are a legal AI assistant. Use the above context to answer the user's query:
{state["query"]}

If web context was used, provide source URLs in your answer.
If user files were used, reference the content appropriately.
Be concise but thorough.
"""
    try:
        response = llm.invoke(prompt)
        return {"final_answer": response.content}
    except Exception as e:
        return {"final_answer": f"Sorry, I'm having trouble connecting to the LLM. Error: {str(e)}"}

def conditioner(state: LegalSchema) -> List[str]:
    tools = state.get("tools", [])
    routes = []
    if "get_legal_context" in tools:
        routes.append("get_legal_context")
    if "get_user_files_context" in tools:
        routes.append("get_user_files_context")
    if "get_web_context" in tools:
        routes.append("get_web_context")
    if not routes:
        routes.append("get_legal_context")
    return routes

#                                        Build the graph

graph = StateGraph(LegalSchema)
graph.add_node("route_query", route_query)
graph.add_node("get_user_files_context", get_user_files_context_node)
graph.add_node("get_legal_context", get_legal_context_node)
graph.add_node("get_web_context", get_web_context_node)
graph.add_node("merge_contexts", merge_contexts)
graph.add_node("generate_answer", generate_answer)

graph.add_edge(START, "route_query")
graph.add_conditional_edges(
    "route_query",
    conditioner,
    {
        "get_legal_context": "get_legal_context",
        "get_user_files_context": "get_user_files_context",
        "get_web_context": "get_web_context"
    }
)
graph.add_edge("get_legal_context", "merge_contexts")
graph.add_edge("get_user_files_context", "merge_contexts")
graph.add_edge("get_web_context", "merge_contexts")
graph.add_edge("merge_contexts", "generate_answer")
graph.add_edge("generate_answer", END)

workflow = graph.compile()

# ---------------------------------------------------------------------------------------
#                            Helper to save uploaded file safely
# ----------------------------------------------------------------------------------
async def save_uploaded_file(file: cl.File) -> str:
    """Save a Chainlit file to a temporary file and return its path."""
    content = None
    if hasattr(file, 'content') and file.content is not None:
        content = file.content
    elif hasattr(file, 'path') and file.path and os.path.exists(file.path):
        with open(file.path, 'rb') as f:
            content = f.read()
    else:
        raise ValueError(f"Unable to read file: {file.name}")

    suffix = Path(file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        return tmp.name

# ----------------------------------------------------------------------
                        # Chainlit Handlers
# ----------------------------------------------------------------------
@cl.on_chat_start
async def start():
    cl.user_session.set("id", cl.context.session.id)
    await cl.Message(
        content="Welcome! You can upload legal documents (PDF, DOCX, TXT) by attaching them to your message. "
                "I will use them together with legal RAG and web search to answer your questions."
    ).send()

@cl.on_message
async def main(message: cl.Message):
    # Process any file attachments
    files = message.elements if message.elements else []
    if files:
        session_id = cl.user_session.get("id")
        temp_paths = []
        for file in files:
            try:
                tmp_path = await save_uploaded_file(file)
                temp_paths.append(tmp_path)
            except Exception as e:
                await cl.Message(content=f"Failed to process file {file.name}: {str(e)}").send()
                continue
        if temp_paths:
            add_user_files_to_chroma(session_id, temp_paths)
            # Cleanup temp files
            for p in temp_paths:
                os.unlink(p)
            await cl.Message(content=f"Processed {len(temp_paths)} file(s). You can now ask questions about them.").send()

    # Run the LangGraph workflow
    initial_state = {
        "query": message.content,
        "tools": [],
        "get_legal_context": "",
        "get_web_context": "",
        "get_user_files_context": "",
        "final_answer": ""
    }
    try:
        final_state = workflow.invoke(initial_state)
        await cl.Message(content=final_state["final_answer"]).send()
    except Exception as e:
        await cl.Message(content=f"An error occurred while processing your request: {str(e)}").send()