from typing import TypedDict
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import json
import os
from dotenv import load_dotenv

load_dotenv()

@tool
def mock_searxng_search(query: str) -> str:
    """Mock search tool"""
    q = query.lower()
    if "crypto" in q or "bitcoin" in q:
        return "Bitcoin hits new all-time high amid regulatory ETF approvals."
    elif "ai" in q or "openai" in q:
        return "OpenAI unveils new autonomous agents capable of complex tasks."
    elif "economy" in q or "rate" in q or "market" in q:
        return "Federal Reserve hints at cutting interest rates in the upcoming quarter."
    return "General markets remain stable amidst geopolitical tensions."

class GraphState(TypedDict):
    persona: str
    search_query: str
    context: str
    post_json: str

class Orchestrator:
    def __init__(self, bot_id: str):
        self.bot_id = bot_id
        hf_llm = HuggingFaceEndpoint(
            repo_id="HuggingFaceH4/zephyr-7b-beta", 
            temperature=0.7, 
            task="text-generation", 
            max_new_tokens=512
        )
        self.llm = ChatHuggingFace(llm=hf_llm)
        self.workflow = self.build_graph()

    def get_search_query(self, state: GraphState) -> GraphState:
        prompt = (f"You are the following persona: {state['persona']}\n"
                  f"Decide on a search query to find real-world context for a post. "
                  f"Output ONLY the search query string.")
        
        resp = self.llm.invoke(prompt)
        query = resp.content.strip().strip('"').strip("'")
        return {"search_query": query}

    def run_search(self, state: GraphState) -> GraphState:
        ctx = mock_searxng_search.invoke({"query": state["search_query"]})
        return {"context": ctx}

    def draft_post(self, state: GraphState) -> GraphState:
        prompt = (f"Persona: {state['persona']}\n"
                  f"Context: '{state['context']}'\n\n"
                  f"Generate an opinionated post (max 280 characters) reacting to this context. "
                  f"Your bot_id is '{self.bot_id}'.\n"
                  f"OUTPUT STRICTLY A VALID JSON OBJECT exactly like this and nothing else: "
                  f"{{\"bot_id\": \"Bot A\", \"topic\": \"Topic here\", \"post_content\": \"Post here\"}}")
        
        resp = self.llm.invoke(prompt)
        content = resp.content.strip().replace("```json", "").replace("```", "").strip()
        return {"post_json": content}

    def build_graph(self):
        wf = StateGraph(GraphState)
        wf.add_node("decide", self.get_search_query)
        wf.add_node("search", self.run_search)
        wf.add_node("draft", self.draft_post)
        
        wf.set_entry_point("decide")
        wf.add_edge("decide", "search")
        wf.add_edge("search", "draft")
        wf.add_edge("draft", END)
        return wf.compile()

    def generate(self, persona: str) -> str:
        res = self.workflow.invoke({
            "persona": persona,
            "search_query": "",
            "context": "",
            "post_json": ""
        })
        return res["post_json"]
