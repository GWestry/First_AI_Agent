from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool
from datetime import datetime

@tool
def search_tool(query: str) -> str:
    """Search the web for information about a topic"""
    search = DuckDuckGoSearchRun()
    return search.run(query)

@tool
def wikipedia_tool(query: str) -> str:
    """Search Wikipedia for information about a topic"""
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    return wikipedia.run(query)