### Tavily Web Search
import os
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()

os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

tool = TavilySearch(
    max_results=5,
    topic="general",
    # include_answer=False,
    # include_raw_content=False,
    # include_images=False,
    # include_image_descriptions=False,
    # search_depth="basic",
    # time_range="day",
    # include_domains=None,
    # exclude_domains=None
)

def run_web_search(query: str):
    results = tool.run(query)
    web_search_results = []
    for result in results['results']:
        web_search_results.append({
            "title": result['title'],
            "content": result['content'],
            "url": result['url']
        })
    return web_search_results

if __name__ == "__main__":
    query = "What is the punishment for theft in ppc?"
    results = run_web_search(query)
    print(results)