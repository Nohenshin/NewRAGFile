from typing import List, Dict
from tavily import TavilyClient
import os

class WebSearch:
    def __init__(self, api_key: str = None):
        # Nếu không truyền api_key, thử lấy từ biến môi trường
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY is required. Set it in .env or pass to constructor.")
        self.client = TavilyClient(api_key=self.api_key)
    
    def search(self, query: str, max_results: int = 3) -> List[Dict]:
        try:
            response = self.client.search(query, max_results=max_results)
            results = []
            for item in response.get('results', []):
                results.append({
                    "title": item.get('title', ''),
                    "url": item.get('url', ''),
                    "content": item.get('content', ''),
                    "score": item.get('score', 0)
                })
            return results
        except Exception as e:
            print(f"Web search error: {e}")
            return []