from typing import List, Dict
from tavily import TavilyClient

class WebSearch:
    def __init__(self, api_key: str):
        self.client = TavilyClient(api_key=api_key)
    
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
        except:
            return []