from typing import List, Dict
import re

class QueryRewriter:
    def __init__(self, llm):
        self.llm = llm
    
    def rewrite(self, original_query: str, history: List[Dict] = None) -> List[str]:
        history_text = ""
        if history:
            recent = history[-5:]
            history_text = "\n".join([f"{m['role']}: {m['content']}" for m in recent])
        prompt = f"""You are a query rewriting assistant. Given the user's question and conversation history, generate 3 alternative questions that ask the same thing but from different angles.

History:
{history_text if history_text else "(No history)"}

Original: {original_query}

Generate 3 alternative questions, one per line, numbered 1-3.
"""
        try:
            response = self.llm.invoke(prompt)
            lines = response.content.split('\n')
            alternatives = []
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                    cleaned = re.sub(r'^[\d\-•\.]+\s*', '', line)
                    if cleaned:
                        alternatives.append(cleaned)
            all_queries = [original_query] + alternatives[:3]
            return all_queries
        except:
            return [original_query]