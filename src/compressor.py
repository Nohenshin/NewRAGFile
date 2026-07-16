from typing import List, Dict

class ContextCompressor:
    def __init__(self, llm, max_tokens: int = 2000):
        self.llm = llm
        self.max_tokens = max_tokens
    
    def compress(self, query: str, documents: List[Dict]) -> str:
        if not documents:
            return ""
        combined = "\n\n---\n\n".join([
            f"Document {i+1}: {doc.get('parent_content', doc.get('child_content', ''))}"
            for i, doc in enumerate(documents)
        ])
        estimated_tokens = len(combined) // 4
        if estimated_tokens <= self.max_tokens:
            return combined
        prompt = f"""Compress the following information into a concise summary relevant to the question.
                Question: {query}

                Information:
                {combined}

                Return only the compressed, relevant information.
                """
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except:
            return combined[:self.max_tokens * 4]