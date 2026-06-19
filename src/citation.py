from typing import List, Dict

class CitationGenerator:
    @staticmethod
    def generate_citations(documents: List[Dict]) -> List[str]:
        citations = []
        for i, doc in enumerate(documents):
            metadata = doc.get('metadata', {})
            source = metadata.get('source', metadata.get('file_path', 'Unknown'))
            page = metadata.get('page', '')
            citations.append(f"[{i+1}] {source}" + (f", p. {page}" if page else ""))
        return citations