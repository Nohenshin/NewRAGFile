from src.llms.cohere_llm import CohereLLM
# from src.llms.google_llm import GoogleLLM  # Comment

class LLMFactory:
    @staticmethod
    def create_llm(llm_type: str, **kwargs):
        llm_type = llm_type.lower()
        if llm_type == "cohere":
            assert "cohere_api_key" in kwargs, "Missing cohere_api_key"
            return CohereLLM(**kwargs)
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")