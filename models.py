# models.py
from pydantic import BaseModel, Field

class AgentInput(BaseModel):
    query: str = Field(description="Initial query from the user.")

class LangchainAgentOutput(BaseModel):
    processed_text: str = Field(description="Text processed by Langchain agent.")
    keywords: list[str] = Field(description="Extracted keywords from the text.")

class PydanticAIAgentOutput(BaseModel):
    structured_response: dict = Field(
        description="Structured response from PydanticAI agent."
    )
    sentiment: str = Field(description="Sentiment of the processed text.")
    openai_enriched_data: str = Field(
        description="Data enriched by OpenAI LLM in PydanticAI agent."
    )