# pydantic_ai_agent.py
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

# Load environment variables from .env file
load_dotenv()
from pydantic import BaseModel, ValidationError
import uvicorn
from textblob import TextBlob # For sentiment analysis
from langchain_openai import ChatOpenAI # For OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough

from models import LangchainAgentOutput, PydanticAIAgentOutput

app = FastAPI(title="PydanticAI Agent (OpenAI)")

# Retrieve API key from environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

# Initialize OpenAI LLM for enrichment
openai_llm = ChatOpenAI(temperature=0.3, openai_api_key=OPENAI_API_KEY)

# Prompt for OpenAI to enrich data
enrichment_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI assistant that enriches given text. Summarize the text and highlight any potential follow-up questions."),
        ("user", "Text to enrich:\n\n{text}"),
    ]
)

enrichment_chain = (
    {"text": RunnablePassthrough()}
    | enrichment_prompt
    | openai_llm
)


@app.post("/validate_and_enrich", response_model=PydanticAIAgentOutput)
async def validate_and_enrich(data: LangchainAgentOutput):
    print(f"PydanticAI Agent (OpenAI) received data: {data.model_dump_json()}")
    try:
        # Pydantic automatic validation on input parsing

        # Sentiment Analysis using TextBlob
        analysis = TextBlob(data.processed_text)
        sentiment_score = analysis.sentiment.polarity
        
        sentiment = "neutral"
        if sentiment_score > 0.1:
            sentiment = "positive"
        elif sentiment_score < -0.1:
            sentiment = "negative"

        # Enrich data using OpenAI LLM
        openai_enriched_data_response = enrichment_chain.invoke({"text": data.processed_text})
        openai_enriched_text = openai_enriched_data_response.content # Access content from AIMessage

        structured_response = {
            "original_query_keywords": data.keywords,
            "summary_from_langchain_agent": data.processed_text,
            "processed_by": "PydanticAI Agent",
            "sentiment_score": sentiment_score
        }

        response = PydanticAIAgentOutput(
            structured_response=structured_response,
            sentiment=sentiment,
            openai_enriched_data=openai_enriched_text # Add enriched data here
        )
        print(f"PydanticAI Agent (OpenAI) sending response: {response.model_dump_json()}")
        return response
    except ValidationError as e:
        print(f"Validation error in PydanticAI Agent: {e.json()}")
        raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e:
        print(f"Error in PydanticAI Agent (OpenAI): {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Ensure OPENAI_API_KEY is set in your environment
    # Example: export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
    uvicorn.run(app, host="127.0.0.1", port=8001)