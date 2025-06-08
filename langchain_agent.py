# langchain_agent.py
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

# Load environment variables from .env file
load_dotenv()
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI # For Gemini
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser
import uvicorn
import httpx

from models import AgentInput, LangchainAgentOutput, PydanticAIAgentOutput

app = FastAPI(title="Langchain Agent (Gemini)")

# Retrieve API key from environment variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

# Initialize LangChain components with Gemini
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, google_api_key=GEMINI_API_KEY)

# Define an output parser for structured output from LangChain
class KeywordsOutput(BaseModel):
    keywords: list[str]
    processed_text: str

parser = JsonOutputParser(pydantic_object=KeywordsOutput)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert text processor. Extract key entities and summarize the text concisely."),
        ("user", "Process the following text:\n\n{text}\n\n{format_instructions}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

chain = (
    {"text": RunnablePassthrough()}
    | prompt
    | llm
    | parser
)

# API endpoint for LangChain Agent to receive initial query
@app.post("/process_query", response_model=LangchainAgentOutput)
async def process_query(input_data: AgentInput):
    print(f"Langchain Agent (Gemini) received query: {input_data.query}")
    try:
        # Simulate LangChain processing with Gemini
        langchain_output_dict = chain.invoke({"text": input_data.query})
        
        # Ensure the output matches our Pydantic model
        langchain_output = LangchainAgentOutput(
            processed_text=langchain_output_dict.get("processed_text", "No processed text from Gemini."),
            keywords=langchain_output_dict.get("keywords", [])
        )
        print(f"Langchain Agent (Gemini) processed data: {langchain_output.model_dump_json()}")

        # A2A: Send processed data to PydanticAI Agent
        async with httpx.AsyncClient() as client:
            pydantic_ai_agent_url = "http://127.0.0.1:8001/validate_and_enrich" # Replace with PydanticAI agent URL
            response = await client.post(
                pydantic_ai_agent_url, json=langchain_output.model_dump()
            )
            response.raise_for_status()  # Raise an exception for bad status codes
            pydantic_ai_response = PydanticAIAgentOutput.model_validate(response.json())
            print(f"PydanticAI Agent response received: {pydantic_ai_response.model_dump_json()}")

        return langchain_output # We return our own output here, or perhaps a combined one
    except Exception as e:
        print(f"Error in Langchain Agent (Gemini): {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Ensure GEMINI_API_KEY is set in your environment
    # Example: export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
    uvicorn.run(app, host="127.0.0.1", port=8000)