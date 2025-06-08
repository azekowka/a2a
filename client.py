# client.py
import httpx
import asyncio
from models import AgentInput

async def main():
    agent_input = AgentInput(query="Tell me about the latest advancements in AI and their ethical implications.")
    
    langchain_agent_url = "http://127.0.0.1:8000/process_query" # LangChain agent URL

    print(f"Sending query to Langchain Agent: {agent_input.query}")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(langchain_agent_url, json=agent_input.model_dump())
            response.raise_for_status() # Raise an exception for bad status codes
            
            # The Langchain agent's response here is its own output,
            # but it has already communicated with the PydanticAI agent internally.
            print(f"Langchain Agent final response to client: {response.json()}")
            
        except httpx.HTTPStatusError as e:
            print(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
        except httpx.RequestError as e:
            print(f"Request error occurred: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())