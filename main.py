
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

class Source(BaseModel):
    """Schema of a source used by Agent"""
    url: str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema of the Agent response"""
    answer: str = Field(description="The agents answer to the user's query")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")

  
# Initialize the Ollama language model
llm = ChatOllama(model="gpt-oss:20b")
    
# Create an agent with the language model and the search tool
tool = [TavilySearch()]  # Using the TavilySearch tool instead of the custom search function]
agent = create_agent(
    model=llm,
    tools=tool,
    response_format=ToolStrategy(AgentResponse),
)

def main():  
    print("welcome to Search agent")
    result = agent.invoke(
        {
            "messages": HumanMessage(
                content="search for the latest news about AI and provide the sources"
            )
        }
    )   
    print(result)

if __name__ == "__main__":   
    main()
