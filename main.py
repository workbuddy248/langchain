from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
#from tavily import TavilyClient
from langchain_tavily import TavilySearch

#tavily = TavilyClient()   

#@tool()
#def search(query: str) -> str:
#    """
#    Tool to search the web for information.
#    Args:        
#        query (str): The search query.
#    Returns:        
#        str: The search results.
#    """
#    print(f"Searching for: {query}")
#    return tavily.search(query=query)


# Initialize the Ollama language model
llm = ChatOllama(model="gpt-oss:20b")
    
# Create an agent with the language model and the search tool
tool = [TavilySearch()]  # Using the TavilySearch tool instead of the custom search function]
agent = create_agent(model=llm, tools=tool)

def main():  
    # Example query to the agent
    print("welcome to Search agent")
    query = "search for 3 job postings for an ai engineer using langchain in India on linkedin and list their details"
    result = agent.invoke({"messages":HumanMessage(content=query)})   
    print(result)

if __name__ == "__main__":   
    main()
