from dotenv import load_dotenv
load_dotenv()

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
# import textwrap

llm = ChatGroq(model="openai/gpt-oss-20b")
search = GoogleSerperAPIWrapper()
memory = MemorySaver()


agent = create_agent(
    model = llm,
    tools = [search.run],
    system_prompt = "You are an agent and can search any ques on google",
    checkpointer = memory
)

while True:
    query = input("User: ")
    
    if query.lower() in ["exit", "quit", "bye"]:
        print("Good bye, hopin to see you again")
        break
    
    res = agent.invoke(
        {"messages":[{"role":"user", "content":query}]},
        {"configurable": {"thread_id": "asd123"}}
    )
    
    print("AI: ", res["messages"][-1].content)
    
    