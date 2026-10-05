from dotenv import load_dotenv
load_dotenv()   


### db, llm, tools, create_agent, system_prompt

from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
import streamlit as st

# SQLDatabaseToolkit gives us pre-built toolkit to communicate with DB
# Provides pre-built tool to llm and by using those tools llm can atek
# action on user query, SO we don't need to build specific tool for SQL.
# Cause we are getting everything prebuilt in langchain
from langchain_community.agent_toolkits import SQLDatabaseToolkit


db = SQLDatabase.from_uri("sqlite:///my_tasks.db")

db.run("""
       CREATE TABLE IF NOT EXISTS tasks (
           id INTEGER PRIMARY KEY AUTOINCREMENT,
           title TEXT NOT NULL,
           description TEXT,
           status TEXT CHECK (status IN ('pending', 'in_progress', 'completed')),
           created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
       );
       
""")

### llm, tools, memory, system prompt
# It takes 2 args in toolkit DB and model

model = ChatGroq(model="openai/gpt-oss-20b")
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()
# memory = InMemorySaver()


# sql_db_query
# sql_db_schema
# sql_db_list_tables
# sql_db_query_checker

# for tool in tools:
#     print(tool.name)


system_prompt = """
You are a task management assistant that interacts with a SQL database containing a 'tasks' table. 

TASK RULES:
1. Limit SELECT queries to 10 results max with ORDER BY created_at DESC
2. After CREATE/UPDATE/DELETE, confirm with SELECT query
3. If the user requests a list of tasks, present the output in a structured table format to ensure a clean and organized display in the browser."

CRUD OPERATIONS:
    CREATE: INSERT INTO tasks(title, description, status)
    READ: SELECT * FROM tasks WHERE ... LIMIT 10
    UPDATE: UPDATE tasks SET status=? WHERE id=? OR title=?
    DELETE: DELETE FROM tasks WHERE id=? OR title=?

Table schema: id, title, description, status(pending/in_progress/completed), created_at.
"""


# This cache helps in whenever streamlit refreshes n request the memory save in cache so that new memory wi not be created
# An this function do not get call againa nd again

@st.cache_resource
def get_agent():
    agent = create_agent (
        model = model,
        tools = tools,
        checkpointer = InMemorySaver(),
        system_prompt = system_prompt
    )
    
    return agent

agent = get_agent()



st.subheader("📝 TaskBot: Manage you todos")

if "messages" not in st.session_state:
    st.session_state.messages = []
    
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])
    

prompt = st.chat_input("Give me your task I'll manage them")


if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("ai"):
        with st.spinner("Working on task..."):
        # query = input("User: ")
            res = agent.invoke(
                {"messages": [{"role": "user", "content": prompt}]},
                {"configurable": {"thread_id": "as12"}}
            )
    
        result = res["messages"][-1].content
        # print("AI: ", result)
        st.markdown(result)
        st.session_state.messages.append({"role": "ai", "content": result})
        




# print("DB createdd Successfully✅")