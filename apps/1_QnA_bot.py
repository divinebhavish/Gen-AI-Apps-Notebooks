from dotenv import load_dotenv 
load_dotenv()

import logging
logging.getLogger("google_genai").setLevel(logging.ERROR)


from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

llm = ChatGoogleGenerativeAI(model = "gemini-3.5-flash-lite")


st.title("🤖 AskBuddy - AI QnA bot")
st.markdown("QnA bot developed with LangChain and Google gemini")


# Session state so that we can store message in on session
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)

query = st.chat_input("Ask Annything") # Placeholder give as Ask anything
if query:
    st.session_state.messages.append({"role":"user", "content":query})
    st.chat_message("user").markdown(query)
    res = llm.invoke(query)
    answer = res.content[0]["text"]
    st.chat_message("AI").markdown(answer)
    st.session_state.messages.append({"role":"AI", "content":answer})


# que = "Who is the Bhavishya Mewara"

# # res = llm.invoke(que)

# while True:
#     query = input("User: ")
    
#     if query.lower in ["exit", "quit", "bye"]:
#         print("Bye Mr. Mewara")
#         break
    
#     res = llm.invoke(query)

#     print("AI: ", res.content[0]["text"], "\n")
    