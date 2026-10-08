import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
load_dotenv()



from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st


### data in st session

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []
    
def process_document(file_paths):

    ### Load the documents
    docs = []
    for p in file_paths:
        docs.extend(PyPDFLoader(p).load())

    ### Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents=docs)

    ### Embedding and Vector DB
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )
    vector_db = InMemoryVectorStore.from_documents(
        documents=docs,
        embedding=embeddings
    )

    ### Create a Agent
    llm = ChatGroq(model="openai/gpt-oss-20b")

    ### Below function is a tool which receives query in the form of string

    @tool
    def retrieve_context(query:str):
        """Retrieve documents relevant to a query form the knowlege base"""
        context = " "
        
        docs = vector_db.similarity_search(query=query, k=3)
        for doc in docs:
            context += doc.page_content + "\n\n"
        
        return context


    system_prompt = """You are a helpfull assistant that answers quetion using retrieved contxt. My knowledge base consist of the details from the uploaded documents. Always use the `retrieve_context` tools for questions requiring external knowledge"""


    memory = InMemorySaver()

    agent = create_agent(
        model=llm,
        tools = [retrieve_context],
        system_prompt = system_prompt,
        checkpointer = memory
    )
    
    st.session_state.agent = agent
    st.session_state.document_uploaded = True
    


# while True:
#     query = input("User: ")
    
#     if query.lower() in ["quit", "exit", "bye"]:
#         break
    
#     response = agent.invoke(
#         {"messages": [{"role": "user", "content": query}]},
#         {"configurable":{"thread_id":1}}
#     )
    
#     result = response["messages"][-1].content
    
#     print("AI: ", result)


### Upload doc UI - Until the doc is uploaded we won't remove this ui
if not st.session_state.document_uploaded:
    uploaded = st.file_uploader(label="Upload you file", type=["pdf"], accept_multiple_files=True)
    
    if uploaded:
        with st.spinner("Uploading..."):
            path = "./doc_files/"
            
            # in uploaded(variable) we have list of files will come so we traverse
            saved = []
            for file in uploaded:
                p = path + file.name
                with open(p, "wb") as f:
                    f.write(file.getvalue())
                saved.append(p)

            process_document(saved)
            st.rerun()
                    



### Upload Chat UI - Until and unless we don't have agent ready with and doc is not uploaded us we won't show chat ui

if st.session_state.document_uploaded and st.session_state.agent:
    for message in st.session_state.messages:
        role = message.get("role")
        content = message.get("content")
        st.chat_message(role).markdown(content)

    
    query = st.chat_input("Ask anything related to you doc...")
    
    if query:
        st.session_state.messages.append({"role":"user", "content": query})
        
        st.chat_message("user").markdown(query)
        response = st.session_state.agent.invoke(
            {"messages": [{"role":"user", "content": query}]},
            {"configurable":{"thread_id":2}}
        )
        
        answer = response["messages"][-1].content
        st.chat_message("ai").markdown(answer)
        st.session_state.messages.append({"role":"ai", "content": answer})