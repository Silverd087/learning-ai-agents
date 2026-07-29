from typing import TypedDict,Sequence,Annotated
from langchain_core.messages import BaseMessage, HumanMessage,SystemMessage,AIMessage,ToolMessage
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.graph import START,END,StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os 
load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    encode_kwargs={"normalize_embeddings": True},
)

llm = ChatAnthropic(model_name="claude-opus-4-8")

pdf_path = "Stock_Market_Performance_2024.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

pdf_loader = PyPDFLoader(pdf_path)

try:
    pages = pdf_loader.load()
    print(f"PDF has been loaded and has {len(pages)} pages")
except Exception as e:
    print(f'Error loading pdf: {e}')
    raise 

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)

chunks = text_splitter.split_documents(pages)
persist_directory = r"C:\Users\ASUS\Documents\Projects\learning-ai-agents"
collection_name = "stock_market"

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

try:
    vectorstore = Chroma.from_documents(
        collection_name=collection_name,
        embedding=embeddings,
        persist_directory=persist_directory,
        documents=chunks
    )
except Exception as e:
    print(f"Error setting up Chroma: {str(e)}")

retriever = vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":5})

@tool
def retriever_tool(query:str):
    """this tool searches and returns information from the stock market performance 2024 document"""
    docs = retriever.invoke(query)

    if not docs:
        return "I found no relevant information in the stock performance 2024 document"

    results = []
    for i,doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")

    return "\n\n".join(results)

tools = [retriever_tool]

llm = llm.bind_tools(tools=tools)

class AgentState(TypedDict):
    messages:Annotated[Sequence[BaseMessage],add_messages]

def should_continue(state:AgentState):
    "check if the last message contains tool calls"
    last_message = state['messages'][-1]
    return len(last_message.tool_calls)>0


system_prompt = """
You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about the stock market performance data. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""

tools_dict = {our_tool.name : our_tool for our_tool in tools}

def call_llm(state:AgentState):
    messages = list(state["messages"])
    messages = [SystemMessage(system_prompt)] + messages
    resposne = llm.invoke(messages)
    return {"messages":[resposne]}

def take_action(state:AgentState):
    """Execute tool calls from the LLM's response."""

    tool_calls = state['messages'][-1].tool_calls
    results= []
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")

        if t["name"] not in tools_dict:
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        else:
            result = tools_dict[t["name"]].invoke(t["args"].get("query",""))

        results.append(ToolMessage(tool_call_id=t['id'],name=t['name'],content=str(result)))
    return {"messages":results}


builder = StateGraph(AgentState)
builder.add_node("llm",call_llm)
builder.add_node("retriever_agent",take_action)

builder.add_edge(START,"llm")
builder.add_conditional_edges("llm",should_continue,{True:"retriever_agent",False:END})
builder.add_edge("retriever_agent","llm")

rag_agent = builder.compile()

def running_agent():
    print("\n=== RAG AGENT===")
    
    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        messages = [HumanMessage(content=user_input)]

        result = rag_agent.invoke({"messages": messages})
        
        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)


running_agent()