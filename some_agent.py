from typing import TypedDict,Sequence,Annotated
from langchain_core.messages import BaseMessage, HumanMessage,SystemMessage,AIMessage,ToolMessage
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.graph import START,END,StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
import os 
load_dotenv()

document_content = ""

@tool
def update(content:str):
    "update the document with the provided content"
    global document_content

    document_content = content
    return f"Document has been updated successfully! the current content is:\n{document_content}"


@tool
def save(filename:str):
    """Save current document to a text file and finish the process
    
    Args:
        filename: Name for the text file
    """
    global document_content
    if not filename.endswith('.txt'):
        filename = f'{filename}.txt'

    try:
        with open(filename,"w") as f:
            f.write(document_content)
        return f"Document has been saved successfully to {filename}"

    except Exception as e:
        return f'Error saving document: {str(e)}'

tools = [save,update]

model = ChatAnthropic(model_name="claude-opus-4-8").bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages]


def call_agent(state:AgentState):
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.
    
    The current document content is:{document_content}
    """)
    if not state["messages"]:
        user_input = "I'm ready to help you to update a document. What would you like to create?"
        user_message = HumanMessage(content=user_input)

    else:
        user_input = input("\n what would you like to do with the document? ")
        print(f"\n user: {user_input}")
        user_message = HumanMessage(content=user_input)


    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    response = model.invoke(all_messages)

    print(f"\n🤖 AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state["messages"]) + [user_message, response]}

def should_continue(state:AgentState):
    messages = state["messages"]

    if not messages:
        return "continue"

    for message in reversed(messages):
        if (isinstance(message,ToolMessage) and
        "saved" in message.content.lower() and
        "document" in message.content.lower()):
            return "end"

    return "continue"



def print_messages(messages):
    """Function I made to print the messages in a more readable format"""
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")

builder = StateGraph(AgentState)

builder.add_node("agent",call_agent)

tool_node = ToolNode(tools=tools)
builder.add_node("tools",tool_node)

builder.add_edge(START,"agent")
builder.add_conditional_edges("agent",should_continue,{"end":END,"continue":"tools"})
builder.add_edge("tools","agent")

graph = builder.compile()

def run_document_agent():
    print("\n ===== DRAFTER =====")
    
    state = {"messages": []}
    
    for step in graph.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    
    print("\n ===== DRAFTER FINISHED =====")

if __name__ == "__main__":
    run_document_agent()