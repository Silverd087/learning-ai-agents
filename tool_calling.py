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
model = ChatAnthropic(model_name="claude-opus-4-8")

@tool
def add(x,y):
    "addition function"
    return x+y
@tool
def substract(x,y):
    "substraction function"
    return x-y
@tool
def multiply(x,y):
    "multiplication function"
    return x*y

tools = [add,substract,multiply]

model = model.bind_tools(tools=tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages]

def ai_node(state:AgentState):
    system_message = SystemMessage(content="you are an AI assistant, please answer to the best of your ability")
    response = model.invoke([system_message] + state["messages"])
    return {"messages":[response]}

def should_continue(state:AgentState):
    last_message = state["messages"][-1]
    if len(last_message.tool_calls)>0:
        return "continue"
    else:
        return "end"


builder = StateGraph(AgentState)

builder.add_node("agent",ai_node)

tool_node = ToolNode(tools=tools)
builder.add_node("tools",tool_node)

builder.add_edge(START,"agent")
builder.add_conditional_edges("agent",should_continue,{"continue":"tools","end":END})

builder.add_edge("tools","agent")

graph = builder.compile()

for event in graph.stream({"messages":[("user","add 2 and 5")]}):
    for v in event.values():
        print(v)