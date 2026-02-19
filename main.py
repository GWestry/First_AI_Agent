from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from tools import search_tool, wikipedia_tool

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# Bind tools to the LLM
llm_with_tools = llm.bind_tools([search_tool, wikipedia_tool])

prompt = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            """You are a research assistant that will help generate a research paper.
            Answer the user query and use tools when needed.
            """,
        ),
        ("human", "{query}"),

    ]
)

#Prompt user for query
query = input("What can I help you research today? ")

# Format the messages
messages = prompt.format_messages(query=query)

# Execution loop
max_iterations = 5
iteration = 0

while iteration < max_iterations:
    iteration += 1
    print(f"\n[Iteration {iteration}]")
    
    #turn verbose on to see thinking and execute
    raw_response = llm_with_tools.invoke(messages)
    
    #check for tool usage
    if raw_response.tool_calls:
        print("Tools used:")
        for tool_call in raw_response.tool_calls:
            print(f"  - {tool_call['name']}: {tool_call['args']}")
        
        # add response to message
        messages.append(AIMessage(content=raw_response.content or "", tool_calls=raw_response.tool_calls))
        
        #execute each tool call
        for tool_call in raw_response.tool_calls:
            tool_name = tool_call['name']
            tool_input = tool_call['args']['query'] if 'query' in tool_call['args'] else str(tool_call['args'])
            
            print(f"\nExecuting {tool_name} with: {tool_input}")
            
            if tool_name == "search_tool":
                result = search_tool.invoke(tool_input)
            elif tool_name == "wikipedia_tool":
                result = wikipedia_tool.invoke(tool_input)
            else:
                result = "Tool not found"
            
            print(f"Result: {result[:200]}...")  # Print first 200 chars
            
            #add tool result to messages
            messages.append(ToolMessage(content=result, tool_call_id=tool_call['id']))
    else:
        response_content = raw_response.content
        print("\nFinal Response:")
        print(response_content)
        break