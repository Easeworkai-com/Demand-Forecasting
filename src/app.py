import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any
from langchain_community.utilities import SQLDatabase
from langchain.agents import AgentType
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from typing import List, Dict, Any
from langchain_experimental.tools import PythonREPLTool
from langchain.memory import ConversationSummaryMemory
#from langchain.memory import ConversationBufferMemory
#from langchain.agents.agent_toolkits import SQLDatabaseToolkit

from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from langchain.chains import LLMChain
from db import engine
from sqlalchemy import create_engine, inspect
from dotenv import load_dotenv
import os

import uuid
from langchain_groq import ChatGroq

load_dotenv()  

# Define the model for the request body
class PromptRequest(BaseModel):
    prompt: str
    session_id: str

# Define the model for creating a new session
class CreateSessionRequest(BaseModel):
    session_name: str

# Initialize FastAPI app
app = FastAPI()

# Initialize OpenAI API key
#openai_api_key = os.getenv("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")  
#groq_api_key = os.getenv("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") 

# Database setup
db = SQLDatabase.from_uri("sqlite:///my_database.db")
print("Available tables:", db.get_usable_table_names())

api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(openai_api_key=api_key, model="gpt-3.5-turbo", temperature=0)

# llm = ChatGroq(
#       temperature=0, 
#       groq_api_key = "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", 
#        model_name="llama3-70b-8192"
#          )

memory = ConversationSummaryMemory(llm=llm)
python_repl_tool = PythonREPLTool()
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

MSSQL_AGENT_PREFIX = """

You are an agent designed to interact with a SQL database.
## Instructions:
- Given an input question, create a syntactically correct {dialect} query
to run, then look at the results of the query and return the answer.
- Unless the user specifies a specific number of examples they wish to
obtain, **ALWAYS** limit your query to at most {top_k} results.
- You can order the results by a relevant column to return the most
interesting examples in the database.
- Never query for all the columns from a specific table, only ask for
the relevant columns given the question.
- You have access to tools for interacting with the database.
- You MUST double check your query before executing it.If you get an error
while executing a query,rewrite the query and try again.
- DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.)
to the database.
- DO NOT MAKE UP AN ANSWER OR USE PRIOR KNOWLEDGE, ONLY USE THE RESULTS
OF THE CALCULATIONS YOU HAVE DONE.
- Your response should be in Markdown. However, **when running  a SQL Query
in "Action Input", do not include the markdown backticks**.
Those are only for formatting the response, not for executing the command.
- ALWAYS, as part of your final answer, explain how you got to the answer
on a section that starts with: "Explanation:". Include the SQL query as
part of the explanation section.
- If the question does not seem related to the database, just return
"I don\'t know" as the answer.
- Only use the below tools. Only use the information returned by the
below tools to construct your query and final answer.
- Do not make up table names, only use the tables returned by any of the
tools below.

## Tools:

"""

MSSQL_AGENT_FORMAT_INSTRUCTIONS = """

## Use the following format:

Question: the input question you must answer.
Thought: you should always think about what to do.
Action: the action to take, should be one of [{tool_names}].
Action Input: the input to the action.
Observation: the result of the action.
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer.
Final Answer: the final answer to the original input question.

Example of Final Answer:
<=== Beginning of example

Action: query_sql_db
Action Input: 
SELECT TOP (10) [death]
FROM covidtracking 
WHERE state = 'TX' AND date LIKE '2020%'

Observation:
[(27437.0,), (27088.0,), (26762.0,), (26521.0,), (26472.0,), (26421.0,), (26408.0,)]
Thought:I now know the final answer
Final Answer: There were 27437 people who died of covid in Texas in 2020.

Explanation:
I queried the `covidtracking` table for the `death` column where the state
is 'TX' and the date starts with '2020'. The query returned a list of tuples
with the number of deaths for each day in 2020. To answer the question,
I took the sum of all the deaths in the list, which is 27437.
I used the following query

```sql
SELECT [death] FROM covidtracking WHERE state = 'TX' AND date LIKE '2020%'"
```
===> End of Example

"""

agent_executor = create_sql_agent(
    llm,
    #db=db,
    verbose=True,
    memory=memory,
    prefix=MSSQL_AGENT_PREFIX,
    format_instructions = MSSQL_AGENT_FORMAT_INSTRUCTIONS,
    toolkit=toolkit,
    handle_parsing_errors=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    tools=[python_repl_tool],
)

session_history: Dict[str, List[Dict[str, str]]] = {}

def is_confident(response: str) -> bool:
    uncertain_keywords = [
        "cannot be determined",
        "Request too large",
        "no such table",
        "no such column",
        "ambiguous column name",
        "syntax error near",
        "did not understand the question in relation to the database"
    ]
    return not any(keyword in response for keyword in uncertain_keywords)

@app.post("/query")
async def query_db(request: PromptRequest):
    session_id = request.session_id
    prompt = request.prompt
    context = (
        "Act as a Data Analyst'. "
        "There is the ONLY table in the database."
        "Given the above conversation generate a search query to lookup in order to get the information only relevant to the conversation."
        "Extract column names and table name and try to map user words with exact column names as user can use synonyms."
        "Use all the data and Run multiple queries if required before giving the final answer."
    )

    inputs = {"prompt": prompt}
    context_window = memory.load_memory_variables(inputs)
    conversation_context = f"Given the context: {context} and the recent chat history {context_window['history']} , Answer the question: {prompt}."

    try:
        response = agent_executor.invoke(conversation_context)

        # Extract the actual response text if it's in a dictionary
        if isinstance(response, dict) and 'output' in response:
            response_text = response['output']
        else:
            response_text = str(response)

        if not is_confident(response_text):
            clarifying_question = f"I didn't quite understand your question about '{prompt}'. Can you please clarify or provide more details?"
            response_text = clarifying_question

        # Save the conversation context with the desired format
        memory.save_context({"prompt": f"{prompt}"}, {"response": f"{response_text}"})

        # Save the conversation context externally
        if session_id not in session_history:
            session_history[session_id] = []
        session_history[session_id].append({"role": "User", "message": prompt})
        session_history[session_id].append({"role": "EaseAI", "message": response_text})

        return {"response": response_text, "conversation": session_history[session_id]}
    except Exception as e:
        # Handling errors
        if "parsing error" in str(e).lower():
            clarifying_question = f"I encountered an error understanding your request: '{prompt}'. Can you please provide more details or clarify your question?"
            memory.save_context({"prompt": f"{prompt}"}, {"response": f"{clarifying_question}"})
            if session_id not in session_history:
                session_history[session_id] = []
            session_history[session_id].append({"role": "User", "message": prompt})
            session_history[session_id].append({"role": "EaseAI", "message": clarifying_question})
            return {"response": clarifying_question, "conversation": session_history[session_id]}
        else:
            # Log the error for debugging
            print(f"Error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset_memory")
async def reset_memory(session_id: str):
    memory.clear()
    if session_id in session_history:
        del session_history[session_id]
    return {"message": "Conversation memory reset successfully"}

@app.get("/history/{session_id}")
async def get_history(session_id: str):
    return {"history": session_history.get(session_id, [])}

@app.post("/create_session")
async def create_session(request: CreateSessionRequest):
    session_id = str(uuid.uuid4())
    session_history[session_id] = []
    return {"session_id": session_id, "session_name": request.session_name}

@app.get("/sessions")
async def get_sessions():
    return {"sessions": list(session_history.keys())}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
