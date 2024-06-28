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

# from langchain.memory import ConversationBufferMemory
# from langchain.agents.agent_toolkits import SQLDatabaseToolkit

from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from langchain.chains import LLMChain

# from db import engine
from sqlalchemy import create_engine, inspect
import os
import uuid
from langchain_groq import ChatGroq
from dotenv import load_dotenv


# Load the .env file

# Access the environment variable
# groq_api_key = os.environ.get(
#     "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"
# )


# Define the model for the request body
class PromptRequest(BaseModel):
    prompt: str
    session_id: str


# Define the model for creating a new session
class CreateSessionRequest(BaseModel):
    session_name: str


# Initialize FastAPI app
app = FastAPI()


agent_executor = create_sql_agent(
    llm,
    # db=db,
    verbose=True,
    memory=memory,
    prefix=MSSQL_AGENT_PREFIX,
    format_instructions=MSSQL_AGENT_FORMAT_INSTRUCTIONS,
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
        "did not understand the question in relation to the database",
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
        if isinstance(response, dict) and "output" in response:
            response_text = response["output"]
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
            memory.save_context(
                {"prompt": f"{prompt}"}, {"response": f"{clarifying_question}"}
            )
            if session_id not in session_history:
                session_history[session_id] = []
            session_history[session_id].append({"role": "User", "message": prompt})
            session_history[session_id].append(
                {"role": "EaseAI", "message": clarifying_question}
            )
            return {
                "response": clarifying_question,
                "conversation": session_history[session_id],
            }
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
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
