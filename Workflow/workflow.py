from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from Workflow.utils.nodes import classify_user_intent, question_answer
from Workflow.utils.state import State

# Initialize FastAPI app
app = FastAPI()

# Define request model
class QuestionRequest(BaseModel):
    question: str

# Initialize LangGraph workflow
class Workflow:
    def __init__(self):
        self.graph_builder = StateGraph(State)
        self.graph_builder.add_node("question_answer", question_answer)

        self.graph_builder.add_conditional_edges(
            START,
            classify_user_intent,
            {
                "medical_related": "question_answer"
            }
        )

        self.graph_builder.add_edge("question_answer", END)

        self.memory = MemorySaver()
        self.graph = self.graph_builder.compile(checkpointer=self.memory)

        self.viz_graph = self.graph.get_graph().draw_mermaid_png()

    def get_response(self, question: str) -> str:
        config = {"configurable": {"thread_id": "2"}}
        
        try:
            events = self.graph.stream(
                {"messages": [{"role": "user", "content": question}]},
                config,
                stream_mode="values",
            )
            
            last_message = None
            for event in events:
                last_message = event["messages"][-1].content
            
            return last_message if last_message else "No results found."
        except Exception as e:
            print(f"An error occurred during workflow execution: {e}")
            return "An error occurred while processing the request."

# Instantiate the workflow
workflow = Workflow()

@app.post("/ask")
def ask_question(request: QuestionRequest):
    try:
        response = workflow.get_response(request.question)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/visualize")
def visualize_workflow():
    try:
        return {"graph": workflow.visualize_graph()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
