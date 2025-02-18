from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    """
    Represents the state structure for the workflow.

    Attributes:
        question: The question input by the user.
        category: The high-level category of the user's query (query_related, medical_related).
        SQLQuery: The query generated during processing (if applicable).
        SQLResult: The result retrieved from the query (if applicable).
        answer: The final answer to the question.
        messages: The list of messages for the workflow.
    """
    question: Annotated[str, "User input question"]
    category: Annotated[str, "Categorized user intent (information_related, complaint_related, or booking_related)"]
    SQLQuery: Annotated[str, "Generated query if applicable"]
    SQLResult: Annotated[str, "Query result if applicable"]
    answer: Annotated[str, "Final answer"]
    messages: Annotated[list, add_messages]