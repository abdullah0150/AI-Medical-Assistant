import re
from Workflow.utils.helper_functions import extract_messages, get_google_api_key, remove_sql_block
from Workflow.utils.state import State

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langsmith import traceable



MODEL_NAME = "gemini-2.0-flash-001"
google_api_key = get_google_api_key()
llm = ChatGoogleGenerativeAI(model=f"models/{MODEL_NAME}", google_api_key=google_api_key, temperature=0)


@traceable(metadata={"llm": MODEL_NAME})
def classify_user_intent(state: State) -> str:
    """
    Uses the LLM to classify the user question into one of four categories:
    - "query_related" for database queries and medical advice.
    - "complaint_related" for user complaints or issues.
    - "booking_related" for appointment bookings.
    - "medical_related" for medical advice or information

    Args:
        state: The current state of the workflow.

    Returns:
        One of the three category strings.
    """

    messages = str(state["messages"][-10:])
    human_messages, _ = extract_messages(messages)

    prompt_template = ChatPromptTemplate([
        (
            "system",
            """
            Determine the category of the latest user question. The question can belong to one of these three categories:

            1. **query_related**: The user wants to retrieve or analyze data from the database.
                - Examples:
                    - How many patients visited the clinic last month?
                    - Show me the appointment schedule for Dr. Smith.
                    - List all available doctors next Monday.

            2. **medical_related**: The user is asking for medical advice or information.
                - Examples:
                    - What are the symptoms of diabetes?
                    - I have a headache. What should I do?
                    - How can I lower my blood pressure?

            Respond with one of the following: 'query_related' or 'medical_related'.
            """
        ), 
        ("user", "{question}")
    ])

    chain = (
        {"question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(human_messages[-1])
    state["category"] = response

    print("Query Category: ", state["category"])
    return state["category"]



@traceable(metadata={"llm": MODEL_NAME, "embedding": "FAISS"})
def question_answer(state: State):

    messages = str(state["messages"][-10:])
    human_messages, _ = extract_messages(messages)


    prompt_template = ChatPromptTemplate([
        (
            "system",
            """
            You are a virtual medical assistant designed to provide general health advices.\n

            **Ignore the context** if it said **I'm sorry** and Respond based on your **Knowledge** and the **The guidelines are as follows:**\n

            Scope of Advice:
            - Answering general health inquiries.
            - Offering guidance based on common symptoms.
            - Avoid giving sensitive medical diagnoses, treatment prescriptions, or any specific therapeutic consultations.

            Responding to Out-of-Scope Questions:
            If you receive a question outside the scope of your role, respond politely as follows:
            "Sorry, I am designed to provide general health advices. For accurate medical consultation, please reach out to a licensed medical professional."

            Compliance and Privacy:
            - Ensure that advice is general, based on reliable information, and complies with medical regulations.

            Response Style:
            - Respond with kindness and professionalism, maintaining a supportive tone.
            - **Respond in the language in which the user asked the question.**
            """
        ), 
        ("user", human_messages)
    ])


    # Define chain
    chain = (
        {"context": RunnablePassthrough(), "messages": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    # Generate response
    response = chain.invoke({"messages": state["messages"]})
    state = {"messages": response}

    return state
