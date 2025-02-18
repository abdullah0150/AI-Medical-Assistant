import re
from Workflow.utils.helper_functions import extract_messages, get_google_api_key, remove_sql_block
from Workflow.utils.state import State

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langsmith import traceable




connection_uri = "mssql+pyodbc://@ASUS/MosefakDB?driver=ODBC+Driver+17+for+SQL+Server&Trusted_Connection=yes"
db = SQLDatabase.from_uri(connection_uri)

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





@traceable(metadata={"llm": MODEL_NAME})
def write_and_execute_query(state: State):
    """Generate an optimized SQL query, ensuring the database can provide the requested data."""

    tables_info = db.get_table_info()

    messages = str(state["messages"][-10:])
    human_messages, ai_messages = extract_messages(messages)

    print("Human Messages:", human_messages)
    print("AI Messages:", ai_messages)

    prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are an SQL expert specializing in SQL Server. Your role is to generate only a valid and optimized SQL query based on the user's request.

            **Key Responsibilities:**
            - Validate if the requested information can be retrieved using the available database schema.
            - If the required data is available, generate an optimized SQL Server query.
            - If the database does not contain relevant tables or fields, return **"Not Available"** as the SQL query.
            - Follow strict SQL Server syntax and best practices.
            - Do **not** provide explanations—return only the SQL query or "Not Available".

            **Context:**
            - **Previous Human Messages Context:** {messages}
            - **Database Schema:** {tables_info}

            **Expected Output:**
            ```sql
            -- Optimized SQL Query (or "Not Available" if the data is not retrievable)
            ```
            """
        ),
        ("user", f"{human_messages[-1]}")
    ])

    # Generate SQL query or determine if the request is unsupported
    chain = (
        prompt_template
        | llm
        | StrOutputParser()
    )

    result = chain.invoke({
        "messages": human_messages,
        "tables_info": tables_info
    })

    print("Generated SQL Query:", result)

    cleaned_query = remove_sql_block(result)

    # If the generated SQL query is "Not Available", return an empty result
    if cleaned_query.strip().lower() == "not available":
        state = {"SQLResult": "No data available for this request.", "SQLQuery": "Not Available"}
    else:
        query_result = db.run(cleaned_query)
        state = {"SQLResult": query_result, "SQLQuery": cleaned_query}

    print("SQL Result:", state["SQLResult"])
    return state



@traceable(metadata={"llm": MODEL_NAME})
def generate_answer(state: State):
    """Generate a professional and structured response, ensuring consistency between the question, query, and result."""
    
    prompt = f"""
    You are a professional AI assistant responding to a client. Your role is to provide clear, accurate, and well-structured answers.

    **Key Responsibilities:**
    1. **Validation:** Ensure that the SQL query and result align with the user's question.
    2. **Answer Generation:**
        - If the SQL result correctly answers the question, provide a **precise and well-structured response**.

    **Client’s Input:**
    - **User Question:** {state["messages"][-1].content}
    - **SQL Query:** {state["SQLQuery"]}
    - **SQL Result:** {state["SQLResult"]}

    **Response Guidelines:**
    - If the data is correct: Provide a **concise and professional answer**.
    - If the query or result is incorrect: Politely aswer that we didn't find the info you are looking for.
    - Maintain a **professional and reassuring tone**.
    - Respond in the same language as the client's question.
    """

    response = llm.invoke(prompt)
    return {"messages": [response]}
