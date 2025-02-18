import os
import re
import textwrap
from dotenv import load_dotenv
load_dotenv()




def get_google_api_key():
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not set. Please set it as an environment variable.")
    
    return GOOGLE_API_KEY
    

def to_markdown(text):
    text = text.replace('•', '  *')
    return "> " + textwrap.indent(text, '> ', predicate=lambda _: True).replace('\n', '\n> ')


def remove_sql_block(input_string: str) -> str:
    """
    Removes the ```sql``` block from the input string.
    
    Args:
        input_string (str): The input string containing the SQL query with ```sql``` block.
    
    Returns:
        str: The SQL query without the ```sql``` block.
    """
    # Remove the leading and trailing ```sql```
    if input_string.startswith("```sql") and input_string.endswith("```"):
        # Strip the ```sql``` and trailing ```
        return input_string[6:-3].strip()
    return input_string



def extract_messages(input_string):
    """
    Extracts human and AI messages from the input string.

    Args:
        input_string (str): The input string containing HumanMessage and AIMessage objects.

    Returns:
        tuple: A tuple containing two lists:
            - human_messages: List of human messages.
            - ai_messages: List of AI messages.
    """
    # Regex pattern to match HumanMessage and AIMessage content
    pattern = r"(HumanMessage|AIMessage)\(content='(.*?)'[^)]*\)"

    # Find all matches
    matches = re.findall(pattern, input_string)

    # Separate variables for human and AI messages
    human_messages = []
    ai_messages = []

    # Organize results
    for match in matches:
        message_type, content = match
        if message_type == "HumanMessage":
            human_messages.append(content)
        elif message_type == "AIMessage":
            ai_messages.append(content)

    return human_messages, ai_messages