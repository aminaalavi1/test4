import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from autogen import ConversableAgent, initiate_chats
import time
from streamlit.scriptrunner import add_script_run_ctx
import threading

# Set up the title of the Streamlit app
st.title("HealthBite Assistant")

# Load OpenAI API key from Streamlit secrets
OPEN_API_KEY = st.secrets["OPENAI_API_KEY"]
config_list = [{"model": "gpt-3.5-turbo", "api_key": OPEN_API_KEY}]

# Initialize ConversableAgents and explicitly disable Docker
onboarding_personal_information_agent = ConversableAgent(
    name="onboarding_personal_information_agent",
    system_message='''You are a helpful patient onboarding agent,
    you are here to help new patients get started with our product.
    Your job is to gather the patient's name, their chronic disease, zip code, and meal cuisine preference.
    When they give you this information, ask them what their meal preferences are, the cuisine they like, and what ingredients they would like to avoid.
    Do not ask for other information. Return 'TERMINATE' when you have gathered all the information.''',
    llm_config={"config_list": config_list},
    code_execution_config={"use_docker": False},
    human_input_mode="NEVER",
)

customer_proxy_agent = ConversableAgent(
    name="customer_proxy_agent",
    llm_config=False,
    code_execution_config={"use_docker": False},
    human_input_mode="ALWAYS",
    is_termination_msg=lambda msg: "terminate" in msg.get("content", "").lower(),
)

# Session state to store the chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from the history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to run chat with timeout
def run_chat_with_timeout(user_input, timeout=30):
    result = [None]
    def wrapper():
        simplified_chat = [
            {
                "sender": onboarding_personal_information_agent,
                "recipient": customer_proxy_agent,
                "message": user_input,
                "summary_method": "reflection_with_llm",
                "max_turns": 2,
                "clear_history": False
            }
        ]
        result[0] = initiate_chats(simplified_chat)

    thread = threading.Thread(target=wrapper)
    add_script_run_ctx(thread)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        st.error("Operation timed out. Please try again.")
        return None
    return result[0]

# Accept user input
if user_input := st.chat_input("You: "):
    # Append the user input to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display the user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Phase 1: Onboarding agent collects info
    st.write("Phase 1: Onboarding agent sending message to customer proxy...")

    # Start simplified chat interaction with timeout
    try:
        with st.spinner('Processing your request...'):
            simplified_results = run_chat_with_timeout(user_input)
        
        # Debugging: Show results from simplified chat
        st.write("Debug: Simplified Chat Results:", simplified_results)

        # Check if there are results from the onboarding agent
        if simplified_results and len(simplified_results) > 0:
            onboarding_response = simplified_results[-1]['message']

            # Display onboarding response
            with st.chat_message("assistant"):
                st.markdown(onboarding_response)

            # Append onboarding response to chat history
            st.session_state.messages.append({"role": "assistant", "content": onboarding_response})

        else:
            st.write("Debug: No response received from onboarding agent or proxy agent.")

    except Exception as e:
        st.error(f"An error occurred during the chat interaction: {str(e)}")
        st.write(f"Debug: Error details: {e}")

# Add a button to clear chat history
if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.experimental_rerun()

# Display current date
st.write(f"Current date: {time.strftime('%A, %B %d, %Y')}")

# Add version information
st.sidebar.write("App Version: 1.0.0")
st.sidebar.write(f"Streamlit Version: {st.__version__}")
