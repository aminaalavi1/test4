import streamlit as st
import logging
from autogen import ConversableAgent, initiate_chats
import time

# Set up logging
logging.basicConfig(level=logging.INFO)

# Set up the title of the Streamlit app
st.title("HealthBite Assistant")

# Load OpenAI API key from Streamlit secrets
OPEN_API_KEY = st.secrets["OPENAI_API_KEY"]
config_list = [{"model": "gpt-3.5-turbo", "api_key": OPEN_API_KEY}]

# Initialize ConversableAgents
onboarding_personal_information_agent = ConversableAgent(
    name="onboarding_personal_information_agent",
    system_message='''You are a helpful patient onboarding agent.''',
    llm_config={"config_list": config_list},
    code_execution_config={"use_docker": False},
    human_input_mode="NEVER",
)

# Change human_input_mode to NEVER for customer_proxy_agent to avoid interaction issues
customer_proxy_agent = ConversableAgent(
    name="customer_proxy_agent",
    llm_config=False,
    code_execution_config={"use_docker": False},
    human_input_mode="NEVER",  # Change this to NEVER
    is_termination_msg=lambda msg: "terminate" in msg.get("content", "").lower(),
)

# Session state to store the chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from the history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
user_input = st.chat_input("You: ")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    st.write("Processing your request...")

    try:
        with st.spinner('Waiting for response...'):
            logging.info("Starting chat initiation...")
            
            # Prepare simplified chat
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
            
            # Log the chat configuration
            logging.info(f"Simplified Chat Configuration: {simplified_chat}")
            
            # Initiate the chat interaction
            result = initiate_chats(simplified_chat)
            logging.info(f"Chat result: {result}")

            # Check for result and process
            if result:
                assistant_response = result[-1]['message']
                with st.chat_message("assistant"):
                    st.markdown(assistant_response)
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            else:
                st.write("No response received from the agent.")
                logging.warning("No response received from initiate_chats.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logging.error(f"Error details: {e}", exc_info=True)

# Add a button to clear chat history
if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

# Display current date and version info
st.write(f"Current date: {time.strftime('%A, %B %d, %Y')}")
st.sidebar.write("App Version: 1.0.5")
st.sidebar.write(f"Streamlit Version: {st.__version__}")
