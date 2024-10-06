import streamlit as st
import time
import logging
from autogen import ConversableAgent, UserProxyAgent, AssistantAgent, GroupChat, GroupChatManager

# Set up logging
logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="HealthBite Assistant", layout="wide")

# Set up the title of the Streamlit app
st.title("HealthBite Assistant")

# Load OpenAI API key from Streamlit secrets
OPEN_API_KEY = st.secrets["OPENAI_API_KEY"]
config_list = [{"model": "gpt-3.5-turbo", "api_key": OPEN_API_KEY}]

# Initialize agents
assistant = AssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant for patient onboarding. Gather the patient's name, chronic disease, zip code, and meal cuisine preference.",
    llm_config={"config_list": config_list},
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=2,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={"use_docker": False},
)

# Create GroupChat
groupchat = GroupChat(agents=[user_proxy, assistant], messages=[], max_round=5)
manager = GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list})

# Session state to store the chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from the history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to run chat with timeout
def run_chat_with_timeout(user_input, timeout=30):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            return manager.initiate_chat(
                manager.groupchat.agents[0],
                message=user_input
            )
        except Exception as e:
            logging.error(f"Error in chat initiation: {e}")
            time.sleep(1)  # Wait a bit before retrying
    raise TimeoutError("Chat initiation timed out")

# Accept user input
user_input = st.chat_input("You: ")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    st.write("Processing your request...")

    try:
        with st.spinner('Waiting for response...'):
            chat_result = run_chat_with_timeout(user_input)
        
        logging.info(f"Chat result: {chat_result}")
        st.json(chat_result)  # Display raw chat result for debugging

        if chat_result:
            assistant_response = chat_result.get("content", "No response from assistant")
            with st.chat_message("assistant"):
                st.markdown(assistant_response)
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        else:
            st.write("No response received from the agent.")

    except TimeoutError:
        st.error("The conversation timed out. Please try again.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logging.error(f"Error details: {e}", exc_info=True)

# Add a button to clear chat history
if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

# Display current date and version info
st.write(f"Current date: {time.strftime('%A, %B %d, %Y')}")
st.sidebar.write("App Version: 1.0.4")
st.sidebar.write(f"Streamlit Version: {st.__version__}")
