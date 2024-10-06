import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from autogen import ConversableAgent, initiate_chats

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
    code_execution_config={"use_docker": False},  # Disable Docker for code execution
    human_input_mode="NEVER",
)

customer_engagement_agent = ConversableAgent(
    name="customer_engagement_agent",
    system_message='''You are a friendly and engaging patient service agent. Your task is to provide the customer with a personalized meal plan for the day. Tailor the meal plan based on the customer's chronic disease. The meal plan should include:
    - Recipes for each meal, detailing the exact ingredients needed and how to cook the meal.
    - A grocery list compiling all the ingredients required for the day.
    - Serving sizes and calorie counts for each meal.
    - Nutritional information like servings of greens, fruits, vegetables, fiber, proteins, etc.
    Additionally, generate a data frame with Date, Meal (like breakfast/Lunch/Dinner), Fat%, calorie intake, and sugar.
    Also, include a plot visualizing the nutritional information.''',
    llm_config={"config_list": config_list},
    code_execution_config={
        "allowed_imports": ["pandas", "matplotlib", "seaborn"],
        "execution_timeout": 60,
        "use_docker": False  # Disable Docker for code execution
    },
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: "terminate" in msg.get("content").lower(),
)

customer_proxy_agent = ConversableAgent(
    name="customer_proxy_agent",
    llm_config=False,
    code_execution_config={"use_docker": False},  # Disable Docker
    human_input_mode="ALWAYS",
    is_termination_msg=lambda msg: "terminate" in msg.get("content").lower(),
)

# Session state to store the chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from the history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if user_input := st.chat_input("You: "):
    # Append the user input to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display the user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Phase 1: Onboarding agent collects info
    st.write("Phase 1: Onboarding agent collecting info...")

    onboarding_chat = [
        {
            "sender": onboarding_personal_information_agent,
            "recipient": customer_proxy_agent,
            "message": user_input,  # First message from user
            "summary_method": "reflection_with_llm",
            "max_turns": 2,  # Let the onboarding agent gather initial information
            "clear_history": False
        }
    ]

    # Start phase 1
    try:
        phase1_results = initiate_chats(onboarding_chat)
        
        # Debugging: Show phase 1 results
        st.write("Phase 1 Results:", phase1_results)

        # Get the result and check if information was gathered
        if phase1_results and len(phase1_results) > 0:
            onboarding_response = phase1_results[-1]['message']

            # Display onboarding response
            with st.chat_message("assistant"):
                st.markdown(onboarding_response)

            # Append onboarding response to chat history
            st.session_state.messages.append({"role": "assistant", "content": onboarding_response})

            # Phase 2: Engagement agent generates meal plan
            st.write("Phase 2: Engagement agent generating meal plan...")

            engagement_chat = [
                {
                    "sender": customer_proxy_agent,
                    "recipient": customer_engagement_agent,
                    "message": "Now that we have the initial information, let's proceed with your meal plan.",
                    "summary_method": "reflection_with_llm",
                    "max_turns": 1,  # Engagement agent generates the meal plan
                }
            ]

            # Start phase 2
            phase2_results = initiate_chats(engagement_chat)

            # Debugging: Show phase 2 results
            st.write("Phase 2 Results:", phase2_results)

            # Get the engagement response and display
            if phase2_results and len(phase2_results) > 0:
                engagement_response = phase2_results[-1]['message']

                # Display engagement response
                with st.chat_message("assistant"):
                    st.markdown(engagement_response)

                # Append engagement response to chat history
                st.session_state.messages.append({"role": "assistant", "content": engagement_response})

            else:
                st.write("No response received from engagement agent.")
        else:
            st.write("No response received from onboarding agent.")

    except Exception as e:
        st.error(f"An error occurred during chat interaction: {e}")
        st.write(f"Error details: {e}")
