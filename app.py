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
    - Nutritional information like servings of greens, fruit
