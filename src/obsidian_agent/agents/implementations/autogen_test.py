from autogen import ConversableAgent
import autogen
import os
from dotenv import load_dotenv

load_dotenv()

config_list_gemini = [
    {
         "model": "gemini-1.5-flash",
         "api_key": os.getenv("GOOGLE_API_KEY"),
         "api_type": "google"
    },
    {
         "model": "gemini-1.5-pro",
         "api_key": os.getenv("GOOGLE_API_KEY"),
         "api_type": "google"
    }
]

# Create specialized agents for plan creation
topic_analyzer = ConversableAgent(
    name="Topic Analyzer",
    system_message="Your task is to analyze the given topic and break it down into key components that need to be addressed in the plan. Provide a structured analysis highlighting main areas of focus.",
    llm_config={"config_list": config_list_gemini},
    code_execution_config=False,
    human_input_mode="NEVER",
    function_map=None
)

plan_drafter = ConversableAgent(
    name="Plan Drafter",
    system_message="Your task is to create a detailed, structured plan based on the topic analysis. Include clear steps, timelines, resource requirements, and potential challenges. Be thorough but practical.",
    llm_config={"config_list": config_list_gemini},
    code_execution_config=False,
    human_input_mode="NEVER",
    function_map=None
)

plan_critic = ConversableAgent(
    name="Plan Critic",
    system_message="Your task is to critically evaluate the proposed plan. Analyze for feasibility, completeness, risks, and potential improvements. Provide specific feedback on what needs to be revised.",
    llm_config={"config_list": config_list_gemini},
    code_execution_config=False,
    human_input_mode="NEVER",
    function_map=None
)

risk_assessor = ConversableAgent(
    name="Risk Assessor",
    system_message="Your task is to identify and analyze potential risks in the plan. Consider various scenarios, dependencies, and provide mitigation strategies for each identified risk.",
    llm_config={"config_list": config_list_gemini},
    code_execution_config=False,
    human_input_mode="NEVER",
    function_map=None
)

plan_refiner = ConversableAgent(
    name="Plan Refiner",
    system_message="Your task is to incorporate all feedback and suggestions to create an improved version of the plan. Focus on addressing critiques while maintaining the plan's core objectives.",
    llm_config={"config_list": config_list_gemini},
    code_execution_config=False,
    human_input_mode="NEVER",
    function_map=None
)

# Start the planning process
# First get topic analysis
initial_topic = "Create a plan for developing an agentic system that can help me manage my knowledge in obsidian. The system should be able to analyze my notes, identify key topics, and suggest links between them semantically and using a bottom-up approach, structure the graph view. I am a beginner in this field. My timeline is 1 week."  # Example topic
topic_analysis = topic_analyzer.generate_reply(messages=[{"content": f"Analyze this topic for planning: {initial_topic}", "role": "user"}])

# Draft initial plan
draft_response = plan_drafter.generate_reply(messages=[{"content": topic_analysis['content'], "role": "user"}])

# Get critique and risk assessment
plan_drafter.initiate_chat(
    plan_critic,
    message=draft_response['content'],
    max_turns=2
)

plan_drafter.initiate_chat(
    risk_assessor,
    message=draft_response['content'],
    max_turns=2
)

# Final refinement
plan_drafter.initiate_chat(
    plan_refiner,
    message=draft_response['content'],
    max_turns=2
)