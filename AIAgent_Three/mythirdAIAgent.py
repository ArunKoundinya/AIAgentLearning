from textwrap import dedent
from phi.agent import Agent
from phi.model.groq import Groq
from dotenv import load_dotenv
from phi.tools.googlesearch import GoogleSearch
from phi.tools.duckduckgo import DuckDuckGo
from phi.model.openai import OpenAIChat


# Load environment variables
load_dotenv()

# Initialize the model
groq_model = Groq(id="llama-3.3-70b-versatile")

# Create an agent with the model
google_agent = Agent(
    name="Google Web Agent",
    model=groq_model,
    tools=[GoogleSearch()],
    description="You are a highly qualified and experienced reader and writer. You can summarize the information from multiple sources. \
        You are best at reading and summarizing the information with the information at hand.",  ## Role
    instructions=[
        ## Task
        "Step1: For the Provided Topic, run 5 different searches",
        "Step2: Gather the information in detail",
        "Step3: Summarize the information in detail",
    ],
    markdown=True,
    add_datetime_to_instructions=True,
)


ducduckgo_agent = Agent(
    name="Duck Duck Go Web Agent",
    model=groq_model,
    tools=[DuckDuckGo()],
    description="You are a highly qualified and experienced reader and writer. You can summarize the information from multiple sources. \
        You are best at reading and summarizing the information with the information at hand.",  ## Role
    instructions=[
        ## Task
        "Step1: For the Provided Topic, run 5 different searches",
        "Step2: Gather the information in detail",
        "Step3: Summarize the information in detail",       
    ],
    markdown=True,
    add_datetime_to_instructions=True,
)
searchengine_team = Agent(
    name="Search Information Summary Team",
    model=OpenAIChat(id="gpt-4o"),
    team=[google_agent, ducduckgo_agent],
    description="You summarize information from Google and DuckDuckGo and present a detailed report.",
    instructions=[
        "Search for the provided topic using Google and DuckDuckGo.",
        "Summarize the information obtained from both searches into a consolidated report.",
    ],
    show_tool_calls=True,
    markdown=True,
    add_datetime_to_instructions=True,
)

# Test searchengine_team
try:
    response = searchengine_team.print_response("Tell me about Pushpa2 Movie Story", stream=True)
    print(response)
except Exception as e:
    print(f"Search Engine Team Error: {e}")
