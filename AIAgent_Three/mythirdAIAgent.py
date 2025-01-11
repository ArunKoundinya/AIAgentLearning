from textwrap import dedent
from phi.agent import Agent
from phi.model.groq import Groq
from dotenv import load_dotenv
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.newspaper4k import Newspaper4k
# Load environment variables
load_dotenv()

# Initialize the model
groq_model = Groq(id="llama-3.3-70b-versatile")

# Create an agent with the model
web_agent = Agent(
    name="Web Agent",
    tools=[DuckDuckGo()],
    description="You are a highly qualified scrapper to extract urls for the given topic",  ## Role
    instructions=[
        ## Task
        "Step1: For the Provided Topic, run 5 different searches",
        "Step2: Gather the information in urls",
        "Step3: Share the Urls",
        ## Specifics
        "If you are unable to find sufficient information, inform the user to search themselves on their own",
        "Your role is very vital for the user. Your assistance and help is needed",
        ## Context
        ## No Business Context
        ## Examples
        ## No Examples
        ## Notes
        "You are world class person in scraping",
    ],
    expected_output=dedent("""\
    - url link 1
    - url link 2
    - url link 3
    - url link 4
    - url link 5

    """),
    show_tool_calls=True,
    markdown=True,
    add_datetime_to_instructions=True,
)


reader_agent = Agent(
    tools=[Newspaper4k()],
    show_tool_calls=True
    )
#agent.print_response("Please summarize https://www.rockymountaineer.com/blog/experience-icefields-parkway-scenic-drive-lifetime")

agent_team = Agent(
    team=[web_agent, reader_agent],
    model=groq_model,
    description="You are a highly qualified and experienced reader and writer. You can summarize the information from multiple sources. \
        You are best at reading and summarizing the information with the information at hand.",  ## Role
    instructions=[
        ## Task
        "Step1: For the Provided Topic, get 5 urls from web_agent",
        "Step2: Gather the information for each url using reader_agent",
        "Step3: Summarize the information in detail",
        "Step4: Focus on the facts and make sure not to hallucinate",
        ## Specifics
        "If you are unable to find sufficient information, inform the user to search themselves on their own",
        "Your role is very vital for the user. Your assistance and help is needed",
        ## Context
        ## No Business Context
        ## Examples
        ## No Examples
        ## Notes
        "You are world class person in scraping, reading, summarizing and writing",
    ],
    expected_output=dedent("""\
    An engaging, informative, and well-structured report in markdown format:

    ## Engaging Title

    ### Overview
    {give a brief introduction of the information and why the user should read this information}
    {make this section engaging and create a hook for the reader}

    ### Section 1
    {break the summary into sections}
    {provide details/facts/processes in this section}

    ... more sections as necessary...

    ### Synopsis
    {provide key takeaways from the information you gathered}

    ### References
    - [Reference 1](link)
    - [Reference 2](link)
    - [Reference 3](link)
    - [Reference 4](link)
    - [Reference 5](link)

    """),
    show_tool_calls=True,
    markdown=True,
)

# Get the response
try:
    #response = web_agent.print_response("Tell me about Pushpa2 Movie Story", stream=True)
    #response = reader_agent.print_response("Please summarize https://dmtalkies.com/pushpa-2-movie-spoilers-full-story-2024/ ", stream=True)
    response = agent_team.print_response("Tell me about Pushpa2 Movie Story", stream=True)
    print(response)
except AttributeError as e:
    print(f"Error: {e}")
    print("Check the documentation for the correct method to generate responses.")


# https://en.wikipedia.org/wiki/Pushpa_2:_The_Rule  
# https://dmtalkies.com/pushpa-2-movie-spoilers-full-story-2024/ 