from textwrap import dedent
from phi.agent import Agent
from phi.model.groq import Groq
from dotenv import load_dotenv
from phi.tools.duckduckgo import DuckDuckGo

# Load environment variables
load_dotenv()

# Initialize the model
groq_model = Groq(id="llama-3.3-70b-versatile")

# Create an agent with the model
web_agent = Agent(
    name="Web Agent",
    model=groq_model,
    tools=[DuckDuckGo()],
    description="You are a highly qualified and experienced reader and writer. You can summarize the information from multiple sources. \
        You are best at reading and summarizing the information with the information at hand.",  ## Role
    instructions=[
        ## Task
        "Step1: For the Provided Topic, run 5 different searches",
        "Step2: Gather the information in detail",
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
    add_datetime_to_instructions=True,
    #save_response_to_file="tmp/{message}.md",
)


# Get the response
try:
    response = web_agent.print_response("Tell me about Pushpa2 Movie Story", stream=True)
    print(response)
except AttributeError as e:
    print(f"Error: {e}")
    print("Check the documentation for the correct method to generate responses.")
