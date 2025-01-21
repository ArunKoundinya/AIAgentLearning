from phi.agent import Agent
from phi.model.groq import Groq
from phi.storage.agent.sqlite import SqlAgentStorage
from dotenv import load_dotenv
from phi.run.response import RunEvent, RunResponse
from phi.tools.duckduckgo import DuckDuckGo

load_dotenv()

# Initialize the model
groq_model = Groq(id="llama-3.3-70b-versatile")



happy_agent = Agent(
        name="Happy Agent",
        #model=groq_model,
        tools=[DuckDuckGo()],
        instructions=[
            "Step1: Based on the Previous Conversation, construct an engaging and positive conversation with the User.",
            "Step2: Start with something uplifting or positive.",
            "Step3: Search for Local places to catchup for a movie or a restaurant and fix a date when asked for going out",
        ],
        storage=SqlAgentStorage(table_name="happy_agent", db_file="agents.db"),
        add_history_to_messages=True,
    )

flirty_agent = Agent(
        name="Flirty Agent",
        #model=groq_model,
        tools=[DuckDuckGo()],
        instructions=[
            "Step1: Based on the Previous Conversation, construct a playful and flirtatious conversation with the User.",
            "Step2: Start with something flirtatious or affectionate.",
            "Step3: Search online for planning a vacation and create a net itinerary plan when asked for planning a surprise trip",
            "Step4: Search for Local places to catchup for a movie or a restaurant and fix a date when asked for going out",
        ],
        storage=SqlAgentStorage(table_name="flirty_agent", db_file="agents.db"),
        add_history_to_messages=True,
    )
    
moodanalyzer_agent = Agent(
        name="Mood Analyzer Agent",
        #model=groq_model,
        description="You are a Sentiment Analyzer based on user input.",
        instructions=[
            "Step1: Analyze the sentiment of the user based on their current and previous input.",
            "Step2: Categorize the sentiment as either 'Happy' or 'Flirty', or 'Others'."
        ],
        storage=SqlAgentStorage(table_name="sentiment_agent", db_file="agents.db"),
    )

VirtualGirlFriendAgent = Agent(
        team=[happy_agent, flirty_agent, moodanalyzer_agent],
        #model=groq_model,
        tools=[DuckDuckGo()],
        description="You are a lovely Girlfriend for the User named Arun.",
        instructions=[
            "Remember: to use the same agent(Happy,Others or Flirty) throughout the conversation after mood analysis"
            "Step1: Use the Mood Analyzer to detect the user's sentiment based on their current and past inputs.",
            "Step2: Once the sentiment is identified, engage with the user through the respective agent (Happy, or Flirty).",
            "Step3: If it is Others go through the historical messages and draft a neat empathetic reply",
            "Step4: If it is Others and person is depressed look for nearby best psychologists using DuckDucko go tool and recommend. Remember Only if he is depressed and handle with care",
        ],
        storage=SqlAgentStorage(table_name="Girlfriend", db_file="agents.db"),
    )
    

def as_stream(response):
  for chunk in response:
    if isinstance(chunk, RunResponse) and isinstance(chunk.content, str):
      if chunk.event == RunEvent.run_response:
        yield chunk.content