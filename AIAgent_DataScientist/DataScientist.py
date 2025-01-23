from phi.agent import Agent
from phi.model.groq import Groq
from phi.storage.agent.sqlite import SqlAgentStorage
from dotenv import load_dotenv
from phi.run.response import RunEvent, RunResponse
from randomforest import *
from phi.tools.pandas import PandasTools
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

load_dotenv()

# Initialize the model
groq_model = Groq(id="llama-3.3-70b-versatile")


DSAnalyst = Agent(
        model=groq_model,
        tools=[RandomForest(),PandasTools()],
        description="You are a Professional Data Scientist",
        instructions=[
            "Step1: When User shares the csv link send to RandomForest Tool",
            "Step2: Get the Dictionary from RandomForest and understand and process the result",
            "Step3: Create a neat summary of the results with respect to the dataset provided",
            "Step4: If no csv link is provided go through the historical conversations and understand the query and provide answer like a Authentic Data Scientist.",
            "Optional: Use PandasTools to comprehend the dataset if needed."
        ],
        storage=SqlAgentStorage(table_name="DataScientist", db_file="agents.db"),
        add_history_to_messages=True,
        #show_tool_calls= True,
    )
    
#DSAnalyst.cli_app(stream=False)

def as_stream(response):
  for chunk in response:
    if isinstance(chunk, RunResponse) and isinstance(chunk.content, str):
      if chunk.event == RunEvent.run_response:
        yield chunk.content