from phi.agent import Agent
from phi.model.groq import Groq
from phi.storage.agent.sqlite import SqlAgentStorage
from dotenv import load_dotenv
from phi.run.response import RunEvent, RunResponse
from randomforest import *
from phi.tools.pandas import PandasTools


load_dotenv()

# Initialize the model
groq_model = Groq(id="llama-3.3-70b-versatile")


DSAnalyst = Agent(
        model=groq_model,
        tools=[RandomForest(),PandasTools()],
        description="You are a Professional Data Scientist",
        instructions=[
            "Step1: When User shares the csv link load the data using PandasTools",
            "Step2: After loading dataset from PandasTools pass the dataframe to RandomForest Tool",
            "Step3: Get the Dictionary from RandomForest and understand and process the result",
            "Step4: Create a neat summary of the results with respect to the dataset provided",
            "Step5: If no csv link is provided go through the historical conversations and understand the query and provide answer like a Authentic Data Scientist.",
        ],
        storage=SqlAgentStorage(table_name="DataScientist", db_file="agents.db"),
        add_history_to_messages=True,
    )
    

def as_stream(response):
  for chunk in response:
    if isinstance(chunk, RunResponse) and isinstance(chunk.content, str):
      if chunk.event == RunEvent.run_response:
        yield chunk.content