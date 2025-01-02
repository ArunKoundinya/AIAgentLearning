from phi.agent import Agent
from phi.model.groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the model
groq_model = Groq(id="llama-3.3-70b-versatile")

# Create an agent with the model
agent = Agent(model=groq_model)

# Get the response
try:
    response = agent.print_response("Write two sentences poem about Chicken & Mutton")
    print(response)
except AttributeError as e:
    print(f"Error: {e}")
    print("Check the documentation for the correct method to generate responses.")
