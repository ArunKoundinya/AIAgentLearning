from phi.agent import Agent
from phi.model.groq import Groq
from phi.storage.agent.sqlite import SqlAgentStorage
from phi.tools.csv_tools import CsvTools
from dotenv import load_dotenv
from pathlib import Path
import httpx
import argparse
from phi.tools.pandas import PandasTools


load_dotenv()

# Initialize the model
groq_model = Groq(id="llama-3.3-70b-versatile")

def load_csv_from_url(url: str, save_path: str) -> Path:
    """
    Downloads a CSV file from the provided URL and saves it to the specified path.

    Args:
        url (str): URL of the CSV file.
        save_path (str): Directory path to save the downloaded file.

    Returns:
        Path: Path to the downloaded CSV file.
    """
    try:
        response = httpx.get(url)
        response.raise_for_status()  # Ensure the request was successful

        # Define the path to save the file
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        csv_file_path = save_dir.joinpath("downloaded_file.csv")

        # Save the file
        csv_file_path.write_bytes(response.content)
        return csv_file_path

    except httpx.RequestError as e:
        print(f"An error occurred while requesting the CSV file: {e}")
        raise
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Download a CSV file and use it in an agent.")
    parser.add_argument("url", type=str, help="URL of the CSV file to download")
    parser.add_argument("--save_path", type=str, default="wip", help="Directory to save the downloaded file")

    args = parser.parse_args()
    url = args.url
    save_directory = args.save_path

    # Download the CSV file
    csv_file_path = load_csv_from_url(url, save_directory)

    # Verify the downloaded file
    print(CsvTools(csvs=[csv_file_path]).list_csv_files())

    # Initialize the agent after the file is available
    csv_agent = Agent(
        name="CSV Agent",
        model=groq_model,
        tools=[CsvTools(csvs=[csv_file_path])],  # Pass the correct file path
        instructions=[
            "List all available files",
            "and run the query to answer the question",
        ],
        storage=SqlAgentStorage(table_name="web_agent", db_file="agents.db"),
        add_history_to_messages=True,
        markdown=True,
        show_tool_calls=True,
    )
    
    panda_agent = Agent(
        name = "Panda Agent",
        model = groq_model,
        tools=[PandasTools()],
        storage=SqlAgentStorage(table_name="web_agent", db_file="agents.db"),
        add_history_to_messages=True,
        )
    
    analysis_agent = Agent(
        name = "Analysis agent",
        model=groq_model,
        description="You are a highly qualified data analyst. You can summarize the information from multiple sources. \
        You are best at analyzing and summarizing the information with the information at hand.",  ## Role
        )

    
    MasterAgent = Agent(
        team=[csv_agent, panda_agent,analysis_agent],
        instructions=[
            "Step1: Use csv agent to load data", 
            "Step2: Use analysis agent to comprehend what user wants",
            "Step3: Use panda agent to process and manipulate data",
            "Step4: Use analysis agent to comprehend the analysis and present the use in a consumable and easy manner"],
        show_tool_calls=True,
        markdown=True,
    )
    # Start the CLI application
    MasterAgent.cli_app(stream=False)

if __name__ == "__main__":
    main()
