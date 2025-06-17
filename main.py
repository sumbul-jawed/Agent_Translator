from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
import os

# Load environment variables
load_dotenv()

# Initialize rich console
console = Console()

# Get Gemini API key from .env
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

# Gemini-compatible OpenAI wrapper
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# Define Agent
agent = Agent(
    name='Writer Agent',
    instructions="""You are a helpful translator. Always translate Urdu sentences into clear and simple English."""
)

# Input Urdu sentence
input_urdu = 'mera nam sumbul jawed hai..mein IT ka course kar rahi ho governor house mein meri slot sunday hai 2 to 5 or sir Ali jawad teacher hai'

# Run agent
response = Runner.run_sync(
    agent,
    input=input_urdu,
    run_config=config
)

# Display output in styled panel
console.print(Panel(response.final_output, title="💬 Urdu to English Translation", style="bold blue"))
