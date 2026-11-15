from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm


def build_agent(model: str, api_base: str) -> LlmAgent:
    return LlmAgent(
        name="writer",
        description="Writes Ukrainian texts and delegates editing to an attached editor.",
        instruction="Write Ukrainian texts. Use editor if that tool is available.",
        model=LiteLlm(model="ollama_chat/" + model, api_base=api_base),
    )
