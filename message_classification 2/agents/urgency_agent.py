"""
Urgency Classification Agent
"""

from openai import OpenAI
from dotenv import load_dotenv
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.json_parser import extract_json

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


PROMPT_URGENCY = """
You are an urgency classification agent.

Given a user message, classify urgency.
Possible values: low, medium, high.

Return ONLY valid JSON:
{{
  "urgency": "low | medium | high"
}}

User message:
{text}
"""


class UrgencyAgent:
    """Agent for urgency classification."""
    
    def __init__(self, model="gpt-4o-mini", temperature=0, verbose=False):
        self.model = model
        self.temperature = temperature
        self.verbose = verbose
        self.client = client
        
    def classify(self, text):
        """Classify urgency level."""
        prompt = PROMPT_URGENCY.format(text=text)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Classify urgency. Output pure JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
        )
        
        raw = response.choices[0].message.content
        
        if self.verbose:
            print(f"[Urgency] {raw}")
        
        result = extract_json(raw)
        
        if "urgency" not in result:
            raise KeyError(f"Missing 'urgency' field: {raw}")
        
        return result


if __name__ == "__main__":
    agent = UrgencyAgent(verbose=True)
    result = agent.classify("WHY is my package still not here!!")
    print(f"Result: {result}")