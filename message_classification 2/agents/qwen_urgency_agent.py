"""
Urgency Classification Agent using Qwen2.5
"""

from openai import OpenAI
from dotenv import load_dotenv
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.json_parser import extract_json

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"), 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

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


class QwenUrgencyAgent:
    """Agent for urgency classification using Qwen2.5."""
    
    def __init__(self, model="qwen2.5-7b-instruct", temperature=0, verbose=False):
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
            print(f"[Qwen Urgency] {raw}")
        
        result = extract_json(raw)
        
        if "urgency" not in result:
            raise KeyError(f"Missing 'urgency' field: {raw}")
        
        return result


if __name__ == "__main__":
    agent = QwenUrgencyAgent(verbose=True)
    result = agent.classify("WHY is my package still not here!!")
    print(f"Result: {result}")