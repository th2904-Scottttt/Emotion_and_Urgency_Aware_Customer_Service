"""
Emotional Tone Classification Agent using Qwen2.5
"""

from openai import OpenAI
from dotenv import load_dotenv
import os
from .json_parser import extract_json

load_dotenv()


PROMPT_EMOTIONAL = """
You are an emotional tone classification agent.

Given a user message, classify the emotional tone.
Possible values: angry, frustrated, anxious, neutral, satisfied, happy.

Return ONLY valid JSON:
{{
  "emotional": "angry | frustrated | anxious | neutral | satisfied | happy"
}}

User message:
{text}
"""


class QwenEmotionalAgent:
    """Agent for emotional tone classification using Qwen2.5."""
    
    def __init__(self, model="qwen2.5-7b-instruct", temperature=0, verbose=False):
        self.model = model
        self.temperature = temperature
        self.verbose = verbose
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
    def classify(self, text):
        """Classify emotional tone."""
        prompt = PROMPT_EMOTIONAL.format(text=text)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Classify emotional tone. Output pure JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
        )
        
        raw = response.choices[0].message.content
        
        if self.verbose:
            print(f"[Qwen Emotional] {raw}")
        
        result = extract_json(raw)
        
        if "emotional" not in result:
            raise KeyError(f"Missing 'emotional' field: {raw}")
        
        return result


if __name__ == "__main__":
    agent = QwenEmotionalAgent(verbose=True)
    result = agent.classify("Thanks so much for your help!")
    print(f"Result: {result}")