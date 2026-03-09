import os
from dotenv import load_dotenv
import json

load_dotenv()

def analyze_sentiment(text: str) -> dict:
    """
    Uses Together API (Qwen3-VL-8B-Instruct) to analyze feedback text sentiment.
    Returns a dictionary with 'score' (0.0 to 1.0) and 'rationale'.
    """
    from together import Together
    
    client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
    
    prompt = f"""Analyze this employee feedback sentiment. Score from 0.0 (negative) to 1.0 (positive). 0.5 is neutral.
Feedback: "{text}"
Respond ONLY with JSON: {{"score": <float>, "rationale": "<1 sentence>"}}"""
    
    response = client.chat.completions.create(
        model="Qwen/Qwen3-VL-8B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=80,
    )
    
    result_str = response.choices[0].message.content.strip()
    
    # Extract JSON from the response
    start = result_str.find('{')
    end = result_str.rfind('}') + 1
    if start >= 0 and end > start:
        result_json = json.loads(result_str[start:end])
    else:
        raise ValueError(f"LLM did not return valid JSON: {result_str}")
    
    score = float(result_json.get("score", 0.5))
    rationale = result_json.get("rationale", "Analysis complete.")
    
    return {
        "score": min(max(score, 0.0), 1.0),
        "rationale": rationale,
        "sentiment_label": "positive" if score >= 0.6 else ("negative" if score <= 0.4 else "neutral")
    }
