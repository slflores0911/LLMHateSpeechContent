import os
import openai
import pandas as pd
from tqdm import tqdm
import time
import json
from dotenv import load_dotenv


load_dotenv()
# ✅ Get OpenAI 
# API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("Set your OpenAI API key as an environment variable: OPENAI_API_KEY")

# ✅ Load tweet dataset (only the tweet column)
df = pd.read_csv("labeled_data.csv")
df = df[["tweet"]].dropna().drop_duplicates().reset_index(drop=True)
df = df.head(1150)
# ✅ Results storage
results = []

# Prompt with label + confidence request
def make_prompt(tweet):
    return f"""
You are a content moderation assistant. Classify the following tweet into one of the categories: "hate_speech", "offensive_language", or "neutral".

Then, estimate your confidence in your classification as a float between 0.0 and 1.0.

Tweet: "{tweet}"

Respond only in JSON in this format:
{{
  "label": "...",
  "confidence": ...
}}
"""

# GPT-3.5 classification function with safe JSON parsing and retry
def classify_tweet(tweet, model="gpt-3.5-turbo", retries=5):
    prompt = make_prompt(tweet)
    backoff = 10  # base wait time
    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful hate speech moderation classifier."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            reply = response["choices"][0]["message"]["content"].strip()

            # Remove possible markdown formatting
            if reply.startswith("```json"):
                reply = reply.strip("```json").strip("```").strip()

            # Safe JSON parse
            data = json.loads(reply)
            label = data.get("label", "").strip().lower()
            confidence = float(data.get("confidence", 0.0))

            # Validate label
            valid_labels = {"hate_speech", "offensive_language", "neutral"}
            if label not in valid_labels:
                label = "error"

            return label, confidence

        except openai.error.RateLimitError as e:
            print(f"Rate limit hit. Sleeping for {backoff} seconds... [{e}]")
            time.sleep(backoff)
            backoff += 10  # exponential-ish backoff

        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e} | Raw reply: {reply[:80]}")
            return "error", 0.0

        except Exception as e:
            print(f"Unexpected error: {e}")
            time.sleep(5)
    return "error", 0.0

#Loop over tweets and classify
for idx, tweet in enumerate(tqdm(df["tweet"], desc="Classifying tweets")):
    label, confidence = classify_tweet(tweet)
    results.append((tweet, label, confidence))

    # Periodic save to avoid data loss
    if idx % 50 == 0 and idx != 0:
        pd.DataFrame(results, columns=["tweet", "gpt_label", "confidence"]).to_csv("partial_classified_output.csv", index=False)

    time.sleep(1)

#Final save
classified_df = pd.DataFrame(results, columns=["tweet", "gpt_label", "confidence"])
classified_df.to_csv("classified_tweets_gpt35_confidence4.csv", index=False)

print("✅ Classification complete. Results saved to 'classified_tweets_gpt35_confidence.csv'")
