import pandas as pd

df = pd.read_csv("classified_tweets_gpt35_confidence4.csv")

df_hate_speech = df[df["gpt_label"] == "hate_speech"]
df_hate_speech = df_hate_speech.sort_values(by="confidence", ascending=False)

df_normal = df[df["gpt_label"] == 'neutral']
df_normal = df_normal.sort_values(by="confidence", ascending=False)

df_hate_speech.to_csv("hate_speech_only.csv", index=False)
df_normal.to_csv("normal_speech_only.csv", index=False)