#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import asyncio
import aiohttp
import pandas as pd
import re
import json
import ast
import nest_asyncio
nest_asyncio.apply()

API_KEY = "API KEY"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "mistralai/mixtral-8x7b-instruct"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Define Task 2 prompt
FLAUSCH_LABELS = [
    "positive feedback", "compliment", "affection declaration",
    "encouragement", "gratitude", "agreement", "ambiguous",
    "implicit", "group membership", "sympathy"
]

PROMPT = f'''You are a precise label span extractor. A short German comment will be provided.
Your task is to decide whether the comment contains *candy speech* — defined as positive, respectful, or encouraging language.
You must identify spans (words or phrases) that contain any form of *candy speech*, defined by these labels:
{', '.join(FLAUSCH_LABELS)}.

Output only a list of dicts. Each dict must have:
- "label": the flausch category
- "start": start character index (inclusive)
- "end": end character index (exclusive)

Strictly output only a JSON list like this:
[{{"label1": "label2",... "start": 0, "end": 10}}, ...]
'''

def extract_json_array(text):
    """Extracts the first JSON array from a string (LLM output)."""
    try:
        match = re.search(r"\[\s*{.*?}\s*]", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        print(f"JSON parsing error: {e}")
        try:
            return ast.literal_eval(match.group())
        except:
            return []
    return []

async def classify_comment(session, comment, comment_id):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": f"Comment: \"{comment}\""}
        ],
        "temperature": 0.0
    }

    try:
        async with session.post(API_URL, headers=HEADERS, json=payload, timeout=60) as response:
            resp_json = await response.json()
            content = resp_json["choices"][0]["message"]["content"].strip()
            spans = extract_json_array(content)

            results = []
            for span in spans:
                if "label" in span and "start" in span and "end" in span:
                    results.append({
                        "comment_id": comment_id,
                        "label": span["label"],
                        "start": span["start"],
                        "end": span["end"]
                    })
            return results
    except Exception as e:
        print(f"Error for comment {comment_id}: {e}")
        return []

async def batch_classify_comments(comment_ids, comments, batch_size=10):
    all_results = []
    async with aiohttp.ClientSession() as session:
        for i in range(0, len(comments), batch_size):
            batch = comments[i:i+batch_size]
            ids = comment_ids[i:i+batch_size]
            tasks = [classify_comment(session, comment, cid) for comment, cid in zip(batch, ids)]
            batch_results = await asyncio.gather(*tasks)
            for spans in batch_results:
                if spans:
                    all_results.extend(spans)
            print(f"Processed {i + len(batch)} / {len(comments)}")
    return all_results



# In[ ]:


# Load your dataset- or
df = pd.read_csv(r"C:\Users\kaoda\Desktop\PHD\Candy speech detection\osfstorage-archive\Data\test data\comments.csv")

# Use all comments (no filtering)
comment_ids = df["comment_id"].tolist()
comments = df["comment"].tolist()

# Run classifier
results = await batch_classify_comments(comment_ids, comments, batch_size=10)

# Convert to DataFrame
if results:
    span_df = pd.DataFrame(results)
    span_df = span_df.merge(df[["comment_id", "comment"]], on="comment_id", how="left")
    span_df = span_df[["comment_id", "comment", "label", "start", "end"]]
    span_df.sort_values(by="comment_id", inplace=True)

    # Save results
    span_df.to_csv(r"task2_spans.csv", index=False)
else:
    print("No candy speech spans found.")


# In[ ]:


# Load your dataset only yes
df = pd.read_csv(r"predictions.csv")

# Only process comments predicted as "yes" in Task 1
df_yes = df[df["task1_prediction"] == "yes"].copy()
comment_ids = df_yes["comment_id"].tolist()
comments = df_yes["comment"].tolist()

# Run classifier
results = await batch_classify_comments(comment_ids, comments, batch_size=10)

# Convert to DataFrame
if results:
    span_df = pd.DataFrame(results)
    span_df = span_df.merge(df[["comment_id", "comment"]], on="comment_id", how="left")
    span_df = span_df[["comment_id", "comment", "label", "start", "end"]]
    # Optional: sort by comment_id
    span_df.sort_values(by="comment_id", inplace=True)

    # Save results
    span_df.to_csv(r"task2_nemo_spans.csv", index=False)
else:
    print("No candy speech spans found.")

