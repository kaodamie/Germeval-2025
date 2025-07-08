#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#new

import asyncio
import aiohttp
import pandas as pd
import re
import json
import ast
import nest_asyncio
nest_asyncio.apply()

API_KEY = "Api key"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "mistralai/mixtral-8x7b-instruct"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

FLAUSCH_LABELS = [
    "positive feedback", "compliment", "affection declaration",
    "encouragement", "gratitude", "agreement", "ambiguous",
    "implicit", "group membership", "sympathy"
]


PROMPT = f"""You are a precise label span extractor for German text. A short comment will be provided.

Your task is to extract only phrases that express *candy speech*. These phrases must be labeled using exactly one of the following categories:
{', '.join(FLAUSCH_LABELS)}.

Output a list of dicts. Each dict must include:
- "label": one of these exact labels only: {', '.join(FLAUSCH_LABELS)}.
- "phrase": the exact phrase from the comment (copy it as-is).

⚠️ VERY IMPORTANT:
- Do not invent new labels.
- Only use one of the above labels.
- Do not return anything else.

Output format (strictly):
[{{"label": "compliment", "phrase": "voll gut"}}, ...]
"""
# PROMPT = f'''You are a precise label span extractor. A short German comment will be provided.
# Your task is to decide whether the comment contains *candy speech* — defined as {', '.join(FLAUSCH_LABELS)}.
# You must identify spans (words or phrases) that contain any form of *candy speech*, strictly defined by the labels:
# {', '.join(FLAUSCH_LABELS)}.

# Output only a list of dicts. Each dict must have:
# - "label": the flausch category
# - "phrase": the exact word or phrase from the comment

# Strictly output only a JSON list like this:
# [{{"label": "compliment", "phrase": "voll gut"}}, ...]
# '''

def extract_json_array(text):
    try:
        match = re.search(r"\[\s*{.*?}\s*]", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        try:
            return ast.literal_eval(match.group())
        except:
            return []
    return []

def find_phrase_offsets(comment, phrase):
    # Find all matches in case the phrase appears more than once
    matches = list(re.finditer(re.escape(phrase), comment))
    return [(m.start(), m.end()) for m in matches]

async def classify_comment(session, row):
    comment = row["comment"]
    comment_id = row["comment_id"]
    document = row["document"]

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
                if "label" in span and "phrase" in span:
                    offsets = find_phrase_offsets(comment, span["phrase"])
                    for start, end in offsets:
                        results.append({
                            "document": document,
                            "comment_id": comment_id,
                            "label": span["label"],
                            "wordspan": span["phrase"],
                            "start": start,
                            "end": end
                        })
            return results
    except Exception as e:
        print(f"Error for comment {comment_id}: {e}")
        return []


# In[ ]:


async def classify_comment(session, row):
    comment = row["comment"]
    comment_id = row["comment_id"]
    document = row["document"]

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

            if "choices" not in resp_json:
                print(f"Warning: No 'choices' field in response for comment {comment_id}. Full response:")
                print(json.dumps(resp_json, indent=2))
                return []

            content = resp_json["choices"][0]["message"]["content"].strip()
            spans = extract_json_array(content)

            results = []
            for span in spans:
                if "label" in span and "phrase" in span:
                    offsets = find_phrase_offsets(comment, span["phrase"])
                    for start, end in offsets:
                        results.append({
                            "document": document,
                            "comment_id": comment_id,
                            "label": span["label"],
                            "wordspan": span["phrase"],
                            "start": start,
                            "end": end
                        })
            return results
    except Exception as e:
        print(f"Error for comment {comment_id}: {e}")
        return []


# In[ ]:


df = preds[preds["task1_prediction"] == "yes"].copy()

# Run Task 2
results = await batch_classify_comments(df, batch_size=5)

# Save if any results
if results:
    span_df = pd.DataFrame(results)
    span_df = span_df[["document", "comment_id", "label", "wordspan", "start", "end"]]
    span_df.to_csv(
        r"task2_spans.csv",
        index=False
    )
    print("Spans saved to task2_spans.csv")
else:
    print("No spans found.")

