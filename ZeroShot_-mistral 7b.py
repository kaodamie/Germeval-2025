#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install aiohttp pandas')


# In[ ]:


import pandas as pd

# Read the two CSV files
task1_df = pd.read_csv(r"task1.csv", sep=",")
comment_df = pd.read_csv(r"comment.csv", sep=",")

# Merge the dataframes on 'comment_id' and 'document'
merged_df = pd.merge(task1_df, comment_df, on=["document", "comment_id"], how="inner")

# Display the merged DataFrame
print(merged_df)

# Optionally, save to a new file
#merged_df.to_csv("merged_task1_comments.csv", sep=",", index=False)


# In[4]:


import asyncio
import aiohttp
import pandas as pd
import re
import nest_asyncio
nest_asyncio.apply()

API_KEY = ("API KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "mistralai/mixtral-8x7b-instruct"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

PROMPT = '''You are a strict binary classifier for short German comments. Each comment is provided in isolation and may be a word or a phrase.

Your task is to decide whether the comment contains *candy speech* — defined as positive, respectful, or encouraging language.

Respond with only: **yes** or **no**. If uncertain, choose 'no'.'''

async def classify_comment(session, comment):
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
            content = resp_json["choices"][0]["message"]["content"].strip().lower()

            if re.search(r"\b(yes|positive|respectful)\b", content):
                return "yes"
            elif re.search(r"\bno\b", content):
                return "no"
            else:
                return content  # catch all fallback
    except Exception as e:
        print(f"Error: {e}")
        return "error"

async def batch_classify_comments(comments, batch_size=10):
    results = []
    async with aiohttp.ClientSession() as session:
        for i in range(0, len(comments), batch_size):
            batch = comments[i:i+batch_size]
            batch_tasks = [classify_comment(session, comment) for comment in batch]
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)
            print(f"Processed {i + len(batch)} / {len(comments)}")
    return results


# In[5]:


# Load your dataset
# df = pd.read_csv(r"/workspace/comments.csv")
df = pd.read_csv(r"comments.csv")

# Apply async classifier
comments = df["comment"].tolist()
predictions = await batch_classify_comments(comments, batch_size=10)

# Attach predictions
df["task1_prediction"] = predictions

# Save predictions
df.to_csv("task1_predictions.csv", index=False)



#predictions = await batch_classify_comments(comments, batch_size=10)


# In[6]:


# Normalize predictions to lowercase and strip whitespace
df["task1_prediction"] = df["task1_prediction"].str.strip().str.lower()

# Replace anything that is not exactly 'yes' with 'no'
df["task1_prediction"] = df["task1_prediction"].apply(lambda x: "yes" if x == "yes" else "no")

# Save cleaned predictions
df.to_csv("zero_shot_predictions_cleaned.csv", index=False)
print("✅ Cleaned predictions saved to zero_shot_predictions_cleaned.csv.")


# In[8]:


df_labels["flausch"]


# In[10]:


from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
df_labels = pd.read_csv(r"/workspace/task1.csv")

# Run predictions on the full dataset
# df_comments["task1_prediction"] = df_comments["comment"].apply(zero_shot_task1)

# Clean predictions and labels
preds = df["task1_prediction"]
true_labels = df_labels["flausch"].str.lower().map({"yes": "yes", "no": "no"})

# Filter to valid values only
valid_mask = preds.isin(["yes", "no"]) & true_labels.isin(["yes", "no"])
preds_clean = preds[valid_mask]
labels_clean = true_labels[valid_mask]

# Evaluation
f1 = f1_score(labels_clean, preds_clean, pos_label="yes")
acc = accuracy_score(labels_clean, preds_clean)
precision = precision_score(labels_clean, preds_clean, pos_label="yes")
recall = recall_score(labels_clean, preds_clean, pos_label="yes")

print("\nEvaluation Results:")
print(f"F1 Score:    {f1:.4f}")
print(f"Accuracy:    {acc:.4f}")
print(f"Precision:   {precision:.4f}")
print(f"Recall:      {recall:.4f}")

