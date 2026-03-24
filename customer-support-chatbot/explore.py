"""explore.py — Understand the Bitext dataset before building the pipeline."""

from datasets import load_dataset
from collections import Counter

# Load the dataset
ds = load_dataset(
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
    split="train",
)

# Basic stats
print(f"Total rows: {len(ds)}")
print(f"Columns: {ds.column_names}")

# Look at a few rows
print("\n--- Sample rows ---")
for i in range(3):
    print(f"\nRow {i}:")
    for key, value in ds[i].items():
        print(f"  {key}: {value}")

# What intents and categories exist?
intents = Counter(ds["intent"])
categories = Counter(ds["category"])

print(f"\n--- Intents ({len(intents)} unique) ---")
for intent, count in intents.most_common():
    print(f"  {intent}: {count}")

print(f"\n--- Categories ({len(categories)} unique) ---")
for cat, count in categories.most_common():
    print(f"  {cat}: {count}")

# How many unique responses are there?
unique_responses = set(ds["response"])
print(f"\nTotal rows: {len(ds)}")
print(f"Unique responses: {len(unique_responses)}")