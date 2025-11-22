import pandas as pd
import json
import os

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

INST = "Given the following passage and question, decide if the question is answerable based on the passage. Reply only \"answerable\" or \"unanswerable\""

prompt = "Instructions: {INST}.\nPassage: {P}\nQuestion: {Q}\nAnswer: "

few_shot_prompt = """Instructions: {INST}.
Passage: The Eiffel Tower was completed in 1889 and stands 330 meters tall.
Question: When was the Eiffel Tower completed?
Answer: answerable

Passage: The Eiffel Tower was completed in 1889 and stands 330 meters tall.
Question: How many visitors does the Eiffel Tower receive each year?
Answer: unanswerable

Passage: {P}
Question: {Q}
Answer: """

def format_question(context, question, few_shot=False):
    if few_shot:
        return few_shot_prompt.format(INST=INST, P=context, Q=question)
    else:
        return prompt.format(INST=INST, P=context, Q=question)

def load_squad_data(json_path="squad_data.json", pkl_path="squad_df.pkl", sample_size=-1):
    """Load and parse SQuAD data. Load from pickle if it exists, otherwise parse from JSON."""
    
    # Check if pickle file exists
    # if os.path.exists(pkl_path):
    #     df = pd.read_pickle(pkl_path)
    #     if sample_size > 0:
    #         df = df.sample(n=sample_size, random_state=1).reset_index(drop=True)
    #     return df
    
    # # Otherwise, parse from JSON
    with open(json_path, "r") as f:
        data = json.load(f)
    
    output = []

    for idx, row in enumerate(data):
        output.append({
            "idx": idx,
            "title": row["title"],
            "context": row["context"],
            "question": row["question"],
            "answers": row["answers"],
            "is_impossible": row["is_impossible"]
        })

    
    df = pd.DataFrame(data)
    df["formatted_question"] = df.apply(lambda row: format_question(row["context"], row["question"]), axis=1)
    df["formatted_question_fewshot"] = df.apply(lambda row: format_question(row["context"], row["question"], few_shot=True), axis=1)
    
    # Save for future use
    df.to_pickle(pkl_path)

    # Subsample for testing
    if sample_size > 0:
        df = df.sample(n=sample_size, random_state=1).reset_index(drop=True)

    return df

def get_last_layer_data(model_name, few_shot=False):
    """Get the last layer's hidden states files from the states directory."""
    files = os.listdir("states/")

    # Get the last layer's hidden states files
    model_files = {}
    for file in files:
        # If few_shot is True, only consider files with 'fewshot' in the name
        if few_shot and "fewshot" not in file:
            continue
        # If few_shot is False, skip files with 'fewshot' in the name
        if not few_shot and "fewshot" in file:
            continue
        base_name = file.split("_layer_")[0]
        layer_num = int(file.split("_layer_")[1].split("_hidden")[0])
        if base_name not in model_files:
            model_files[base_name] = ("", 0)
        if layer_num > model_files[base_name][1]:
            model_files[base_name] = (file, layer_num)

    # Sort model_files by base_name
    model_files = dict(sorted(model_files.items()))

    safe_name = model_name.replace("/", "_").replace("-", "_")
    all_hidden_states = np.load("states/" + model_files[safe_name][0])
    return all_hidden_states

def load_model_and_tokenizer(model_name, generation=False):
    """Load model and tokenizer from Hugging Face."""
    padding_side = "left" if generation else "right"
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side=padding_side)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer