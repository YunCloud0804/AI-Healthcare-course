import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


data = pd.read_csv("synthetic_data_diabetes.csv")

# load gpt2 tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name, padding_side='left')  # left-padding, why??? I get this from the warning
model = GPT2LMHeadModel.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def classify_blood_sugar(value):
    if value < 140:
        return "Low Risk"
    elif 140 <= value <= 199:
        return "Risk"
    else:
        return "High Risk"

def create_prompt(row):
    return f"""Subject ID: {row['SUBJECT_ID']}Blood Sugar Level: {row['RANDOM_BLOOD_SUGAR']} mg/dL"""

generated_outputs = []
hidden_states_data = []

for idx, row in data.iterrows():
    prompt = create_prompt(row)
    # tonkenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # get the last layer
    hidden_states = outputs.hidden_states[-1]
    pooled_hidden_states = hidden_states.mean(dim=1).squeeze().numpy()
    classification = classify_blood_sugar(row["RANDOM_BLOOD_SUGAR"])
    generated_outputs.append({"prompt": prompt, "output": classification})
    hidden_states_data.append({
        "SUBJECT_ID": row["SUBJECT_ID"],
        "RANDOM_BLOOD_SUGAR": row["RANDOM_BLOOD_SUGAR"],
        "CLASSIFICATION": classification,
        "HIDDEN_STATES": pooled_hidden_states.tolist()
    })
    if (idx + 1) % 100 == 0:
        print(f"Processed rows {idx + 1}/{len(data)}") # show process

# save the prompt outputs csv file
output_file_prompts = "generated_prompts_outputs.csv"
pd.DataFrame(generated_outputs).to_csv(output_file_prompts, index=False, encoding="utf-8")

# save the json file
output_file_hidden_states = "gpt2_features.json"
pd.DataFrame(hidden_states_data).to_json(output_file_hidden_states, orient="records", lines=True)


