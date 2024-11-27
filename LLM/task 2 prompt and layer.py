import pandas as pd
import random
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import json
# define mapping
value_mapping = {
    1: ("Normal Sinus", "Low Risk", "Normal heart rhythm; no significant abnormalities."),
    2: ("AV Paced", "Mild Symptoms", "Indicates AV node pacing; monitor for complications."),
    3: ("Vent. Tachy", "Severe Symptoms", "High risk of cardiac arrest if untreated."),
    4: ("Atrial Fib", "Severe Symptoms", "Risk of blood clots, stroke, and heart failure."),
    5: ("Sinus Tachy", "Mild Symptoms", "Fast heart rate; could indicate physical stress or early disease."),
    6: ("V Paced", "Mild Symptoms", "Ventricular pacing; monitor for electrical abnormalities."),
    7: ("Sinus Brady", "Mild Symptoms", "Slow heart rate; may indicate medication effects or underlying issues."),
    8: ("Supravent Tachy", "Severe Symptoms", "High heart rate originating above the ventricles, requiring intervention."),
    9: ("Ventricular Fib", "Critical Condition", "Life-threatening condition; immediate medical attention needed."),
    10: ("A Paced", "Mild Symptoms", "Atrial pacing; monitor for atrial abnormalities."),
    11: ("1st Deg AV Block", "Low Risk", "Delay in electrical conduction; usually asymptomatic."),
    12: ("2nd AVB/Mobitz I", "Mild Symptoms", "Intermittent conduction failure; mild but requires monitoring."),
    13: ("Atrial Flutter", "Severe Symptoms", "Rapid atrial contractions; risk of stroke or heart failure."),
    14: ("Sinus Arrhythmia", "Low Risk", "Normal variant; often benign."),
    15: ("Other/Remarks", "Low Risk", "General remarks; no significant findings indicated."),
    16: ("Multifocal Atr Tachy", "Severe Symptoms", "Rapid irregular rhythm from multiple foci; needs medical evaluation."),
    17: ("Junctional", "Severe Symptoms", "Originating near AV node; requires investigation for underlying causes."),
    18: ("Asystole", "Critical Condition", "Complete lack of cardiac electrical activity; emergency."),
    19: ("Idioventricular", "Severe Symptoms", "Slow rhythm originating from the ventricles; requires immediate attention."),
    20: ("Comp Heart Block", "Critical Condition", "Complete block of electrical conduction; emergency pacemaker required."),
    21: ("Wandering Atrial Pace", "Mild Symptoms", "Pacemaker activity shifts; mild but worth monitoring."),
    22: ("Paroxysmal Atr Tachy", "Severe Symptoms", "Sudden bursts of rapid rhythm; symptomatic management required."),
    23: ("Paced", "Low Risk", "Stable pacing; no immediate concerns."),
    24: ("2nd AVB Mobitz II", "Critical Condition", "Advanced conduction block; high risk of sudden cardiac arrest."),
    25: ("Zoll Paced", "Mild Symptoms", "External pacing; monitor for stabilization.")
}


synthetic_data_path = 'D:\\homework\\AI Healthcare\\LLM\\synthetic_data.csv'
synthetic_data = pd.read_csv(synthetic_data_path)

# generate cot
def generate_reasoning(value, symptoms, mapping):
    test_type, risk_level, explanation = mapping.get(
        value,
        ("Unknown", "Unknown Risk", "No specific reasoning available.")
    )
    return (
        f"- The test type '{test_type}' indicates: {explanation}\n"
        f"- Symptoms reported: {symptoms}\n"
        f"- Classification: [{risk_level}]."
    )

cot_prompts = []
for _, row in synthetic_data.iterrows():
    subject_id = row["SUBJECT_ID"]
    heart_rate = row["VALUE"]
    symptoms = row["TEXT"]
    risk_class = row["RISK_CLASS"]
    reasoning = generate_reasoning(heart_rate, symptoms, value_mapping)
    prompt = (
        f"Patient ID: {subject_id}\n"
        f"Heart Rate Value: {heart_rate}\n"
        f"Symptoms: {symptoms}\n"
        "Reasoning:\n"
        f"{reasoning}\n"
    )
    cot_prompts.append(prompt)

num_examples = 5
icl_examples = random.sample(cot_prompts, num_examples)

# combine CoT and ICL prompts
icl_prompt = "Examples:\n" + "\n\n".join(icl_examples)
new_patient = random.choice(synthetic_data.to_dict(orient="records"))
cot_prompt = (
    "Classify the following patient:\n"
    f"Patient ID: {new_patient['SUBJECT_ID']}\n"
    f"Heart Rate Value: {new_patient['VALUE']}\n"
    f"Symptoms: {new_patient['TEXT']}\n"
    "Reasoning:\n"
)
final_prompt = icl_prompt + "\n\n" + cot_prompt
cot_icl_prompt_file = 'D:\\homework\\AI Healthcare\\LLM\\cot_icl_combined_prompt.txt'
with open(cot_icl_prompt_file, 'w') as file:
    file.write(final_prompt)


# load gpt2
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name, output_hidden_states=True)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# load cot-icl
cot_icl_prompt_file = 'D:\\homework\\AI Healthcare\\LLM\\cot_icl_combined_prompt.txt'
with open(cot_icl_prompt_file, 'r') as file:
    cot_icl_prompt = file.read()
# tokenize
inputs = tokenizer(
    cot_icl_prompt,
    return_tensors="pt",
    truncation=True,
    max_length=1024,
    padding=True
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
inputs = {key: val.to(device) for key, val in inputs.items()}
with torch.no_grad():
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=150,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_hidden_states=True
    )
    predictions = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    last_hidden_state = outputs.hidden_states[-1][0].tolist()  # extract last layer
predictions_file = 'D:\\homework\\AI Healthcare\\LLM\\cot_icl_predictions.txt'
with open(predictions_file, 'w') as file:
    file.write(predictions)

# save json
hidden_states_file = 'D:\\homework\\AI Healthcare\\LLM\\gpt2_hidden_states_features.json'
with open(hidden_states_file, 'w') as file:
    json.dump(last_hidden_state, file)
