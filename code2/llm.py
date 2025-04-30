import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import openai
from datasets import load_dataset

models_to_test = [
    "deepseek-v2",
    "gpt-4o-mini",
    "qwen2.5-coder-32b-instruct"
]

dataset = load_dataset("code_x_glue_ct_python", split="test")

results = {model: [] for model in models_to_test}

def evaluate_model(model_name, tokenizer, model, examples):
    for example in examples:
        input_text = example["input"]
        target = example["target"]
        
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=128)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        results[model_name].append(1 if prediction.strip() == target.strip() else 0)

for model_name in models_to_test:
    if model_name == "gpt-4o-mini":
        openai.api_key = "your_api_key"
        def generate_gpt(prompt):
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return response.choices[0].message.content
        
        evaluate_model(model_name, None, None, dataset, generate_gpt)
        
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model.eval()
        
        evaluate_model(model_name, tokenizer, model, dataset)

import pandas as pd
results_df = pd.DataFrame({
    model: [sum(scores)/len(scores) for scores in results.values()]
}, index=["Accuracy"])
results_df.to_csv("model_comparison_results.csv")