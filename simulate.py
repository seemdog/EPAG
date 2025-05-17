# batch_dialogue_simulator.py
import os
import json
import time
import argparse
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from openai import OpenAI, RateLimitError
import anthropic


# === Utility Functions ===
def open_txt(file_path):
    with open(file_path, "r") as file:
        return file.read()



    
def safe_openai_call(model, messages, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = openai_client.chat.completions.create(
                model=model,
                messages=messages,
            )
            return response.choices[0].message.content
        except RateLimitError:
            print(f"[OpenAI] Rate limit hit. Waiting {2**attempt} seconds...")
            time.sleep(2**attempt)
    raise RuntimeError("OpenAI Rate Limit Retry Failed")

def safe_claude_call(model, messages, max_tokens=8192, max_retries=5):
    for attempt in range(max_retries):
        try:
            completion = claude_client.messages.create(
                model=model,
                system=messages[0]["content"],
                messages=messages[1:],
                max_tokens=max_tokens,
            )
            return completion.content[0].text
        except anthropic.RateLimitError:
            print(f"[Anthropic] Rate limit hit. Waiting {2**attempt} seconds...")
            time.sleep(2**attempt)
    raise RuntimeError("Anthropic Rate Limit Retry Failed")

def ask(messages, model, hf_model=None, hf_tokenizer=None):
    if len(messages) % 2 == 0:
        if any(word in model for word in ["gpt", "mini", "o1"]):
            return safe_openai_call(model, messages)
        elif any(word in model for word in ["claude", "haiku"]):
            return safe_claude_call(model, messages)
        elif "/" in model and hf_model and hf_tokenizer:
            text = hf_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = hf_tokenizer([text], return_tensors="pt").to(hf_model.device)
            with torch.no_grad():
                outputs = hf_model.generate(**inputs, max_new_tokens=1024)
            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            return hf_tokenizer.decode(generated_ids, skip_special_tokens=True)
        else:
            return "Wrong Model."
    else:
        return safe_openai_call("gpt-4o-mini", messages)

def batch_generate(messages_list, hf_model, hf_tokenizer, max_new_tokens=512):
    texts = [
        hf_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        ) for messages in messages_list
    ]
    inputs = hf_tokenizer(texts, return_tensors="pt", padding=True).to(hf_model.device)
    with torch.no_grad():
        outputs = hf_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            #do_sample=False
        )
    generated = []
    for i, input_ids in enumerate(inputs.input_ids):
        generated_ids = outputs[i][len(input_ids):]
        generated_text = hf_tokenizer.decode(generated_ids, skip_special_tokens=True)
        generated.append(generated_text)
    return generated

def map_messages(mode, history, doctor_prompt_tmp, patient_prompt_tmp):
    prompt = doctor_prompt_tmp if mode == "doctor" else patient_prompt_tmp
    system = {"role": "system", "content": prompt}
    messages = [system]
    for j in range(len(history)):
        role = "user" if (j % 2 == 0 and mode == "doctor") or (j % 2 != 0 and mode == "patient") else "assistant"
        messages.append({"role": role, "content": history[j]})
    return messages

def preprocess_patient_info_doctor(patient_info):
    info = patient_info["simple_info"]
    return "\n".join(
        f"{k}: {', '.join(v) if isinstance(v, list) else v}" for k, v in info.items()
    )

def preprocess_patient_info_patient(patient_info):
    patient_info_copy = patient_info.copy()
    patient_info_copy.pop("token", None)
    patient_info_copy.pop("pid", None)
    return str(patient_info_copy)

def one_turn(history, doctor_prompt_tmp, patient_prompt_tmp, model, hf_model=None, hf_tokenizer=None):
    messages = map_messages("doctor", history, doctor_prompt_tmp, patient_prompt_tmp)
    response = ask(messages, model, hf_model, hf_tokenizer)
    history.append(response)
    messages = map_messages("patient", history, doctor_prompt_tmp, patient_prompt_tmp)
    response = ask(messages, model)
    history.append(response)
    return history

def one_turn_batch(histories, doctor_prompts, patient_prompts, hf_model, hf_tokenizer):
    doctor_messages_batch = [
        map_messages("doctor", history, doc_prompt, pat_prompt)
        for history, doc_prompt, pat_prompt in zip(histories, doctor_prompts, patient_prompts)
    ]
    doctor_responses = batch_generate(doctor_messages_batch, hf_model, hf_tokenizer)
    for i in range(len(histories)):
        histories[i].append(doctor_responses[i])

    patient_messages_batch = [
        map_messages("patient", history, doc_prompt, pat_prompt)
        for history, doc_prompt, pat_prompt in zip(histories, doctor_prompts, patient_prompts)
    ]
    patient_responses = [ask(msg, "gpt-4o-mini") for msg in patient_messages_batch]
    for i in range(len(histories)):
        histories[i].append(patient_responses[i])

    return histories

save_lock = Lock()
def process_patient(i, save_path):
    try:
        info_doc = preprocess_patient_info_doctor(patient_profiles[i])
        info_pat = preprocess_patient_info_patient(patient_profiles[i])
        try:
            history = [patient_profiles[i]["simple_info"]["주증상"]]
        except:
                history = [patient_profiles[i]["simple_info"]["Chief Complaint"]]
        doc_prompt = doctor_prompt.replace("{patient_information}", info_doc)
        pat_prompt = patient_prompt.replace("{patient_information}", info_pat)
        for _ in range(num_turns):
            history = one_turn(history, doc_prompt, pat_prompt, model, hf_model, hf_tokenizer)
        pid = patient_profiles[i]["pid"]
        disease = patient_profiles[i]["disease_name"]
        row = pd.DataFrame([{"pid": pid, "disease": disease, "QA": history}])
        with save_lock:
            row.to_csv(save_path, mode='a', index=False, header=False)
    except Exception as e:
        print(f"Error with patient {i}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt-4o-mini')
    parser.add_argument('--num_turns', type=int, default=9)
    parser.add_argument('--language', type=str, default='korean')
    parser.add_argument('--num_process', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    model = args.model
    num_turns = args.num_turns
    language = args.language
    num_process = args.num_process
    BATCH_SIZE = args.batch_size

    model_name_clean = model.split("/")[-1]
    save_path = f"./dialogue/{model_name_clean}_turn_{num_turns}.csv"

    if language == 'korean':
        with open("./data/patient_profile.json", 'r') as file:
            patient_profiles = json.load(file)
        patient_prompt = open_txt("./prompt/patient_prompt.txt")
    else:
        with open("./data/patient_profile_en.json", 'r') as file:
            patient_profiles = json.load(file)
        patient_prompt = open_txt("./prompt/patient_prompt_en.txt")

    doctor_prompt = open_txt("./prompt/doctor_prompt.txt")
    print(f"Number of patient profiles: {len(patient_profiles)}")
    print("Prompts loaded!")
    
    load_dotenv("./key.env")
    print("Keys loaded!")

    openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), timeout=20)
    hf_model = hf_tokenizer = None


    if any(word in model for word in ["claude", "haiku"]):
        claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), timeout=10)
    elif "/" in model:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from huggingface_hub import login as hf_login
        import torch
        print(f"CUDA Available: {torch.cuda.is_available()}")
        try:
            hf_login(os.getenv("HF_TOKEN"))
            print("Huggingface Login successful.")
        except:
            print("Huggingface Login failed. You might not be able to load model.")
            pass
        hf_tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, padding_side='left')
        hf_tokenizer.pad_token = hf_tokenizer.eos_token
        hf_model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
    print("Model to evaluate: ", model)

        # Load or initialize DataFrame
    if os.path.exists(save_path):
        data = pd.read_csv(save_path)
        print("Loaded existing data.")
    else:
        data = pd.DataFrame(columns=["pid", "disease", "QA"])
        with save_lock:
            data.to_csv(save_path, index=False)
        print("No existing data to load. Generating empty dataframe...!")

    start_idx = len(data)

    if "/" in model:
        for i in tqdm(range(start_idx, len(patient_profiles), BATCH_SIZE), desc="Processing Simulation"):
            batch_profiles = patient_profiles[i:i + BATCH_SIZE]
            histories, doctor_prompts, patient_prompts = [], [], []
            for profile in batch_profiles:
                try:
                    history = [profile["simple_info"]["주증상"]]
                except:
                    history = [profile["simple_info"]["Chief Complaint"]]
                histories.append(history)
                doctor_prompts.append(doctor_prompt.replace("{patient_information}", preprocess_patient_info_doctor(profile)))
                patient_prompts.append(patient_prompt.replace("{patient_information}", preprocess_patient_info_patient(profile)))
            for _ in range(num_turns):
                histories = one_turn_batch(histories, doctor_prompts, patient_prompts, hf_model, hf_tokenizer)
            with save_lock:
                for idx, profile in enumerate(batch_profiles):
                    row = pd.DataFrame([{
                        "pid": profile["pid"],
                        "disease": profile["disease_name"],
                        "QA": histories[idx]
                    }])
                    row.to_csv(save_path, mode='a', index=False, header=False)
    else:
        with ThreadPoolExecutor(max_workers=num_process) as executor:
            futures = [executor.submit(process_patient, i, save_path) for i in range(start_idx, len(patient_profiles))]
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing Simulation"):
                pass
