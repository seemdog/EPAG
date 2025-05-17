from openai import OpenAI, RateLimitError
import pandas as pd
from tqdm import tqdm
import pickle
import argparse
import os
import ast
from dotenv import load_dotenv
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from multiprocessing import cpu_count
import re
import threading

random.seed(42)

load_dotenv("./key.env")

# Thread-local storage for OpenAI client
thread_local = threading.local()

def get_openai_client():
    if not hasattr(thread_local, "client"):
        thread_local.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=10)
    return thread_local.client
    
def open_txt(file_path):
    with open(file_path, "r") as file:
        content = file.read()
    return content


def safe_openai_call(user_input, max_retries=10):
    for attempt in range(max_retries):
        try:
            client = get_openai_client()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.0
            )
            return response.choices[0].message.content
        except RateLimitError:
            wait_time = 2 ** attempt
            print(f"[OpenAI] Rate limit hit. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
    raise RuntimeError("OpenAI Rate Limit Retry Failed")


def process_dialogue(index, row, organizer_prompt, num_turn):
    
    try:
        disease = row["disease"]
        dialogue = ast.literal_eval(row["QA"])
        main_symptom = dialogue[0]

        dialogue = dialogue[1:num_turn*2+1]

        if num_turn < "".join(dialogue).count("Question:"):
            return "QA format Error."
        elif num_turn > "".join(dialogue).count("Question:"):
            return "Early Stop Error."

        input_dialogue = "Main Symptom: " + main_symptom + "\n"# + "\n".join(dialogue)[:num_turn*2]
        for m in range(num_turn*2):
            input_dialogue += ("\n" + dialogue[m])
        input_dialogue = input_dialogue.strip()
        organizer_prompt_tmp = organizer_prompt.replace("{input}", input_dialogue)
        organizer_prompt_tmp = organizer_prompt_tmp.replace("{disease}", disease)

        response = safe_openai_call(organizer_prompt_tmp)

        if "체중" not in input_dialogue:
            response = response.replace("- 체중 변화는 없다.\n", "")
            response = response.replace("- 체중 변화가 없다.\n", "")
        if "호흡" not in input_dialogue:
            response = response.replace("- 호흡 곤란은 없다.\n", "")
            response = response.replace("- 호흡 곤란이 없다.\n", "")
        if "피로" not in input_dialogue and "피곤" not in input_dialogue:
            response = response.replace("- 피로감이 없다.\n", "")
            response = response.replace("- 피로감은 없다.\n", "")
        for word in ["발열", "구토", "메스꺼움", "기침", "설사", "두통"]:
            if word not in input_dialogue:
                response = response.replace(f"- {word}은 없다.\n", "")
                response = response.replace(f"- {word}는 없다.\n", "")
                response = response.replace(f"- {word}이 없다.\n", "")
                response = response.replace(f"- {word}가 없다.\n", "")
        
        return response

    except Exception as e:
        print(f"Error at index {index}: {e}")
        return "Error"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dialogue_file', type=str, default='gpt-4o-mini_turn_9.csv')
    parser.add_argument('--num_processes', type=int, default=cpu_count(), help='Number of processes')
    parser.add_argument('--language', type=str, default='korean', help='korean or english')
    parser.add_argument('--cut_turn', type=int)
    args = parser.parse_args()
    
    dialogue_file = args.dialogue_file
    num_processes = args.num_processes
    language = args.language
    num_turn = args.cut_turn

    if language == 'korean':
        organizer_prompt = open_txt("./prompt/organizer_prompt.txt")
    else: # english
        organizer_prompt = open_txt("./prompt/organizer_prompt_en.txt")
        
    dialogue_dir = "./dialogue/" + dialogue_file

    data = pd.read_csv(dialogue_dir)
    data = data[["pid", "disease", "QA"]]
    print(f"Dialogue data loaded from {dialogue_dir.split("/")[-1][:-4]}...!")

    dialogue_dir = re.sub(r'turn_\d+', f'turn_{num_turn}', dialogue_dir)#
    dialogue_dir_ = dialogue_dir.split("/")[-1][:-4]#

    new_dir = f"./dialogue/result_pkl"
    os.makedirs(new_dir, exist_ok=True)

    try:
        with open(f'{new_dir}/{dialogue_dir_}_organized.pkl', 'rb') as f:
            organized_dict = pickle.load(f)
        print("Loaded existing data.")
    except:
        organized_dict = {}
        print("No existing data to load.")

    start_idx = len(organized_dict)


    print(f"Organization starting at {time.strftime('%m-%d-%H-%M')}...")
    cut_dialogue_list = []
    with ThreadPoolExecutor(max_workers=num_processes) as executor:
        futures = {
            executor.submit(process_dialogue, i, data.iloc[i], organizer_prompt, num_turn): i
            for i in range(start_idx, len(data))
        }
        

        for future in tqdm(as_completed(futures), total=len(futures)):
            index = futures[future]  # future -> index
            result = future.result()
            organized_dict[index] = result  # index로 저장

            with open(f'{new_dir}/{dialogue_dir_}_organized.pkl', 'wb') as f:
                pickle.dump(organized_dict, f)


        organized_list = [organized_dict.get(i, "Error") for i in range(len(data))]

        missing = [i for i in range(len(data)) if i not in organized_dict]
        if missing:
            print(f"Missing {len(missing)} items: {missing[:5]}...")

    data["QA"] = [ast.literal_eval(dialogue)[:num_turn*2+1] for dialogue in data["QA"]]
    data["organized_QA"] = organized_list
    data.to_csv(dialogue_dir, index=False)
    print(f"Organization finished and file saved at {time.strftime('%m-%d-%H-%M')}...")

    os.remove(f"{new_dir}/{dialogue_dir_}_organized.pkl")
    print("pkl file removed.")
