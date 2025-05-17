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
import shutil

random.seed(42)

def open_txt(file_path):
    with open(file_path, "r") as file:
        content = file.read()
    return content

def safe_openai_call(user_input, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages = [{"role": "system", "content": "You are a helpful assistant."},
               {"role": "user", "content":user_input}] ,
                temperature=0.0
            )
            return response.choices[0].message.content
        except RateLimitError:
            print(f"[OpenAI] Rate limit hit. Waiting {2**attempt} seconds...")
            time.sleep(2**attempt)
    raise RuntimeError("OpenAI Rate Limit Retry Failed")

def process_row(i, data, gold, comparer_prompt, score_list, weighted_score_list, penalty_score_list, reward_score_list, gold_label_list, reason_list, dialogue_dir_):
    disease = data["disease"][i]
    organized_qa = data["organized_QA"][i]
    
    score = 0
    weighted_score = 0
    penalty_score = 0
    reward_score = 0
    referred_gold_label = ""
    reasons = ""
    
    if "Error" not in organized_qa:
        organized_qa_list = organized_qa.split("\n")
        gold_ = gold[gold["disease"] == disease].reset_index(drop=True)

        gold_list = ""
        for n in range(len(gold_)):
            gold_list += ("-" + gold_["symptom"][n] + "\n")
        gold_list = gold_list.strip()

        for m in range(len(organized_qa_list)):
            comparer_prompt_tmp = comparer_prompt.replace("{disease}", disease)
            comparer_prompt_tmp = comparer_prompt_tmp.replace("{a}", gold_list)
            sentence = organized_qa_list[m].strip()
            comparer_prompt_tmp = comparer_prompt_tmp.replace("{b}", sentence)

            response = safe_openai_call(comparer_prompt_tmp)

            response = response.replace("Reason:", "").strip()
            final_response_index = response.find("Final Response:")
            reason = response[:final_response_index].strip().replace("\n", "").replace("-", "")
            final_response = response[final_response_index + len("Final Response:"):].strip().replace("*", "")

            if final_response in gold_list:
                score += 1
                try:
                    weight = gold_[gold_['symptom'] == final_response]['weight'].iloc[0]
                    if weight == "h":
                        weighted_score += 2
                    else:
                        weighted_score += 1
                except:
                    weighted_score = float("-inf")
                    score = float("-inf")
                    break

                if final_response in referred_gold_label:
                    penalty_score += 0.5
                    reward_score += 1.5
                else:
                    penalty_score += 1
                    reward_score += 1
            else:
                final_response = "Not in Gold Label"


            reasons += ("- " + reason + "\n")
            referred_gold_label += ("- " + final_response + "\n")

        if weighted_score == float("-inf"):
            weighted_score = -1
            score = -1

    else: # QA format Error.인 경우
        score = -1
        weighted_score = -1
        penalty_score = -1
        reward_score = -1
        referred_gold_label = "Error."
        reasons = "Error."

    # Save results in shared lists
    score_list[i] = score
    weighted_score_list[i] = weighted_score
    penalty_score_list[i] = penalty_score
    reward_score_list[i] = reward_score
    gold_label_list[i] = referred_gold_label
    reason_list[i] = reasons

    return (score_list, weighted_score_list, penalty_score_list, reward_score_list, gold_label_list, reason_list)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dialogue_file', type=str, default='04-11-14-06_gpt-4o-mini_turn_2.csv')
    parser.add_argument('--num_processes', type=int, default=cpu_count(), help='Number of processes')
    parser.add_argument('--language', type=str, default='korean', help='korean or english')
    args = parser.parse_args()
    
    dialogue_file = args.dialogue_file
    num_processes = args.num_processes
    language = args.language

    dialogue_dir = "./dialogue/" + dialogue_file
    dialogue_dir_ = dialogue_dir.split("/")[-1][:-4]

    if language == 'korean':
        organizer_prompt = open_txt("./prompt/comparer_prompt.txt")
        gold = pd.read_csv("./data/gold_label.csv")
    else: # english
        organizer_prompt = open_txt("./prompt/comparer_prompt_en.txt")
        gold = pd.read_csv("./data/gold_label_en.csv")
    data = pd.read_csv(dialogue_dir)
    data = data[["pid", "disease", "QA", "organized_QA"]]
    print(f"Dialogue data loaded from {dialogue_dir_}...!")

    new_dir = f"./dialogue/result_pkl"
    os.makedirs(new_dir, exist_ok=True)

    try:
        with open(f'{new_dir}/{dialogue_dir_}_score.pkl', 'rb') as f:
            score_list = pickle.load(f)
        with open(f'{new_dir}/{dialogue_dir_}_weighted_score.pkl', 'rb') as f:
            weighted_score_list = pickle.load(f)
        with open(f'{new_dir}/{dialogue_dir_}_penalty_score.pkl', 'rb') as f:
            penalty_score_list = pickle.load(f)
        with open(f'{new_dir}/{dialogue_dir_}_reward_score.pkl', 'rb') as f:
            reward_score_list = pickle.load(f)
        with open(f'{new_dir}/{dialogue_dir_}_referred_gold.pkl', 'rb') as f:
            gold_label_list = pickle.load(f)
        with open(f'{new_dir}/{dialogue_dir_}_reason.pkl', 'rb') as f:
            reason_list = pickle.load(f)
    except:
        score_list = [None] * len(data)
        weighted_score_list = [None] * len(data)
        penalty_score_list = [None] * len(data) # 이미 존재하는 gold label이 또 매칭된 경우 0.5점 주기
        reward_score_list = [None] * len(data) # 이미 존재하는 gold label이 또 매칭된 경우 1점이 아니라 1.5점 주기
        gold_label_list = [None] * len(data)
        reason_list = [None] * len(data)

    load_dotenv("./key.env")
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    print("API key loaded!")
    openai_client = OpenAI(
        api_key=OPENAI_API_KEY,
        timeout=10,
    )

    # Only process rows that are not done yet
    pending_indices = [i for i in range(len(data)) if score_list[i] is None]

    with ThreadPoolExecutor(max_workers=num_processes) as executor:
        futures = [
            executor.submit(
                process_row, i, data, gold, organizer_prompt,
                score_list, weighted_score_list, penalty_score_list, reward_score_list, gold_label_list, reason_list, dialogue_dir_
            )
            for i in pending_indices
        ]
        for future in tqdm(as_completed(futures), total=len(pending_indices)):
            score_list, weighted_score_list, penalty_score_list, reward_score_list, gold_label_list, reason_list = future.result()

    # Save results
    with open(f'{new_dir}/{dialogue_dir_}_score.pkl', 'wb') as f:
        pickle.dump(score_list, f)
    with open(f'{new_dir}/{dialogue_dir_}_weighted_score.pkl', 'wb') as f:
        pickle.dump(weighted_score_list, f)
    with open(f'{new_dir}/{dialogue_dir_}_penalty_score.pkl', 'wb') as f:
        pickle.dump(penalty_score_list, f)
    with open(f'{new_dir}/{dialogue_dir_}_reward_score.pkl', 'wb') as f:
        pickle.dump(reward_score_list, f)
    with open(f'{new_dir}/{dialogue_dir_}_referred_gold.pkl', 'wb') as f:
        pickle.dump(gold_label_list, f)
    with open(f'{new_dir}/{dialogue_dir_}_reason.pkl', 'wb') as f:
        pickle.dump(reason_list, f)

    # Final DataFrame 저장
    data["referred_gold_label"] = gold_label_list
    data["reasons"] = reason_list
    data["score"] = score_list
    data["weighted_score"] = weighted_score_list
    data["penalty_score"] = penalty_score_list
    data["reward_score"] = reward_score_list
    data.to_csv(dialogue_dir, index=False)

    # 임시 pkl 파일 제거
    os.remove(f'{new_dir}/{dialogue_dir_}_score.pkl')
    os.remove(f'{new_dir}/{dialogue_dir_}_weighted_score.pkl')
    os.remove(f'{new_dir}/{dialogue_dir_}_penalty_score.pkl')
    os.remove(f'{new_dir}/{dialogue_dir_}_reward_score.pkl')
    os.remove(f'{new_dir}/{dialogue_dir_}_referred_gold.pkl')
    os.remove(f'{new_dir}/{dialogue_dir_}_reason.pkl')

    if os.path.isdir("./dialogue/result_pkl") and not os.listdir("./dialogue/result_pkl"): shutil.rmtree("./dialogue/result_pkl")