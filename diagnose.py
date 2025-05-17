import json
import pandas as pd
from pathlib import Path
from ast import literal_eval
from openai import OpenAI
import yaml
from multiprocessing import Pool, Manager
import argparse
from tqdm import tqdm
import argparse
import random
import os
from dotenv import load_dotenv

random.seed(42)

def preprocess_patient_info_doctor(patient_info) -> str:
    patient_info = patient_info["simple_info"]
    lst = list(patient_info.keys())
    text = ""
    for i in range(len(lst)):
        if type(patient_info[lst[i]])==list:
            lst_text = ", ".join(patient_info[lst[i]])
            text += f"{lst[i]}: {lst_text}\n"
        else:
            text += f"{lst[i]}: {patient_info[lst[i]]}\n"
    return text

def disease_predict(data, pid2profile, turn, model='gpt-4o-mini'):
    pid = f'{data.pid:08d}'
    patient_info = pid2profile[pid]
    patient_info = preprocess_patient_info_doctor(patient_info)
    history = '\n\n'.join([line.replace('\n\n', '\n') for line in literal_eval(data.QA)[:turn*2+1]])
    input_data = f'patient_info: \n{patient_info}\nmedical_history: \n{history}'    
    input_message = {
        "role": 'user',
        "content": input_data,
    }
    for _ in range(5):
        try:
            completion = openai_client.chat.completions.create(
                model = model,
                messages = disease_predict_system + [input_message],
                temperature = 0.0
            )
            result = completion.choices[0].message.content
            if '```yaml' in result:
                result = result.split('```yaml')[1].split('```')[0]
            result = yaml.safe_load(result)['Diseases']
            
        except Exception as e:
            continue
        
        break
    return result


def disease_eval(data, model_predictions, model='gpt-4o-mini'):
    golden_standard = data.disease
    input_data = f'model_predictions: \n{model_predictions}\ngolden_standard: \n{golden_standard}'    
    input_message = {
        "role": 'user',
        "content": input_data,
    }
    for _ in range(5):
        try:
            completion = openai_client.chat.completions.create(
                model = model,
                messages = disease_eval_system + [input_message],
                temperature = 0.0
            )
            result = completion.choices[0].message.content
            if '```yaml' in result:
                result = result.split('```yaml')[1].split('```')[0]
            result = yaml.safe_load(result)
            assert 'Reasoning' in result
            assert 'Result' in result
            
        except Exception as e:
            continue
        
        break
    return result

def disease_predict_eval(data, pid2profile, turn, model='gpt-4o-mini'):
    if data["score"] == -1:
        return {
        "model_predictions": -1,
        "topk_reason": -1,
        "topk_result": False,
        "top1_reason": -1,
        "top1_result": False
    }
        
    model_predictions = disease_predict(data, pid2profile, turn, model)
    result = disease_eval(data, model_predictions, model)
    topk_reason = result['Reasoning']
    topk_result = result['Result']
    result = disease_eval(data, model_predictions[:1], model)
    top1_reason = result['Reasoning']
    top1_result = result['Result']
    return {
        "model_predictions": model_predictions,
        "topk_reason": topk_reason,
        "topk_result": topk_result,
        "top1_reason": top1_reason,
        "top1_result": top1_result
    }


if __name__ == "__main__":    

    parser = argparse.ArgumentParser()
    parser.add_argument('--dialogue_file', type=str, default='', help='File name.')
    args = parser.parse_args()
    dir = args.dialogue_file

    if dir == "":
        paths = [Path("./dialogue/"+f) for f in os.listdir("./dialogue") if f.endswith('.csv')]
    else:
        paths = [Path(f"./dialogue/{dir}")]    

    profiles = json.loads(Path('./data/patient_profile.json').read_text())
    print("Patient Profile loaded...!")
    datas = [pd.read_csv(p) for p in paths]
    print("Data to evaluate loaded...!")
    pid2profile = {}
    for p in profiles:
        pid2profile[p['pid']] = p
        
    num_cores = 50    

    disease_predict_system = [{
    "role": 'system',
    "content": Path('./prompt/disease_prediction_prompt.txt').read_text(),
    }]
    disease_eval_system = [{
    "role": 'system',
    "content": Path('./prompt/disease_evaluation_prompt.txt').read_text(),
    }]

    load_dotenv("./key.env")
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    print("API key loaded!")
    openai_client = OpenAI(
        api_key=OPENAI_API_KEY,
        timeout=10,
    )
    
    model_name = 'gpt-4o-mini'    
    
    for path, data in zip(paths, datas):
        print(path)
        turn = int(str(path)[str(path).find("turn")+len("turn_"):-4])
        with Manager() as manager:
            with Pool(num_cores) as pool:
                    results = pool.starmap(
                        disease_predict_eval,
                        [(data.iloc[i], pid2profile, turn, model_name) for i in tqdm(range(len(data)))]
                    )
                    merge_results = {
                        "model_predictions": [],
                        "topk_reason": [],
                        "topk_result": [],
                        "top1_reason": [],
                        "top1_result": []
                    }
                    for result in results:
                        merge_results['model_predictions'].append(result['model_predictions'])
                        merge_results['topk_reason'].append(result['topk_reason'])
                        merge_results['topk_result'].append(result['topk_result'])
                        merge_results['top1_reason'].append(result['top1_reason'])
                        merge_results['top1_result'].append(result['top1_result'])
                    merge_results = pd.DataFrame(merge_results)
                    new_data = pd.concat([data, merge_results], axis=1)
                    new_data.to_csv(path, index=False)