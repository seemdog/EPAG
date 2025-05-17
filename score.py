import pandas as pd
import json
import argparse
import os

def stat(data):

    filtered_data = data[data['score'] != -1]

    mean = filtered_data['score'].mean()  # 평균
    median = filtered_data['score'].median()  # 중앙값
    std_dev = filtered_data['score'].std()  # 표준편차
    describe = filtered_data['score'].describe()  # describe() 함수로 기본 통계

    result = {}
    result["mean"] = filtered_data['score'].mean()  # 평균
    result["median"] = filtered_data['score'].median()  # 중앙값
    result["std_dev"] = filtered_data['score'].std()  # 표준편차
    result["describe"] = describe = filtered_data['score'].describe().to_dict()  # describe() 함수로 기본 통계

    dic = {}
    dic["score"] = result

    w_result = {}
    w_result["mean"] = filtered_data['weighted_score'].mean()  # 평균
    w_result["median"] = filtered_data['weighted_score'].median()  # 중앙값
    w_result["std_dev"] = filtered_data['weighted_score'].std()  # 표준편차
    w_result["describe"] = describe = filtered_data['weighted_score'].describe().to_dict()  # describe() 함수로 기본 통계

    dic["weighted_score"] = w_result

    p_result = {}
    p_result["mean"] = filtered_data['penalty_score'].mean()  # 평균
    p_result["median"] = filtered_data['penalty_score'].median()  # 중앙값
    p_result["std_dev"] = filtered_data['penalty_score'].std()  # 표준편차
    p_result["describe"] = describe = filtered_data['penalty_score'].describe().to_dict()  # describe() 함수로 기본 통계

    dic["penalty_score"] = p_result

    r_result = {}
    r_result["mean"] = filtered_data['reward_score'].mean()  # 평균
    r_result["median"] = filtered_data['reward_score'].median()  # 중앙값
    r_result["std_dev"] = filtered_data['reward_score'].std()  # 표준편차
    r_result["describe"] = describe = filtered_data['reward_score'].describe().to_dict()  # describe() 함수로 기본 통계

    dic["reward_score"] = r_result

    top_1 = len(filtered_data[filtered_data["top1_result"]==True])/len(filtered_data)
    top_k = len(filtered_data[filtered_data["topk_result"]==True])/len(filtered_data)

    dic["top1_accuracy"] = top_1
    dic["topk_accuracy"] = top_k

    dic["clean_len"] = len(data[data['score'] != -1])
    dic["early_stop_len"] = len(data[data['organized_QA'] == "Early Stop Error." ])
    dic["format_error_len"] = len(data[data['organized_QA'] == "QA format Error." ])
    dic["matching_error_len"] = len(data[(~data['organized_QA'].str.contains('Error', na=False)) & (data['score'] == -1)])

    return dic




parser = argparse.ArgumentParser(description="All path should be relative.")

parser.add_argument('--dialogue_file', type=str, help='Doctor-Patient dialogue file path(csv)', default = "04-11-14-06_gpt-4o-mini_turn_2.csv")

args = parser.parse_args()
dialogue_dir = args.dialogue_file
dialogue_dir_ = dialogue_dir[:-4]

new_dir = f"./dialogue/result_score"
os.makedirs(new_dir, exist_ok=True)

data = pd.read_csv(f"./dialogue/{dialogue_dir}")
stat = stat(data)

with open(f'{new_dir}/{dialogue_dir_}.json', 'w') as json_file:
    json.dump(stat, json_file, indent=4)


