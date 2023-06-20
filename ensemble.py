import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from collections import Counter
import argparse
import json
import collections

# ML 내부에 inferences 라는 폴더 생성해 주시고, 해당 inferences 폴더 내부에 앙상블 하고자 하는 output.csv 파일을 넣으면
# 자동으로 긁어와서 ensemble_output.csv 파일을 생성하게 됩니다.
# *주의* inferences 폴더 내부의 파일은 <파일이름_score.csv> 형식을 지켜주셔야 합니다.
# ex) inferences/output_0.97.csv
# '_' 를 기준으로 filename 과 score 를 인식하기 때문에 언더바 사용에 주의해주세요
# inferences 폴더와 내부 파일 생성 후에는 그냥 python code/ensemble.py 로 작동하시면 됩니다~!
# -> python src/ensemble.py -m sw


class Ensemble():
    def __init__(self):
        self.ENSEMBLE_DIR = os.path.join(os.getcwd(),'ensemble')

    def json_to_pandas(self,path):
        with open(path, 'r') as f:
            data = json.load(f)
        ids = data.keys()
        predictions = data.values()
        return pd.DataFrame({"id":ids,"prediction":predictions})

    def json_load(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        return data
        
    def hard_voting(self):
        print('Mode : hard voting')
        HARD_DIR = os.path.join(self.ENSEMBLE_DIR,'files_hard')
        self.files = os.listdir(HARD_DIR)
        dfs = [self.json_to_pandas(HARD_DIR+'/'+file_name) if file_name.find('json')!=-1 else pd.read_csv('./ensemble/files_hard/'+file_name) for file_name in self.files]
        self.scores = [float(file_name.split(',')[0]) for file_name in self.files]
        self.df_list = [dfs[i] for i in np.array(self.scores).argsort()[::-1]]

        total_predictions = []
        prediction = []
        for i in range(len(self.df_list[0])):
            predictions = []
            for df in self.df_list:
                predictions.append(df.iloc[i]['prediction'])
            prediction.append(Counter(predictions).most_common(1)[0][0])
            total_predictions.append(predictions)

        concated_df = pd.DataFrame()
        concated_df = self.df_list[0]['id'].copy()
        concated_df = pd.concat([concated_df,pd.Series(total_predictions, name='predictions')], axis=1)
        concated_df['prediction'] = prediction
        ensemble_json = collections.OrderedDict()
        for data in concated_df.iterrows():
            ensemble_json.update({data[1]['id']:data[1]['prediction']})
        
        output = os.path.join('./ensemble/outputs/',"[hard]ensemble_output.json")
        with open(output, "w", encoding="utf-8") as writer:
            writer.write(
                json.dumps(ensemble_json, indent=4,
                           ensure_ascii=False) + "\n"
            )

    def soft_voting(self):
        print('Mode : soft voting')
        SOFT_DIR = os.path.join(self.ENSEMBLE_DIR,'files_soft')
        self.files = os.listdir(SOFT_DIR)
        self.js_list = [self.json_load(SOFT_DIR+'/'+file_name) if file_name.find('json')!=-1 else pd.read_csv('./ensemble/files_hard/'+file_name) for file_name in self.files]
        ids = self.js_list[0].keys()

        prediction = []
        for id in ids:
            predictions = {}
            for js in self.js_list: # 파일마다 돌면서
                for prd in js[id]: # n 개의 best prediction 을 뽑아오고 걔네를 모두 dict 에 저장 & 누적 확률
                    if prd['text'] not in predictions:
                        predictions[prd['text']] = 0
                    predictions[prd['text']] += prd['probability']
            sorted_pred = sorted(predictions.items(), key = lambda item: item[1], reverse = True)
            prediction.append(sorted_pred[0][0])

        ensemble_json = collections.OrderedDict()
        for id,pred in zip(ids, prediction):
            ensemble_json.update({id:pred})
        breakpoint()
        output = os.path.join('./ensemble/outputs/',"[soft]ensemble_output.json")
        with open(output, "w", encoding="utf-8") as writer:
            writer.write(
                json.dumps(ensemble_json, indent=4,
                           ensure_ascii=False) + "\n"
            )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default="sw")
    args = parser.parse_args()
    mode = args.mode
    e = Ensemble()
    if mode == 'soft':
        e.soft_voting()
    else:
        e.hard_voting()