import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from collections import Counter
import argparse
import json
import collections

# ML 내부에 ensemble 이라는 폴더 필요합니다. 해당 폴더는 files_hard/ files_soft/ outputs 의 하위 폴더를 가집니다.
# files_hard : hard voting 을 위한 prediction 파일들이 들어가는 곳입니다. 파일 명에 Score 를 꼭 포함해 주셔야 정상적으로 작동합니다.
# - ex) 70,prediction.json
# - json 파일이 필요하며 파일 명에 _가 포함되는 경우가 많아 이번에는 score 와 file_name 을 구분하기 위해 , 를 사용하였습니다.
# files_soft : soft voting 을 위한 nbest_predictions 파일들이 들어가는 곳 입니다. 파일명 수정하실 필요 없습니다. (점수 포함할 필요 X)
# outputs : ensemble 결과 파일이 저장되는 곳 입니다.



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
        print(self.files,'\n','*'*20)
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
        print(self.files,'\n','*'*20)
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