import pandas as pd
import json
import torch
from tqdm.notebook import tqdm
from transformers import Trainer
from torch.utils.data import DataLoader

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
# src에서 실행
from datasets import load_from_disk
dataset = load_from_disk("../data/train_dataset/")
print(dataset)


df = pd.DataFrame(dataset['train'])

with open('./train_nbest_predictions.json','r', encoding='utf-8-sig') as f:
    answer_set = json.load(f)

df_train_answer_set = pd.DataFrame(answer_set)

answer_list = []
for column in df_train_answer_set.columns:
    answer = []
    for text_dict in df_train_answer_set[column]:
        answer.append(text_dict['text'])
    answer_list.append(answer)

df['answer_list'] = answer_list
df['answer'] = df['answers'].apply(lambda x : x['text'][0])

ans_rank = []
for i in range(len(df)):
    try:
        ans = df.iloc[i]['answer_list'].index(df.iloc[i]['answer']) + 1
    except ValueError:
        ans = -1
    ans_rank.append(ans)

df['answer_rank'] = ans_rank
df['answer_rank'].value_counts().sort_index()

df_val = pd.read_csv('val.csv')
df_val['answer_rank'].value_counts().sort_index()

class Reader_Dataset(torch.utils.data.Dataset):
    """ Dataset 구성을 위한 class."""

    def __init__(self, pair_dataset, labels):
        self.pair_dataset = pair_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach()
                for key, val in self.pair_dataset.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
def tokenizing_data(df, tokenizer, max_len):
    """ tokenizer에 따라 sentence를 tokenizing 합니다."""
    query_passage_sents = []
    answers_candidate = []
    answers = []
    for idx, row in tqdm(df.iterrows(),desc = '토크나이징'):
        
        # passage
        passages = row['context'].replace("\\n", "")
        passages = passages.split('. ')
        # top 5개의 정답만 학습용으로 추출
        for i in range(5):
            sent= ''
            ans = row['answer_list'][i]
            try:
                passage_idx = -1
                for p in passages:
                    if ans in p:
                        passage_idx = passages.index(p)
            except ValueError:
                passage_idx = -1

            if passage_idx == -1:
                continue

            # passage + query 데이터셋 생성
            if passage_idx !=0:
                if passage_idx+1 < len(passages):
                    sent += passages[passage_idx-1]+ '. ' + passages[passage_idx]+ '. ' + passages[passage_idx+1]
                else:
                    sent += passages[passage_idx-1]+ '. ' + passages[passage_idx]
            else:
                if passage_idx+1 < len(passages):
                    sent +=  passages[passage_idx]+ '. ' + passages[passage_idx+1]
                else:
                    sent += passages[passage_idx]

            sent += '. *'+row['question']
            answers_candidate.append(ans)
            
            # 정답이 맞는 데이터셋이라면
            if ans == row['answer']:
                answers.append(1)
            else:
                answers.append(0)
            
            query_passage_sents.append(sent)

    tokenized_sentences = tokenizer(
        query_passage_sents,
        answers_candidate,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    )

    return tokenized_sentences , answers

tokenizer = AutoTokenizer.from_pretrained(
    'klue/roberta-large',
    use_fast = True
)

tokenized_train, train_label = tokenizing_data(df,tokenizer,512)
tokenized_valid, valid_label = tokenizing_data(df_val,tokenizer,512)
train_label = torch.tensor(train_label).float()
valid_label = torch.tensor(valid_label).float()


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device

train_dataset = Reader_Dataset(tokenized_train, train_label)
dev_dataset = Reader_Dataset(tokenized_valid, valid_label)

# 데이터로더 생성
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(dev_dataset, batch_size=4, shuffle=False)

MODEL_NAME = 'klue/roberta-large'
model_config = AutoConfig.from_pretrained(MODEL_NAME)
model_config.num_labels = 1


model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, config=model_config)

model.to(device)

training_args = TrainingArguments(
    output_dir = './here',
    per_device_train_batch_size = 4,
    save_total_limit= 3,
    num_train_epochs = 3,
    weight_decay = 0.01,
    logging_steps = 10,
    eval_steps = 10,
    do_eval=True
    )

from sklearn.metrics import mean_squared_error

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    rmse = mean_squared_error(labels, predictions, squared=False)
    return {"rmse": rmse}



import wandb
wandb.init(project='MRC_meta_reader')
trainer = Trainer(
        # the instantiated 🤗 Transformers model to be trained
        model=model,
        args=training_args,                 # training arguments, defined above
        train_dataset=train_dataset,     # training dataset
        eval_dataset=dev_dataset,        # evaluation dataset
        compute_metrics=compute_metrics,     
    )

trainer.train()
wandb.finish()