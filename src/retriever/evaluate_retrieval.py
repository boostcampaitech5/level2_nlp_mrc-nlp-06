# %%
import pandas as pd
import re

from datasets import Dataset, concatenate_datasets, load_from_disk
from bm25 import BM25Retrieval
from konlpy.tag import Mecab

# %%
org_dataset = load_from_disk('../data/train_dataset')
full_ds = concatenate_datasets(
    [
        org_dataset["train"].flatten_indices(),
        org_dataset["validation"].flatten_indices(),
    ]
)  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
print("*" * 40, "query dataset", "*" * 40)
print(full_ds)

# preprocessing
do_process = False
def preprocess(context):
    # \n 처리
    context = re.sub(r'\\n\\n', ' ', context)
    context = re.sub(r'\n\n', ' ', context)
    context = re.sub(r'\\n', ' ', context)
    context = re.sub(r'\n', ' ', context)
    # () 괄호 안 내용 지우기
    # parenthesis = r'\([^)]*\)'
    # context = re.sub(parenthesis, '', context)
    return {'context':context}

if do_process:
    full_ds = full_ds.map(preprocess, input_columns='context')

# Set tokenizer
# tokenizer = lambda x: x.split(' ')
tokenizer = Mecab().morphs
tokenizer_name = 'Mecab'
topk=40

# %%
retriever = BM25Retrieval(tokenizer)

# %%
result_df = retriever.retrieve(full_ds, topk)

# %%
len(result_df)

# %%
result_df

# %% [markdown]
# ## MRR, Accuracy 계산

# %%
RR = []

for i, item in result_df.iterrows():
    int_list = [int(x) for x in item['doc_idx'].split()] # top-k document index list
    try:
        ans_idx = int_list.index(int(item['true_doc'])) # index of gold document in int_list
    except ValueError:
        ans_idx = -1 # no gold document retrieved in top-k
    RR.append(1 / (ans_idx + 1) if ans_idx + 1 > 0 else 0)

result_df['RR'] = RR

# %%
print(result_df.head())

# %%
cal = result_df.loc[result_df['RR'] > 0]['RR']
MRR = (1 / len(result_df)) * cal.sum()
Acc = len(cal) / len(result_df)

print(f"MRR : {MRR}")
print(f"Accuracy : {Acc}")

# %% [markdown]
# ## Retrieval 결과 저장

# %%
result_df.to_csv(f"./{tokenizer_name}_top{topk}_MRR{MRR.round(5)}_Acc{Acc.round(5)}.csv", index=False)
