import json
import os
import pickle
import time
from contextlib import contextmanager
from typing import List, Optional, Tuple, Union

import wandb
import shutil
import faiss
import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm
import argparse

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    BertModel, BertPreTrainedModel,
    AdamW, get_linear_schedule_with_warmup,
    TrainingArguments,
)
import random

def set_seed(seed: int = 42):
    """
    seed 고정하는 함수 (random, numpy, torch)

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

class BertEncoder(BertPreTrainedModel):

    def __init__(self, config):
        super(BertEncoder, self).__init__(config)
        self.bert = BertModel(config)
        self.init_weights()
      
    def forward(
            self,
            input_ids, 
            attention_mask=None,
            token_type_ids=None
        ): 

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        pooled_output = outputs[1]
        return pooled_output
    

class DenseRetrieval:
    def __init__(
        self,
        model_name: Optional[str],
        dataset, # 전체 데이터 셋
        q_encoder : Optional[BertEncoder],
        p_encoder : Optional[BertEncoder],
        args, # training_args
        data_path: Optional[str] = "./data/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> None:

        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)
        
        # self.ToPostIdx = {문서 내용 : wiki 문서 출처}    
        self.ToPostIdx = {}
        for v in wiki.values():
            self.ToPostIdx[v['text']] = v['document_id']

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        
        self.ids = list(range(len(self.contexts)))

        #임베딩 인코더 만들기
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.p_encoder = p_encoder
        self.q_encoder = q_encoder
        self.args = args
        self.num_neg = 3
        self.dataset = dataset
        
        self.p_embedding = None

    def prepare_in_batch_negative(self,dataset):
        ''' 
        In batch negative가 적용된 train_dataloader를 만들어주는 함수
        '''     
        corpus = np.array(self.contexts)
        p_with_neg = []
        for idx, context in enumerate(dataset['context']):
            p_neg = []
            while True:
                neg_idxs = np.random.randint(len(corpus), size=self.num_neg)

                if not context in corpus[neg_idxs]:
                    p_neg = corpus[neg_idxs]
                    break
                
            for i in range(self.args.per_device_train_batch_size):
                if(idx % 4 == i):
                    tmp_p_neg = list(p_neg.copy())
                    tmp_p_neg.insert(i,context)
                    p_with_neg.extend(tmp_p_neg)
            
                                
        # q_seps = (corpus_len, 512)
        q_seqs = tokenizer(dataset['question'], padding="max_length", truncation=True, return_tensors='pt')
        # p_seps = (corpus_len * num_eng + 1, 512)
        p_seqs = tokenizer(p_with_neg, padding="max_length", truncation=True, return_tensors='pt')

        max_len = p_seqs['input_ids'].size(-1)
        
        # p_seps = (corpus_len, num_eng + 1, 512)
        p_seqs['input_ids'] = p_seqs['input_ids'].view(-1, self.num_neg+1, max_len)
        p_seqs['attention_mask'] = p_seqs['attention_mask'].view(-1, self.num_neg+1, max_len)
        p_seqs['token_type_ids'] = p_seqs['token_type_ids'].view(-1, self.num_neg+1, max_len)

        train_dataset = TensorDataset(
            p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'], 
            q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids']
        )

        return DataLoader(train_dataset, shuffle=True, batch_size=self.args.per_device_train_batch_size)
    
    def train(self, args=None):
        
        self.train_dataloader = self.prepare_in_batch_negative(self.dataset['train'])
        self.valid_dataloader = self.prepare_in_batch_negative(self.dataset['validation'])

        if args is None:
            args = self.args
        batch_size = args.per_device_train_batch_size
        # Optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.p_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in self.q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.q_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        t_total = len(self.train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
        
        global_step = 0
        eval_step = 100
        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()
        
        train_iterator = tqdm(range(int(args.num_train_epochs)), desc="Epoch")
        for _ in train_iterator:
            with tqdm(self.train_dataloader, unit="batch") as tepoch:
                for batch in tepoch:
                    p_encoder.train()
                    q_encoder.train()
            
                    targets = torch.arange(batch_size).long() # positive example은 전부 첫 번째에 위치하므로
                    targets = targets.to(args.device)
                    p_inputs = {
                        'input_ids': batch[0].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                        'attention_mask': batch[1].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                        'token_type_ids': batch[2].view(batch_size * (self.num_neg + 1), -1).to(args.device)
                    }
            
                    q_inputs = {
                        'input_ids': batch[3].to(args.device),
                        'attention_mask': batch[4].to(args.device),
                        'token_type_ids': batch[5].to(args.device)
                    }

                    p_outputs = self.p_encoder(**p_inputs)  # (batch_size*(num_neg+1), emb_dim)
                    q_outputs = self.q_encoder(**q_inputs)  # (batch_size*, emb_dim)
                    p_outputs = p_outputs.view(batch_size, self.num_neg + 1, -1)
                    q_outputs = q_outputs.view(batch_size, 1, -1)
                    
                    del p_inputs, q_inputs
                    
                    # Calculate similarity score & loss
                    #(batch_size, num_neg + 1)
                    sim_scores = torch.bmm(q_outputs, torch.transpose(p_outputs, 1, 2)).squeeze() 
                    sim_scores = sim_scores.view(batch_size, -1)
                    sim_scores = F.log_softmax(sim_scores, dim=1)
                    loss = F.nll_loss(sim_scores, targets)
                    tepoch.set_postfix(loss=f'{str(loss)}')
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    self.p_encoder.zero_grad()
                    self.q_encoder.zero_grad()

                    global_step += 1
                    
                    
                    if global_step % eval_step ==0:
                        p_encoder.eval()
                        q_encoder.eval()
                        valid_loss = 0.
                        valid_steps = 0
                        with torch.no_grad():
                            for valid_batch in tqdm(self.valid_dataloader, desc = 'eval_step'):
                                valid_targets = torch.zeros(batch_size).long()
                                valid_targets = valid_targets.to(args.device)
                                valid_p_inputs = {
                                    'input_ids': valid_batch[0].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                                    'attention_mask': valid_batch[1].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                                    'token_type_ids': valid_batch[2].view(batch_size * (self.num_neg + 1), -1).to(args.device)
                                }
                        
                                valid_q_inputs = {
                                    'input_ids': valid_batch[3].to(args.device),
                                    'attention_mask': valid_batch[4].to(args.device),
                                    'token_type_ids': valid_batch[5].to(args.device)
                                }
         
                                valid_p_outputs = self.p_encoder(**valid_p_inputs)
                                valid_q_outputs = self.q_encoder(**valid_q_inputs)
                                valid_p_outputs = valid_p_outputs.view(batch_size, self.num_neg + 1, -1)
                                valid_q_outputs = valid_q_outputs.view(batch_size, 1, -1)
                                valid_sim_scores = torch.bmm(valid_q_outputs, torch.transpose(valid_p_outputs, 1, 2)).squeeze()
                                valid_sim_scores = valid_sim_scores.view(batch_size, -1)
                                valid_sim_scores = F.log_softmax(valid_sim_scores, dim=1)
                                valid_loss += F.nll_loss(valid_sim_scores, valid_targets)
                                valid_steps += 1
                                del valid_p_inputs, valid_q_inputs                                
                        
                        valid_loss /= valid_steps
                        wandb.log({'valid_loss': valid_loss})                        
                    wandb.log({"train_loss": loss, "global_step": global_step})
                    torch.cuda.empty_cache()

    def get_dense_embedidng(self, pickle_name):
        '''
        knowledge base인 wiki 데이터셋의 passage embedding 결과 불러온는 곳
        self.p_embedding에 들어가게 되는데 파일이 존재하지 않을 경우 12분 정도 소요
        '''
        pickle_name = pickle_name
        emd_path = os.path.join('./src/retriever/', pickle_name)
        
        if os.path.isfile(emd_path):
                with open(emd_path, "rb") as file:
                    self.p_embedding = pickle.load(file)
                print("Embedding pickle load.")
        
        else:
            print("Build passage embedding")
            with torch.no_grad():
                self.p_encoder.eval()

                wiki_p_seqs = self.tokenizer(self.contexts, padding="max_length", truncation=True, return_tensors='pt').to(self.args.device)
                wiki_p_dataset = TensorDataset(
                    wiki_p_seqs['input_ids'], wiki_p_seqs['attention_mask'], wiki_p_seqs['token_type_ids']
                )
                wiki_p_dataloader = DataLoader(wiki_p_dataset, batch_size=1)

                p_embs = []
                for batch in tqdm(wiki_p_dataloader,desc = 'wiki passage encoding'):
                    batch = tuple(t.to(self.args.device) for t in batch)
                    p_inputs = {
                        'input_ids': batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2]
                    }
                    p_emb = self.p_encoder(**p_inputs).to('cpu')
                    p_embs.append(p_emb)

                p_embs = torch.stack(p_embs, dim=0).view(len(self.contexts), -1)  # (num_wiki_passage, emb_dim)
            self.p_embedding = p_embs
            with open(emd_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            print("Embedding pickle saved.")
            
    def retrieve(self, dataset, k = 100, pickle_name = ''):
        total = []
        
        self.get_dense_embedidng(pickle_name)
        
        
        self.q_encoder.eval()
        q_seqs_val = self.tokenizer(dataset['question'], padding="max_length", truncation=True, return_tensors='pt').to(self.args.device)
        q_emb = self.q_encoder(**q_seqs_val).to('cpu')  # (num_query=1, emb_dim)
        dot_prod_scores = torch.matmul(q_emb, torch.transpose(self.p_embedding, 0, 1))
        rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
        doc_indices = rank[:,:k]
        
        for idx, example in enumerate(
                tqdm(dataset, desc="Dense retrieval: ") 
            ):
                tmp = {
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp['true_doc'] = self.ToPostIdx[example["context"]]
                    tmp["answers"] = example["answers"]
                    tmp['context_doc'] = " ".join([f'{self.ToPostIdx[self.contexts[pid]]}' for pid in doc_indices[idx]])
                    int_list = [int(x) for x in tmp['context_doc'].split()]
                    try:
                        ans_idx = int_list.index(int(tmp['true_doc']))
                    except ValueError:
                        ans_idx = -1
                        
                    tmp['MRR'] = 1 / (ans_idx+1) if ans_idx +1 >0 else 0
                total.append(tmp)

        cqas = pd.DataFrame(total)
        return cqas
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="")
    
    parser.add_argument(
        "--dataset_name", metavar="./data/train_dataset", type=str, default="./data/train_dataset", help=""
    )
    
    parser.add_argument(
        "--model_name_or_path",
        metavar="bert-base-multilingual-cased",
        default ="klue/bert-base",
        type=str,
        help="",
    )
    parser.add_argument("--data_path", metavar="./data", type=str, default="./data", help="")
    
    parser.add_argument(
        "--context_path", metavar="wikipedia_documents", type=str, default = "wikipedia_documents.json", help=""
    )
    parser.add_argument("--use_faiss", metavar=False, type=bool, default = False, help="")

    args = parser.parse_args()

    # Test sparse
    org_dataset = load_from_disk(args.dataset_name)
    train_dataset = org_dataset["train"].flatten_indices()
    valid_dataset = org_dataset["validation"].flatten_indices()


    model_args = TrainingArguments(
        output_dir="dense_retireval",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size= 4,
        gradient_accumulation_steps = 1,
        num_train_epochs=5,
        weight_decay=0.01
    )
    
    if model_args.device.type!='cuda':
        print('gpu 사용 불가')
    
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,)
    
    p_encoder = BertEncoder.from_pretrained(args.model_name_or_path).to(model_args.device.type)
    q_encoder = BertEncoder.from_pretrained(args.model_name_or_path).to(model_args.device.type)
    
    retriever = DenseRetrieval(model_name = 'klue/bert-base',
                           dataset = org_dataset,
                           q_encoder = q_encoder,
                           p_encoder = p_encoder,
                           args = model_args)
    
    wandb.init(project='MRC_Retriever', name='yh_dpr')

    retriever.train()
    retriever.get_dense_embedidng('dense_embedding')
    torch.save(retriever.p_encoder.state_dict(),'./src/retriever/passage_encoder/model.pt')
    torch.save(retriever.q_encoder.state_dict(),'./src/retriever/query_encoder/model.pt')

    wandb.finish()
    shutil.rmtree('./wandb')    
