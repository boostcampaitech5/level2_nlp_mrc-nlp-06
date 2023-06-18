import time
import pickle
import pandas as pd

from contextlib import contextmanager
from datasets import Dataset, concatenate_datasets, load_from_disk
from tqdm import tqdm
from elasticsearch import Elasticsearch

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

class ESRetrieval:
    def __init__(
        self,
        index_name : str):
        
        self.es = Elasticsearch(["http://127.0.0.1:9200"])
        print(self.es.info())
        self.index = index_name
        print(f'INDEX : {self.index}')
        
        with open('./elastic_search/convert_id.pickle', 'rb') as fr:
            self.convertid = pickle.load(fr)

        assert self.es.indices.exists(index=[index_name])

    def search(self, query:str, topk:int):
        body = {
                "query": {
                    "bool": {
                        "must": [{"match": {"document_text": query}}],
                    }
                }
            }
        res = self.es.search(index=self.index, body=body, size=topk)
        ids = []
        contexts = []
        for hit in res['hits']['hits']:
            ids.append(hit['_id'])
            contexts.append(hit['_source']['document_text'])
        return ids, contexts # topk retrieved document id, context. sorted by score

    def retrieve(self, dataset: Dataset, topk: int):

        total = []
        for i, item in enumerate(tqdm(dataset, desc="Elastic Search Retrieval")):
            query = item['question']
            ids, contexts = self.search(query, topk)
            tmp = {
                'question' : query,
                'id' : item['id'],
                'context' : " ".join(contexts)
            }
            if "context" in item.keys() and "answers" in item.keys():
                tmp['original_context'] = item['context']
                tmp['true_doc'] = self.convertid[item['document_id']]
                tmp['answers'] = item['answers']
                tmp['context_doc'] = ids

                try : ans_idx = item['topk'].index(f"{item['gold_doc']}")
                except : ans_idx = -1
                if ans_idx >= 0:
                    tmp['RR'] = 1 / (ans_idx + 1) 
                else:
                    tmp['RR'] = 0

            total.append(tmp)

        df = pd.DataFrame(total)

        return df
    
if __name__=='__main__':

    # Test sparse
    org_dataset = load_from_disk('../../data/train_dataset')
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    print("*" * 40, "query dataset", "*" * 40)
    print(full_ds)

    INDEX = 'wiki-contexts'
    topk = 40

    retriever = ESRetrieval(INDEX)
    with timer("Elasticsearch retrieval"):
        df = retriever.retrieve(full_ds, topk)

    tmp = df.loc[df['RR'] > 0]['RR']
    MRR = tmp.sum() / len(df)
    Acc = len(tmp) / len(df)

    print(f'MRR : {MRR}')
    print(f'Acc : {Acc}')
