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
                        "must": [{"match_phrase": {"document_text": query}}],
                    }
                }
            }
        res = self.es.search(index=self.index, body=body, size=topk)
        return self.query_result(res)

    def search_ner(self, query:str, topk:int, nouns:str):
        # Query의 명사를 활용하여 복합 쿼리로 검색이 이루어집니다. 
        del_list = ['어떤', '무엇', '누구', '어디', '곳', '이름', '사람', '인물', '것', '언제']
        nouns = nouns.split(' ')
        for i in del_list:
            if i in nouns:
                nouns.remove(i)

        body = {
                "query": {
                    "bool": {
                        "must": [{"match": {"document_text": query}}],
                        "should": [{"match_phrase": {"document_text": " ".join(nouns)}}]
                    }
                }
            }
        res = self.es.search(index=self.index, body=body, size=topk)
        return self.query_result(res)

    def retrieve(
            self, dataset: Dataset, topk: int, search_mode:str
            ) -> pd.DataFrame:
        '''
        Arguments:
            dataset: Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
            topk: 쿼리당 retrieve할 passage 개수
            search_mode: 'base' or 'noun'

        Returns:
            pd.DataFrame

        Note:
            Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
            Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
        '''

        assert search_mode in ['base', 'noun']
        
        total = []
        for i, item in enumerate(tqdm(dataset, desc="Elastic Search Retrieval")):
            query = item['question']

            if search_mode=='base':
                ids, contexts = self.search(query, topk)
            elif search_mode=='noun':
                ids, contexts = self.search_ner(query, topk, item['nouns'])
                
            tmp = {
                'question' : query,
                'id' : item['id'],
                'context' : " ".join(contexts)
            }
            if "context" in item.keys() and "answers" in item.keys():
                # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                tmp['original_context'] = item['context']
                tmp['true_doc'] = self.convertid[item['document_id']]
                tmp['answers'] = item['answers']
                tmp['context_doc'] = ids

                try : ans_idx = ids.index(f"{tmp['true_doc']}")
                except : ans_idx = -1
                if ans_idx >= 0:
                    tmp['RR'] = 1 / (ans_idx + 1) 
                else:
                    tmp['RR'] = 0

            total.append(tmp)

        df = pd.DataFrame(total)
        return df
    
    def query_result(self, res):
        ids = []
        contexts = []
        for hit in res['hits']['hits']:
            ids.append(hit['_id'])
            contexts.append(hit['_source']['document_text'])
        return ids, contexts # topk retrieved document id, context. sorted by score
    
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

    INDEX = 'wiki-contexts' # Elasticsearch index name
    topk = 40
    search_mode = 'base'

    retriever = ESRetrieval(INDEX)
    with timer("Elasticsearch retrieval"):
        df = retriever.retrieve(full_ds, topk, search_mode)

    tmp = df.loc[df['RR'] > 0]['RR']
    MRR = tmp.sum() / len(df)
    Acc = len(tmp) / len(df)

    print(f'MRR : {MRR}')
    print(f'Acc : {Acc}')
