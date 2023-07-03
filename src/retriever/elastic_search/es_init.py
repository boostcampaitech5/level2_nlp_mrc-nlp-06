import json
import re
import pickle

from tqdm import tqdm
from elasticsearch import Elasticsearch

def es_setting():
    # Elasticsearch 서버 접속
    es = Elasticsearch(["http://127.0.0.1:9200"])
    print('Elastic search ping:', es.ping())
    print('Elastic search info:')
    es.info()
    return es

def create_index(es, index):
    # Elasticsearch 서버에 인덱스 생성
    INDEX_SETTINGS = {
        "settings": {
            "analysis": {
                "filter": {
                    "my_shingle": {
                        "type": "shingle"
                    }
                },
                "analyzer": {
                    "my_analyzer": {
                        "type": "custom",
                        "tokenizer": "nori_tokenizer", # nori tokenizer 사용
                        "decompound_mode": "mixed",
                        "filter": ["my_shingle"]
                    }
                },
                "similairty": {
                    "my_similarity": {
                        "type": "BM25" # BM25로 유사도 측정
                    }
                }
            }
        },

        "mappings": {
            "properties": {
                "document_text": { # wiki context 부분
                    "type": "text",
                    "analyzer": "my_analyzer"
                },
            }
        }
    }
    if es.indices.exists(index=index):
        es.indices.delete(index=index)
    return es.indices.create(index=index, body=INDEX_SETTINGS)

def preprocess(text):
    # Elasticsearch 인덱스에 삽입하기 전 context 전처리
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\\n", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"[^A-Za-z0-9가-힣.?!,()~‘’“”"":%&《》〈〉''㈜·\-\'+\s一-龥サマーン≪ㅋい\"よ」≫な＜・うし＞』äら°の>「∧/\\xadに<『]", "", text)  
    text = re.sub(r"\s+", " ", text).strip()  # 두 개 이상의 연속된 공백을 하나로 치환
    
    return text

def load_wiki_data():
    dataset_path = "../../../data/wikipedia_documents.json"
    with open(dataset_path, "r") as f:
        wiki = json.load(f)

    tmp_contexts = [preprocess(v["text"]) for v in wiki.values()]
    wiki_contexts = list(dict.fromkeys(tmp_contexts))
    convertid = {} # document id가 달라져서 매핑이 필요. key(바뀌기전 id):value(바뀐 후 id)
    for i, item in enumerate(tmp_contexts):
        convertid[i] = wiki_contexts.index(item)

    wiki_articles = [
        {"document_text": wiki_contexts[i]} for i in range(len(wiki_contexts))
    ]
    return wiki_articles, convertid

def main():
    es = es_setting()
    INDEX_NAME = "wiki-contexts"
    create_index(es, INDEX_NAME)
    wiki_articles, convertid = load_wiki_data()

    # save conver id data
    with open('convert_id.pickle','wb') as fw:
        pickle.dump(convertid, fw)

    # insert data
    for i, text in enumerate(tqdm(wiki_articles)):
        try:
            es.index(index=INDEX_NAME, id=i, body=text)
        except:
            print(f"Unable to load document {i}.")

    print("ElasticSearch setting Done.")

if __name__=='__main__':
    main()