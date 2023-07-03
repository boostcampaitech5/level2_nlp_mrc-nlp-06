# Elasticsearch 시작하기

- [Elasticsearch](https://www.elastic.co/kr/downloads/past-releases/elasticsearch-7-17-10)를 설치해 주세요.
- 압축을 푼 폴더를 `elastic_search/.` 에 위치시켜 주세요.

# Elasticsearch 실행하기
### 실행
```console
cd elasticsearch-7.17.10
bin/elasticsearch
```
### 환경 세팅(한 번만 실행)
```console
pip install elasticsearch
bin/elasticsearch-plugin install analysis-nori
python es_init.py
```

# Search mode
## base
Question에 대한 passage를 retrieve합니다. 
## nouns
Question에 대한 명사들을 추출한 컬럼이 필요합니다. 