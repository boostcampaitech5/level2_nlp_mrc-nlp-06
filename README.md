# 🔍 Open-Domain Question Answering

# 🙌🏻 1. 프로젝트 개요
### 1.1. 개요
- Question Answering(QA)은 다양한 종류의 질문에 대해 대답하는 인공지능을 만드는 연구 분야이다. 다양한 QA 시스템 중, Open-Domain Question Answering(ODQA)은 주어지는 지문이 따로 존재하지 않고 사전에 구축되어있는 Knowledge resource 에서 질문에 대답할 수 있는 문서를 찾는 과정이 추가된다.
- 본 ODQA 대회에서 우리가 만들 모델은 two-stage로 구성되어 있다. 첫 단계는 질문에 관련된 문서를 찾아주는 "retriever" 단계이고, 다음으로는 관련된 문서를 읽고 적절한 답변을 찾거나 만들어주는 "reader" 단계이다. 두 가지 단계를 각각 구성하고 그것들을 적절히 통합하게 되면, 어려운 질문을 던져도 답변을 해주는 ODQA 시스템을 만드는것이 이 프로젝트의 목표이다.
# 1.2. 평가 지표
두 가지 평가지표가 있다. EM 기준으로 리더보드 등수가 반영되고, F1은 참고용으로만 활용된다.
- Exact Match(EM)
  - 모델의 예측과, 실제 답이 정확하게 일치할 때만 점수가 주어진다. 즉 모든 질문은 0점 아니면 1점으로 처리된다.
- F1 Score
  - EM과 다르게 부분 점수를 제공한다. 예를 들어, 정답은 "Barack Obama"지만 예측이 "Obama"일 때, EM의 경우 0점을 받겠지만 F1 Score는 겹치는 단어도 있는 것을 고려해 부분 점수를 받을 수 있다.
 
# 👨‍👩‍👧‍👦 2. 프로젝트 팀 구성 및 역할

### 2.1. 팀 구성

|<img src='https://avatars.githubusercontent.com/u/74442786?v=4' height=100 width=100px></img>|<img src='https://avatars.githubusercontent.com/u/99644139?v=4' height=100 width=100px></img>|<img src='https://avatars.githubusercontent.com/u/50359820?v=4' height=100 width=100px></img>|<img src='https://avatars.githubusercontent.com/u/85860941?v=4' height=100 width=100px></img>|<img src='https://avatars.githubusercontent.com/u/106165619?v=4' height=100 width=100px></img>|
|:---:|:---:|:---:|:---:|:---:|
| [김민호](https://github.com/GrapeDiget) | [김성은](https://github.com/seongeun-k) | [김지현](https://github.com/jihyeeon) | [서가은](https://github.com/gaeun0112) | [홍영훈](https://github.com/MostlyFor) |

### 2.2. 역할
- 김민호 : 프로젝트 리팩토링, 핵심 문장 강조, Curriculum learning
- 김성은 : BM25 retriever, Elasticsearch, 데이터 전처리
- 김지현 : Custom model for question answering, Self-distillation, Ensemble
- 서가은 : 모델 & topk 변경 실험, 추가 데이터 fine tuning, balanced sampling
- 홍영훈 : TF-IDF retriever, DPR 구현, post-processing

# 💽 3. Data Analysis
### Data
- Train data : 3952
  - Feature: 'title', 'context', 'question', 'id', 'answers', 'document_id'
- Train과 validation 모두 유사한 분포의 context 길이를 갖는다.

# 📂 4. 프로젝트 수행
### 4.1. Data Preprocessing
- Wiki data 전처리
### 4.2. Retriever
- Top-k 변경
- TF-IDF
- 자체 토크나이징
- BM25
- ElasticSearch
- DPR(Dense Passage Retrieval)
### 4.3 Reader
- klue/roberta-large 사용
- 핵심 문장 강조
- Custom model
  - LSTM
  - Bi-LSTM
  - SDS-CNN & MLP
- 추가 데이터셋을 활용한 전이학습
  - KorQuAD 1.0
  - KorQuAD 2.0
  - 기계독해 데이터셋
  - 뉴스기사 기계독해 데이터셋
  - 일반상식 데이터셋
- Balanced sampling
- Curriculum learning

### 4.4 Ensemble
- Hard voting
- Soft voting
- K-fold

### 4.5. 기타
- Self-distillation
- Post-processing
