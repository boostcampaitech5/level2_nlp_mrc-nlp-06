pip install elasticsearch
cd elasticsearch-7.17.10                        # 8 이전 버전 권장
bin/elasticsearch                               # elasticsearch 실행. 여기서 Permission denied 오류가 난다면 local에서 돌리는 것을 권장. 
bin/elasticsearch-plugin install analysis-nori  # nori tokenizer 설치
cd ..
sleep(5)
python es_init.py