apt-get install -y build-essential openjdk-8-jdk python3-dev curl git automake

pip install konlpy "tweepy<4.0.0"

pip install JPype1-py3 mecab-ko mecab-ko-dic mecab-python

/bin/bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)