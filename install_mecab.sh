# java 부터 설치
apt-get install g++
apt-get update
apt-get install default-jre
apt-get install default-jdk
apt-get install make

# mecab 깔기
apt install curl
bash < curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh

# 먼저 아무폴더나 만들어서 이동해 주세요( 예시>mkdir mecab ) 
mkdir mecab
cd ./mecab

wget https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz
wget https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz

tar xvfz mecab-0.996-ko-0.9.2.tar.gz
tar xvfz mecab-ko-dic-2.1.1-20180720.tar.gz

cd mecab-0.996-ko-0.9.2
./configure
make
make check
make install
ldconfig


cd ..
cd mecab-ko-dic-2.1.1-20180720
#./configure: line 1696: syntax error near unexpected token `mecab-ko-dic,'에러시
./configure # [잘 진행되는지 봐야함] error시 ./autogen.sh 후 다음 과정 진행
make
make install

# pip install mecab-python3
cd ..
git clone https://bitbucket.org/eunjeon/mecab-python-0.996.git
cd mecab-python-0.996
python setup.py build
python setup.py install


# cd ../../
# 설치가 끝나고 해당 폴더에서 이 쓰래기들 다 밀어버립니다.
# rm -r mecab