#0512 NLP 강의 필기

"""
* Co- occurrence Matrix
-> 토큰화한 벡터들을 Matrix로 만들어서 붙어있는 단어관계면 1, 아니면 0 부여, 여러번 나오면 그 숫자만큼
-> 행렬이 전치행렬과 같은 구조를 갖는 매트릭스
-> 단어 주변에 어떤 단어가 나왔는지 정보를 담고있는 매트릭스


* GloVe :Co- occurrence 확률을 Word2Vec에 embedding
-> 중심 단어와 주변 단어 벡터의 내적을 Co-occurence prob가 되도록  = Corpus 동시 등장 확률
(이때 내적은 a,b둘의 norm값 x cos0 값임 = 즉 내적값이 높으면 연관성인 cos0도 높음)

* GolVe : Nomenclature
-> X : A co-accurrence Matrix
-> 단어 i 등장시, 다음 단어가 k일 확률 = Xik / Xi 조건부 확률
-> Wi : 중심 단어의 embedding vector for word i
-> Wi햇 : 주변 단어의 embedding vector

* GloVe : Main Idea
-> 중심 단어와 주변 단어 벡터의 내적 = Corpus 동시 등장 확률

        <Wi,Wk> (내적의 의미)        logp(k|i)

* GloVe : Loss Function
->  상대 비율로 만들기 위해 중심단어 후보1 후보2값을 빼주고
    벡터와 벡터를 연산해서 스칼라값이 나와야하기 때문에 벡터를 내적시킴
    그 값이  p(i|k) / p(j|k)
    중심 단어와 주변 단어의 내적에 함수를 씌우면 = 확률이 된다.

--> 주변단어, 중심단어의 벡터의 내적이 우리가 구한 copus의 동시등장 확률,
    co-occurence matrix를 만들어서 나온 확률값의 log값이  됨
                    wit X Whatk = log(i|k)

* GloVe loss function의 문제점
1. log Xik 값의 발산 --> log(1+Xik)
2. Co-occurrence 행렬 X가 spares인 경우(단어는 많지만 문장갯수는 적어서 몰리거나 단어 쌍이 많을 경우)
  --> weighted prob 사용

* GolVe
--> 중심Wi, 주변 Wk 가 있을때 둘의 내적이 log Xik + bias
--> co-occurrence X : 중심-주변 관계일 경우 +1 값을 부여하는 matrix

Word2Vec의 알고리즘 : CBOW, skip-gram , GloVe --> dataset 만들어서 학습시켜야함
"""


#~~~~텍스트 마이닝2~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""
* Fasttext 
--> 페이스북에서 개발한 Word2Vec의 알고리즘
    단어 단위에서 더 쪼갠 "subword"의 개념을 도입
    글자 단위의 n-gram
    ex>mouse의 3-gram 표현은 = <mo, mou, ous, use, se>  5개의 임베디드 벡터 필요
    연산량과 시간이 오래걸림에도 사용하는 이유-?
    
    1. OOV (Out- of Vocabulary)
    -dataset으로 학습해도 모든 단어 학습 불가능
    -모르는 단어 OOV가 등장할 경우 처리 불가능! : 토큰화, 인코딩 임베딩이 되어있지 않기 때문 
    -Word2Vec, GloVe에서 처리 불가능
        --> 단어의 정확한 뜻은 몰라도 단어를 자르면 느낌을 살려서 학습 가능 
        backdrop = back + drop      downside = down + side 
    
    2. Rare words
    -빈도수가 적은 단어들은 전처리 과정에서 제외
        -때문에 Word2Vec 임베딩 결과도 좋지 않음 학습을 거의 하지 않기 때문
        
    3. FastText는 Typo(오타)에 대해서도 강함
    

*  FastText : Typo?
    -없는 단어에 대한 유사도는 어떻게 구하는가?
    -벡터를 많이 만든 다음에 한 두개 정도는 맞지 않아도 다른 것들과의 유사도가 높기 때문에 가능
    
    
* Word2Vec Algorithms 요약

- CBOW , SkipGram (주->중, 중->주)
- SGNS: SkipGram with Negative Sampling  (주변에 없는 단어를 가져와서 부정 레이블링 후 이진분류)
- NNLM : 이전에서 다음을 예측
- GloVe : 중, 주 단어의 내적을 co-coccurence의 등장확률과 비슷하게 만드는 임베딩 방법
- FastText : n-gram word embed 사용 (사전훈련모델)


* Pre=Trained Word Embedding : Procedure
1. 사전 훈련된 Word2Vec 데이터셋을 불러온다
2. 데이터셋의 특성과 자료형을 파악한다 (visualization 등)
3. 원하는 Task 수행


* PCA : Principal Component Analysis 
-> Word2Vec의 시각화를 위한 분석 방법Tool
-> 3D-world에서 이해하기
    ex> 3차원의 데이터를 mapping을 통해서 2차원으로 만들기 고->저차원 데이터로 차원축소

Word2Vec 임베딩 100차원 -> 2차원으로 줄이기 (평면으로 시각화하여 단어 분포 확인 가능)
from sklearn.decomposition import PCA
pca = PCA(n_componenet = 2)
pcafit = pca.fit_transform(word_vec_list)
"""

#~~~~~~~~~~~~~~~~spares 분석을 위한 RNN~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 # 실습 02:36


