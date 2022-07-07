"""

* BERT & GPT

- transformer 구조를 가지고있는 사전 학습 모델 encoder,  decoder를 차용해서 사용

* BERT : Bidirectional Encoding Representations Transformer -> 양방향성 인코딩 구조를 갖는 트랜스포머

* GPT : Generate Pretrained Transformer : 만들어내는거



* BERT
-> transformer 구조를 활용한 사전 훈련 언어 모델
-> transformer 구조의 encoder 부분을 사용함

---transformer의 encoder 구조 --
(-> input -> embedding layer
-> LStm 대신에 positional encoding
-> self attention (각각 weight가 다른)
-> Feed forward nueral network>

*BERT 특징

1. 대용량 corpus data로 모델(word embedding : GLove, FastText, SGNS, CBoW)을 학습시킨 후,
   task에 알맞게 "전이학습"을 하는 모델 ->trnasfer learning
2. Bi-derectional Language Model -> 빈칸을 추론할때 앞뒤문장을 참조해서 추론가능

-BERT도 word embedding의 한 방법
    장점: BERT로 구성된 언어 모델은 LSTM, CMNN, Attention 없이 ANN(FCL)만 이어서 task를 수행해도
        성능이 잘 나오기 때문

-BERT는 pre-trained model : 모델을 가져와서 task에 알맞게 fine-tuning만 하면됨 (nural 가져와서 마지막에 FCL)

*BERT 구조

-문장 a , b 들어가서 encoding을 거쳐서 NSP, MLM 2개의 목적 함수를 학습시킴
-BERT는 두가지 방식의 pre-training 구조
1. Masked Alnguage Model ( MLM ) -> 가려서 학습시키는 언어 모델, 단어를 가림, 단어 레벨로 학습
2. Next Sentence Prediction ( NSP ) -> a, b 두 문장을 넣었을때 두 문장이 서로 연결되어있는가, 이어질때 어색함이 있는지 없는지
                                    -> 지도학습, 이어지면 1, 아니면0 으로 labeling가능
                                    -> 문장 레벨로 학습


* MLM : Masked Language Model
- 입력 단어의 15%를 masking 함
    -이 중 80%는 그대로 masking 수행
    -10%는 masking 대신 random word로 치환한다
    -10%는 원래 단어를 그대로 둔다

- masking한 단어를 예측하도록 인공신경망에 요구
- BERT에 2 문장을 넣음 -> 일부 masking하고 맞춰봐 -> 정답을 알려주는 supervised learning

ex> 어제 정말 " 재미있게 " 놀았어
-> 어제, 정말, 놀았어 input -> self attention 진행 -> transformer encoder안에서 계산
-> loss 구해서 학습
-> 단어 사이 self attention 으로 module 참고하기 때문에 양방향 학습 방법


*NSP : Next Sentence Prediction (supervised bianary classcification )
- 두 개의 문장이 주어졌을때 이 둘이 이어지는 문장인가?
- A : I studied hard today
- B : It is hard ti predict stock market
- Label : No ( 0 )

* BERT Structure
- BERT_bsae 구조
-12개의 transformer block
-786개의 hidden size
-12개의 self- attention

-3개의 embedding layer를 거침
1. token embedding
2. segment embedding :문장 단위
3. position embedding : 위치 가중치
--> 합한게 bert의 embedding

문장마다 sep로 문장 나누기 -> transformer 통과 -> fine tuning

*Ablation Study
-BERT에서의 각 구성 요소를 하나씩 제거 하면서 각 요소들이 얼마나 중요한지를 판단할 수 있는 study

* GPT  : Genertae Pre-Trained Transformer -> 앞 단어만 보고 뒷단어 생성해내기 때문에 generate
-bert vd gpt
-> 양방향 참고하는 bert와 다르게 GPT는 무조건 앞만 참조
- transformer의 decoder구조만 ㄴ사용 즉, self-attention 구조를 사용


* GPT - Masked Self Attention
- 순차적으로 계산하는 단방향성 pre-train 모델
- GPT는 문장 생성과 관련된 task
- BERT 는 문장 의미 및 감정 분석 추출과 관련된 task



"""