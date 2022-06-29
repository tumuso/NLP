#0520 NLP 강의 필기

#CNN : 공간정보 , conv 사용해서 공간 사이의 weight를 공유
#RNN: 시간정보, weight 공유

"""


* RNN 기본원리 -> Recurrent Neural Network -> 재귀성 네트워크, 시간에 따라 같은 weight 사용
  -> 시계열 데이터를 처리하기에 좋은 뉴럴 네트워크 (음성인식, 음악생성기, DNA염기서열 분석, 번역기, 감정 분석)
  -> CNN이 이미지 구역별로 같은 필터를 사용해서 같은 weight를 공유한다면,
  -> RNN은 시간별로 같은 weight를 공유한다. 즉 과거와 현재는 같은 weight를 공유함

*First Order System
 ->현재 시간의 상태가 이전 시간의 상태와 관련이 있다고 가정:
        x : 상태, t: 시간 xt : 시간 t일때의 x의 상태값
                        xt = f(xt-1)
 ->이전 상태에만 영향을 미침
 ->이 시스템은 외부 입력 없이 초기 조건만으로 홀로 돌아감
 ->자기 자신의 이전 상태에 대한 함수
 ->현재 시간의 상태가 이전 시간의 상태와 현재의 입력에 관계가 있는 경우,
         x : 상태, t: 시간 xt : 시간 t일때의 x의 상태값, ut: 현재 입력
                        xt = f(xt-1, ut)  <-input이 2개

* State-Space Model
                     xt = f(xt-1, ut)  <-input이 2개

  ->모든 시간 t에서 모든 상태 xt가 관측이 가능한가?
  ->날씨 예측, 주가 예측의 xt는 일부만 예측이 가능함!
    ->따라서 관측가능한 변수들의 모음을 따로 만들어줘야 함

 ->각 시간에서 관측 가능한 상태의 모음 : 출력 yt (output)
                    yt = h(xt) : 비선형 함수

                     ut           xt            yt
                    input -> hiddden state -> output

* State-Space Model 정리
 -> 어떤 시스템을 해석하기 위한 3요소 : 입력, 상태, 출력
                    xt = f(xt-1, ut)
                    yt = h(xt)

* RNN ANN 차이
 -> RNN은 hidden state인 x가 self feedback을 하지만 ANN는 하지 않음
 -> but 함수 f로 x의 weight를 공유한다 : Wxx -> 초기 조건을 정의해줘야함 (random initializing, 0 , 1)
 -> xt는 앞의 state인 xt-1을 참조한다
 -> 상태 Xt는 이전까지의 상태와 이전까지의 입력을 대표할 수 있는 압축본이라고 할 수 있다
 -> 상태 Xt는 시계열로 들어오는 입력들을 최대한 상세히 표현할 수 있어야 함

* State-Space Model as RNN

원래 풀고 싶었던 문제: Xt = f(Ut, Ut-1 ... U0)
   ex) I like eating [pizza]   -> input 3 , output 1

대신해서 풀 문제: Xt = f(Xt-1, Ut)
   모든 input이 hidden state를 간접적으로 거쳐서 마지막 state에서 모든 input의 정보를 함축
   --> First-order Markov Model


함수 f와 h를 근사하기 위해서 뉴럴 네트워크를 사용함
-뉴럴 네트워크에서 비선형 함수를 표현하는 방법 :
            y = a(wtx + b)    -> a: 비선형성 부여activation function , w: weight, b: bias
-뉴럴 네트워크 셋팅으로 함수 근사:
            Xt = a(Wxx*Xt-1 + Wxu*Ut + bx)
            yt = a(Wyx*Xt + by)

* a: activation function
* Wxx : self feedback weight
* Wxu : 초기조건 input weight
* b : bias
* Wyx : wight
--> 사용하는 parameter matrix는 총 "5"개

4:07

==================================================================

RNN: Basic Structure

Xt = f(xt-1, ut)
Yt = h(xt)

--> BPTT (Back - propagation trough time )사용

RNN : Problem types
1. many to many -> 번역  (RNN에서는 잘 사용하지 x)
2. many to one -> 예측
3. one to many -> 단어 하나로 문장 생성
4. sequence to sequence -> rnn 2개 연결  번역!

*RNN의 한계점

- self feedback 진행하면서 exploding gradient / vanishing gradient 문제점 발생

1. exploding gradient -> Nan , inf 발생
  ->학습 도중 loss가 inf가 뜰 경우 : 학습이 더 이상 진행이 불가능함
  ->gradient cliiping으로 해결 가능

2. vanishing gradient -> 0 발생
  -> 학습 도중에 파악 자체가 어려움
  ->다른 네트워크 구조를 쓰는게 훨씬 편함 -> Gated Rnns : LSTM, GRM

------------------------------------

* RNN 단점 보완 -> GRM / LSTM

* LSTM -> gradient flow 를 제엉할 수 있는 밸브 역할
       -> state space의 입력 상태 출력 구조는 동일
       -> gate 구조의 추가
       -> 4개의 MLP구조
       -> input gate : 정보를 얼마나 활용할 건가를 결정
       -> forget gate : 얼마나 잊어버릴 것인가
       -> input, forget cell gate 합쳐서 output으로 내보냄

* GRU -> LSTM의 간소화 버전
      -> cell state가 없음
      -> 대신에 hidden layer 넣기
      -> lstm보다 파라미터 수가 적으므로 training time이 절약된다

* BiLSTM : 시간에 따라 뒤로 가는 model

* RNN Dataset : 5개 짜리 RNN 만들시 u 5개, self feedback 하면서 출력됨
                y도 5개의 출력이 나오지만 Yt의 추정값만 사용 실값의 loss값을 사용해서 BPTT진행

* RNN : encoder , ANN(FCL) : Decoder


* MAchine Translation : Sequence to Sequence
                        -> 전체 문제에서 특징을 추출한 뒤 번역된 문장을 재생성

-> Encoder RNN, -(context vector)-> Decoder RNN 2개를 사용
->  training시에는 답으로 모든 문장을 보여줌

1. word tokenization & encoding
2. training
3. validation & testing

* 1D - CNN Model : 시계열 분석하는 자연어처리용 CNN word2vec filter를 conv에 넣음
-> pooling 작업을 통해 feature 추출 -> ANN(FCL) -> y
-> Max Pooling 통해 이미지 처리하듯이 사용 가능
-> Max pooling 한 결과를 concatenation -> ANN에 넣기


* Bi-LSTM
   -> causal system : 뒤에 단어만으로 추론 --> 반대로 가는 구조도 추가하자 해서 나온 모델
   -> self feedback 부분의 가중치가 늘어남
   -> causalty를 무시해도 될 때 사용가능


* Attention & Transformer

* Attention : sequence to sequence의 단점 개선
1. context vector에 모든 정보가 함축되어있음 -> 정보 손실
2. Vanishing , exploding Gradient
3. 단어를 생성할 때마다 context vector를 모두 사용한다 ( 비효율적)
--> encoder input에 들어가는 정보를 좀 더 해결했으면 좋겠응

* output을 사용하지 않고 hidden sell을 context vector로 사용
-> 단어 하나가 어떤건지 유사도를 통해 파악
-> decoder의 hidden state를 encoder의 hidden state에 내적함 : 내적값이 attention sxore가 됨
->attention score에 softmax에 통과시킴 ( 가중치로 사용, 가중합계를 하기 위해)
-> 가중치 참고하여 어디에 집중하는지 알 수 있음
-> query : decoder의 hidden state , key : encoder의 hidden state, value

--> attention 매커니즘은 현재 시점 예측에서 입력의 특정 부분에 보다 집중할 수 있도록 설계됨

* Attention in Sequence , Attention Score을 구하는 유사도

  -> Bahdanau Attention (score를 구하는 방법 중 하나)


* Transfomer

"""
