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
"""