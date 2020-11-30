
# 국건영 데이터 (2018)을 활용한 청소년(만13세~18세) 비만 예측 모델

## Data : 국민건강데이터 2018년 version

## ADM_final_Male_model은 남성 모델
- 변수
  * 청건행 데이터에서 비만 예측에 유의하다고 판단된 변수들과 비슷한 변수들을 국건영 데이터에서 가져옴
- 전처리
  * 설문 답변들의 범주화를 통해 Data Reduction 과정 수행
  * na행이 10개 밖에 없어서 dropna 해줌
=> 타겟 변수 포함 10개 , 타겟 제외 9개가 input

- Modeling
  * X,y 를 data, target이라는 변수로 나눠 줌
  * astype('category')으로 팩터화
  * 정규화
  * Grid Search(SVM with Linear Kernel) + K-fold(k = 5)
  * Grid Search(SVM with Linear Kernel) + K-fold(k = 10)
  * Grid Search(SVM with RBF Kernel) + K-fold(k = 5)
  * Grid Search(SVM with RBF Kernel) + K-fold(k = 10) 
- 모델을 구성한 주요 변수 파악
  * 이 경우 RBF은 비선형이기에 Linear 모델만 가능
  => 결과로 주관적 체형인식이 굉장히 주된 변수라는 것을 알 수 있음.
  
## ADM_final_integ_sex은 성별 통합모델
- 변수
  * 청건행 데이터에서 비만 예측에 유의하다고 판단된 변수들과 비슷한 변수들을 국건영 데이터에서 가져옴
- 전처리
  * 설문 답변들의 범주화를 통해 Data Reduction 과정 수행
  * na행이 10개 밖에 없어서 dropna 해줌
  * 남성모델과 차이점은 성별을 변수로 넣어줌
=> 타겟 변수 포함 11개 , 타겟 제외 10개가 input

- Modeling
  * X,y 를 data, target이라는 변수로 나눠 줌
  * astype('category')으로 팩터화
  * 정규화
  * Grid Search(SVM with Linear Kernel) + K-fold(k = 5)
  * Grid Search(SVM with Linear Kernel) + K-fold(k = 10)
  * Grid Search(SVM with RBF Kernel) + K-fold(k = 5)
  * Grid Search(SVM with RBF Kernel) + K-fold(k = 10) 
- 모델을 구성한 주요 변수 파악
  * 이 경우 RBF은 비선형이기에 Linear 모델만 가능
  => 결과로 주관적 체형인식과 나이가 큰 영향을 끼친다는 것을 알 수 있음
