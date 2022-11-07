# README.md

[https://img.shields.io/badge/python-3.8.5-blue](https://img.shields.io/badge/python-3.8.5-blue)

### **[RecSys] Book Rating Prediction**

사용자의 책 평점 데이터를 바탕으로 사용자가 어떤 책을 더 선호할지 예측

> **Contents**
> 
> 
> Team Members
> 
> Project Introduction) 
> 
> Architecture
> 
> Score Record (RMSE)
> 
> Getting Started
> 

### Team Members

| 강수헌_T4003 | 박경준_T4076 | 박용욱_T4088 | 오희정_T4129 | 정소빈_4196 |
| --- | --- | --- | --- | --- |
| Github | Github | Github | Github | Github |
| soso6079@naver.com | rudwns708.14564@gmail.com | oceanofglitta@gmail.com | ohhj1999@gmail.com | sobing98@gmail.com |

## Project Introduction

| 프로젝트 주제 | 사용자의 책 평점 데이터를 바탕으로 사용자가 어떤 책을 더 선호할지 예측 |
| --- | --- |
| 프로젝트 개요 | 부스트캠프 Level1-U stage 강의를 통해 배운 내용을 바탕으로, 모델을 설계하고 학습하며 추론을 통해 나온 결과를 바탕으로 순위 산정하는 방식 |
| 활용 장비 및 재료 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| • 서버: Tesla V100, 88GB RAM Server<br>• 개발 IDE: Jupyter Notebook, VS Code<br>• 협업 Tool: Notion, Slack, Zoom |
| Metric | RMSE Score |
| Dataset | • books.csv : 149,570개의 책(item)에 대한 정보를 담고 있는 메타데이터<br>• users.csv : 68,092명의 고객(user)에 대한 정보를 담고 있는 메타데이터<br>• train_ratings.csv : 59,803명의 사용자(user)가 129,777개의 책(item)에 대해 남긴 306,795건의 평점(rating) 데이터
 |
| 기대 효과 | 사용자의 책 평점을 예측하는 모델을 개발하고, 이 모델이 사용자에게 책을 추천할 때 좋은 기준이 될 수 있을 것이다.  |

![프로젝트 구조도](https://user-images.githubusercontent.com/76675506/200248697-c05d655e-e983-463c-b7ee-973edde90af9.png)


프로젝트 구조도

![데이터 구조도](https://user-images.githubusercontent.com/76675506/200249158-5f0d97f7-1990-47e0-935a-ef975dc432ce.png)

데이터 구조도

## Architecture

| 분류 | 내용 |
| --- | --- |
| 아키텍처 | FactorizationMachineModel + FieldAwareFactorizationMachineModel + DeepCrossNetworkModel |
| LB점수(8/14등) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| • public : 2.1407<br>• private : 2.1409 |
| Training Feature | user_id, isbn, age, publisher, language, location country, year of publication, book author, category<br>(book title, city, state를 제외하고 나머지를 학습에 사용함) |
| 데이터 | • user_id: 고유번호<br>• location: city이용해 state, country결측치 처리<br>• age: pseudo labeling로 결측치 처리<br>• publisher, language: isbn이용해 결측치 처리 |
| 앙상블 방법 | 1번 학습 방법으로 FM+FFM+HOFM+DCN을 optimal_weighted로 묶고 2번 학습 방법으로 FM+FFM+DCN을 optimal_weighted로 묶고 (1+2)/2 방식으로 앙상블을 진행함. |

### **Score Record (RMSE)**

![private board evaluation](https://user-images.githubusercontent.com/76675506/200249433-a41ea708-9040-4060-9c75-b6ad03814039.png)

private board evaluation

![public board evaluation](https://user-images.githubusercontent.com/76675506/200249732-8e304813-ed2a-4a67-8a15-1034276d3cb6.png)

public board evaluation

## Getting Started

- requirements : `install requirements`

```
pip install -r requirements.txt
```

- train & Inference : `main.py`

```
python main.py --MODEL FM --DATA_PATH data
```

![options](https://user-images.githubusercontent.com/76675506/200249850-d40a0e84-5710-4bbd-8972-22655a935382.png)

options
