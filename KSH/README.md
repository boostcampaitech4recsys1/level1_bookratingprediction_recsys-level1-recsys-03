## EDA
---
- 고전 명서의 인기와 평점은 어떨까?
- 파레토 법칙
- 콜드 스타트 분포

## fill_location
---
user 데이터의 변수들 중, location 변수의 null 값을 처리하는 코드

pseudo algorithm:
```python
for user in user_list:
    if country is null:
        ref (same city user)

    if state is null:
        ref (same country & city user)
```

## handling_category
---
변수들의 적절한 카디널리티를 찾는 코드
- 변수 별로 다른 threshold 적용
  - 클러스터의 개수를 정할 때 사용하는 elbow method를 참고해서 적절한 thresold를 찾음
- 변수 별로 동일한 threshold 적용

## contents_based_similarity
---
- 유저 유사도 계산을 위한 사전 클러스터링
    - DBSCAN, K-Means를 테스트한 결과 실루엣 스코어가 더 높은 K-Means 선택
- 유저 유사도 기반 cold start 문제에 대한 rmse: 2.4
    - FM 모델은 동일 문제에 대해 2.3이 나왔으므로 사용할 수 없닥 판단
