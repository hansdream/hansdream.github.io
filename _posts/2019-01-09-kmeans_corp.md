---
layout: post
title: "[Python] K-means를 통한 기업 패턴에 따른 분류"
subtitle: "비지도학습을 통한 기업 패턴 분석"
author: "MK"
comments: true
sitemap :
  changefreq : daily
  priority : 1.0
tags: [머신러닝]
---



4차산업혁명 시대에 이르면서 기업마다 다양한 디지털 산업을 영위하고, 디지털업계가 유통업, 금융업에 진출하면서 산업의 경계가 허물어져 가고 있다.

점점 기존의 산업분류체계에 대한 의구심을 갖고 새로운 기업 패턴 분류에 대한 니즈가 생겨나게 되었다.

이러한 관점에서 가장 눈에 띈 시도를 한 기업이 켄쇼이다.
켄쇼는 기업의 패턴분석을 통해 4차산업혁명 기업들을 분류하고 인덱스를 만들어 관리하였다.

이러한 산업 트렌드 변화의 출발선에서 가장 기본적인 기업데이터를 가지고 패턴분석으 해보고자 한다.

데이터 분류는 임의로 5개로 분류하였다.

<br>

### 1. 데이터 가져오기
---
**산업정보를 제외하고 해당 기업의 정보만으로 구성된 순수 기업 데이터** 를 pickle 변수로 생성해 두었다.
`corp_features`은 freature데이터들만으로 구성된 데이터이다.

```python
import pickle
import pandas as pd
import numpy as np

with open('./pickles/dataset.p', 'rb') as file:  
    data = pickle.load(file)
    corp_features = pickle.load(file)
```

<br><br>

### 2. 데이터 스케일링
---
모델에 넣기전에는 항상 스케일링을 체크해준다. 여기서는 min/max 스케일링법을 사용하였다.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit_transform(corp_features)
feature = scaler.fit_transform(corp_features)
```

<br><br>

### 3. K-means 모델실행
---
`n_clusters` 파라메터를 통해 분류 `class수`를 결정할 수 있다.
```python
from sklearn.cluster import KMeans
import matplotlib.pyplot  as plt

# create model and prediction
model = KMeans(n_clusters=5,algorithm='auto')
model.fit(feature)
predict = pd.DataFrame(model.predict(feature))
predict.columns=['predict']
```

<br><br>

### 4. 예측결과 합치기
---
```python
# 결과 합치기
final_df = pd.DataFrame(np.hstack((predict, feature)))
# 컬럼명 지정
cols = list(corp_features.columns.values)
cols.insert(0,'group')
final_df.columns = cols
```

참고로 생성한 결과 **그룹의 값을 0~숫자가 아닌, 지정값으로 변경할 수도 있다.**
여기서는 plot그래프를 그릴예정이므로 사용하지 않았다.

```python
# 숫자 to 그룹명 변경
group_name = {0: 'gr01',
               1: 'gr02',
               2: 'gr03',
               3: 'gr04',
               4: 'gr05'}

final_df['group'] = final_df['group'].replace(group_name)
```

<br><br>

### 5. 결과 시각화 확인(t-SNE활용)
---
데이터의 분류를 시각화해서 확인하고 싶으나, 다차원의 데이터를 눈으로 확인하기란 불가능에 가깝다. 단, 차원을 줄이면 가능하다. 그래서 차원을 축소하여 feature를 2,3로 줄인 후 그래프를 그려 확인해볼 수 있다.

주로 PCA, tSNE기법을 많이 사용하고 있다.
여기서는 tSNE 코드를 사용하였다.

먼저 생성된 결과를 데이터 프레임으로 변환한다.
```python
feature_df = pd.DataFrame(feature)
```


**차원축소**
```python
import numpy as np
from sklearn.manifold import TSNE

# 2개의 차원으로 축소
transformed = TSNE(n_components=2).fit_transform(feature_df)
transformed.shape
```


**시각화**
```python
xs = transformed[:,0]
ys = transformed[:,1]
plt.scatter(xs,ys, c=final_df['group'])  #라벨은 색상으로 분류됨

plt.show()
```


![img_area](/img/posting/2019-01-09-002-tsne.PNG){: .post-img}

결과를 확인해보면 상당부분 노란색 영역에 분포하지만 어느정도 5개 분류가 명확한 것을 확인할 수 있다.
향후 추가 데이터 특징 분석을 통해 분류된 기업들간 어떤 패턴이 있는지 확인해보면 의미있는 데이터를 얻을 수도 있을 것이다.


<br>
