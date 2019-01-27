---
layout: post
title: "[Python] 데이터 전처리 과정"
subtitle: "Kaggle 데이터를 활용한 전처리 과정"
author: "MK"
comments: true
sitemap :
  changefreq : daily
  priority : 1.0
categories : [Python]
tags: [전처리]
---


데이터는 **Kaggle의 home credit default risk** 를 사용했다.

<br>

### 1. 데이터 가져오기
---
Kaggle Data [보러가기](https://www.kaggle.com/c/home-credit-default-risk/data)

 `application_train.csv`파일 다운로드

```python
data = pd.read_csv('./datas/kaggle_homecredit/application_train.csv')
```

<br><br>

### 2. 변수 정보 확인
---

```python
# 컬럼별 type 확인 및 결측치 확인
data.info()
data.isnull().sum()
```

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 307511 entries, 0 to 307510
Columns: 122 entries, SK_ID_CURR to AMT_REQ_CREDIT_BUREAU_YEAR
dtypes: float64(65), int64(41), object(16)
memory usage: 286.2+ MB

SK_ID_CURR         0
TARGET             0
NAME_CONTRACT_TYPE 0
CODE_GENDER        0
FLAG_OWN_CAR       0
FLAG_OWN_REALTY    0
CNT_CHILDREN       0
AMT_INCOME_TOTAL   0
AMT_CREDIT         0
AMT_ANNUITY       12
AMT_GOODS_PRICE  278
```

결측치 수에 따라 항목을 포함할지 여부를 결정한다.

<br><br>

### 3. 결측값 처리
---
```python
# 문자전환
data = data.replace(' ', '')

# 만약 결측치가 문자열 스페이스(' ')로 되어 있다면, np.nan으로 바꾸어 Pandas 라이브러리가 인식할수 있도록 변환
data = data.replace('', np.nan)

# 결측 row 제거하는 방법
data.dropna(how='all') # 'all':한 행이 모두 missing value이면 제거, any': 행 내에서 하나라도

# 결측치 처리
data.fillna(0, inplace=True)
```

<br><br>

### 4. 범주형 변수 처리
---
`남자/여자`, `소형/중형/대형` 등 문자로 표현되는 범주형 데이터는 `One & Hot 인코딩`으로 처리된다.

**One & Hot 인코딩** 을 하게 되면 특정 변수에서 나올수 있는 **문자열 수대로 새로운 항목이 생성** 된다.

다시말해 **소형/중형/대형** 데이터를 같는 변수의 인코딩 처리시 **소형여부,중형여부, 대형여부 총 3가지 항목값을 매핑해야 한다.**
즉, 값이 `중형`일 경우 소형여부, 중형여부, 대형여부의 매핑값은 `0, 1, 0`이 된다.


하지만 남자/여자처럼 **0,1 두가지 코드로 해결할 수 있는 경우** 는 굳이 남자여부 여자여부로 나누지 않고 한 항목을 사용해 0,1로 표현할 수 있으므로 유사변수는 `category_list`로 따로 분류하였다.

변수별 유효값을 확인하여 판단한 결과는 아래와 같다.

```python
# 처리 방식에 따른 컬럼 정리
category_list = ['NAME_CONTRACT_TYPE','FLAG_OWN_CAR','FLAG_OWN_REALTY']
one_hot_list = ['NAME_TYPE_SUITE','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS',
                'NAME_HOUSING_TYPE','OCCUPATION_TYPE','WEEKDAY_APPR_PROCESS_START','ORGANIZATION_TYPE',
                'FONDKAPREMONT_MODE','HOUSETYPE_MODE','WALLSMATERIAL_MODE','CODE_GENDER','EMERGENCYSTATE_MODE']
```

<br>

**_이제 2개의 리스트를 각각, 다른 로직을 적용한다._**

```python
# categorize
categories_encoded = pd.DataFrame()
cate_cols = []

for x in category_list:

    print(x)
    X = data[x]
    x_encoded, x_categories = X.factorize()

    # dataframe
    temp_df = pd.DataFrame(x_encoded)
    categories_encoded = pd.concat([categories_encoded,temp_df],axis=1)

    # 컬럼명 추가
    cate_cols.append(x + '_1')

# 컬럼명 수정
categories_encoded.columns = [cate_cols]

```

아래와 같이 항목이 생성된 것을 확인할 수 있다.
```python
categories_encoded.head()
```


|   | NAME_CONTRACT_TYPE_1 | FLAG_OWN_CAR_1 | FLAG_OWN_REALTY_1 |
|---|----------------------|----------------|-------------------|
| 0 | 0                    | 0              | 0                 |
| 1 | 0                    | 0              | 1                 |
| 2 | 1                    | 1              | 0                 |
| 3 | 0                    | 0              | 0                 |
| 4 | 0                    | 0              | 0                 |

<br>

항목별로 max값을 확인해 0,1값 이상을 갖는지 확인한다. 모두 1이면 정상, 2이상이 나오는 값은 `one_hot_list`로 넘겨서 처리한다.
```python
# 항목별 맥스값 체크 (확인용)
for x in category_list:
    col = x + '_1'
    print(x, ' max : ', max(categories_encoded[col].values)[0])
```

```
NAME_CONTRACT_TYPE  max :  1
FLAG_OWN_CAR  max :  1
FLAG_OWN_REALTY  max :  1
```

<br><br>

다음은 `one_hot_list` 처리방식이다.
```python
# One-Hot-Encoder
onehot_encoded = pd.DataFrame()
onehot_cols = []

for x in one_hot_list:

    print(x)
    X = data[x]
    x_encoded, x_categories = X.factorize()
    x_1hot = encoder.fit_transform(x_encoded.reshape(-1,1))
    x_1hot = x_1hot.toarray()    

    # dataframe
    temp_df = pd.DataFrame(x_1hot)
    onehot_encoded = pd.concat([onehot_encoded,temp_df],axis=1)

    # 컬럼명 추가
    for i in range(1, temp_df.shape[1] +1):
        onehot_cols.append(x + '_' + str(i))

# 컬럼명 수정
onehot_encoded.columns = [onehot_cols]
```

<br>

```python
# 항목별 맥스값 체크 (확인용)
for x in one_hot_list:
    col = x + '_1'
    print(x, ' max : ', max(onehot_encoded[col].values)[0])

# 모두 1이면 정상
```


```
NAME_TYPE_SUITE  max :  1.0
NAME_INCOME_TYPE  max :  1.0
NAME_EDUCATION_TYPE  max :  1.0
NAME_FAMILY_STATUS  max :  1.0
NAME_HOUSING_TYPE  max :  1.0
OCCUPATION_TYPE  max :  1.0
WEEKDAY_APPR_PROCESS_START  max :  1.0
ORGANIZATION_TYPE  max :  1.0
FONDKAPREMONT_MODE  max :  1.0
HOUSETYPE_MODE  max :  1.0
WALLSMATERIAL_MODE  max :  1.0
CODE_GENDER  max :  1.0
EMERGENCYSTATE_MODE  max :  1.0
```

<br><br>

### 5. 최종 항목 정리
---
이제 변환한 데이터의 원본은 제거하고 변환한 데이터 항목을 병합하는 일만 남았다.

원본의 전체 컬럼을 `total_cols`, 변환한 컬럼을 제거한 후 남은 컬럼을 `final_cols`로 정의한다.

```python
total_cols = set(data.columns.values)
final_cols = list(total_cols - set(category_list) - set(one_hot_list))
```

최종 데이터셋은 변환한 항목을 병합하여 구한다.
```python
final_df = pd.concat([categories_encoded, onehot_encoded, data[final_cols] ], axis=1)
```

```python
print("최초컬럼->최종컬럼")
print(len(total_cols),len(final_cols))

print("병합 전 3개 테이블")
print(categories_encoded.shape, onehot_encoded.shape, data[final_cols].shape)

print("병합 후 1개 테이블")
final_df.shape
```

```
최초컬럼->최종컬럼
122 106
병합전 3개 테이블
(307511, 3) (307511, 140) (307511, 106)
병합 후 1개 테이블
(307511, 249)
```


간혹 컬럼명이 이상하게 나올때가 있다. 아래 코드로 재정비한다.
```python
# 컬럼명 재지정
final_df.columns = cate_cols + onehot_cols + final_cols
final_df.head()
```



![img_area](/img/posting/2019-01-09-001-progressing.PNG){: .post-img}

<br><br>

### 6. 결과 저장하기
---
원하는 형태로 저장한다.

**csv 저장**
```python
final_df.to_csv('./datas/kaggle_homecredit/home_credit_risk.csv')
```

**파이썬 객체 저장**
```python
import pickle

with open("./pickles/home_credit_risk.p", 'wb') as file:  # hello.txt 파일을 바이너리 쓰기 모드(wb)
    pickle.dump(final_df, file)
```


<br>

### **Reference**
---
- <https://datascienceschool.net/view-notebook/f43be7d6515b48c0beb909826993c856>
- <https://wikidocs.net/16582>

<br>
