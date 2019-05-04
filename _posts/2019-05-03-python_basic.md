---
layout: post
title: "[Python] 파이썬 유용 코드 & Tips"
subtitle: "유용 코드 Top"
author: "MK"
comments: true
sitemap :
  changefreq : daily
  priority : 1.0
categories : [Python]
tags:
---


### 1. 타입변경
---
**데이터프레임 문자타입변경**
```phthon
df.iloc[1][1].astype(str)
```

**날짜 타입변경**
```phthon
start_date.strftime("%Y%m%d")
```

<br><br>
### 2. 통계값 확인
---
```phthon
print(stockReturn.describe())
print('skeness: ', stockReturn.skew(axis=0))
print('kurtosis: ', stockReturn.kurtosis(axis=0))
print('autocorrelation: ', stockReturn.autocorr())
```


<br><br>
### 3. 변경
---
```phthon
str(1).zfiil(2)  # 숫자앞에 영으로 채우기
```

```phthon
'2011.01.01'.replace('.', '-') # 점을 하이픈으로 바꾸기
```

```phthon
'   2011-01-01 ~ 2011-12-31          '.strip()  # 공백 제거하기
```


<br><br>
### 4. 날짜 연산
---
**날짜 차이 계산**
```phthon
from datetime import timedelta, date  
# timedelta() 시간 차이를 계산하는 함수, date() 시간변수를 만드는 함수

def daterange(start_date, end_date):

    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)
```


<br><br>
### 5. 결측치 관련
---
```phthon
alco_with_nan.notnull() # null이 아닌 경우 True
```
