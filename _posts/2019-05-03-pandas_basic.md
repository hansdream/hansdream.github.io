---
layout: post
title: "[Python] Pandas 활용 기본"
subtitle: "Pandas를 활용한 데이터 처리"
author: "MK"
comments: true
sitemap :
  changefreq : daily
  priority : 1.0
categories : [Python]
tags: [파이썬기초]
---

`pandas`는 데이터 분석에 용이한 파이썬 패키지이다.
테이블과 유사한 형태인 `DataFrame` 위주의 유용한 코드를 정리해본다.


<br><br>
### 1. DataFrame 생성
---
**1) 리스트로 생성하기**
```phthon
import pandas as pd
df = pd.DataFrame({"a" : list1,"b" :list2})
```

**2) Dict로 생성하기**
```phthon
test_dict = {'names':['Jake','Eunice','Philip','Karen']}
df= pd.DataFrame(test_dict)
```
```phthon
frame_test_column = pd.DataFrame.from_dict(test_dict, orient='columns')
```

|   | name   |
|---|--------|
| 0 | Jake   |
| 1 | Eunice |
| 2 | Philip |



```phthon
frame_test_index = pd.DataFrame.from_dict(test_dict, orient='index')
```

|      | 0    | 1      | 2      | 3     |
|------|------|--------|--------|-------|
| name | Jake | Eunice | Philip | Karen |



<br><br>
### 2. 접근과 변경
---
**1) 데이터프레임 값 반올림**
```phthon
df.round(2)
```

**2) 조건 추출-쿼리사용**
```phthon
df.query("(a > 5) and (a < 10)")
```

**3) 조건 추출-인덱스사용**
```phthon
result_df = result_df.iloc[:, result_df.columns != "del"]  # del 항목 제외
```
**4) 조건 추출-컬럼명사용**
```phthon
df[frame_test['names']=='Eunice']
```

**5) 값 패딩**
```phthon
data['code'] = data['code'].map(lambda x: str(x).rjust(6, '0'))[:]  # 6자리로 0패딩하기
```



<br><br>
### 3. iloc vs loc
---
```phthon
# iloc 은 숫자 인덱스 사용
default_df.iloc[1]

# loc 은 실제 설정된 인덱스 사용!
default_df.loc[2011]
```




<br><br>
### 4. 행/열 삭제
---
```phthon
default_df.drop(['first_name'],axis=1) # 열삭제
del(default_df['2017']) # 열삭제
default_df.drop([2017]) # 행삭제

```
```phthon
df.drop_duplicates() # 중복제거
```

<br><br>
### 5. 피봇테이블 생성
---
```phthon
data_pivot=stock_df.pivot_table('Change',index='Code',columns='Year', aggfunc='mean')
data_pivot
```

<br><br>
### 6. 데이터프레임 컬럼
---
**컬럼명 가져오기**
```phthon
cols = data.columns.values
```

**컬럼명으로 인덱스 설정**
```phthon
data = data.reindex(cols,axis = 1)
```

**컬럼명 변경**
```phthon
df.columns = ["c1", "c2", "c3"]
```

```phthon
df.rename(columns={"first_name":"성"}, inplace = True)
```



<br><br>
### 7. 연산
---
**행 합산**
```phthon
df['sum'] = df.sum(axis=1)
```
**lambda 연산**
```phthon
# Year과 quarter를 합쳐서 period항목 만들기
df['period'] = df[['Year', 'quarter']].apply(lambda x: ' '.join(x.astype(str)), axis=1)
```

<br><br>
### 8. 요약보기
---
```phthon
df.describe()
```

|        | name   |
|--------|--------|
| count  | 4      |
| unique | 4      |
| top    | Eunice |
| freq   | 1      |




**범주형 데이터 확인**
```phthon
print(data.gender.value_counts())
```

<br><br>
### 9. 인덱스 변경
---
**특정컬럼 인덱스 지정**
```phthon
index_df = default_df.set_index('name')
index_df.set_index('name',inplace=True)  # 바로 변경
```

**기존 인덱스 컬럼으로 살리기**
```phthon
index_df.reset_index().set_index('enter')
```

**컬럼 순서 바꿀때 활용**
```phthon
df.reindex([2011, 2012, 2013, 2014, 2015, 2016, 2017])
```
**로우와 컬럼 함꺼번에 재색인**
```phthon
df.reindex(index=[2010,2011, 2012, 2013, 2014, 2015, 2016, 2017], columns=['first_name','last_name'])
```


<br><br>
### 10. 결측치 처리
---
**1) 채우기 메서드 활용**
```phthon
# ffill, pad
# bfill, backfill
df.sort_index().reindex([2010,2011, 2012, 2013, 2014, 2015, 2016, 2017], method='ffill')
```
**2) fillna 활용**
```phthon
df.fillna(0) # 지정된 값으로 채우기
```

**3) replace 활용**
```phthon
df.replace(np.NaN, 'NULL Value', inplace=True)
```

**4) fill_value 활용**
```phthon
default_df.reindex([2011, 2012, 2013, 2014, 2015, 2016, 2017], fill_value='NoName')  # 없는 값은 'NoName'으로 채우기
```
**5) NA행 삭제**
```phthon
# how = any, all
df.dropna(how='all') # 행삭제
df.dropna(how='all', axis=1) # 열삭제
```

**6) 나머지의 평균으로 채우기**
```phthon
d1 = alco_with_nan['D1']
clean = d1.notnull()
d1[-clean] = d1[clean].mean()
```




<br><br>
### 11. 정렬하기
---
**인덱스 기준**
```phthon
default_df.sort_index(ascending=False)
```

**특정열 기준**
```phthon
default_df.sort_values(by='first_name', ascending=True)
```



<br><br>
### 12. 순위지정
---
```phthon
rank_test = pd.DataFrame(np.random.randn(10))
```


<br><br>
### 13. 병합하기
---
```phthon
# 인덱스를 원래 열로 돌려두고 다시 합친다.
pd.merge(merge1.reset_index(), merge2.reset_index()).set_index('State')

# 특정 인덱스 지정
pd.merge(merge1, merge2, left_index=True, right_index=True)

# 인덱스를 그대로 사용할거면 join을 쓰기도 한다.
merge1.join(merge2).head()

# how = left, right, outer, inner
pd.merge(merge1[:10], merge2, left_index=True, right_index=True, how='left')
```


<br><br>
### 14. 데이터 연결하기
---
```phthon
# concat 아래로 이어 붙이기
pd.concat([merge1,merge2])

# concat 옆으로 이어 붙이기
pd.concat([merge1,merge2],axis=1)
```



<br><br>
### 15. 그룹만들기(group by)
---
```phthon
merge1.groupby('column')
merge1.groupby(['column','column2'])

# 5개씩 묶기
merge1.rolling(windows=5)

# 묶어서 계산하기
merge1.rolling(windows=5).mean() #평균
```



<br><br>
