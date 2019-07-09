---
layout: post
title: "[Python] Korean preprocessing"
subtitle: 한국어 전처리
author: "MK"
comments: true
sitemap :
  changefreq : daily
  priority : 1.0
categories : [Python]
tags: [자연어처리]
---


한국어에 대한 자연어 처리는 영문과 유사하지만 조금 더 특별한 전처리를 갖는다.<br>
불필요한 심볼을 제거한 후 한국어 분석에 대표적인 라이브러리 `konlpy`를 사용하여 형태소를 분석한다.<br>
이중에서 데이터 분석에 가장 큰 의미를 갖는 형태소인 `명사`만을 추출해 실제로 데이터 분석에 활용한다.<br>
사용 목적이나 데이터 처리 상태에 따라 다른 형태소를 포함해볼수도 있을 것이다.<br>
명사를 추출하기 전에 추출 대상에서 제거할 불용어 사전을 미리 만들어 쓸 수도 있는데, 리스트 변수를 활용해 제거 후 추추한다.<br>

<br>


```python
import konlpy.tag
```
<br>

### 1. 분석할 텍스트 읽어오기

```python
with open('news.txt', 'r', encoding='utf8') as f:
    content = f.read()
```
에러가 날 경우 변환 불가능한 문자가 포함되어 있을 수 있다.
적용가능한 다른 encoder를 사용하거나 해당 옵션 항목을 제외해보기.

<br><br>
### 2. 불필요한 심볼 없애기

```python
filtered_content = content.replace('.', '').replace(',','').replace("'","").replace('·', ' ').replace('=','').replace('\n','')
```

`replace('\n','')`가 처리되지 않을 경우, 이하 코드에서 `java.lang.NullPointerException` 에러 발생 가능


<br><br>
### 3. 형태소 분석 및 단어 추출
```python
Okt = konlpy.tag.Okt()
Okt_morphs = Okt.pos(filtered_content)  # 튜플반환
print(Okt_morphs)
```
과거 Twitter가 Okt로 변경됨.

아래와 같이 형태소가 분석된다.
```
[('금융감독원', 'Noun'), ('이', 'Josa'), ('내년', 'Noun'), ('부터', 'Josa'), ('상', 'Noun'), ('장사', 'Noun'), ('200', 'Number'), ('곳', 'Noun'), ('에', 'Josa'), ('대한', 'Noun'), ('재무제표', 'Noun'), ('심사', 'Noun'), ('감리', 'Noun'), ('를', 'Josa'), ('실시', 'Noun'), ('하기로', 'Verb'), ('했다', 'Verb'), ('이', 'Determiner'), ('달', 'Noun'), ('부터', 'Josa'), ('심사', 'Noun'), ('감리', 'Noun'), ('가', 'Josa'), ('진행', 'Noun'), ('되는', 'Verb'), ('회사', 'Noun'), ('가', 'Josa'), ('113', 'Number'), ('곳', 'Noun'), ('인', 'Josa'), ('점', 'Noun'), ('을', 'Josa'), ('감안', 'Noun'), ('하면', 'Verb'), ('당장', 'Noun'), ('2', 'Number'), ('배', 'Noun'), ('로', 'Josa'), ('늘리는', 'Verb'), ('셈', 'Noun'), ('이다', 'Josa'), ('금융', 'Noun'), ('당국', 'Noun'), ('은', 'Josa'), ('이를', 'Verb'), ('통해', 'Noun'), ('상', 'Noun'), ('장사', 'Noun'), ('감리', 'Noun'), ('주기', 'Noun'), ('를', 'Josa'), ('기존', 'Noun'), ('20년', 'Number'), ('에서', 'Foreign'), ('단번', 'Noun'), ('에', 'Josa'), ('10년', 'Number'), ...]
```


명사, 조사 네이밍 표현을 다시 변경한다.
```python
komoran = konlpy.tag.Komoran()
komoran_morphs = komoran.pos(filtered_content)
print(komoran_morphs)
```


<br><br>
### 4. 명사만 추출하기
```python
Noun_words = []
for word, pos in Okt_morphs:
    if pos == 'Noun':
        Noun_words.append(word)
print(Noun_words)
```

```
['금융감독원', '내년', '상', '장사', '곳', '대한', '재무제표', '심사', '감리', '실시', '달', '심사', '감리', '진행', '회사', '곳', '점', '감안', '당장', '배', '셈', '금융', '당국', '통해', '상', '장사', '감리', '주기', '기존', '단번', '복안', '다만', '시장', '연간', '상', '장사', '항상', '감리', '의미', '만큼', '자칫', '강도', '사정', '치', '우려', '금융', '당국', '금감원', '최근', '증권', '선물', '위원회', '올해', '상', '장사', '곳', '대한', '재무제표', '심사', '감리', '착수', '안', '논의', '뒤', '내년', '연간', '곳', '심사', '감리', '결정', '금감원', '상', '장사', '감리', '주기', '축소', '위해', '올해', '곳', '안팎', '대한', '심사', '감리', '추진', '인력', '문제', '감리', '회사', '수', '곳', '것', '앞서', '금감원', '올해', '감리', '진행', '중이', '거나', '진행', '기업', '총', '개', '발표', '바', '이', '중', '여', '곳', '정밀', '감리', '수준', '강도', '감리', '곳', '이보', '수준', '심사', '감리', '셈', '핵심', '내년', '부터', '내년', '재무제표', '심사', '곳', '전', '감리', '과정', '문제', '발생', '정밀', '감리', '기업', '등', '총', '여', '개', '기업', '대한', '감리', '실시', '피', '심사', '감리', '업체', '수', '배', '전체', '피', '감리', '업체', '수', '올해', '대비', '가량', '증가', '것', '금감원', '심사', '감리', '대폭', '확대', ...]
```


<br><br>
### 5. 불용어 제거 전 별도 사전 구축

분석대상에서 제외하고 싶은 단어를 stopwords에 명시한다.

```python
stopwords = ['매일경제', '서울', '기자','상']
unique_Noun_words = set(Noun_words)
for word in unique_Noun_words:
    if word in stopwords:
        while word in Noun_words: Noun_words.remove(word)  # 최종결과 : Noun_words
```


<br><br>
### 6. 빈도분석
각 단어들이 몇번 사용되었는지 분석한다.
```python
from collections import Counter
c = Counter(Noun_words)
print(c.most_common(10)) # 상위 10개 출력하기
```

```
[('감리', 46), ('심사', 19), ('기업', 13), ('곳', 12), ('정밀', 8), ('회계', 8), ('장사', 7), ('금감원', 7), ('수', 7), ('재무제표', 6)]
```


<br><br>
### 7. 워드클라우드

```python
import wordcloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from os import path

FONT_PATH = 'C:/Windows/Fonts/malgun.ttf' # For Korean characters

noun_text = ''
for word in Noun_words:
    noun_text = noun_text +' '+word

wordcloud = WordCloud(max_font_size=60, relative_scaling=.5, font_path=FONT_PATH).generate(noun_text) # generate() 는 하나의 string value를 입력 받음
plt.figure()
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
```


![img_area](/img/posting/2019-07-09-001-wordcloud.PNG){: .post-img}


<br>
