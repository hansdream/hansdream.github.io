---
layout: post
title: "[Django] Models 클래스 만들기"
subtitle: "장고를 활용한 웹개발 방식"
author: "MK"
comments: true
sitemap :
  changefreq : daily
  priority : 1.0
tags: [Django]
---

[이전포스팅] [Django] 장고를 활용한 웹 구동방식의 이해  [보러가기](https://mkjjo.github.io/2019/01/05/django_operation_method.html)



![img_area](/img/posting/2019-01-05-001-djangoflow2.PNG){: .post-img}

위 그림에서처럼 `models.py`를 통해 데이터를 접근, 관리할 수 있다.

Django에서 기본 제공하는 `db.sqlite3`와 연동하여 models.py에서 정의한 **models 클래스 데이터들을 DB형태로 관리** 하게 된다.


### 1. 모델 생성하기
---
이전 포스팅 예제에서 후보들을 투표할 수 있는 App `elections`를 만들었다.

후보들의 정보를 관리하기 위한 DB가 필요하므로 `Candidate`모델을 생성해본다.

클래스를 생성하고 저장하고 싶은 필드정보와 길이를 넣는다.

`mkdjango > elections > models.py`

```python
from django.db import models

# Create your models here.
class Candidate(models.Model):
    name = models.CharField(max_length=10)
    introduction = models.TextField()
    area = models.CharField(max_length=15)
    party_number = models.IntegerField(default=0)

    :# 항목을 대표하는 이름이 후보자의 name이 되도록 오버라이트
    def __str__(self):
        return self.name
```


---
### 2. DB 마이그레이션
모델클래스를 DB형태로 저장하기 위해 몇가지 셋팅이 필요하다.

`mkjango/settings.py` 파일에 해당 App명을 추가한다.

```python
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'elections' # add
]
```

shell에서 해당 폴더로 이동한 후 `db.sqlites3`의 수정일자를 확인한다.
그런 후 아래 명령을 사용해 **마이그레이션 한다.**

```
(base) C:\Users\MK\projects\mkdjango>python manage.py makemigrations
```

![img_area](/img/posting/2019-01-05-002-pycham.png){: .post-img}

`0001_initial.py`가 생성된 걸 확인할 수 있다.<br>
생성된 파일은 migration시 어떻게 DB에 정리할지를 정의해둔 파일이다.

이제 이 파일을 가지고 실제 DB에 공간을 마련한다.

```
(base) C:\Users\MK\projects\mkdjango>python manage.py migrate
```

이제 다시  `db.sqlites3`의 수정일자를 확인해보면 변경되었음을 확인할 수 있다.

DB가 모델을 받아들일 준비가 되었다.



### 3. DB에 로드하기
---
**1) admin을 이용하기**

```
(base) C:\Users\MK\projects\mkdjango>python manage.py createsuperuser
```
위 명령을 실행하여 `username`과 `password`를 지정한다.<br>
이제 웹 브라우저를 열어 `localhost:8000/admin`에 접속 후 지정한 name과 password를 입력한다.


admin에서 model파일을 불러다 Candidate를 입력하겠다는 내용이 작성되어야 한다.

`mkjango>elections>admin.py` 파일을 수정한다.

```python
from django.contrib import admin
from .models import Candidate
# Register your models here.
admin.site.register(Candidate)
```


![img_area](/img/posting/2019-01-05-002-adminlogin.PNG){: .post-img}

아래와 같이 **Candidate를 등록** 할 수 있는 폼이 생성되었다.
![img_area](/img/posting/2019-01-05-002-admin_candidate1.PNG){: .post-img}

**Add버튼을 클릭하여 후보를 등록한다.**
![img_area](/img/posting/2019-01-05-002-admin_candidate_add.PNG){: .post-img}

![img_area](/img/posting/2019-01-05-002-admin_candidate2.PNG){: .post-img}

**2) Shell 이용하기**

```
(base) C:\Users\MK\projects\mkdjango>python manage.py shell
```

```
In [1]: from elections.models import Candidate
In [2]: Candidate.objects.all()
Out[2]: <QuerySet [<Candidate: 힐러리>, <Candidate: 트럼프>]>
In [3]: new_candidate = Candidate(name="루비오")
In [4]: new_candidate.save()
Out[5]: <QuerySet [<Candidate: 힐러리>, <Candidate: Out[5]: <QuerySet [<Candidate: 힐러리>, <Candidate: 트럼프>, <Candidate: 루비 오>]>
```

```
In [6]: no1 = Candidate.objects.filter(party_number=1
    ...: )

In [7]: no1
Out[7]: <QuerySet [<Candidate: 힐러리>]>
```



### 4. 데이터 보여주기
---
DB에 저장된 Candidate를 화면에 보여주기 위해 `elections/view.py` 파일을 수정한다.


```python
from django.shortcuts import render
from django.http import HttpResponse
from .models import Candidate

# Create your views here.
def index(request):
    candidates = Candidate.objects.all() #해당 테이블의 모든 정보를 가져온다.

    str = ''
    for candidate in candidate:
        str += "<p>{} 기호{}번({})".format(candidate.name,
                                     candidate.party_number,
                                     candidate.area)
        str += candidate.introduction+"</p>"
    return HttpResponse(str)
```

### 5. 템플릿 사용하기

해당 App폴더(elections) 안에 `templates`라는 폴더를 만든다.<br>
또 다시 그 안에 `elections/templates/` App이름(elections) 폴더를 만든다. `elections/templates/elections` <br>
이제 마지막으로 그 안에 `index.html`을 생성한다.


![img_area](/img/posting/2019-01-05-002-pycham2.PNG){: .post-img}


templates폴더 안에 같은 이름의 폴더를 하나 더 만드는 이유는 아래와 같이 다양한 앱에서 index.html을 사용할 때 혼선을 막기 위해서이다.

![img_area](/img/posting/2019-01-05-002-app_indexes.PNG){: .post-img}

아래 처럼 **원하는 포맷 코드를 작성** 해둔다.

```hteml
<!-- C\Code\mysite\elections\templates\elections\index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
  <title>선거 후보</title>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" href="http://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/css/bootstrap.min.css">
  <script src="https://ajax.googleapis.com/ajax/libs/jquery/1.12.0/jquery.min.js"></script>
  <script src="http://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/js/bootstrap.min.js"></script>
</head>
<body>
<div class="container">
    <table class="table table-striped">
        <thead>
        <tr>
            <td><B>이름</B></td>
            <td><B>소개</B></td>
            <td><B>출마지역</B></td>
            <td><B>기호</B></td>
        </tr>
        </thead>
        <tbody>
        {% for candidate in candidates %}
        <tr>
            <td>{{candidate.name}}</td>
            <td>{{candidate.introduction}}</td>
            <td>{{candidate.area}}</td>
            <td>기호{{candidate.party_number}}번</td>
        </tr>
        {% endfor %}
        <tbody>
    </table>
</body>
```

이제 `elections/view.py` 파일을 다시 수정한다.

```python
from django.shortcuts import render
from .models import Candidate

# Create your views here.
def index(request):
    candidates = Candidate.objects.all()

    # DB에서 전달받을 Dict
    context = {'candidates':candidates}

    return render(request, 'elections/index.html', context)  # 템플릿 이용
```


![img_area](/img/posting/2019-01-05-002-template.PNG){: .post-img}

<br>

### **Reference**
---
- <https://programmers.co.kr/learn/courses/6/>

<br>
