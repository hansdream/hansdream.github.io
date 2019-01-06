---
layout: post
title: "[Djagno] 장고를 활용한 웹 구동방식의 이해"
subtitle: "장고를 활용한 웹개발 방식"
author: "MK"
comments: true
sitemap :
  changefreq : daily
  priority : 1.0
tags: [Django]
---

Django는 파이썬 코드를 활용한 Web Framework이다.  

무료 오픈소스인데다가 다양한 커뮤니티가 활성화되어 많이 활용되고 있다.

Django는 문서관리시스템과 Wiki부터 SNS에 이르기까지 `다양한 종류의 웹 사이트`를 빌드하는데 사용할 수 있다. Django를 통해 개발된 대표적인 사이트는 아래와 같다.

![img_area](/img/posting/2019-01-05-001-examples.png){: .post-img}


원하는 대부분의 기능들(각종 데이터베이스, 템플릿 엔진 등)을 제공하지만, 필요하다면 다른 컴포넌트들을 사용하기 위해 `확장`될 수 있다.

또한 웹개발의 `보안` 측면에서도 유용한 기능을 제공하며, 아키텍처 확장시 유동적으로 Django의 크기를 조절 가능하여 트래픽이 급증하는 상황에 대한 대처가 가능하다.

Django는 파이썬 베이스로 리눅스, 윈도우 그리고 맥 OS X 등등 `다양한 운영체제`에서 작동할 수 있다.



### 1. Django 설치 및 프로젝트 생성
---

shell에서 pip를 활용해 설치할 수 있다.

이전에 이미 python, pip 설치는 완료되어야 한다.

```shell
(base) C:\Users\MK>pip install django
```

project라는 폴더를 미리 생성해두었다. 해당 폴더로 이동해 **Django 프로젝트를 생성** 한다.

```shell
(base) C:\Users\MK>cd projects
(base) C:\Users\MK\projects>django-admin startproject mkdjango
```

```
2019-01-05  오후 11:43    <DIR>          .
2019-01-05  오후 11:43    <DIR>          ..
2019-01-05  오후 11:43               555 manage.py
2019-01-05  오후 11:43    <DIR>          mkdjango
               1개 파일                 555 바이트
               3개 디렉터리  12,264,472,576 바이트 남음
```


### 2. Django 서버구동
---

```shell
(base) C:\Users\MK\projects\mkdjango>python manage.py runserver       
```

```
Performing system checks...

System check identified no issues (0 silenced).

You have 15 unapplied migration(s). Your project may not work properly until you apply the migrations for app(s): admin, auth, contenttypes, sessions.
Run 'python manage.py migrate' to apply them.
January 05, 2019 - 23:45:50
Django version 2.1.5, using settings 'mkdjango.settings'
Starting development server at http://127.0.0.1:8000/
Quit the server with CTRL-BREAK.
```

`http://127.0.0.1:8000` 혹은 `localhost:8000`로 접속할 수 있다.


### 3. 디렉토리 및 파일 구조
---
하나의 프로젝트를 생성하면 다음과 같은 파일들이 생성된다.

```
mkdjango/
  mkdjango/
    __init__.py
    setting.py
    urls.py
    wsgi.py
  db.sqlite3
  manage.py
```

이 밖에도 개발자들간 공통으로 명명하는 모듈들이 있다.

```
urls.py: 앱의 URL 패턴 선언
forms.py: 입력 폼 선언
behaviors.py: 모델 믹스인 위치에 대한 옵션
constants.py: 앱에 쓰이는 상수 선언
decorators.py: 데코레이터
db/: 여러 프로젝트에서 용되는 커스텀 모델이나 컴포넌트
fields.py: 폼 필드
factories.py: 테스트 데이터 팩토리 파일
helpers.py: 뷰와 모델 파일을 가볍게 하기 위해 유틸리티 함수 선언
managers.py: models.py가 너무 커질 경우 커스텀 모델 매니저가 위치
signals.py: 커스텀 시그널
viewmixins.py: 뷰 모듈과 패키지를 더 가볍게하기 위해 뷰 믹스인을 이 모듈로 이전
```


### 4. App 추가하기
---
Django에는 기능별로 App을 구분하고 생성한다.

shell에서 elections라는 App을 생성하였다.

```shell
(base) C:\Users\MK\projects\mkdjango>python manage.py startapp elections
```
프로젝트 폴더 아래, App 폴더가 생성된 것을 확인할 수 있다.
```
mkdjango/
  elections/
      __pycache__
      migrations
      __init__.py
      admin.py
      apps.py
      models.py
      tests.py
      views.py
  mkjango/
      __init__.py
      setting.py
      urls.py
      wsgi.py
  db.sqlite3
  manage.py
```

App을 생성한 다음, `mkjango/settings.py` 파일에 해당 App명을 추가한다.

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

### 5. HTML 구동방식 절차
---
특정 App에 있는 특정 함수를 실행해 HTML화면에 나타내기 위해선 어떻게 해야할까?

구조상 아래와 같은 레이어단계의 view에서 output을 정의를 해야한다.

`mkdjango > elections > views.py`

![img_area](/img/posting/2019-01-05-001-layers.png){: .post-img}

우선, App에 함수를 생성해 놓은 후

```python
from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.
def index(request):
    return render("Hello World")
```

이 함수가 실행되도록 만들어 보자!
`urls.py`를 통해 상호 호출 될 수 있도록 아래와 같은 구조로 파일을 수정한다.

![img_area](/img/posting/2019-01-05-001-layers_code.png){: .post-img}



다시 서버를 구동 후, `localhost:8000`에 접속해보면 `Hello World` 문구를 확인할 수 있다.

`mkdjango>urls.py`코드를 다시 살펴보면,

```python
urlpatterns = [
    url(r'^', include('elections.urls')),
    url(r'^admin/', admin.site.urls),
]
```

화면에 접근이 이루어 졌을때 누가 처리할 것인지를 명명한다.
`admin`이면 `admin.site.urls` 호출하게 된다.


### 6. MVC 패턴 이해하기
---
아직, DB와 탬플릿에 대한 내용은 다루지 않았지만 Django는 db.sqlite3라는 파일을 통해 `DataBase기능`을 제공한다.

나아가 App내에 `temppates을 생성`해 출력 포맷을 지정하기도 한다.

큰 단위의 프로젝트로 나아갈 수록 이러한 역할들을의 묶음을 구분해야할 필요성이 있다.

그렇지 않으면 여러 개발자간 혼선과 크고 작은 오류가 발생할 것이다.

MVC패턴은 다음과 같이 model, view, controller가 분리되는 개념을 말한다.

Django는 더 나아가 MTV를 추구한다. 표준 MVC 패턴과 비교했을 때 Django의 디자인은 Model-Template-View + Controller 라고도 부른다. Controller는 이미 프레임 워크의 일부이기 때문에 종종 생략된다.

즉, 이렇게 3가지 패턴으로 분류함으로써 효율적인 개발 생태계를 유지해나간다.

![img_area](/img/posting/2019-01-05-001-mvc.png){: .post-img}



결론적으로 전체 flow는 아래와 같이 흘러가게 된다.

![img_area](/img/posting/2019-01-05-001-djangoflow.PNG){: .post-img}

```
URLs:
분리된 뷰 함수를 작성하는 것이 각각의 리소스를 유지보수하기 훨씬 쉽다. URL mapper는 요청 URL을 기준으로 HTTP 요청을 적절한 view로 보내주기 위해 사용된다.

View:
HTTP 요청을 수신하고 HTTP 응답을 반환하는 요청 처리 함수이다. View는 Model을 통해 요청을 충족시키는 데 필요한 데이터에 접근하며 탬플릿에게 응답의 서식 설정을 맡긴다.

Models:
application의 데이터 구조를 정의하고 데이터베이스의 기록을 관리(추가, 수정, 삭제)하고 query하는 방법을 제공하는 파이썬 객체이다.

Templates:
파일의 구조나 레이아웃을 정의하고(예: HTML 페이지), 실제 내용을 보여주는 데 사용되는 플레이스홀더를 가진 텍스트 파일이다. view는 HTML 탬플릿을 이용하여 동적으로 HTML 페이지(.js, .html)를 만들고 model에서 가져온 데이터로 채운다. 탬플릿이 꼭 HTML 타입일 필요는 없다.
```


이벤트 발생에 따른 처리 주체 및 리소스들은 아래와 같다.

![img_area](/img/posting/2019-01-05-001-djangoflow2.PNG){: .post-img}

>주로 녹색네모들이 우리가 실질적으로 다루는것들이다. 다만, WSGI는 특별히 조작할 것은 없다.
>미들웨어(Middleware)라는것은 우리가 느낄수는 없지만 장고뒤에서 다양한 처리를 도와준다.
>WSGI는 웹서버와 장고를 적절하게 결합해주는 역할을 담당한다.
>urls.py파일은 정규표현식으로 구성되어 있다.


<br>

### **Reference**
---
- <https://developer.mozilla.org/ko/docs/Learn/Server-side/Django>
- <https://wikidocs.net/6606>
- <https://eunhyejung.github.io/python/2018/07/31/django-basic-concept.html>
- <https://programmers.co.kr/learn/courses/6/>

<br>
