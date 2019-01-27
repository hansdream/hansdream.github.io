---
layout: post
title: "[Python] 학습 데이터 가져오기"
subtitle: "학습데이터 가져오기"
author: "MK"
comments: true
sitemap :
  changefreq : daily
  priority : 1.0
categories : [Python]
tags: [파이썬기초]
---


모델링에 주로 사용되는 학습데이터가 있다.

<br>

### 1. mnist
---
`mnist`는 **0~9까지의 숫자 모음** 으로 비교적 많은 이미지 학습 예제에서 다뤄지고 있다.

MNIST 학습데이터는 28x28 사이즈에 총 784개의 픽셀로 이루어진 흑백이미지이다. 각 픽셀마다 0~255의 값으로 표현되는데, 0에 가까울수록 검정, 255에 가까울수록 하얀색이다. 데이터 맨 앞에는 label이 붙어 모델링에 정답지로 활용된다.

![img_area](/img/posting/2019-01-09-001-mnist.jpg){: .post-img}

**데이터 가져오기**

[input_data.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/input_data.py)
이 파일을 다운 받은 후 아래 코드를 통해 데이터를 가져올 수 있다.

```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
```

다운로드된 데이터는 55,000개의 학습 데이터(`mnist.train`), 10,000개의 테스트 데이터(`mnist.text`), 그리고 5,000개의 검증 데이터(`mnist.validation`) 이렇게 세 부분으로 나뉜다.
라벨과 이미지데이터를 별도로 갖게 되는데 학습 이미지는 `mnist.train.images`이며, 학습 라벨은 `mnist.train.labels`이다.

<br><br>

### 2. Iris
---
아이리스라는 **꽃의 종을 분류하기 위한 데이터셋** 이다.
아래 코드를 통해 얻는 `X`에는 **꽃받침 길이, 꽃받침 너비, 꽃잎 길이, 꽃잎 너비** 가 변수로 들어가 있으며 `Y`가 종자를 판별하는 label이다.

![img_area](/img/posting/2019-01-09-001-iris.png){: .post-img}


**데이터 가져오기**

```python
from sklearn import datasets
# import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target
```

<br>
