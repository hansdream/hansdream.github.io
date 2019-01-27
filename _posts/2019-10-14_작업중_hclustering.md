---
layout: post
title: "[Python] Hierarchical clustering을 통한 기업 패턴 분류"(진행중)
subtitle: "비지도학습을 통한 기업 패턴 분석"
author: "MK"
comments: true
sitemap :
  changefreq : daily
  priority : 1.0
categories : [Finance]
tags: [머신러닝]
---

K-means, DBSCAN 알고리즘과 마찬가지로 label이 없는데이터의 비지도 학습이 가능한 알고리즘이다.

계층적 군집분석은 하나의 케이스가 될때까지 비슷한 군집끼리 묶어나가는 방식이다. K-means와 달리 군집수를 미리 지정하고 모델을 돌리지 않아도 된다.

다만 모델링 후 Class수를 결정해 output을 확인할 수 있다.

아래와 같은 계층구조 트리에서 원하는 수준의 가지수(Classes)에서 잘라내면 되기 때문이다.

![img_area](/img/posting/2019-01-14-001-htree.PNG){: .post-img}




http://bcho.tistory.com/1204
