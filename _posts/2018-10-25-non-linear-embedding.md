---
layout: post
title: "Nonlinear embedding: ISOMAP, LLE, t-SNE"
subheading: "Dimensionality reduction"
thumbnail_img: "http://www.turingfinance.com/wp-content/uploads/2014/10/Dimensionality.png"
date: 2018-10-25 13:00:00 -0900
categories: ml
comments: true
---

차원 축소는 기계학습에서 중요한 부분이다. 일반적으로 공개된 데이터들은 feature 수에 비해 데이터가 적은 경우가 많다. 흔히 이야기하는 [차원의 저주](https://en.wikipedia.org/wiki/Curse_of_dimensionality)에 해당한다. 이런 상황에선 데이터에 noise가 포함될 확률이 높아지고, 학습하는데 computational cost가 높고, 학습 자체도 충분히 이루어지지 않을 수 있다. 이 때 정말 중요한 feature들만 적절히 고르거나, 만들어서 사용할 수 있다면 데이터를 representation하는 것이 훨씬 쉬워진다. Supervised, Unsupervised method가 존재하는데 이번 글에선 Unsupervised nonlinear embeding의 방법들인 ISOMAP과 LLE, t-SNE에 대해서 알아볼 것이다.

## 1. ISOMAP

<img src="/img/isomap_explain.png" />

ISOMAP은 isometric mapping을 의미하는 말로 geodesic distance를 보존하는 방향으로 학습된다. 위 이미지의 맨 위 도표처럼 데이터가 분포되어있고 선으로 이어진 것처럼 그래프가 만들어졌을 때 저차원에서의 Euclidean 거리는 적절하지 못하다. a와 c가 실제론 매우 먼 거리를 가지고 있는데 가깝다고 결론을 낼 수 있기 때문이다. 원래 데이터에서 만들어진 그래프를 따라 graph distance를 구하게 되면 좀 더 geodesic distance를 잘 반영할 수 있다.

### 1.1 Procedure

- Neighborhood graph 만들고
- Shortest path 계산하고
- MDS를 활용해 저차원의 embedding을 만든다.

먼저 Neighborhood graph를 만드는 것은 $\epsilon$-ISOMAP과 $k$-ISOMAP 두 가지 방법이 있다. 기존 데이터셋 자체만으로 만들어지는 graph이다. $\epsilon$-ISOMAP은 두 데이터 포인트가 $\epsilon$보다 작으면 연결하는 방식이고, $k$-ISOMAP은 두 데이터 포인트가 k-nearest neighbor라면 연결하는 방식이다.

그래프가 만들어지면 distance matrix, 즉 dissimilarity matrix를 만든다. 연결된 모든 데이터 포인트에 대하여 거리를 계산하기 때문에 계산 복잡도가 $O(N^2)$로 매우 높다. 이후 계산된 거리를 바탕으로 shortest path를 구하는데 널리 알려진 Dykstra algorithm같은 것을 사용한다. 이 두 단계(distance matrix, shortest path)가 계산복잡도가 높기 때문에 ISOMAP의 bottleneck에 해당한다.

이후 MDS 방법을 통해 저차원의 embedding을 구한다.

### 1.2 Code sample

<img src="/img/isomap_result.png" />

```py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)

digits = datasets.load_digits(n_class=6)
X = digits.data
y = digits.target
n_samples, n_features = X.shape
n_neighbors = 30


def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


# Isomap projection of the digits dataset
X_iso = manifold.Isomap(n_neighbors, n_components=2).fit_transform(X)
plot_embedding(X_iso, "Isomap projection")
```

- `X_iso = manifold.Isomap(n_neighbors, n_components=2).fit_transform(X)`
    + isomap 인스턴스를 생성한다. 생성할 때 2개의 파라미터를 지정해주는데 `n_neighbors`는 주변 몇 개의 데이터포인트를 고려할 것인지를 정하는 것이고, `n_components`는 임베딩할 저차원의 dimension을 의미한다.
    + 생성할 때 바로 X 데이터를 매개변수로 넣어 transform 시킨다.
- `plot_embedding(X_iso, "Isomap projection")` : plotting할 때 사용하는 코드를 함수화해놓은 것으로 다른 알고리즘에서도 동일하게 사용할 수 있다.

## 2. LLE(Locally Linear Embedding)

<img src="/img/lle_main.png" />

LLE 알고리즘은 manifold learning의 일종이다. Topology preservation이라고도 하는데 기존 데이터의 structure를 유지하는 형태로 학습된다.

### 2.1 Assumptions

- 데이터가 밀도 높게 sampling 되어있을 것
- 데이터가 smooth한 manifold에 위치

밀도 높다는 말은 neighborhood 포인트가 최소 20개 이상 여러개 있어야한다는 의미이고, smooth하게 위치한다는 것은 데이터의 분포가 급격하게 꺾이는 부분이나 특정 부분에 위치하지 않는 구멍이 있거나 하면 좋지 않다는 의미이다.

### 2.2 Procedure

- Neighborhood graph 생성(ISOMAP의 neighborhood 그래프 생성 방식과 동일)
    + $\epsilon$-neiborhood : $\epsilon$보다 작은 거리일 때 연결
    + k-nearest neighborhood : k-nearest 알고리즘으로 판단된 주변 데이터 포인트들을 이웃이라 생각하고 연결한다.
- Weight matrix construction
    + 각 데이터 포인트는 neighborhood 포인트들이 weighted sum으로 reconstruction된다. 이 neighborhood 포인트들의 weight들을 학습하는 것이고, 이웃이 아닌 포인트들의 weight는 0이다.
    + 이렇게 이웃들만 사용하기 때문에 **locally**라고 하고, 이웃 데이터 포인트들의 weight를 linear coefficients로 활용하기 때문에 **linear**라고 한다. LLE라고 이름붙여진 이유이다.
    + 각 데이터 포인트마다 그 데이터 포인트의 이웃들의 weighted sum과의 차이를 구하고, 모든 포인트에 대해서 이 차이를 합산한 것이 cost이다.

$$
E(W) = \sum_i{|x_i - \sum_i{W_{ij}x_j|^1}}
$$

- Embedding
    + 두 번째 단계에서 구해진 weight를 가지고, 아래 수식에서처럼 저차원에서의 데이터 포인트들을 계산해낸다.

$$
\Phi(W) = \sum_i{|y_i - \sum_j{W_{ij}y_j}|^2}
$$

### 2.3 Code sample

<img src="/img/lle_result.png" />

```py
clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
                                      method='standard')
X_lle = clf.fit_transform(X)
plot_embedding(X_lle, "Locally Linear Embedding of the digits")
print("Reconstruction error: %g" % clf.reconstruction_error_)
```

- `clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='standard')`
    + `n_neighbor` : 각 데이터 포인트에서 고려할 neighbor point의 개수
    + `n_components` : 임베딩할 차원 수
    + `method` : ‘standard’, ‘hessian’, ‘modified’ 등을 넣을 수 있으며 locally linear embedding에서 조금씩 내부 알고리즘을 바꿔가며 확인할 수 있다.
- `X_lle = clf.fit_transform(X)` : input X를 transform한다.
- `plot_embedding(X_lle, "Locally Linear Embedding of the digits")` : 저차원으로 임베딩한 결과물을 plotting 한다.

## 3. t-SNE

### 3.1 SNE: Stochastic Neighbor Embedding

SNE에 대해서 먼저 알아보겠다. SNE는 고차원의 데이터를 저차원으로 임베딩할 때 neighborhood structure를 유지하려한다. 각 데이터 포인트마다 Gaussian distribution을 이용해 neighborhood의 확률값을 계산하며, 이 확률값이 고차원에서나 저차원에서나 비슷하기를 바란다.

$$
p_{j|i} = {e^{-{||x_i - x_j||^2 \over 2\sigma_i^2}} \over \sum_{k \neq i}e^{-{||x_i - x_k||^2 \over 2\sigma_i^2}}}
$$

$$
q_{j|i} = {e^{-||y_i - y_j||^2} \over \sum_{k \neq i}e^{-||y_i-y_k||^2}}
$$

위 수식에서 $p$는 고차원에서 $i$라는 데이터 포인트에 대해 $j$라는 데이터 포인트가 이웃으로 뽑힐 확률값이다. 같은 의미로 $q$는 저차원에서의 확률값을 의미한다. 여기서 $p$는 기존 데이터셋을 통해서 구할 수 있는 값이지만 $q$에서 y가 미지수이기 때문에 우리는 이 y를 학습해야한다.

학습은 두 확률분포 $p, q$가 비슷해지도록 y값을 업데이트하면 된다. 두 확률분포의 차이를 측정하는 방법으로 **Kullback-Leibler divergence**를 사용하고 계산 수식은 다음과 같다.

$$
Cost = \sum_i{KL(P_i||Q_i)} = \sum_i\sum_j p_{j|i} log{p_{j|i}\over q_{j|i}}
$$

위 Cost를 우리가 학습하고자 하는 y에 대하여 gradient를 계산하면 원래는 첫 번째 수식처럼 나오지만, i와 j를 바꿨을 때의 p, q 값이 크게 다르지 않다는 점을 고려하여 수식을 좀 더 간결히 했을 때 두 번째 수식이 나온다. 이를 Symmetric SNE라고 하며 해당 gradient를 가지고 우리가 알고있는 최적화 방법을 통해 학습하면 된다.

$$
\text{(1): } {\partial C \over \partial y_i} = 2\sum_j(y_j - y_i)(p_{j|i} - q_{j|i} + p_{i|j} - q+{i|j}) \\
\text{(2): } {\partial C \over \partial y_i} = 4\sum_j(y_j - y_i)(p_{ij} - q_{ij})
$$

### 3.2 t-SNE

SNE를 활용하면 Crowding problem이 발생한다. 데이터를 저차원으로 축소했을 때 각 포인트들이 너무 밀집되어 구분이 어려운 문제이다. SNE가 가우시안 분포를 사용하기 때문에 발생하는데 가우시안 분포가 중심에서 멀어질수록 PDF 값이 급격하게 떨어지기 대문이다. 즉 z-score로 봤을 때 중심에서 5 정도 떨어진 값이나 10 떨어진 값이나 큰 차이가 없다. 실제로는 꽤 거리 차이가 있고 구분해주면 좋은데 말이다. 그래서 좀 더 꼬리가 두터운 t 분포를 활용한다. t 분포를 활용했다고 해서 t-SNE라고 불린다.

고차원에서의 확률값 p는 그대로 가우시안을 활용하고 저차원의 확률값 q만 아래 수식처럼 t 분포를 활용한다.

$$
q_{j|i} = {(1+||y_i - y_j||^2)^{-1} \over \sum_{k \neq j}(1+||y_k-y_j||^2)^{-1}}
$$

### 3.3 Code sample

<img src="/img/tsne_result.png" />

```py
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
X_tsne = tsne.fit_transform(X)
plot_embedding(X_tsne, "t-SNE embedding of the digits")
```

- `tsne = manifold.TSNE(n_components=2, init='pca')`
    + 몇 가지 파라미터를 주어서 tsne 인스턴스를 생성한다.
    + `n_components` : 몇 차원으로 임베딩할 것인지를 정한다.
    + `init` : embedding을 초기화하는 방법을 설정하는 것이고 기본값은 'random'이다. 'pca'로 설정해서 초기화하는 것이 더 안정적이다.
- `X_tsne = tsne.fit_transform(X)`
    + tsne 인스턴스를 활용해 input X를 저차원으로 임베딩한다.
- `plot_embedding(X_tsne, "t-SNE embedding of the digits")` : 이전에 사용했던 함수를 그대로 사용했다.
