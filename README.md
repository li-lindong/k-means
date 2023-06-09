# k-means
This repository contains the k-means method implemented in Pytorch.

## 1. Objective Function
The objective function of the clustering algorithm tries to find the clustering centers, so that the samples can be divided into the corresponding clustering center, and make the distance between the samples and its closest clustering center as samll as possible.

Given a set of samples ${x_1, x_2,..., x_n}$ and a positive integer $k$, this algorithm tries to find $k$ clustering centers $C_1, C_2,..., C_k$ and minimizes the objective function:

$$
E=\sum_{i=1}^k \sum_{x \in C_i}\left\|x-\mu_i\right\|_2^2
$$

where $mu_i$ is the centroid of the samples belonging to class $i$, and it can be expressed as $\mu_i=\frac{1}{\left|C_i\right|} \sum_{x \in C_i} x$.

## 2. Block Disagram
The block diagram is as follows,

<center><img src="https://github.com/li-lindong/k-means/blob/main/block%20diagram.png" width=65%></center>
