# k-means
This repository contains the k-means method implemented in Pytorch.

## Objective Function
The objective function of the clustering algorithm tries to find the clustering centers, so that the samples can be divided into the corresponding clustering center, and make the distance between the samples and its closest clustering center as samll as possible.

$$
E=\sum_{i=1}^k \sum_{x \in C_i}\left\|x-\mu_i\right\|_2^2
$$

The block diagram is as follows,

<center><img src="https://github.com/li-lindong/k-means/blob/main/block%20diagram.png" width=65%></center>
