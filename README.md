# <a href="[http//www.google.com](https://user.ceng.metu.edu.tr/~gcinbis/courses/Spring24/CENG796)"><p style="text-align:center">METU CENG 796 / Spring 24</p></a>

# <p style="text-align:center">Discrete Latent Variable Models</p>

## <p style="text-align:center">Topic Summary</p>


#### <p style="text-align:center">Topic Summary Authors</p>

### <p style="text-align:center">Umut Ozyurt <br> (umuttozyurt@gmail.com, umut.ozyurt@metu.edu.tr)</p>
### <p style="text-align:center">Melih Gokay Yigit <br> (gokay.yigit@metu.edu.tr)</p>


## Table of contents
1.  [Why Discrete Latent Variables?](#why-discrete-latent-variables)
2.  [Stochastic Optimization](#stochastic-optimization)
3.  [REINFORCE Method](#reinforce-method)
4.  [Variational Learning of Latent Variable Models](#variational-learning-of-latent-variable-models)
5.  [Neural Variational Inference and Learning (NVIL)](#neural-variational-inference-and-learning-nvil)
6.  [Towards Reparameterized, Continuous Relaxations](#towards-reparameterized-continuous-relaxations)
7.  [Categorical Distributions and Gumbel-Softmax](#categorical-distributions-and-gumbel-softmax)
8.  [Combinatorial, Discrete Objects: Permutations](#combinatorial-discrete-objects-permutations)
9.  [Plackett-Luce (PL) Distribution](#plackett-luce-pl-distribution)
10. [Relaxing PL Distribution to Gumbel-PL](#relaxing-pl-distribution-to-gumbel-pl)
11. [Summary and Conclusions](#summary-and-conclusions)
12. [References](#references)


## Why Discrete Latent Variables?

Discrete latent variables are hidden variables in models that take on a finite set of distinct values. They are crucial in decision-making processes and learning structures because they help capture the inherent discreteness in various types of data and systems. <br>

The most basic understanding for this question stems from the real-world data representations. One can simply observe how the “data” is represented in the examples below:


<div style="text-align: center;">
    <figure>
    <img src="figures/A-human-DNA-and-Part-of-DNA-sequence-28-29.jpg" alt="DNA Sequence">
    <figcaption>Fig 1. DNA sequence data representation https://www.researchgate.net/figure/A-human-DNA-and-Part-of-DNA-sequence-28-29_fig1_341901570 </figcaption>
    </figure>
</div>


<div style="text-align: center;">
    <figure>
    <img src="figures/Screenshot from 2024-05-14 18-41-41.png" alt="Game state">
    <figcaption>Fig 2. Game state data representation from the game named sokoban https://medium.com/deepgamingai/game-level-design-with-reinforcement-learning-52b02bb94954  </figcaption>
    </figure>
</div>


TODO: ADD GRAPH FIGURE:
<div style="text-align: center;">
    <figure>
    <img src="" alt="Graph example">
    <figcaption>Fig 3. TODO: ADD EXPLANATION AND CITATION </figcaption>
    </figure>
</div>





In addition to the DNA sequence, game state and graph representation examples above, we have many additional data representation domains (e.g. text data, images, speech and audio, molecules, geographical data, market basket items, programming codes, healthcare records, financial transactions, e-commerce clickstream data…) where the data is/has inherently discrete representations.<br>

This natural (abundant) appearance of this discreteness forces us to use discrete latent variable models since some assumptions may fail in classic latent variable models due to the discontinuity of the data. Hence, if one wants to work with the real-world data using latent variable models, they probably will confront some assumption fails and change their perception to the discrete latent variables realm.


## Stochastic Optimization
Slide 5:<br>


$$
\max_{\phi} E_{q_{\phi}(z)}[f(z)]
$$

$$
\max_{\theta, \phi} E_{q_{\phi}(z|x)} \left[ \log \frac{p_{\theta}(x, z)}{q(z|x)} \right]
$$

$$
p_{\theta}(x) = \sum_{\text{All possible values of } z} p_{\theta}(x, z) = \sum_{z \in \mathcal{Z}} \frac{q(z)}{q(z)} p_{\theta}(x, z) = E_{z \sim q(z)} \left[ \frac{p_{\theta}(x, z)}{q(z)} \right]
$$


Slide 6:<br>


$$
\max_{\phi} E_{q_{\phi}(z)}[f(z)]
$$

$$
\max_{\theta, \phi} E_{q_{\phi}(z|x)} \left[ \log \frac{p_{\theta}(x, z)}{q(z|x)} \right]
$$

$$\nabla_{\theta} E_{q(z; \phi)} \left[ \log p(z, \mathbf{x}; \theta) - \log q(z; \phi) \right] = E_{q(z; \phi)} \left[ \nabla_{\theta} \log p(z, \mathbf{x}; \theta) \right]$$



$$\approx \frac{1}{k} \sum_{k} \nabla_{\theta} \log p(z^k, \mathbf{x}; \theta)$$


## REINFORCE Method
Slide 7, 8:<br>


$$
\max_{\phi} E_{q_{\phi}(z)}[f(z)]
$$

Slide 9-14:<br>


$$E_{q_{\phi}(z)} [f(z)] = \sum_{z} q_{\phi}(z) f(z)$$

$$\frac{\partial}{\partial \phi_{i}} E_{q_{\phi}(z)} [f(z)] = \sum_{z} \frac{\partial q_{\phi}(z)}{\partial \phi_{i}} f(z) = \sum_{z} q_{\phi}(z) \frac{1}{q_{\phi}(z)} \frac{\partial q_{\phi}(z)}{\partial \phi_{i}} f(z)$$

$$= \sum_{z} q_{\phi}(z) \frac{\partial \log q_{\phi}(z)}{\partial \phi_{i}} f(z) = E_{q_{\phi}(z)} \left[ \frac{\partial \log q_{\phi}(z)}{\partial \phi_{i}} f(z) \right]$$

Slide 15, 16:<br>


$$E_{q_{\phi}(z)} [f(z)] = \sum_{z} q_{\phi}(z) f(z)$$


$$\nabla_{\phi} E_{q_{\phi}(z)} [f(z)] = E_{q_{\phi}(z)} [f(z) \nabla_{\phi} \log q_{\phi}(z)]$$

$$\nabla_{\phi} E_{q_{\phi}(z)} [f(z)] \approx \frac{1}{K} \sum_{k} f(z^k) \nabla_{\phi} \log q_{\phi}(z^k)$$




## Variational Learning of Latent Variable Models

Slide 17:<br>


$$\mathcal{L}(x; \theta, \phi) = \sum_{z} q_{\phi}(z|x) \log p(z, x; \theta) + H(q_{\phi}(z|x))$$

$$= E_{q_{\phi}(z|x)}[\log p(z, x; \theta) - \log q_{\phi}(z|x)]$$

$$E_{q_{\phi}(z|x)} [f(\phi, \theta, z, x)] = \sum_{z} q_{\phi}(z|x) f(\phi, \theta, z, x)$$


$$\nabla_{\phi} E_{q_{\phi}(z|x)} [f(\phi, \theta, z, x)] = E_{q_{\phi}(z|x)} [f(\phi, \theta, z, x) \nabla_{\phi} \log q_{\phi}(z|x) + \nabla_{\phi} f(\phi, \theta, z, x)]$$

Slide 18:<br>


$$E_{q_{\phi}(z)} [f(z)] = \sum_{z} q_{\phi}(z) f(z)$$

$$\nabla_{\phi} E_{q_{\phi}(z)} [f(z)] = E_{q_{\phi}(z)} [f(z) \nabla_{\phi} \log q_{\phi}(z)]$$

$$\nabla_{\phi} E_{q_{\phi}(z)} [f(z)] \approx \frac{1}{K} \sum_{k} f(z^k) \nabla_{\phi} \log q_{\phi}(z^k) :<br>
= f_{MC}(z^1, \cdots , z^K)$$

$$E_{z^1, \cdots , z^K \sim q_{\phi}(z)} [f_{MC}(z^1, \cdots , z^K)] = \nabla_{\phi} E_{q_{\phi}(z)} [f(z)]$$


Slide 19:<br>


$$\nabla_{\theta} E_{q} [x^2]$$

$$q_{\theta}(x) = N(\theta, 1)$$

$$E_{q} [x^2 \nabla_{\theta} \log q_{\theta}(x)] = E_{q} [x^2 (x - \theta)]$$

$$x = \theta + \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1)$$

$$\nabla_{\theta} E_{q} [x^2] = \nabla_{\theta} E_{p} [(\theta + \epsilon)^2] = E_{p} [2(\theta + \epsilon)]$$


## Neural Variational Inference and Learning (NVIL)

Slide 27:<br>


$$\mathcal{L}(x; \theta, \phi) = \sum_{z} q_{\phi}(z|x) \log p(z, x; \theta) + H(q_{\phi}(z|x))$$

$$= E_{q_{\phi}(z|x)} [\log p(z, x; \theta) - \log q_{\phi}(z|x)]$$

$$:<br>
= E_{q_{\phi}(z|x)} [f(\phi, \theta, z, x)]$$

Slide 28:<br>


$$\mathcal{L}(x; \theta, \phi, \psi, B) = E_{q_{\phi}(z|x)} [f(\phi, \theta, z, x) - h_{\psi}(x) - B]$$

$$\nabla_{\phi} \mathcal{L}(x; \theta, \phi, \psi, B) = E_{q_{\phi}(z|x)} [(f(\phi, \theta, z, x) - h_{\psi}(x) - B) \nabla_{\phi} \log q_{\phi}(z|x) + \nabla_{\phi} f(\phi, \theta, z, x)]$$




## Towards Reparameterized, Continuous Relaxations

Slide 29:<br>


$$
\max_{\phi} E_{q_{\phi}(z)}[f(z)]
$$

Slide 30:<br>


$$g = \max \{y_1, y_2, \ldots, y_n\}$$

$$F(g; \mu, \beta) = \exp \left( - \exp \left( - \frac{g - \mu}{\beta} \right) \right)$$

## Categorical Distributions and Gumbel-Softmax

Slide 31:<br>


$$\mathbf{z} = \text{onehot} \left( \arg \max_{i} (g_i + \log \pi_i) \right)$$

Slide 32:<br>


$$\mathbf{z} = \text{one\_hot} \left( \arg \max{_i} (g_i + \log \pi) \right)$$

$$\hat{\mathbf{z}} = \text{soft} \max_{i} \left( \frac{g_i + \log \pi}{\tau} \right)$$

Slide 33:<br>


$$\hat{\mathbf{z}} = {\text{soft} \max_{i}} \left( \frac{g_i + \log \pi}{\tau} \right)$$

Slide 35:<br>


$$\max_{\phi} E_{q_{\phi}(z)} [f(z)]$$

$$\max_{\phi} E_{q_{\phi}(\hat{z})} [f(\hat{z})]$$

## Combinatorial, Discrete Objects:<br>
 Permutations

Slide 36:<br>


$$\max_{\phi} E_{q_{\phi}(z)} [f(z)]$$






## Plackett-Luce (PL) Distribution

Slide 37:<br>


$$p(z_1 = i) \propto s_i$$

$$q_s(z) = \dfrac{s_{z1}}{Z} \dfrac{s_{z2}}{Z - s_{z1}} \dfrac{s_{z3}}{Z - \sum_{i=1}^{2}s_{zi}} \cdots \dfrac{s_{zk}}{Z - \sum_{i=1}^{k-1}s_{zi}}$$

where $Z = \sum_{i=1}^{k} s_i$ is the normalizing constant.



## Relaxing PL Distribution to Gumbel-PL

Slide 38:<br>



$$\tilde{s}_i = g_i + \log s_i$$











## Summary and Conclusions









## References
TODO: make the references appropriate <br>
<br>
<br>
<br>
[1]: Discrete Latent Variable Models Francesco Bartolucci,1 Silvia Pandolfi,1 and Fulvia Pennoni2 - https://www.annualreviews.org/docserver/fulltext/statistics/9/1/annurev-statistics-040220-091910.pdf?expires=1715703760&id=id&accname=ar-240193&checksum=8413D1CC7F1ED750E3BD979D48DE1B52


