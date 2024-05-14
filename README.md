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





## REINFORCE Method





## Variational Learning of Latent Variable Models





## Neural Variational Inference and Learning (NVIL)





## Towards Reparameterized, Continuous Relaxations





## Categorical Distributions and Gumbel-Softmax





## Combinatorial, Discrete Objects: Permutations





## Plackett-Luce (PL) Distribution





## Relaxing PL Distribution to Gumbel-PL













## Summary and Conclusions









## References
TODO: make the references appropriate <br>
<br>
<br>
<br>
[1]: Discrete Latent Variable Models Francesco Bartolucci,1 Silvia Pandolfi,1 and Fulvia Pennoni2 - https://www.annualreviews.org/docserver/fulltext/statistics/9/1/annurev-statistics-040220-091910.pdf?expires=1715703760&id=id&accname=ar-240193&checksum=8413D1CC7F1ED750E3BD979D48DE1B52


