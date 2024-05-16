# <a href="https://user.ceng.metu.edu.tr/~gcinbis/courses/Spring24/CENG796"><p style="text-align:center">METU CENG 796 / Spring 24</p></a>

# <p style="text-align:center">Discrete Latent Variable Models</p>

## <p style="text-align:center">Topic Summary</p>


#### <p style="text-align:center">Topic Summary Authors</p>

### <p style="text-align:center">Umut Ozyurt <br> (umuttozyurt@gmail.com, umut.ozyurt@metu.edu.tr)</p>
### <p style="text-align:center">Melih Gokay Yigit <br> (gokay.yigit@metu.edu.tr)</p>



<br>
<br>

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



<br>
<br>

## Why Discrete Latent Variables?

Discrete latent variables are hidden variables in models that take on a finite set of distinct values. They are crucial in decision-making processes and learning structures because they help capture the inherent discreteness in various types of data and systems. <br>

The most natural answer for the "Why should we use them?" question stems from the real-world data representations. One can simply observe how the “data” is represented in the examples below:


<div style="text-align: center;">
    <figure>
    <img src="figures/A-human-DNA-and-Part-of-DNA-sequence-28-29.jpg" alt="DNA Sequence">
    <figcaption>Fig 1. DNA sequence data representation https://www.researchgate.net/figure/A-human-DNA-and-Part-of-DNA-sequence-28-29_fig1_341901570 </figcaption>
    </figure>
</div>

<br>

<div style="text-align: center;">
    <figure>
    <img src="figures/Screenshot from 2024-05-14 18-41-41.png" alt="Game state">
    <figcaption>Fig 2. Game state data representation from the game named sokoban https://medium.com/deepgamingai/game-level-design-with-reinforcement-learning-52b02bb94954  </figcaption>
    </figure>
</div>

<br>

TODO: ADD GRAPH FIGURE:
<div style="text-align: center;">
    <figure>
    <img src="" alt="Graph example">
    <figcaption>Fig 3. TODO: ADD EXPLANATION AND CITATION </figcaption>
    </figure>
</div>





In addition to the DNA sequence, game state and graph representation examples above, we have many additional data domains (e.g. text data, images, speech and audio, molecules, geographical data, market basket items, programming codes, healthcare records, financial transactions, e-commerce clickstream data) where the data is/has inherently discrete representations.<br>

This natural and abundant appearance of the discreteness shifts us to use discrete latent variable models, allowing the models to capture the inner meanings/representations of the data better. Moreover, if one wants to work with the <u>real-world</u> data using latent variable models, they probably will confront some <u>assumption fails due to the discontinuoity of the data</u> and change their perception to the discrete latent variables realm.



<br>
<br>

## Stochastic Optimization

The terms "stochastic optimization" is used for the process of minimizing or maximizing an objective function in the case it involves <u>randomness</u>. This non-deterministic optimization process can be useful in many cases, specifically when the data is too large to fit into memory, or when the data is too complex to be processed in a deterministic way (which can reduce the chance of converging to a local minimum if gradient descent is used).<br>

Recap from VAE content:<br>

---

We model our data as $p_{\theta}(x)$, where $x$ is the data and $\theta$ is the parameter of the model. We introduce a latent variable $z$, and also introduce a distribution $q(z|x)$ to the model, which leads to:

$$
p_{\theta}(x) = \sum_{\text{All possible values of } z} p_{\theta}(x, z)$$
$$ = \sum_{z \in \mathcal{Z}} \frac{q(z)}{q(z)} p_{\theta}(x, z)$$
$$ = E_{z \sim q(z)} \left[ \frac{p_{\theta}(x, z)}{q(z)} \right]
$$

We can pick a $q(z)$ that is easy to sample from and outputs related values to the true posterior $p(z|x)$. One of the ways to do this is making $q$ has a parameter $\phi$, and trying to optimize it, minimizing the KL divergence between $q(z|x)$ and $p(z|x)$, which can be made by maximizing the ELBO (Evidence Lower Bound) objective function.<br>
Roughly, the objective is to maximize the following function (maximizing the ELBO):

$$
\max_{\theta, \phi} E_{q_{\phi}(z|x)} \left[ \log \frac{p_{\theta}(x, z)}{q(z|x)} \right]
$$



---
<br>

Now, we can consider the following objective:<br>

$$
\max_{\phi} E_{q_{\phi}(z)}[f(z)]
$$

Here, if we assume $z$ is <b>continuous</b>, $q$ is reparametrizable, and $f$ is differentiable, we can use the <b>reparametrization trick</b> to get the gradient of the expectation w.r.t. $\theta$ :

$$
\max_{\theta, \phi} E_{q_{\phi}(z|x)} \left[ \log \frac{p_{\theta}(x, z)}{q(z|x)} \right]
$$

We can derive:

$$\nabla_{\theta} E_{q(z; \phi)} \left[ \log p(z, \mathbf{x}; \theta) - \log q(z; \phi) \right] = E_{q(z; \phi)} \left[ \nabla_{\theta} \log p(z, \mathbf{x}; \theta) \right]$$

And we can approximate this expectation by Monte Carlo sampling:

$$\approx \frac{1}{k} \sum_{k} \nabla_{\theta} \log p(z^k, \mathbf{x}; \theta)$$


<p style="font-size:19px">But, what if the assumptions above fails? (z is not continuous).</p>
In this case, we can utilize the REINFORCE Method, explained in th next section.


<br>
<br>

## REINFORCE Method

REINFORCE is a monte carlo variation of a policy gradient method which is used to update a network's weights [[1]](#1). It was introduced by Ronald J. Williams in 1992, and the name "<b>REINFORCE</b>" is an acronym for : <br>
<b>RE</b>ward <b>I</b>ncrement = <b>N</b>on-negative <b>F</b>actor × <b>O</b>ffset <b>R</b>einforcement × <b>C</b>haracteristic <b>E</b>ligibility

It's main goal is to optimize the expected reward, but it can be used in the problems where discrete latent variables or discrete actions exist. We are going to use the REINFORCE method to optimize the following objective function:

$$
\max_{\phi} E_{q_{\phi}(z)}[f(z)]
$$

Here, our policy is $q_{\phi}(z)$, having the parameters $\phi$. With REINFORCE, we try to maximize the expected reward by updating the policy parameters $\phi$.<br>

But first, we need some mathematical tricks in the computation of the gradients. Normally we want to calculate (or estimate):

$$\frac{\partial}{\partial \phi_{i}} E_{q_{\phi}(z)} [f(z)]$$

But with this form, we cannot calculate the gradient directly (since it is infeasible to calculate the expectation with all possible z values). Furthermore, we cannot estime the gradient by sampling z values from $q_{\phi}(z)$, because the expectation $E_{q_{\phi}(z)}$ is not in the gradient. So, we need to get it out. We can use the following trick to do this:


$$\frac{\partial}{\partial \phi_{i}} E_{q_{\phi}(z)} [f(z)]$$

Expand the expectation:

$$= \frac{\partial}{\partial \phi_{i}} \sum_{z} q_{\phi}(z) f(z)$$

Take the derivative inside the sum:

$$ = \sum_{z} \frac{\partial q_{\phi}(z)}{\partial \phi_{i}} f(z)$$

Introduce $\dfrac{q_{\phi}(z)}{q_{\phi}(z)}$:

$$ = \sum_{z} q_{\phi}(z) \frac{1}{q_{\phi}(z)} \frac{\partial q_{\phi}(z)}{\partial \phi_{i}} f(z)$$

Here is the important part. We have the log-derivative rule $\dfrac{\partial \log q_{\phi}(z)}{\partial \phi_{i}} = \dfrac{1}{q_{\phi}(z)} \dfrac{\partial q_{\phi}(z)}{\partial \phi_{i}}$. So, we can rewrite the equation above as:

$$= \sum_{z} q_{\phi}(z) \frac{\partial \log q_{\phi}(z)}{\partial \phi_{i}} f(z)$$

Now, we can rewrite the equation as an expectation:

$$ = E_{q_{\phi}(z)} \left[ \frac{\partial \log q_{\phi}(z)}{\partial \phi_{i}} f(z) \right] = E_{q_{\phi}(z)} [f(z) \nabla_{\phi} \log q_{\phi}(z)]$$

By making these operations, we can now estimate the gradient by sampling z values from $q_{\phi}(z)$, taking the gradient of the log-probability of the z values, and multiplying it with the function value of z. This is the main idea of the REINFORCE method.<br>

We are able to sampke $K$ times from $q_{\phi}(z)$, and estimate the gradient as:

$$\nabla_{\phi} E_{q_{\phi}(z)} [f(z)] \approx \frac{1}{K} \sum_{k} f(z^k) \nabla_{\phi} \log q_{\phi}(z^k)$$

This form can be interpreted as a gradient update, weighted in a way which maximizes the expected reward.

TODO: add citations and more comments



<br>
<br>

## Variational Learning of Latent Variable Models


TODO: ADD MORE COMMENTS AND CITATIONS


Now, we want to work with the ELBO (Evidence Lower Bound) objective function, which is defined as:


$$\mathcal{L}(x; \theta, \phi) = \sum_{z} q_{\phi}(z|x) \log p(z, x; \theta) + H(q_{\phi}(z|x))$$
$$= E_{q_{\phi}(z|x)}[\log p(z, x; \theta) - \log q_{\phi}(z|x)]$$

As it can be observed, the $- \log q_{\phi}(z|x)$ part is also dependent on the $\phi$ parameter. So, we have to consider a function $f$ which also includes the $\phi$ parameter (also the $\theta$ and $x$). So, instead of $E_{q_{\phi}(z)}[f(z)]$, the objective function can be rewritten as:

$$E_{q_{\phi}(z|x)} [f(\phi, \theta, z, x)] = \sum_{z} q_{\phi}(z|x) f(\phi, \theta, z, x)$$

Now, we can re-write the REINFORCE formula to be used for this objective function:

$$\nabla_{\phi} E_{q_{\phi}(z|x)} [f(\phi, \theta, z, x)] = E_{q_{\phi}(z|x)} [f(\phi, \theta, z, x) \nabla_{\phi} \log q_{\phi}(z|x) + \nabla_{\phi} f(\phi, \theta, z, x)]$$

Here, this REINFORCE rule is more generic since $f$ is also dependent on the $\phi$ parameter. Now, we can construct a Monte Carlo estimation for the ELBO's gradient w.r.t. $\phi$, just as before.

<br>
<br>

We have seen that the ELBO objective function can be optimized (maximized) using the REINFORCE method even when the latent variable $z$ is discrete. But, what are the downsides of this method? The main problem is the high variance in the gradient estimates, which can lead to slow convergence. 


$$\nabla_{\phi} E_{q_{\phi}(z)} [f(z)] \approx \frac{1}{K} \sum_{k} f(z^k) \nabla_{\phi} \log q_{\phi}(z^k)$$

For the REINFORCE rule above (or the more generic version for the ELBO, which includes $\phi$, $\theta$ and $x$), the variance can be compared with the variance of the reparameterization trick.<br>


$$\nabla_{\theta} E_{q} [x^2]$$

$$q_{\theta}(x) = N(\theta, 1)$$

Here, $q_{\theta}(x)$ is a Gaussian distribution parameterized by $\theta$. Calculating the variance:

$$E_{q} [x^2 \nabla_{\theta} \log q_{\theta}(x)] = E_{q} [x^2 (x - \theta)]$$

However, in the reparameterization trick, the variance is much lower, which leads to faster convergence:

$$x = \theta + \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1)$$

$$\nabla_{\theta} E_{q} [x^2] = \nabla_{\theta} E_{p} [(\theta + \epsilon)^2] = E_{p} [2(\theta + \epsilon)]$$


But still, we cannot use the reparameterization trick for discrete latent variables. The question is: "Are there any other methods to reduce the variance in the gradient estimates for discrete latent variables?" The answer is "Yes", and we are going to discuss the "Gumbel-Softmax" method in the next sections.


<div style="text-align: center;">
    <figure>
    <img src=figures/comparison_of_techniques.png alt="technique comparisons">
    <figcaption>Fig 4. Comparison of the Score function estimator (REINFORCE), Reparametrization trick and other methods https://gabrielhuang.gitbooks.io/machine-learning/content/reparametrization-trick.html </figcaption>
    </figure>
</div>

<br>
<br>

## Neural Variational Inference and Learning (NVIL)

TODO: I SKIPPED HERE. We have to talk about the control variates if we are going to talk about NVIL. <br>

Slide 27:<br>


$$\mathcal{L}(x; \theta, \phi) = \sum_{z} q_{\phi}(z|x) \log p(z, x; \theta) + H(q_{\phi}(z|x))$$

$$= E_{q_{\phi}(z|x)} [\log p(z, x; \theta) - \log q_{\phi}(z|x)]$$

$$:<br>
= E_{q_{\phi}(z|x)} [f(\phi, \theta, z, x)]$$

Slide 28:<br>


$$\mathcal{L}(x; \theta, \phi, \psi, B) = E_{q_{\phi}(z|x)} [f(\phi, \theta, z, x) - h_{\psi}(x) - B]$$

$$\nabla_{\phi} \mathcal{L}(x; \theta, \phi, \psi, B) = E_{q_{\phi}(z|x)} [(f(\phi, \theta, z, x) - h_{\psi}(x) - B) \nabla_{\phi} \log q_{\phi}(z|x) + \nabla_{\phi} f(\phi, \theta, z, x)]$$





<br>
<br>

## Towards Reparameterized, Continuous Relaxations

The term "continuos relaxation" is a technique to approximate the discrete variables with continuous variables. Since we have discrete latent variables and cannot use the reparametrization technique directly, and we have high variance in the REINFORCE method if we do not meticulously adjust the control variates, it is a plausible choice to go with the continuous relaxation to utilize the reparametrization trick. <br>

To get a continuous relaxation, there is a distribution utilized for this purpose. It is called "Gumbel" distribution, and it is useful in representing the extreme events.<br>

The CDF of the Gumbel distribution is:

$$F(x; \mu, \beta) = e^{-e^{-(x - \mu)/\beta}}$$

Again, we want to optimize the following objective function by maximizing it:

$$
\max_{\phi} E_{q_{\phi}(z)}[f(z)]
$$


If we choose the Gumbel distribution to be standard ($\mu = 0$ and $\beta = 1$), we can write the Standard Gumbel distribution as:

$$G(x) = e^{-e^{-x}}$$


Here are the example of the Gumbel distribution PDF visualizations for a better understanding:

<div style="text-align: center;">
    <figure>
    <img src=figures/Plot-of-the-Gumbel-distribution-for-various-m-and-s-values.png alt="Gumbel distribution PDF">
    <figcaption>Fig 5. Gumbel Distribution Probability Distribution Function Visualization (beta is replaced with sigma here here) https://www.researchgate.net/figure/Plot-of-the-Gumbel-distribution-for-various-m-and-s-values_fig1_318519731 </figcaption>
    </figure>
</div>

Additional knowledge: <br>
Note that the mean is not equal to $\mu$. Actually, when the calculations are done, the mean is equal to $\mu + \beta \gamma$, where $\gamma$ is the Euler-Mascheroni constant ($\approx 0.577$). That is why when Gumbel with $\mu = 0 , \beta = 1$ has a different mean than the Gumbel with $\mu = 0 , \beta = 2$<br>



<br>
<br>


## Categorical Distributions and Gumbel-Softmax

Slide 31:<br>


$$\mathbf{z} = \text{onehot} \left( \arg \max_{i} (g_i + \log \pi_i) \right)$$

Slide 32:<br>


$$\mathbf{z} = \text{onehot} \left( \arg \max_{i} (g_i + \log \pi) \right)$$

$$\hat{\mathbf{z}} = \text{soft} \max_{i} \left( \frac{g_i + \log \pi}{\tau} \right)$$

Slide 33:<br>


$$\hat{\mathbf{z}} = {\text{soft} \max_{i}} \left( \frac{g_i + \log \pi}{\tau} \right)$$

Slide 35:<br>


$$\max_{\phi} E_{q_{\phi}(z)} [f(z)]$$

$$\max_{\phi} E_{q_{\phi}(\hat{z})} [f(\hat{z})]$$


<br>
<br>

## Combinatorial, Discrete Objects:<br>
 Permutations

Slide 36:<br>


$$\max_{\phi} E_{q_{\phi}(z)} [f(z)]$$







<br>
<br>

## Plackett-Luce (PL) Distribution

Slide 37:<br>


$$p(z_1 = i) \propto s_i$$

TODO: Fix the equation below (sum signs fails to render properly only in the github)<br>

$$
q_s(z) = \frac{s_{z1}}{Z} \frac{s_{z2}}{Z - s_{z1}} \frac{s_{z3}}{Z - \sum_{i=1}^{2}s_{zi}} \cdots \frac{s_{zk}}{Z - \sum_{i=1}^{k-1}s_{zi}}
$$

where $Z = \sum_{i=1}^{k} s_i$ is the normalizing constant.

One way to fix it is below, but it is not a good solution:<br>

![Equation](https://latex.codecogs.com/svg.latex?q_s(z)=\frac{s_{z1}}{Z}\frac{s_{z2}}{Z-s_{z1}}\frac{s_{z3}}{Z-\sum_{i=1}^{2}s_{zi}}\cdots\frac{s_{zk}}{Z-\sum_{i=1}^{k-1}s_{zi}})



where ![Equation](https://latex.codecogs.com/svg.latex?Z=\sum_{i=1}^{k}s_i)
is the normalizing constant.




<br>
<br>

## Relaxing PL Distribution to Gumbel-PL

Slide 38:<br>



$$\tilde{s}_i = g_i + \log s_i$$












<br>
<br>

## Summary and Conclusions




## References

<br>
<br>

## References
Example citation generator:<br>
https://www.scribbr.com/citation/generator/ <br> <br>
TODO: make the references appropriate <br>
Example reference usage : [[5]](#5) <br>
Example reference usage : ([Yigit, 2022](#6)) <br>
Example reference usage : ([Ozyurt et al., 2023](#2))
<br>
<br>
<br>


<a id="1">[1]</a> 
Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 8(3–4), 229–256. https://doi.org/10.1007/bf00992696



<br>

<a id="2">[2]</a> 
Reinforcement learning: An introduction. Sutton & Barto Book: Reinforcement Learning: An Introduction. (n.d.). http://incompleteideas.net/book/the-book-2nd.html 



<br>

<a id="3">[3]</a> 
Maddison, C. J., Tarlow, D., & Minka, T. (2014, October 31). A* sampling. arXiv.org. https://arxiv.org/abs/1411.0030



<br>

<a id="4">[4]</a> 
Jang, E., Gu, S., & Poole, B. (2022, July 21). Categorical Reparameterization with Gumbel-Softmax. OpenReview. https://openreview.net/forum?id=rkE3y85ee¬eId=S1LB3MLul



<br>

<a id="5">[5]</a> 
FILLER



<br>

<a id="6">[6]</a> 
FILLER



<br>

<a id="7">[7]</a> 
FILLER



<br>

<a id="8">[8]</a> 
FILLER





<br>
<br>

### Additional Notes:

* For the most of the content, slides from cs236 lecture in "Stanford University, prepared by "Stefano Ermon" and "Aditya Grover" have been utilized. <br> 

* Gpt4o is used to strengthen the text in some places, and to obtain equations (from images).