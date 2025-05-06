# cs229-problem-set-3-deep-learning-unsupervised-learning-solved
**TO GET THIS SOLUTION VISIT:** [CS229 Problem Set #3-Deep Learning & Unsupervised learning Solved](https://www.ankitcodinghub.com/product/cs229-problem-set-3-deep-learning-unsupervised-learning-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;96212&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;2&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (2 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;CS229 Problem Set #3-Deep Learning \u0026amp; Unsupervised learning Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (2 votes)    </div>
    </div>
<div class="page" title="Page 1">
<div class="layoutArea">
<div class="column">
&nbsp;

Notes: (1) These questions require thought, but do not require long answers. Please be as concise as possible. (2) If you have a question about this homework, we encourage you to post your question on our Piazza forum, at http://piazza.com/stanford/fall2018/cs229. (3) If you missed the first lecture or are unfamiliar with the collaboration or honor code policy, please read the policy on Handout #1 (available from the course website) before starting work. (4) For the coding problems, you may not use any libraries except those defined in the provided environment.yml file. In particular, ML-specific libraries such as scikit-learn are not permitted. (5) To account for late days, the due date listed on Gradescope is Nov 03 at 11:59 pm. If you submit after Oct 31, you will begin consuming your late days. If you wish to submit on time, submit before Oct 31 at 11:59 pm.

All students must submit an electronic PDF version of the written questions. We highly recom- mend typesetting your solutions via LATEX. If you are scanning your document by cell phone, please check the Piazza forum for recommended scanning apps and best practices. All students must also submit a zip file of their source code to Gradescope, which should be created using the make zip.py script. In order to pass the auto-grader tests, you should make sure to (1) restrict yourself to only using libraries included in the environment.yml file, and (2) make sure your code runs without errors when running p05 percept.py and p06 spam.py. Your submission will be evaluated by the auto-grader using a private test set.

</div>
</div>
</div>
<div class="page" title="Page 2">
<div class="section">
<div class="layoutArea">
<div class="column">
CS229 Problem Set #3 2 1. [20 points] A Simple Neural Network

Let X = {x(1), · · · , x(m)} be a dataset of m samples with 2 features, i.e x(i) ∈ R2. The samples are classified into 2 categories with labels y(i) ∈ {0, 1}. A scatter plot of the dataset is shown in Figure 1:

</div>
</div>
<div class="section">
<div class="layoutArea">
<div class="column">
4.0 3.5 3.0 2.5 2.0 1.5 1.0 0.5 0.0

</div>
</div>
<div class="layoutArea">
<div class="column">
0.0 0.5 1.0

</div>
<div class="column">
1.5 2.0 x1

</div>
<div class="column">
2.5 3.0 3.5 4.0

</div>
</div>
</div>
<div class="layoutArea">
<div class="column">
Figure 1: Plot of dataset X.

The examples in class 1 are marked as as “×” and examples in class 0 are marked as “◦”. We want to perform binary classification using a simple neural network with the architecture shown in Figure 2:

</div>
</div>
<div class="layoutArea">
<div class="column">
Inputs Hidden layer

Output

</div>
</div>
<div class="layoutArea">
<div class="column">
Figure 2: Architecture for our simple neural network.

Denote the two features x1 and x2, the three neurons in the hidden layer h1,h2, and h3, and

the output neuron as o. Let the weight from xi to hj be w[1] for i ∈ {1,2},j ∈ {1,2,3}, and the i,j

weight from hj to o be w[2]. Finally, denote the intercept weight for hj as w[1] , and the intercept j 0,j

weight for o as w[2]. For the loss function, we’ll use average squared loss instead of the usual 0

negative log-likelihood:

1 􏰈m

(o(i) − y(i))2, where o(i) is the result of the output neuron for example i.

</div>
</div>
<div class="layoutArea">
<div class="column">
l = m

</div>
<div class="column">
i=1

</div>
</div>
</div>
<div class="section">
<div class="layoutArea">
<div class="column">
x2

</div>
</div>
</div>
</div>
<div class="page" title="Page 3">
<div class="layoutArea">
<div class="column">
CS229 Problem Set #3 3 (a) [5 points] Suppose we use the sigmoid function as the activation function for h1,h2,h3 and

o. What is the gradient descent update to w[1] , assuming we use a learning rate of α? Your 1,2

answer should be written in terms of x(i), o(i), y(i), and the weights.

<ol start="2">
<li>(b) &nbsp;[10 points] Now, suppose instead of using the sigmoid function for the activation function
for h1,h2,h3 and o, we instead used the step function f(x), defined as

􏰆1, x ≥ 0 0, x &lt; 0

Is it possible to have a set of weights that allow the neural network to classify this dataset with 100% accuracy?

If it is possible, please provide a set of weights that enable 100% accuracy by completing optimal step weights within src/p01 nn.py and explain your reasoning for those weights in your PDF.

If it is not possible, please explain your reasoning in your PDF. (There is no need to modify optimal step weights if it is not possible.)

Hint: There are three sides to a triangle, and there are three neurons in the hidden layer.
</li>
<li>(c) &nbsp;[10 points] Let the activation functions for h1,h2,h3 be the linear function f(x) = x and the activation function for o be the same step function as before.

Is it possible to have a set of weights that allow the neural network to classify this dataset with 100% accuracy?

If it is possible, please provide a set of weights that enable 100% accuracy by complet- ing optimal linear weights within src/p01 nn.py and explain your reasoning for those weights in your PDF.

If it is not possible, please explain your reasoning in your PDF. (There is no need to modify optimal linear weights if it is not possible.)
</li>
</ol>
</div>
</div>
<div class="layoutArea">
<div class="column">
f(x) =

</div>
</div>
</div>
<div class="page" title="Page 4">
<div class="layoutArea">
<div class="column">
CS229 Problem Set #3 4 2. [15 points] KL divergence and Maximum Likelihood

The Kullback-Leibler (KL) divergence is a measure of how much one probability distribution is different from a second one. It is a concept that originated in Information Theory, but has made its way into several other fields, including Statistics, Machine Learning, Information Geometry, and many more. In Machine Learning, the KL divergence plays a crucial role, connecting various concepts that might otherwise seem unrelated.

In this problem, we will introduce KL divergence over discrete distributions, practice some simple manipulations, and see its connection to Maximum Likelihood Estimation.

The KL divergence between two discrete-valued distributions P(X),Q(X) over the outcome space X is defined as follows1:

DKL(P∥Q) = 􏰈 P(x)log P(x) x∈X Q(x)

For notational convenience, we assume P(x) &gt; 0,∀x. (One other standard thing to do is to adopt the convention that “0log0 = 0.”) Sometimes, we also write the KL divergence more explicitly as DKL(P||Q) = DKL(P(X)||Q(X)).

Background on Information Theory

Before we dive deeper, we give a brief (optional) Information Theoretic background on KL divergence. While this introduction is not necessary to answer the assignment question, it may help you better understand and appreciate why we study KL divergence, and how Information Theory can be relevant to Machine Learning.

We start with the entropy H(P) of a probability distribution P(X), which is defined as H(P) = − 􏰈 P(x)logP(x).

x∈X

Intuitively, entropy measures how dispersed a probability distribution is. For example, a uni- form distribution is considered to have very high entropy (i.e. a lot of uncertainty), whereas a distribution that assigns all its mass on a single point is considered to have zero entropy (i.e. no uncertainty). Notably, it can be shown that among continuous distributions over R, the Gaussian distribution N(μ,σ2) has the highest entropy (highest uncertainty) among all possible distributions that have the given mean μ and variance σ2.

To further solidify our intuition, we present motivation from communication theory. Suppose we want to communicate from a source to a destination, and our messages are always (a sequence of) discrete symbols over space X (for example, X could be letters {a,b,…,z}). We want to construct an encoding scheme for our symbols in the form of sequences of binary bits that are transmitted over the channel. Further, suppose that in the long run the frequency of occurrence of symbols follow a probability distribution P(X). This means, in the long run, the fraction of times the symbol x gets transmitted is P(x).

A common desire is to construct an encoding scheme such that the average number of bits per symbol transmitted remains as small as possible. Intuitively, this means we want very frequent symbols to be assigned to a bit pattern having a small number of bits. Likewise, because we are

1If P and Q are densities for continuous-valued random variables, then the sum is replaced by an integral, and everything stated in this problem works fine as well. But for the sake of simplicity, in this problem we’ll just work with this form of KL divergence for probability mass functions/discrete-valued distributions.

</div>
</div>
</div>
<div class="page" title="Page 5">
<div class="layoutArea">
<div class="column">
CS229 Problem Set #3 5

interested in reducing the average number of bits per symbol in the long term, it is tolerable for infrequent words to be assigned to bit patterns having a large number of bits, since their low frequency has little effect on the long term average. The encoding scheme can be as complex as we desire, for example, a single bit could possibly represent a long sequence of multiple symbols (if that specific pattern of symbols is very common). The entropy of a probability distribution P(X) is its optimal bit rate, i.e., the lowest average bits per message that can possibly be achieved if the symbols x ∈ X occur according to P(X). It does not specifically tell us how to construct that optimal encoding scheme. It only tells us that no encoding can possibly give us a lower long term bits per message than H(P).

To see a concrete example, suppose our messages have a vocabulary of K = 32 symbols, and each symbol has an equal probability of transmission in the long term (i.e, uniform probability distribution). An encoding scheme that would work well for this scenario would be to have log2 K bits per symbol, and assign each symbol some unique combination of the log2 K bits. In fact, it turns out that this is the most efficient encoding one can come up with for the uniform distribution scenario.

It may have occurred to you by now that the long term average number of bits per message depends only on the frequency of occurrence of symbols. The encoding scheme of scenario A can in theory be reused in scenario B with a different set of symbols (assume equal vocabulary size for simplicity), with the same long term efficiency, as long as the symbols of scenario B follow the same probability distribution as the symbols of scenario A. It might also have occured to you, that reusing the encoding scheme designed to be optimal for scenario A, for messages in scenario B having a different probability of symbols, will always be suboptimal for scenario B. To be clear, we do not need know what the specific optimal schemes are in either scenarios. As long as we know the distributions of their symbols, we can say that the optimal scheme designed for scenario A will be suboptimal for scenario B if the distributions are different.

Concretely, if we reuse the optimal scheme designed for a scenario having symbol distribution Q(X), into a scenario that has symbol distribution P(X), the long term average number of bits per symbol achieved is called the cross entropy, denoted by H(P,Q):

H(P,Q) = − 􏰈 P(x)logQ(x). x∈X

To recap, the entropy H(P) is the best possible long term average bits per message (optimal) that can be achived under a symbol distribution P(X) by using an encoding scheme (possibly unknown)specificallydesignedforP(X). ThecrossentropyH(P,Q)isthelongtermaveragebits per message (suboptimal) that results under a symbol distribution P(X), by reusing an encoding scheme (possibly unknown) designed to be optimal for a scenario with symbol distribution Q(X).

Now, KL divergence is the penalty we pay, as measured in average number of bits, for using the optimal scheme for Q(X), under the scenario where symbols are actually distributed as P(X). It is straightforward to see this

DKL(P,Q) = 􏰈 P(x)log P(x) x∈X Q(x)

= 􏰈 P (x) log P (x) − 􏰈 P (x) log Q(x) x∈X x∈X

=H(P,Q)−H(P). (differenceinaveragenumberofbits.)

</div>
</div>
</div>
<div class="page" title="Page 6">
<div class="layoutArea">
<div class="column">
CS229 Problem Set #3 6

If the cross entropy between P and Q is zero (and hence DKL(P||Q) = 0) then it necessarily means P = Q. In Machine Learning, it is a common task to find a distribution Q that is “close” to another distribution P. To achieve this, we use DKL(Q||P) to be the loss function to be optimized. As we will see in this question below, Maximum Likelihood Estimation, which is a commonly used optimization objective, turns out to be equivalent minimizing KL divergence between the training data (i.e. the empirical distribution over the data) and the model.

Now, we get back to showing some simple properties of KL divergence. (a) [5 points] Nonnegativity. Prove the following:

</div>
</div>
<div class="layoutArea">
<div class="column">
and

</div>
<div class="column">
∀P,Q DKL(P∥Q)≥0

DKL(P∥Q)=0 ifandonlyifP =Q.

</div>
</div>
<div class="layoutArea">
<div class="column">
[Hint: You may use the following result, called Jensen’s inequality. If f is a convex function, and X is a random variable, then E[f(X)] ≥ f(E[X]). Moreover, if f is strictly convex (f is convex if its Hessian satisfies H ≥ 0; it is strictly convex if H &gt; 0; for instance f(x) = −logx is strictly convex), then E[f(X)] = f(E[X]) implies that X = E[X] with probability 1; i.e., X is actually a constant.]

(b) [5 points] Chain rule for KL divergence. The KL divergence between 2 conditional distributions P (X |Y ), Q(X |Y ) is defined as follows:

􏰎 P(x|y)􏰏 DKL(P(X|Y)∥Q(X|Y))=􏰈P(y) 􏰈P(x|y)logQ(x|y)

yx

This can be thought of as the expected KL divergence between the corresponding conditional distributions on x (that is, between P(X|Y = y) and Q(X|Y = y)), where the expectation is taken over the random y.

Prove the following chain rule for KL divergence:

DKL(P (X, Y )∥Q(X, Y )) = DKL(P (X)∥Q(X)) + DKL(P (Y |X)∥Q(Y |X)).

(c) [5 points] KL and maximum likelihood. Consider a density estimation problem, and suppose we are given a training set {x(i); i = 1, . . . , m}. Let the empirical distribution be Pˆ(x) = m1 􏰁mi=1 1{x(i) = x}. (Pˆ is just the uniform distribution over the training set; i.e., sampling from the empirical distribution is the same as picking a random example from the training set.)

Suppose we have some family of distributions Pθ parameterized by θ. (If you like, think of Pθ(x) as an alternative notation for P(x;θ).) Prove that finding the maximum likelihood estimate for the parameter θ is equivalent to finding Pθ with minimal KL divergence from Pˆ. I.e. prove:

m

arg min DKL(Pˆ∥Pθ) = arg max 􏰈 log Pθ(x(i))

θθ

i=1

</div>
</div>
</div>
<div class="page" title="Page 7">
<div class="layoutArea">
<div class="column">
CS229 Problem Set #3 7

Remark. Consider the relationship between parts (b-c) and multi-variate Bernoulli Naive Bayes parameter estimation. In the Naive Bayes model we assumed Pθ is of the following form: Pθ(x,y) = p(y)􏰍ni=1 p(xi|y). By the chain rule for KL divergence, we therefore have:

n

DKL(Pˆ∥Pθ) = DKL(Pˆ(y)∥p(y)) + 􏰈 DKL(Pˆ(xi|y)∥p(xi|y)).

i=1

This shows that finding the maximum likelihood/minimum KL-divergence estimate of the parameters decomposes into 2n + 1 independent optimization problems: One for the class priors p(y), and one for each of the conditional distributions p(xi|y) for each feature xi given each of the two possible labels for y. Specifically, finding the maximum likelihood estimates for each of these problems individually results in also maximizing the likelihood of the joint distribution. (If you know what Bayesian networks are, a similar remark applies to parameter estimation for them.)

</div>
</div>
</div>
<div class="page" title="Page 8">
<div class="layoutArea">
<div class="column">
CS229 Problem Set #3 8 3. [25 points] KL Divergence, Fisher Information, and the Natural Gradient

As seen before, the Kullback-Leibler divergence between two distributions is an asymmetric measure of how different two distributions are. Consider two distributions over the same space given by densities p(x) and q(x). The KL divergence between two continuous distributions, q and p is defined as,

􏰒∞ p(x) DKL(p||q) = p(x) log q(x) dx

−∞

􏰒∞ 􏰒∞

= p(x) log p(x)dx − p(x) log q(x)dx −∞ −∞

= Ex∼p(x)[log p(x)] − Ex∼p(x)[log q(x)].

A nice property of KL divergence is that it invariant to parametrization. This means, KL divergence evaluates to the same value no matter how we parametrize the distributions P and Q. For e.g, if P and Q are in the exponential family, the KL divergence between them is the same whether we are using natural parameters, or canonical parameters, or any arbitrary reparametrization.

Now we consider the problem of fitting model parameters using gradient descent (or stochastic gradient descent). As seen previously, fitting model parameters using Maximum Likelihood is equivalent to minimizing the KL divergence between the data and the model. While KL divergence is invariant to parametrization, the gradient w.r.t the model parameters (i.e, direction of steepest descent) is not invariant to parametrization. To see its implication, suppose we are at a particular value of parameters (either randomly initialized, or mid-way through the optimization process). The value of the parameters correspond to some probability distribution (and in case of regression, a conditional probability distribution). If we follow the direction of steepest descent from the current parameter, take a small step along that direction to a new parameter, we end up with a new distribution corresponding to the new parameters. The non- invariance to reparametrization means, a step of fixed size in the parameter space could end up in a distribution that could either be extremely far away in DKL from the previous distribution, or on the other hand not move very much at all w.r.t DKL from the previous distributions.

This is where the natural gradient comes into picture. It is best introduced in contrast with the usual gradient descent. In the usual gradient descent, we first choose the direction by calculating the gradient of the MLE objective w.r.t the parameters, and then move a magnitude of step size (where size is measured in the parameter space) along that direction. Whereas in natural gradi- ent, we first choose a divergence amount by which we would like to move, in the DKL sense. This effectively gives us a perimeter around the current parameters (of some arbitrary shape), such that points along this perimeter correspond to distributions which are at an equal DKL-distance away from the current parameter. Among the set of all distributions along this perimeter, we move to the distribution that maximizes the objective (i.e minimize DKL between data and itself) the most. This approach makes the optimization process invariant to parametrization. That means, even if we chose a new arbitrary reparametrization, by starting from a particular distribution, we always descend down the same sequence of distributions towards the optimum.

In the rest of this problem, we will construct and derive the natural gradient update rule. For that, we will break down the process into smaller sub-problems, and give you hints to answer them. Along the way, we will encounter important statistical concepts such as the score function and Fisher Information (which play a prominant role in Statistical Learning Theory as well). Finally, we will see how this new natural gradient based optimization is actually equivalent to Newton’s method for Generalized Linear Models.

</div>
</div>
</div>
<div class="page" title="Page 9">
<div class="layoutArea">
<div class="column">
CS229 Problem Set #3 9 Let the distribution of a random variable Y parameterized by θ ∈ Rn be p(y; θ).

<ol>
<li>(a) &nbsp;[3 points] Score function
The score function associated with p(y; θ) is defined as ∇θ log p(y; θ), which signifies the

sensitivity of the likelihood function with respect to the parameters. Note that the score

function is actually a vector since it’s the gradient of a scalar quantity with respect to the

vector θ.

Recall that Ey∼p(y)[g(y)] = 􏰇 ∞ p(y)g(y)dy. Using this fact, show that the expected value −∞

of the score is 0, i.e.

Ey∼p(y;θ)[∇θ′ logp(y;θ′)|θ′=θ]=0
</li>
<li>(b) &nbsp;[2 points] Fisher Information

Let us now introduce a quantity known as the Fisher information. It is defined as the covariance matrix of the score function,

I(θ)=Covy∼p(y;θ)[∇θ′ logp(y;θ′)|θ′=θ]

Intuitively, the Fisher information represents the amount of information that a random variable Y carries about a parameter θ of interest. When the parameter of interest is a vector (as in our case, since θ ∈ Rn), this information becomes a matrix. Show that the Fisher information can equivalently be given by

I(θ)=Ey∼p(y;θ)[∇θ′ logp(y;θ′)∇θ′ logp(y;θ′)⊤|θ′=θ]

Note that the Fisher Information is a function of the parameter. The parameter of the Fisher information is both a) the parameter value at which the score function is evaluated, and b) the parameter of the distribution with respect to which the expectation and variance is calculated.
</li>
<li>(c) &nbsp;[5 points] Fisher Information (alternate form)

It turns out that the Fisher Information can not only be defined as the covariance of the

score function, but in most situations it can also be represented as the expected negative Hessian of the log-likelihood.

Show that Ey∼p(y;θ)[−∇2θ′ log p(y; θ′)|θ′=θ] = I(θ).

Remark. The Hessian represents the curvature of a function at a point. This shows that the expected curvature of the log-likelihood function is also equal to the Fisher information matrix. If the curvature of the log-likelihood at a parameter is very steep (i.e, Fisher Information is very high), this generally means you need fewer number of data samples to a estimate that parameter well (assuming data was generated from the distribution with those parameters), and vice versa. The Fisher information matrix associated with a statistical model parameterized by θ is extremely important in determining how a model behaves as a function of the number of training set examples.
</li>
<li>(d) &nbsp;[5 points] Approximating DKL with Fisher Information

As we explained at the start of this problem, we are interested in the set of all distributions that are at a small fixed DKL distance away from the current distribution. In order to calculate DKL between p(y; θ) and p(y; θ + d), where d ∈ Rn is a small magnitude “delta” vector, we approximate it using the Fisher Information at θ. Eventually d will be the natural gradient update we will add to θ. To approximate the KL-divergence with Fisher</li>
</ol>
</div>
</div>
</div>
<div class="page" title="Page 10">
<div class="layoutArea">
<div class="column">
CS229 Problem Set #3 10 Infomration, we will start with the Taylor Series expansion of DKL and see that the Fisher

Information pops up in the expansion.

Show that DKL(pθ||pθ+d) ≈ 21dT I(θ)d.

Hint: Start with the Taylor Series expansion of DKL(pθ||pθ ̃) where θ is a constant and θ ̃ is a variable. Later set θ ̃ = θ + d. Recall that the Taylor Series allows us to approximate a scalar function f(θ ̃) near θ by:

f(θ ̃)≈f(θ)+(θ ̃−θ)T∇θ′f(θ′)|θ′=θ + 21(θ ̃−θ)T 􏰋∇2θ′f(θ′)|θ′=θ􏰌(θ ̃−θ)

(e) [8 points] Natural Gradient

Now we move on to calculating the natural gradient. Recall that we want to maximize the log-likelihood by moving only by a fixed DKL distance from the current position. In the previous sub-question we came up with a way to approximate DKL distance with Fisher Information. Now we will set up the constrained optimization problem that will yield the natural gradient update d. Let the log-likelihood objective be l(θ) = logp(y;θ). Let the DKL distance we want to move by, be some small positive constant c. The natural gradient update d∗ is

d∗ =argmaxl(θ+d) subjectto DKL(pθ||pθ+d)=c (1) d

First we note that we can use Taylor approximation on l(θ + d) ≈ l(θ) + dT ∇θ′ l(θ′)|θ′=θ. Also note that we calculated the Taylor approximation DKL(pθ||pθ+d) in the previous sub- problem. We shall substitute both these approximations into the above constrainted opti- mization problem.

In order to solve this constrained optimization problem, we employ the method of Lagrange multipliers. If you are familiar with Lagrange multipliers, you can proceed directly to solve for d∗. If you are not familiar with Lagrange multipliers, here is a simplified introduction. (You may also refer to a slightly more comprehensive introduction in the Convex Opti- mization section notes, but for the purposes of this problem, the simplified introduction provided here should suffice).

Consider the following constrained optimization problem

d∗ = argmaxf(d) subject to g(d) = c

d

The function f is the objective function and g is the constraint. We instead optimize the Lagrangian L(d, λ), which is defined as

L(d, λ) = f (d) − λ[g(d) − c]

with respect to both d and λ. Here λ ∈ R+ is called the Lagrange multiplier. In order to

optimize the above, we construct the following system of equations:

∇dL(d, λ) = 0, (a) ∇λL(d, λ) = 0. (b)

</div>
</div>
</div>
<div class="page" title="Page 11">
<div class="layoutArea">
<div class="column">
CS229 Problem Set #3 11

So we have two equations (a and b above) with two unknowns (d and λ), which can be sometimes be solved analytically (in our case, we can).

The following steps guide you through solving the constrained optimization problem:

<ul>
<li>Construct the Lagrangian for the constrained optimization problem (1) with the Taylor approximations substituted in for both the objective and the constraint.</li>
<li>Then construct the system of linear equations (like (a) and (b)) from the Lagrangian you obtained.</li>
<li>From (a), come up with an expression for d that involves λ.

At this stage we have already found the “direction” of the natural gradient d, since λ is only a positive scaling constant. For most practical purposes, the solution we obtain here is sufficient. This is because we almost always include a learning rate hyperparameter in our optimization algorithms, or perform some kind of a line search for algorithmic stability. This can make the exact calculation of λ less critical. Let’s call this expression d ̃ (involving λ) as the unscaled natural gradient. Clearly state what is d ̃ as a function of λ.

The remaining steps are to figure out the value of the scaling constant λ along the direction of d, for completeness.</li>
<li>Plugin that expression for d into (b). Now we have an equation that has λ but not d. Come up with an expression for λ that does not include d.</li>
<li>Plug that expression for λ (without d) back into (a). Now we have an equation that has d but not λ. Come up with an expression for d that does not include λ.
The expression fof d obtained this way will be the desired natural gradient update d∗. Clearly state and highlight your final expression for d∗. This expression cannot include λ.
</li>
</ul>
(f) [2 points] Relation to Newton’s Method

After going through all these steps to calculate the natural gradient, you might wonder if this is something used in practice. We will now see that the familiar Newton’s method that we studied earlier, when applied to Generalized Linear Models, is equivalent to natural gradient on Generalized Linear Models. While the two methods (Netwon’s and natural gradient) agree on GLMs, in general they need not be equivalent.

Show that the direction of update of Newton’s method, and the direction of natural gradient, are exactly the same for Generalized Linear Models. You may want to recall and cite the results you derived in problem set 1 question 4 (Convexity of GLMs). For the natural

̃

gradient, it is sufficient to use d, the unscaled natural gradient.

</div>
</div>
</div>
<div class="page" title="Page 12">
<div class="layoutArea">
<div class="column">
CS229 Problem Set #3 12 4. [30 points] Semi-supervised EM

Expectation Maximization (EM) is a classical algorithm for unsupervised learning (i.e., learning with hidden or latent variables). In this problem we will explore one of the ways in which EM algorithm can be adapted to the semi-supervised setting, where we have some labelled examples along with unlabelled examples.

In the standard unsupervised setting, we have m ∈ N unlabelled examples {x(1),…,x(m)}. We wish to learn the parameters of p(x,z;θ) from the data, but z(i)’s are not observed. The classical EM algorithm is designed for this very purpose, where we maximize the intractable p(x;θ) indirectly by iteratively performing the E-step and M-step, each time maximizing a tractable lower bound of p(x;θ). Our objective can be concretely written as:

m

lunsup(θ) = 􏰈 log p(x(i); θ)

i=1 m

= 􏰈 log 􏰈 p(x(i), z(i); θ) i=1 z(i)

Now, we will attempt to construct an extension of EM to the semi-supervised setting. Let us suppose we have an additional m ̃ ∈ N labelled examples {(x(1),z(1)),…,(x(m ̃),z(m ̃))} where both x and z are observed. We want to simultaneously maximize the marginal likelihood of the parameters using the unlabelled examples, and full likelihood of the parameters using the labelled examples, by optimizing their weighted sum (with some hyperparameter α). More concretely, our semi-supervised objective lsemi-sup(θ) can be written as:

m ̃ 􏰈

(i) (i) lsup(θ)= logp(x ̃ ,z ̃ ;θ)

i=1

lsemi-sup(θ) = lunsup(θ) + αlsup(θ)

We can derive the EM steps for the semi-supervised setting using the same approach and steps as before. You are strongly encouraged to show to yourself (no need to include in the write-up) that we end up with:

</div>
</div>
<div class="layoutArea">
<div class="column">
E-step (semi-supervised)

For each i ∈ {1,…,m}, set

M-step (semi-supervised)

</div>
<div class="column">
Q(t)(z(i)) := p(z(i)|x(i); θ(t)) i

</div>
</div>
<div class="layoutArea">
<div class="column">
θ

</div>
<div class="column">
(t+1)

</div>
<div class="column">
:= arg max

θ

</div>
<div class="column">
(t) Qi

</div>
<div class="column">
(z

</div>
<div class="column">
(i)

</div>
<div class="column">
) log

</div>
<div class="column">
+ α

</div>
<div class="column">
􏰈 (i) log p(x ̃

i=1

</div>
<div class="column">
(i) , z ̃

</div>
<div class="column">
; θ)

</div>
</div>
<div class="layoutArea">
<div class="column">
􏰐m􏰎 􏰈 􏰈

i=1 z(i)

</div>
<div class="column">
p(x(i), z(i); θ) (t)

</div>
<div class="column">
􏰏 􏰎m ̃

</div>
<div class="column">
􏰏􏰑

</div>
</div>
<div class="layoutArea">
<div class="column">
Qi (z(i))

</div>
</div>
<div class="layoutArea">
<div class="column">
(a) [5 points] Convergence. First we will show that this algorithm eventually converges. In order to prove this, it is sufficient to show that our semi-supervised objective lsemi-sup(θ) monotonically increases with each iteration of E and M step. Specifically, let θ(t) be the parameters obtained at the end of t EM-steps. Show that lsemi-sup(θ(t+1)) ≥ lsemi-sup(θ(t)).

</div>
</div>
</div>
<div class="page" title="Page 13">
<div class="layoutArea">
<div class="column">
CS229 Problem Set #3 13 Semi-supervised GMM

Now we will revisit the Gaussian Mixture Model (GMM), to apply our semi-supervised EM al- gorithm. Let us consider a scenario where data is generated from k ∈ N Gaussian distributions, with unknown means μj ∈ Rd and covariances Σj ∈ Sd+ where j ∈ {1,…,k}. We have m data points x(i) ∈ Rd,i ∈ {1,…,m}, and each data point has a corresponding latent (hidden/un- known) variable z(i) ∈ {1, . . . , k} indicating which distribution x(i) belongs to. Specifically, z(i) ∼ Multinomial(φ), such that 􏰁kj=1 φj = 1 and φj ≥ 0 for all j, and x(i)|z(i) ∼ N (μz(i) , Σz(i) ) i.i.d. So, μ, Σ, and φ are the model parameters.

We also have an additional m ̃ data points x ̃(i) ∈ Rd, i ∈ {1, . . . , m ̃ }, and an associated observed

In summary we have m + m ̃ examples, of which m are unlabelled data points x’s with unobserved

Our task now will be to apply the semi-supervised EM algorithm to GMMs in order to leverage the additional m ̃ labelled examples, and come up with semi-supervised E-step and M-step update rules specific to GMMs. Whenever required, you can cite the lecture notes for derivations and steps.

<ol start="2">
<li>(b) &nbsp;[5 points] Semi-supervised E-Step. Clearly state which are all the latent variables that need to be re-estimated in the E-step. Derive the E-step to re-estimate all the stated latent variables. Your final E-step expression must only involve x,z,μ,Σ,φ and universal constants.</li>
<li>(c) &nbsp;[5 points] Semi-supervised M-Step. Clearly state which are all the parameters that need to be re-estimated in the M-step. Derive the M-step to re-estimate all the stated parameters. Specifically, derive closed form expressions for the parameter update rules for μ(t+1), Σ(t+1) and φ(t+1) based on the semi-supervised objective.</li>
<li>(d) &nbsp;[5 points] [Coding Problem] Classical (Unsupervised) EM Implementation. For this sub-question, we are only going to consider the m unlabelled examples. Follow the instructions in src/p03 gmm.py to implement the traditional EM algorithm, and run it on the unlabelled data-set until convergence.
Run three trials and use the provided plotting function to construct a scatter plot of the resulting assignments to clusters (one plot for each trial). Your plot should indicate clus- ter assignments with colors they got assigned to (i.e., the cluster which had the highest probability in the final E-step).

Note: You only need to submit the three plots in your write-up. Your code will not be autograded.
</li>
<li>(e) &nbsp;[7 points] [Coding Problem] Semi-supervised EM Implementation. Now we will consider both the labelled and unlabelled examples (a total of m + m ̃ ), with 5 labelled examples per cluster. We have provided starter code for splitting the dataset into a ma- trices x of labelled examples and x tilde of unlabelled examples. Add to your code in src/p03 gmm.py to implement the modified EM algorithm, and run it on the dataset until convergence.
Create a plot for each trial, as done in the previous sub-question.
</li>
</ol>
</div>
</div>
<div class="layoutArea">
<div class="column">
variable z ̃ ∈ {1, . . . , k} indicating the distribution x ̃

constants (in contrast to z(i) which are unknown random variables). As before, we assume

(i) (i)

x ̃ |z ̃ ∼ N (μz ̃(i) , Σz ̃(i) ) i.i.d.

</div>
</div>
<div class="layoutArea">
<div class="column">
(i) (i)

belongs to. Note that z ̃ are known

</div>
</div>
<div class="layoutArea">
<div class="column">
(i) (i)

with corresponding observed labels z ̃ . The traditional

</div>
</div>
<div class="layoutArea">
<div class="column">
z’s, and m ̃ are labelled data points x ̃

EM algorithm is designed to take only the m unlabelled examples as input, and learn the model parameters μ, Σ, and φ.

</div>
</div>
</div>
<div class="page" title="Page 14">
<div class="layoutArea">
<div class="column">
CS229 Problem Set #3 14 Note: You only need to submit the three plots in your write-up. Your code will not be

autograded.

(f) [3 points] Comparison of Unsupervised and Semi-supervised EM. Briefly describe

the differences you saw in unsupervised vs. semi-supervised EM for each of the following: i. Number of iterations taken to converge.

ii. Stability (i.e., how much did assignments change with different random initializations?) iii. Overall quality of assignments.

Note: The dataset was sampled from a mixture of three low-variance Gaussian distribu- tions, and a fourth, high-variance Gaussian distribution. This should be useful in deter- mining the overall quality of the assignments that were found by the two algorithms.

</div>
</div>
</div>
<div class="page" title="Page 15">
<div class="layoutArea">
<div class="column">
CS229 Problem Set #3 15 5. [20 points] K-means for compression

In this problem, we will apply the K-means algorithm to lossy image compression, by reducing the number of colors used in an image.

We will be using the files data/peppers-small.tiff and data/peppers-large.tiff.

The peppers-large.tiff file contains a 512×512 image of peppers represented in 24-bit color. This means that, for each of the 262144 pixels in the image, there are three 8-bit numbers (each ranging from 0 to 255) that represent the red, green, and blue intensity values for that pixel. The straightforward representation of this image therefore takes about 262144 × 3 = 786432 bytes (a byte being 8 bits). To compress the image, we will use K-means to reduce the image to k = 16 colors. More specifically, each pixel in the image is considered a point in the three-dimensional (r, g, b)-space. To compress the image, we will cluster these points in color-space into 16 clusters, and replace each pixel with the closest cluster centroid.

Follow the instructions below. Be warned that some of these operations can take a while (several minutes even on a fast computer)!

(a) [15 points] [Coding Problem] K-Means Compression Implementation. From the data directory, open an interactive Python prompt, and type

<pre>       from matplotlib.image import imread; import matplotlib.pyplot as plt;
</pre>
and run A = imread(’peppers-large.tiff’). Now, A is a “three dimensional matrix,” and A[:,:,0], A[:,:,1] and A[:,:,2] are 512×512 arrays that respectively contain the red, green, and blue values for each pixel. Enter plt.imshow(A); plt.show() to display the image.

Since the large image has 262144 pixels and would take a while to cluster, we will instead run vector quantization on a smaller image. Repeat (a) with peppers-small.tiff. Treating each pixel’s (r,g,b) values as an element of R3, run K-means2 with 16 clusters on the pixel data from this smaller image, iterating (preferably) to convergence, but in no case for less than 30 iterations. For initialization, set each cluster centroid to the (r,g,b)-values of a randomly chosen pixel in the image.

Take the matrix A from peppers-large.tiff, and replace each pixel’s (r, g, b) values with the value of the closest cluster centroid. Display the new image, and compare it visually to the original image. Include in your write-up all your code and a copy of your compressed image.

(b) [5 points] Compression Factor. If we represent the image with these reduced (16) colors, by (approximately) what factor have we compressed the image?

</div>
</div>
<div class="layoutArea">
<div class="column">
2Please implement K-means yourself, rather than using built-in functions.

</div>
</div>
</div>
