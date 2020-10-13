---
layout: post
title: "Debiasing a facial detection system using VAEs: detailed overview"
---

Author: [Siddhant Kandge](www.linkedin.com/in/siddhant-kandge-967a46192)




This blog post is a detailed description and overview of a research paper on **mitigating bias in the machine learning-based system published at AIES’19, January 27–28, 2019.** It also includes an overview of Autoencoders and Variational autoencoder which I referred from a blog post by
JEREMY JORDAN on autoencoder and VAEs. Link to references, [here](http://introtodeeplearning.com/AAAI_MitigatingAlgorithmicBias.pdf). I also referred the introduction to deep   learning, by MIT videos and PPT.

#### What is a biased model:
Sometimes machine learning models or algorithms result in being biased on some of the factors. i.e. some times model learns only that features which are over-represented in data and ignore rare feature from data and result in being biased toward that over-represented features.

You must have heard of Amazon attempt to build a [resume filtering tool](https://in.reuters.com/article/amazon-com-jobs-automation/insight-amazon-scraps-secret-ai-recruiting-tool-that-showed-bias-against-women-idINKCN1MK0AH) that ends up being biased against women. Also in 2018, [Amazon’s Facial Recognition](https://www.aclu.org/blog/privacy-technology/surveillance-technologies/amazons-face-recognition-falsely-matched-28) Software ends up being racial biased. It Matches 28 U.S. Congresspeople with Criminal Mugshots. Many similar cases happened due to bias in ML systems.
Theses bias problems may arise due to the dataset on which a model is trained or the algorithm behind the approach to solving a particular problem and [many more].(https://techcrunch.com/2016/12/10/5-unexpected-sources-of-bias-in-artificial-intelligence/)
There are many processing techniques commonly used to mitigate the bias problem in ML predictive model. Those are grouped into 3 groups:

1. **Pre-processing:** 
This is the mitigation approach that occurs before the creation of the model. In this technique, the dataset is transformed such that underlying bias present in data should be removed before modeling. i.e we simply remove sensitive attribute or features which create bias from data and then feed that newly transformed data to a model. But, this approach sometimes doesn’t work due to the presence of correlated attributes in data which will level the effect of removing sensitive attributes.
E.g., if we remove ethnicity attribute from the dataset, then also an individual’s zip code can inadvertently encode partial info of the likelihood of certain ethnicity.

2. **In-processing:**
This technique can be considered as a modification to the traditional training process or algorithm to address bias during model training takes place. E.g., the Resample dataset during training such that features that are over-represented in data should be dropped and the probability of selecting features with rare representation should be increased and the dataset is augmented. This augmented dataset is then fed to a model.

3. **Post-processing:**
In this technique, the processing of the dataset is done after training a model. The adversarial debiasing is an example of post-processing which makes use of GANs for this purpose.

### Bias in Facial Detection system:

In this blog, we are going to look at an approach to mitigate the bias problem in facial detection tasks as stated in a paper. For our particular task, we are only addressing the issue of racial and gender bias in facial detection systems.

In the dataset given for facial detection, we may be given a dataset with different faces and we may not know an exact distribution of faces in terms of different features like race, gender, etc. Due to this, our dataset may end up being biased on particular instances of these features that are over-represented in the dataset.

The one solution to this bias issue is to learn an underlying latent structure present in data using **generative models** and use this learned **latent variables**/representation to adaptively resample the dataset during training. This is the in-process approach to mitigate bias in a dataset. This solution can be implemented with the help of **Generative modeling.**

Note: The Generative models itself can be biased and there are ways to solve this problem. But, in this blog, we are not focusing on it.


#### Let’s start from basic:
What are the latent variables?

**latent variables** are variables that are not directly observed but are rather inferred(through a mathematical model) from other variables that are observed/learned (directly measured).

Mathematical models that aim to explain observed variables in terms of latent variables are called **latent variable models**. A latent variable model (LVM) is a probabilistic graphical model (PGM) of observed variables that includes latent variables.

#### What is generative modeling?

Generative modeling is a powerful way of learning any kind of data distribution in an unsupervised manner. In unsupervised learning, only data is given and labels are not. Latent representations are the essence of deep generative models.

So, in our task of facial detection, to uncover data we will learn latent variables or latent space which gives underlying info about data in a compressed version(through dimensionality reduction) and that can be used to train an unbiased classification model.

There are two main classes of models in latent variable models.

1. Auto-Encoders and Variational Auto-Encoders
2. Generative Adversarial Networks(GANs)

#### What are encoders and decoders?
An encoder is a series of neural networks such as CNN, DNN, etc. It takes an input(E.g image) and passes it through a series of a neural network to downscale it and outputs an encoding which is a compressed version of input which is called a latent variable. I.e we simply reduce the number of nodes in a successive hidden layer to get a compressed version of the input.

Encoders give underlying info about the dataset in a more compressed way(through dimensionality reduction) as it is easy to process.

Where decoder is also a series of neural networks but instead of lowering a dimension of input, it will upscale input by increasing the number of nodes in a successive hidden layer to match the input dimension.

## Autoencoders:
 An autoencoder is a simple generative model and a type of artificial neural network used to learn efficient data codings in an unsupervised manner and thus can be used for generative modeling and debiasing tasks.

<center>
<img src="https://miro.medium.com/max/875/1*4Q6iQ9b7IQfkmgMzGj5ygw.png">
</center>

So, autoencoder is a combination of an encoder and decoder structure where to an encoder (in our task)we pass our original image from a dataset that generates lower-dimensionality feature representation, a latent feature(z in fig) and learns a mapping from input x to latent variable z. This latent variable will contain a single value to describe to each latent attributes such as **skin tone, gender,** etc.

Then decoder takes this latent feature vector as an input and learns a mapping back from latent z and tries to reconstruct an original image by upscaling z to train a model.

Here(In the facial detection task), we convert data(images) into a lower-dimensional feature vector as it is easy to process and we get a more compressed version rather than a large high dimensional matrix.

The **bottleneck hidden layer** is a key feature in the structure of autoencoder as this bottleneck layer forces the network to learn compressed latent representation.

Without this hidden bottleneck layer, our network will simply memorize the input values by passing them through the network but not in compressed form.

### Loss Function:

The network structure mention above is often trained by minimizing reconstruction error L(x,x^) which is a difference between the original input image and reconstructed image output by decoder.
Here, L(x,x^) is nothing but **mean squared error** (MSE) and it is defined as the mean of the squared difference between our network output and therefore the ground truth. The MSE between output image I¯ and ground truth image I i’ll be defined as

<center>
<img src="https://miro.medium.com/max/606/1*Pg18wYXrcrBBH3s7hMpD_w.png">
</center>

Thereby penalizing a network consistent with the reconstruction error, the model can learn the foremost important attributes of the input data and how to best reconstruct the original input image from an “encoded” state(z). Ideally, this encoding will **learn and describe the latent attributes(such as skin tone, gender, hair color, smile, etc) of the input data.**

Let’s talk about Variational Autoencoders:
#### What is meant by variational?

In VAE, variational is referred to as optimization via [variational inference](https://medium.com/@jonathan_hui/machine-learning-variational-inference-273d8e6480bb). In short, it is a method to approximate maximum likelihood when the probability density is complicated. In **variational inferencing**, Given the observation X, we build a probability model q for latent variables z, i.e. q ≈p(z|X) where we make an assumption for prior(p) on how the q should look like.

That is, instead of outputting a latent variable containing a single value for each attribute directly, the encoder will output probability distribution for each latent attribute depending on prior.

Why there is a need for this?

In traditional autoencoders, we feed images to a network we will get the same output as long as the weights are the same. This is a deterministic encoding that allows us to reproduce input as best as possible.
But what if we want to generate a more smooth representation of latent space and use this generate a new image and sample a new image that is similar to input data just like generators in GANs. For this purpose, this Variational part is added.

### Variational autoencoder:

<center>
<img src="https://miro.medium.com/max/875/1*3S2-o92z0Ue3vbjhXe-VVw.png">
</center>
That is, In VAEs instead of deterministic bottleneck layer (z), a stochastic sampling operation is done. i.e instead of learning latent variable directly for each variable we **learn a mean and standard deviation that parameterized probability distribution for each of this latent variable.**

Here. an encoder computes a probability distribution q(z|x), while a decoder is doing a reverse inference, is computing a new probability distribution p(x|z).

Each time encoder will output a probability distribution on latent attributes and we will randomly sample latent variable(z) from that distribution which will be given as input to the decoder. Then the decoder will try to reconstruct an image that should similar to the input for any sample from the latent state distribution. The following image may help you to understand better.

<!--center>
<img src="https://miro.medium.com/max/875/1*GhZ9231C_nxmZfVhMxlx2Q.png">
</center-->

### Loss Function:

Since we’ve introduced this probabilistic aspect, a loss function gets slightly change. Our loss function will now contain two terms one is a reconstruction error and also the second is a regularization term.

<center>
<img src="https://miro.medium.com/max/875/1*h4m3HRLKBi9QX2WR1poHzw.png">
</center>

The reconstruction loss is analogous to an autoencoder’s loss, but here regularization term is added to make sure that our learned distribution q(z|x) should be almost like true prior distribution p(z) and it also prevents a model to not overfit on certain parts of latent space.

What prior do?:
1. It will help to distribute encodings evenly around the center of the latent space.
2. Penalize(punish) the network when it tries to “cheat” by clustering points in specific regions (ie. memorizing the data).

The common choice for prior may be a **Normal Gaussian distribution**. i.e by putting prior, we simply put a constraint on an output of encoder that, to not go far away from a typical Gaussian distribution.

when we accompany Gaussian/ normal distribution, then normally regularization term is formulated as KL divergence between two distribution. it’s nothing but a measure of the difference between two probability distributions which is stated as follows.As a convention, $$mAP$$ is expressed as a percent value.

<center>
<img src="https://miro.medium.com/max/486/1*txvf3csUJMb7xcDlw4ewUA.png">
</center>

Here, LKL(μ,σ) = D(q(z|x)||p(z)) as we would like to attenuate difference between q(z|x) and p(z) and KL divergence will do it for us.

Now, Let’s talk a little bit about the use of KL- divergence in VAE:
It’s quite some technical stuff so one can skip this part if not interested. So, KL divergence is especially wont to measure the difference between two probability distributions over the same variable x. Specifically, the Kullback-Leibler (KL) divergence of q(x) from p(x), denoted DKL(p(x), q(x)), is a measure of the knowledge lost when q(x) is employed to approximate p(x).
Let p(x)=N(μ1,σ1) and q(x)=N(μ2,σ2). Typically p(x) represents the “true” distribution of data, observations, or a precisely calculated theoretical distribution. The measure q(x) typically represents a theory, model, description, or approximation of p(x).
Therefore KL divergence between p and q is defined as:
KL(p,q)=−∫p(x)logq(x)dx + ∫p(x)logp(x)dx.
After computing these integrals we will get out KL loss as follows, the complete is [here](https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians)
KLloss=log(σ2/σ1)+(σ¹²+(μ1−μ2)²)/2*σ²² −1/2
But, in VAE we assume our prior is to normal unit gaussian i.e μ2=0 and σ2=1. If we put these values within the above equation then we get our loss function as
KLloss=−1/2 * (log(σ1)−σ1−μ¹²+1)
Although the KL divergence measures the “distance” between two distributions, it’s not a distance measure. this is often because the KL divergence isn’t a metric measure. it’s not symmetric: the KL from p(x) to q(x) is usually not an equivalent to the KL from q(x) to P(x). Furthermore, it needn’t satisfy triangular inequality. Nevertheless, DKL(P||Q) is a non-negative measure. DKL(P||Q) ≥ 0 and DKL(P||Q) = 0 if and only if P = Q.
So, by minimizing this KL loss we will learn latent representation more accurately.

Let’s train our network. But there is one question,

can we train our VAE network now?

The answer is NO!!. As we can see in our structure of VAE, we have latent variable z which not deterministic as in autoencoders. Therefore we cannot train our network. so, Now what is a solution then?
A solution to this is a **Reparametrizing sampling layer**.

### Reparametrizing sampling layer:

Since we cannot backpropagate gradient through the sampling layer as here z is not deterministic, instead z is a probabilistic result of the stochastic sampling operation(i.e each time it gets randomly sampled).
Therefore, instead of drawing z directly from the natural distribution which parameterizes by **mu** and **sigma**, we can do, is to consider sampled latent vector z as
<center>
<img src="https://miro.medium.com/max/614/1*q4A5sLIFDnBO92GlUmGFPg.png">
</center>

where mu and sigma are fixed vectors and sigma is scaled by epsilon which is a random constant drawn randomly from a normal distribution. So, this can allow us to backpropagates through a network and to updates the weights of the network.
The graphical representation will be,

<center>
<img src="https://miro.medium.com/max/875/1*8dHRvOx5tLMcost0CBdHhA.png">
</center>

Note: In the image above z ~q(z|x) and not p(z|x) according to my way of implementation.

By using this trick we can train our VAE network to learn a latent representation and to accurately reconstruct an input.
Here, we complete all the basics requirements needed for our task.
Let’s now discuss the DB-VAE model as proposed in the paper which is a modified version of VAE, made particularly for our task of debiasing.

## Debiasing variational autoencoders:
As mention at the beginning of the article, we are getting to use an adaptive resampling of training data to mitigate bias. So, therefore we’ve to slightly change VAE architecture.

Now, we’ll use the overall idea behind the VAE architecture to create a model named DB-VAE, to mitigate (potentially) unknown biases present within the training data.

In this section, we’ll undergo the algorithm presented within the paper for the adaptive resampling of the training data supported the latent structure learned by our DB-VAE model. By dropping over-represented regions of the latent space consistent with their frequency of occurrence, we increase the probability of choosing rarer data for training.

This is done adaptively (means during training itself)as the latent variables themselves are being learned during training. Thus, our debiasing approach accounts for the entire distribution of the underlying features within the training data.

#### ALGORITHM:

STEP1: VAE network is employed to find out underlying features of the training dataset (in this case images of faces) in an unbiased and unsupervised manner.

STEP2: From this learned latent distribution, we estimate that the probability distribution of every learned latent variable.
Certain distribution of these variables could also be over-represented in our dataset (like skin color, pose), and some instances with lower probability (like faces with dark skin, glasses, etc)fall kind of on the tail of this distribution.
so if our dataset has many images of faces with certain skin color then selecting a picture with those feature during training is extremely high and this cause generating an unwanted bias and the likelihood of choosing the image with rare features during sampling could also be very low

STEP3: Then use this inferred distribution to adaptively re-sample data during training.
Specifically, we’ll alter the probability that a given image is employed during training supported how often its latent features appear within the dataset. So, faces with rarer features (like dark skin, sunglasses, or hats) should become more likely to be sampled during training, while the sampling probability for faces with features that are over-represented within the training dataset should decrease (relative to uniform sampling across the training data).

STEP4: this may be used to generate a more balanced and more fair dataset which will be fed to a network to end in an unbiased classifier.
<center>
<img src="https://miro.medium.com/max/875/1*Q-WWoUKKwd_fm_3RqqlqWA.png">
<p>New proposed model architecture for debiasing tasks.</p>
</center>

In our task of facial detection, we are getting to apply our DB-VAE to a supervised classification problem. Importantly, Now the encoder portion within the DB-VAE architecture also outputs one supervised variable, zo, like the category prediction — face or not face. **Usually, VAEs haven’t trained to output any supervised variables (such as a category prediction)! this is often another key distinction between the DB-VAE and a standard VAE.**
we are only getting to learn the latent representation of faces, as that is what we’re ultimately debiasing against, although we are training a model on a binary classification problem. We’ll get to make sure that, for faces, our DB-VAE model learns both a representation of the unsupervised latent variables, captured by the distribution q(z|x), and outputs a supervised class prediction zo, but, for negative examples, it should only output a category prediction zo.
Therefore, the loss function for our task is getting slightly changed.

### DB-VAE loss function:

The form of the loss will depend upon whether the input image includes a face in it or not. Because images of non-faces shouldn’t be used to learn a latent space or variable. Latent representation should depend only on images with faces.

For face images, our loss function will have two components:

1. **VAE loss** (LVAE): consists of the latent loss(KLloss)and the reconstruction loss.
2. **Classification loss** (Ly(y,y^)): standard cross-entropy loss for a binary classification problem.

while, for images of **non-faces**, our loss function is simply the classification loss.
We can write one expression for the loss by defining an indicator variable that tells which training data are images of faces (If(y)=1 ) and which are images of non-faces (If(y)=0). Using this, we obtain:

**Ltotal=Ly(y,y^)+If(y)[LVAE]**

By minimizing this loss function and by resampling data during training will give us an unbiased facial detection system.

Great, Here we obtain a strong model architecture and algorithm to get rid of an unwanted bias from our dataset. during this article, I considered this algorithm specific to the facial detection task as mention in PAPER but, you’ll be able to apply this algorithm to the other task which has this algorithmic bias problem.

To check that model isn’t biased on dataset or algorithm is extremely important in ML and DL tasks. And to make sure this there are some ways and one among those ways is explained during this blog.
This blog is more on concept and overview of the research paper. It covers all theoretical concepts about algorithms and basic concepts require to know it. In my next blog, I will be able to explain the implementation of this algorithm from scratch.

Please give your responses within the comments and suggestions are always welcome.

Thank You….
