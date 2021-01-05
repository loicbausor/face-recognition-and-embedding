# Benchmarking deep face recognition algorithm for one shot learning usage.

## Introduction 

The idea of the project was to implement (with transfer learning and some retraining) different face recognition loss functions on a pretrained network architecture, test their performance and show their particulatrites. The pictures used for (re)training the models is a subset of the [Labelled Face in the Wild Dataset](http://vis-www.cs.umass.edu/lfw/).

In a second time,  we use pretrained Facenet embeding (as ours were not enough powerful) as inputs to do an open set face recognition task the dataset used is a small personnal celebrity dataset (**all links to download are below**).

## Summary of the tasks done
1) Face detection and image cropping of the entire dataset
2) Training and benchmark of three algorithm loss for face embedding (SphereFace, CenterLoss and Triplet loss)
3) Test a pretrained face embedding for a open set face recognition problem on a the celebrity dataset.

## How to use this repo
1) **Clone the repo** :
`git clone `
2) Download the data and the pretrained architectures 
    - Put  the data in the  data folder
    - Put the models in the model folder
3) Run the notebook you want (as the two are independant)
## Useful definitions of face recognitions problems
**Face verification :** 
A face verification problem consist in checking if two input images are the same person or not  

**Face recognition :**
A face recognition problem consist in checking if the input image corresponds to one of the person in database. It can be done, for example, running multiple face verifications between the input image and the entire dataset. This method can be assimilated to _one shot learning_   

**Open set problem :** 
Open-set classification is a problem of handling unknown classes that are not contained in the training dataset, whereas traditional classifiers assume that only known classes appear in the test environment.

**One shot learning problem :**
*One-shot learning is an object categorization problem, found mostly in computer vision. Whereas most machine learning based object categorization algorithms require training on hundreds or thousands of samples/images and very large datasets, one-shot learning aims to learn information about object categories from one, or only a few, training samples/images.* ([Wikipedia](https://en.wikipedia.org/wiki/One-shot_learning))

## Main approach for face recognition problems :
Most of the face recognition use cases are open set problems, we cannot train a neural network to solve the problem. Also it can be assimilated as one shot learning problems in the sense it often uses only few training examples to classify the new ones. 

The main appraoch to solve those challenges, is to first learn *face embedings* one a closed set of identities. Then we can use those vectorial/hidden representations to compute similiraty between faces as *a face verification* problem : if the similarity between to faces is strong we can infer than the two faces are the same. To finish this approach can be extended to a  *open set classification problem* by taking the greater similarity among a different identities.

## Part 1 :  loss comparison for face embedding

In this first part, we tried to fit three different losses found in the litterature to learn face embeddings on a subset of LFW dataset. You can find our work in the **Part 1** notebook.

### Pipeline for training features

Before to present theoritically the losses used, here it is  the pipeline we did to do the training part : 
1) Face cropping using an [haar cascade classifier](https://towardsdatascience.com/computer-vision-detecting-objects-using-haar-cascade-classifier-4585472829a9) 
2) Image resizing to be fed into the network
3) Image normalizing. To do so we substracted every pixels by 128 and divided them 127.5 as done in a lot of litterature protocols (eg . [this paper](https://arxiv.org/pdf/1704.08063.pdf) )

To enhance this  pipeline we should have done face alignement and image augmentation (rotations, deformation, add noise ...).

### Loss presentation
We implemented, above a pre trainded Mobile net architecture those three losses and trained the models.
#### Triplet loss
Let A the hidden representation of a face, P the hidden representation of a face with the same identity and N the hidden representation of a face with another identity.
The triplet loss aim to minimize : 
$${\displaystyle {\mathcal {L}}\left(A,P,N\right)=\operatorname {max} \left({\|\operatorname {f} \left(A\right)-\operatorname {f} \left(P\right)\|}^{2}-{\|\operatorname {f} \left(A\right)-\operatorname {f} \left(N\right)\|}^{2}+\alpha ,0\right)}$$
Where $$\alpha$$ is a margin parameter (the highest it is the more loss is "permissive"). The idea of this loss function is to pass three examples through the neural network and (an Anchor a Postive and a Negative) instead of one to try to separate as much as possible the hidden representations of the different identities.

#### Center loss
Based on this [paper](https://ydwen.github.io/papers/WenECCV16.pdf) and helped from this [repo](https://github.com/handongfeng/MNIST-center-loss), we implemented the center face loss to learn embeddings on our dataset.

The center loss is defiend as the following :

$$L = L_S + \lambda L_C = \frac{1}{N}\sum^{N}_{i =1} log \frac{e^{W^{T}_{y_i}x_i + b_{y_i}}}{\sum_{j=1}^{p}e^{W^{T}_{j}x_i + b_{j}}} + \lambda \sum^{N}_{i =1}|| x_i - c_{y_i}||^2_2$$

Where
 - $$c_{y_i}$$ denotes the yith class center of deep features.
 - $$\lambda$$ is the tradeoff parameter between the softmax loss and the variance of the hidden represnetations


#### Sphere Face loss
Based on this [paper](https://arxiv.org/pdf/1801.07698.pdf) and helped from this [repo](https://github.com/4uiiurz1/keras-arcface), we implemented the sphere face loss to learn embeddings on our dataset.

The sphere face loss is slightly [A-softmax loss function](https://towardsdatascience.com/additive-margin-softmax-loss-am-softmax-912e11ce1c6b) described as the following : 
L = $$\frac{1}{N}\sum^{N}_{i =1} log \frac{e^{s(cos(m\theta_{y_i})}}{e^{s(cos(m\theta_{y_i}))} + \sum_{j=1, j \neq y_i} e^{s(cos(m\theta_{j})}}$$
Where
 - $$\theta_{j}$$ represents the angle between $$W_j$$ (weights of the previous layer) and the feature $$x_i$$
 - m is a multiplicative margin  penality
 - s is scale penality 

We did not understand all the geometric specifities of this loss, however it seems to force the network to separate as much as possible the embedding of different classes.

### Results
We trained for 25 epochs two versions of each network/loss (changing the losses hyperparameters). Our embeding size is of 256 neurons. We tried to vizualize our representations with a TSNE algorithm. You can find all the losses and embeding graphs in the **Part1 Notebook**. To sum up, none of our networks achieved to good results (in term of accuracy or embedings).

### Possible explanations of the "poor" results
Many things could have been enhanced to have best results.
#### Data
- We only use 10% of whole LFW dataset due to computationnal cost of the algorithm. This dataset is already considered as small for a face regonition task.
- Haar cascade is not the best face detector we could have taken. Some face cropping is failing in our training set. We should have used better models like MTCNN model.
- We did not do face alignement
- The dataset is really imbalanced, pushing the network to overspecialize  

#### Hyperparameters
- Due to the computationnal cost, we did not have tested most combinations of hyperparameters.

#### Implementation
- We may have failed at  building the embdedding models properly.

## Part 2 : Open set recognition with a pretrained facenet model

For this part, we used the celebrity dataset present in the repo. It is composed of pictures of 5 celibrities and a bunch a of pictures of unidentified people.

The idea is to take a little set of pictures of each celebrities (1,2,3 or 4) as an identification set (which can be assimilated to training set) and try to see if we can classify the identity of the other pictures based on this small sample.

To do so we use we embed each face using a pre trained model and try to see measure cosine similarity between the embedings of the test set and the one of the identification set. If the similarity is below a certain threshold we can consider that the picture have the same identity. If no pictures have came below the threshold we consider that the person is unknown.

Playing with the threshold hyper parameter and the number of pictures by identities in the identification set we achieve an accurracy arrond 98% for 4 pictures by identities. For only one picture by identity, we obtain almost 90% accuracy. You can find all the protocol, confusions matrix and results in the **Part2 Notebook**.





