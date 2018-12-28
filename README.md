# Machine Learning in Python from scratch 
All the work done by me as a part of the excellent Stanford University's Machine Learning Course on Coursera + A Vectorized Python implementation resembling **as closely as possible** to both provided and personally-completed code in the octave/matlab. 
The course is taught by [Andrew Ng](https://en.wikipedia.org/wiki/Andrew_Ng) a genius and an excellent popularizer, which is a rare combination.

Discover my blog about [Common Machine Learning Problems](https://www.linkedin.com/pulse/common-machine-learning-algorithms-hamed-zitoun/). 

## Python Implementation
Since the course uses Octave/Matlab in programming assignments, I reimplemented [all the assignments in Python]( https://github.com/hzitoun/coursera_machine_learning_course/tree/master/algorithms_in_python) using only [NumPy](http://www.numpy.org/), [SciPy](https://www.scipy.org/), and [Matplotlib](https://matplotlib.org/).
After that, I've converted each assignment to a [Jupyter Notebooks](https://github.com/hzitoun/coursera_machine_learning_matlab_python/tree/master/jupyter_notebooks).
## Supervised Learning
Given a set of labeled observations, find a function f which can be used to assign a class or value to unseen observations. Predictions should be similar to real labels.
### Regression
In a regression problem, we are instead trying to predict results within a **continuous output**, meaning that we are trying to map input variables to some continuous function.
#### 1.  **Linear regression** with one variable to predict proﬁts for a food truck 
- 🐍 [Demo | Linear Regression Notebook](https://github.com/hzitoun/coursera_machine_learning_matlab_python/blob/master/jupyter_notebooks/week_2/ex1.ipynb) 
- ▶️ [Demo | Linear Regression Matlab](https://github.com/hzitoun/coursera_machine_learning_matlab_python/tree/master/algorithms_in_matlab/week_2/ex1/ex1.m)
<p align="center">
    <img src ="./figures/1_linear_regression.png" alt="Linear regression with one variable"/>
</p>

#### 2.  **Regularized Linear regression** with multiple variables to predict the prices of houses 
- 🐍 [Demo | Linear Regression with multiple variables Notebook](https://github.com/hzitoun/coursera_machine_learning_matlab_python/blob/master/jupyter_notebooks/week_2/ex1MultiFeatures.ipynb)
- ▶️ [Demo | Linear Regression with multiple variables Matlab](https://github.com/hzitoun/coursera_machine_learning_matlab_python/blob/master/algorithms_in_matlab/week_2/ex1/ex1_multi.m)
<p align="center">
    <img src ="./figures/1_linear_regression_3d.png" alt="Regularized Linear regression with multiple variables"/>
</p>

### Classification 

In a classification problem, we are trying to predict results in a **discrete output**. In other words, we are trying to map input variables into discrete categories.

#### 3. Regularized logistic regression to predict whether microchips passes quality assurance (QA)

- 🐍 [Demo | Regularized Logistic Regression Notebook](https://github.com/hzitoun/coursera_machine_learning_matlab_python/blob/master/jupyter_notebooks/week_3/ex2_reg.ipynb)
- ▶️ [Demo | Regularized Logistic Regression Matlab](https://github.com/hzitoun/coursera_machine_learning_matlab_python/blob/master/algorithms_in_matlab/week_3/ex2/ex2_reg.m)

<p align="center">
    <img src ="./figures/2_logistic_regression.png" alt="Regularized logistic regression"/>
</p>

#### 4.  **Multi-class Logistic regression** to recognize handwritten digits 
- 🐍 [Demo | Multi-class Logistic regression Notebook](https://github.com/hzitoun/coursera_machine_learning_matlab_python/blob/master/jupyter_notebooks/week_4/ex3.ipynb)
- ▶️ [Demo | Multi-class Logistic regression Matlab](https://github.com/hzitoun/coursera_machine_learning_matlab_python/blob/master/algorithms_in_matlab/week_4/ex3/ex3.m)
<p align="center">
   <img src ="./figures/3_one_vs_all_classification.png" alt="Multi-class Logistic regression" />
</p>

#### 5.  **Neural Networks** (MLP) to recognize handwritten digits 
- 🐍 [Demo | Neural Networks Notebook Part I](https://github.com/hzitoun/coursera_machine_learning_matlab_python/blob/master/jupyter_notebooks/week_4/ex3_nn.ipynb), [Demo | Neural Networks Notebook Part II](https://github.com/hzitoun/coursera_machine_learning_matlab_python/blob/master/jupyter_notebooks/week_5/ex4.ipynb)
- ▶️ [Demo | Neural Networks Matlab Part I](https://github.com/hzitoun/coursera_machine_learning_matlab_python/blob/master/algorithms_in_matlab/week_4/ex3/ex3_nn.m), [Demo | Neural NetworksMatlab Part II](https://github.com/hzitoun/coursera_machine_learning_matlab_python/blob/master/algorithms_in_matlab/week_5/ex4/ex4.m)
<p align="center">
    <img src ="./figures/4_viz_nn.png" alt="Neural Networks"/>
</p>

#### 6.  **Support Vector Machines SVM** ( with and without Gaussian Kernels)

- 🐍 [Demo | Support Vector Machines Notebook](https://github.com/hzitoun/coursera_machine_learning_matlab_python/blob/master/jupyter_notebooks/week_7/ex6.ipynb)
- ▶️ [Demo | Support Vector Machines Matlab](https://github.com/hzitoun/coursera_machine_learning_matlab_python/blob/master/algorithms_in_matlab/week_7/ex6/ex6.m)

<p align="center">
    <img src ="./figures/6_svms.png" />
</p>    

- 🐍 [Demo | SVM for Spam Notebook](https://github.com/hzitoun/coursera_machine_learning_matlab_python/blob/master/jupyter_notebooks/week_7/ex6_spam.ipynb)
- ▶️ [Demo | SVM for Spam Matlab](https://github.com/hzitoun/coursera_machine_learning_matlab_python/blob/master/algorithms_in_matlab/week_7/ex6/ex6_spam.m)

<p align="center">
   <img src ="./figures/6_spam.png" />
</p>

## Metrics to evaluate ML algorithms
Tackling Overfitting and Underfitting problems.

### 7.  High Bias vs High Variance 
- 🐍 [Demo | High Bias vs High Variance Notebook](https://github.com/hzitoun/coursera_machine_learning_matlab_python/blob/master/jupyter_notebooks/week_6/ex5.ipynb)
- ▶️ [Demo | High Bias vs High Variance Matlab](https://github.com/hzitoun/coursera_machine_learning_matlab_python/blob/master/algorithms_in_matlab/week_6/ex5/ex5.m)
<p align="center">
    <img src ="./figures/5_learning_curves.png" alt="learning curves"/>
</p>

## Unsupervised Learning
Labeling can be tedious (too long, too slow), often done by humans and no real labels to compare.
Unsupervised learning allows us to approach problems with little or no idea what our results should look like.
We can derive structure from data where we don't necessarily know the effect of the variables. We can derive this structure by clustering the data based on relationships among the variables in the data.
With unsupervised learning there is no feedback based on the prediction results.

### Clustering
Group objects in clusters, similar within cluster, dissimilar between clusters.

#### 8. K-means clustering algorithm for image compression

- 🐍 [Demo | K-means Notebook](https://github.com/hzitoun/coursera_machine_learning_matlab_python/blob/master/jupyter_notebooks/week_8/ex7.ipynb) 
- ▶️ [Demo | K-means Matlab](https://github.com/hzitoun/coursera_machine_learning_matlab_python/blob/master/algorithms_in_matlab/week_8/ex7/ex7.m)
<p align="center">
   <img src ="./figures/7_kmeans.png" />
    <img src ="./figures/7_keams_image_compression.png" />
</p>

### Dimensionality reduction

Reduce data set dimensions. Used for ata compression or big data visualization.

#### 9.  **Principal Component Analysis (PCA)** to perform dimensionality reduction
- 🐍 [Demo | Principal Component Analysis Notebook](https://github.com/hzitoun/coursera_machine_learning_matlab_python/blob/master/jupyter_notebooks/week_8/ex7_pca.ipynb) 
- ▶️ [Demo | Principal Component Analysis Matlab](https://github.com/hzitoun/coursera_machine_learning_matlab_python/blob/master/algorithms_in_matlab/week_8/ex7/ex7_pca.m)
<p align="center">
 <img src ="./figures/8_pca_datasets_before.png" />
</p>
<p align="center">
   <img src ="./figures/8_pca_faces.png" />
</p>

### Anomaly detection
Identifies rare items (outliers) which raise suspicions by differing significantly from the majority of the data.

#### 10.  **Anomaly detection algorithm** to detect anomalous behavior in server computers of a data center
- 🐍 [Demo | Anomaly detection algorithm Notebook](https://github.com/hzitoun/coursera_machine_learning_matlab_python/blob/master/jupyter_notebooks/week_9/ex8.ipynb)
- ▶️ [Demo | Anomaly detection algorithm Matlab](https://github.com/hzitoun/coursera_machine_learning_matlab_python/blob/master/algorithms_in_matlab/week_9/ex8/ex8.m)
<p align="center">
    <img src ="./figures/9_anomaly_detection.png" />
</p>

### Recommender System
Predicts the rating or preference a user would give to an item.

#### 11. **Collaborative ﬁltering recommender system** applied to a dataset of movie ratings

- 🐍 [Demo | Collaborative ﬁltering recommender system Notebook](https://github.com/hzitoun/coursera_machine_learning_matlab_python/blob/master/jupyter_notebooks/week_9/ex8_cofi.ipynb)
- ▶️ [Demo | Collaborative ﬁltering recommender system Matlab](https://github.com/hzitoun/coursera_machine_learning_matlab_python/blob/master/algorithms_in_matlab/week_9/ex8/ex8_cofi.m)
<p align="center">
    <img src ="./figures/9_collaborative_filtering.png" />
</p>
