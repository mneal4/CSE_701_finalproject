# EM Algorithm for Gaussian Mixture Model-Based Clustering
Mackenzie Neal

## Introduction
This program implements the EM algorithm in order to perform Gaussian mixture model-based clustering. This program uses two classes in order to do so. The first is a matrix class to allow for matrices to be constructed and various matrix operations and functions to be implemented. The second class, called EMalgo, is derived from the matrix class so that matrix operations and functions can be used by EMalgo. This class contains functions needed to perform and assess the EM algorithm.

## Background Information

### Model-Based Clustering

Model-based clustering uses finite mixture models to classify observations. Finite mixture models arise from the assumption that a population contains sub-populations that can be modelled by a finite number of densities. A random vector X belongs to finite mixture model if, for all x ⊂ X we can write the density as

$$f(x|\vartheta) = \sum_{g=1}^{G} \pi_g f_g(x|\theta_g)$$,

where $\pi_g$ are the mixing proportions such that $$\sum_{g=1}^{G} \pi_g =1$$ and $f_{g}(x|\theta_{g})$ are the component densities.

The EM algorithm is used to estimate these mixture models. The EM is a natural choice due to the presence of latent variables (cluster memberships).

### EM Algorithm for Gaussian Model-Based Clustering

The EM algorithm performs maximum likelihood estimation in the presence of missing data. In this case the missing data is cluster memberships and the complete data log-likelihood is:

$$l(\vartheta) = \sum_{i=1}^n log \sum_{g=1}^G [\pi_g*\phi(x_i|\mu_g,\Sigma_g)]$$.

The algorithm iterates between two steps in order to estimate cluster memberships and maximize the log-likelihood above. The first step is the E step, detailed below.

Let $x_1,...x_n$ represent our p-dimensional observations. Let $z_{ig}$ represent the cluster membership of observation i. 

E Step:
Given the model parameters calculate the probability of observation i belong to group g, using Bayesian statistics. This is done via the following formula:

```math
\hat{z}_{ig} = \frac{\hat{\pi}_g \phi(x_i|\hat{\mu}_g,\hat{\Sigma}_g)}{ \sum_{h=1,..,G}\hat{\pi}_h \phi(x_i|\hat{\mu}_h,\hat{\Sigma}_h).}
```

Once cluster memberships are estimated then maximization of the log-likelihood can be performed by obtaining the following MLE's.

M Step:

$$\hat{\pi}_g = \frac{1}{n}\sum_{i=1}^n \hat{z}_{ig}$$

$$\hat{\mu}_{g} = \frac{1}{n_g}\sum_{i=1}^n \hat{z}_{ig}x_i$$

$$\hat{\Sigma}_g = \frac{1}{n_g}\sum_{i=1}^n \hat{z}_{ig}(x_i-\hat{\mu}_g)(x_i-\hat{\mu}_g)^t$$

The algorithm iterates between these two steps until convergence of the log-likelihood is achieved. 

All the above background information comes from [Expectation-Maximization Algorithm Step-by-Step](https://medium.com/analytics-vidhya/expectation-maximization-algorithm-step-by-step-30157192de9f) and [Mixture Model-Based Classification](https://www.taylorfrancis.com/books/mono/10.1201/9781315373577/mixture-model-based-classification-paul-mcnicholas).

### Adjusted Rand Index (ARI)

ARI is a performance measure used to compare the similarity between two partitions. Prior to ARI, the Rand index (RI) was used to compare partitions [Rand, 1971](https://www.jstor.org/stable/2284239#metadata_info_tab_contents):
$$RI = \frac{\text{number of agreements}}{\text{number of agreements + number of disagreements}}$$
ARI was proposed as to force the index to have expected value of zero under random assignment [Hubert and Arabie, 1985](https://link.springer.com/article/10.1007/BF01908075). The corrected index can be found below.
$$ARI = \frac{\text{RI} − \text{Expected RI}}{\text{Max RI} − \text{Expected RI}}$$
Thus, ARI equals one when there is perfect agreement between partitions and is negative when the assignment is worse than random.

## Program Structure

### Matrix Class
Contains constructors for creating a matrix object. These objects are 1D arrays but the elements of said object can be assessed as matrix elements normally would be i.e. A(1,2). The following overloaded operators and member functions are included in the class.
1. Matrix Addition (+, +=)
2. Matrix Subtraction (-, -=)
3. Matrix Negation (-)
4. Matrix Multiplication (*)
5. Matrix Division (/)
6. Matrix Printing (<<)
7. Member Access ()
8. get_rows(): To obtain number of rows.
9. get_cols(): To obtain number of columns.
10. extract_row(): To obtain a row from a matrix.
11. extract_col(): To extract a column from a matrix.
12. determinant(): To calculate the determinant of a matrix.
13. cofactor(): To get a co-factor matrix
14. inverse(): To calculate matrix inverse
15. mean(): To obtain a mean matrix from a matrix.
16. covariance(): To obtain a covariance matrix from a matrix.
17. transpose(): To obtain the transpose of a matrix.
18. zero_matrix(): To obtain a zero matrix.

This class contains three private members rows, cols, and elements to store the number of rows, the number of columns, and the matrix elements.

### EMalgo Class
This is a derived class from the matrix class to allow the functions in this class to inherit the matrix objects. The functions included in this class are listed below.
1. dmvnorm(): Calculates the density of a multivariate normal.
2. compute_log_likelihood(): Calculates the log-likelihood.
3. vec_mat(): To create a vector of matrices.
4. e_step(): To perform the E step of the algorithm.
5. m_step(): To perform the M step of the algorithm.
6. random_soft(): To generate a matrix of soft classifications.
7. EM(): To implement the EM algorithm.
8. classify(): To convert soft classifications to hard classifications.
9. binomial_coeff(): To calculate binomial coefficients.
10. ARI(): To calculate the adjusted Rand index.

## Running the Program

The user has three options when running the program. The first is that no arguments are provided, in this case, a description of the program will be printed. The second is when the user has data they want to cluster on but they do not have the true classifications to compare to, in which case, the following three inputs are needed in the following order.
1. The data (.csv file) you are clustering on (these must be numeric)
2. The number of clusters (this must be a positive integer)
3. Name of the file you want to export the estimated clusters to (i.e. output.txt)

If the user does have the true clusters than four inputs are needed, as follows in the following order.
1. The data (.csv file) you are clustering on (these must be numeric)
2. The number of clusters (this must be a positive integer)
3. The data file (.csv) containing the true clusters
4. Name of the file you want to export the estimated clusters to (i.e. output.txt).



## Examples

In terminal:
```
clang++ -Wall -Wextra -Wconversion -Wsign-conversion -Wshadow -Wpedantic -std=c++20 EMalgo.cpp -o EMalgo 
./EMalgo input_sim_data.csv 2 trueclass.csv output.txt
```


Output: 

EM Algorithm Results (results may vary slightly due to random initialization scheme): 

zigs (soft classifications): [The output is shortened here for readability.]

$$ \begin{matrix} 0 & 1\\
0 & 1\\
  . & . \\
. & . \\
. & . \\
 0 & 1\\
 1 & 0 \\
. & . \\
. & . \\
. & . \\
 1 & 0\\
\end{matrix}$$

Group 1 

Mean vector:
$$[4.93344, 9.93986]$$


Mixing proportion:
$$0.56$$

Covariance matrix:
```math
\bigg[\begin{matrix} 1.80233  & 0.419065\\0.419065 & 2.15444\\\end{matrix}\bigg]
```



Group 2 

Mean vector:
$$[1.01969, 0.630914]$$

Mixing proportion:
$$0.44$$

Covariance matrix:
```math
\bigg[\begin{matrix} 0.797828  & -0.0779682\\-0.0779682 & 0.884848 \\\end{matrix}\bigg]
```

ARI: 1.0

## Discussion of Results
The data used was simulated from a two dimensional, two component Gaussian mixture model. Model specification are found below.

Group 1:
```math
\pi_1 = 0.4
```
```math
\mu_1 = [1,1]
```
```math
\Sigma_1 = \bigg[\begin{matrix}1 & 0\\0& 1\\\end{matrix}\bigg]
```

Group 2:
```math
\pi_1 = 0.6
```
```math
\mu_1 = [5,10]
```
```math
\Sigma_1 =  \bigg[\begin{matrix}2 & 0.5\\0.5& 2\\ \end{matrix}\bigg]
```
We can see that the true model parameters are very close to the estimated model parameters. Additionally, classification was perfect with an ARI of 1.
