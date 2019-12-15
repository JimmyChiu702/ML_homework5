# Machine Learning HW5 Report
## Code Explanation
### Part I
### 1. Apply Gaussian Process Regression

* **Gaussian Process Regressor**
1. Initialize kernel, C, and inverse of C
```python
self.C = self.kernel.k(self.train_X, self.train_X) + 
         np.ones((self.train_X.size, self.train_X.size))*self.beta
self.inv_C = np.linalg.inv(self.C)
```
2. Predict Y and sigma
![](https://i.imgur.com/GSgTEmm.png)

```python
k = np.array([self.kernel.k(X, _X) for _X in self.train_X]).reshape(-1, 1)
Y = k.T.dot(self.inv_C).dot(self.train_Y)
k_star = self.kernel.k(X, X) + self.beta
sigma = k_star - k.T.dot(self.inv_C).dot(k)
```
### 2. Optimize the kernel parameters
Not finished yet...

### Part II
### 1. Use different models
* **Train model** 
-t kernel_type (0: linear kernel, 1: polynomial kernel, 2: RBF kernel)
```python
model_1 = svm_train(train_Y, train_X, '-t 0') # linear kernel
...
```
* **Test model**
eval -> evaluations result(ACC, MSE, SCC)
```python
pred_1, eval_1, val_1 = svm_predict(test_Y, test_X, model_1) # linear model
...
```
### 2. Tuning parameters using grid search
* **Initialize candidates list of C and gamma**
use default parameters setting in libsvm/tools/grid.py
```python
C_list = [pow(2, i) for i in range(-5, 16, 2)]
gamma_list = [pow(2, i) for i in range(3, -16, -2)]
```
* **Grid search**
With cross-validation (4-fold), calculate ACC of every combination of C and gamma, and record the best C and gamma which produce the hieghest ACC. Finally, train the model with the best C and gamma.
```python
for C in C_list:
    for gamma in gamma_list:
        acc = svm_train(train_Y, train_X, 
                        '-s 0 -t 2 -v 4 -c {} -g {}'.format(C, gamma))
        
        if acc > max_acc:
            max_acc = acc
            best_C = C
            best_gamma = gamma
            
model=svm_train(train_Y, train_X, 
                '-s 0 -t 2 -c {} -g {}'.format(best_C, best_gamma))
```
### 3. Use linear kernel + RBF kernel
* **Define linear kernel, RBF kernel, and the combination of these two kernel**
```python
def linear_kernel(X, Y):
    k = np.zeros((len(X), len(Y)+1))
    k[:, 1:] = np.dot(X, Y.T)
    k[:, :1] = np.arange(len(X))[:,np.newaxis]+1
    return np.dot(X, Y.T)

def RBF_kernel(X, Y, gamma=0.05):
    return np.exp(-gamma*cdist(X, Y, 'euclidean'))

def mixed_kernel(X, Y):
    k = np.zeros((len(X), len(Y)+1))
    k[:, 1:] = linear_kernel(X, Y) + RBF_kernel(X, Y)
    k[:, :1] = np.arange(len(X)).reshape(-1, 1)+1
    return k
```
* **Precompute the input data**
```python
train_K = mixed_kernel(train_X, train_X)
test_K = mixed_kernel(test_X, train_X)
```
* **Train (-t 4)**
```python
model = svm_train(train_Y, train_K, '-t 4')
```

## Results
### Part I
#### Result
![](https://i.imgur.com/NZXq9r8.png)

### Part II
#### 1. Use different kernel
[Result] (using default parameters)
||ACC|MSE|SCC|
|-|-|-|-|
|Linear kernel|95.08%|0.1404|0.9311|
|Polynomial kernel|34.68%|2.6212|0.1489|
|RBF kernel|95.32%|0.1492|0.9272|
#### 2. Grid search
[Candidates]
C: [2^-5, 2^-3, 2^-1, ..., 2^15]
gamma: [2^3, 2^1, 2^-1, ..., 2^-15]

[Search result]
Best C: 32
Best gamma: 0.03125
ACC in training data(cross validation): 98.62%
ACC in testing data: 98.52%
#### 3. Use linear kernel + RBF kernel
||ACC|MSE|SCC|
|-|-|-|-|
|Linear kernel|95.08%|0.1404|0.9311|
|RBF kernel|95.32%|0.1492|0.9272|
|Linear kernel + RBF kernel|95.24%|0.1364|0.9331|