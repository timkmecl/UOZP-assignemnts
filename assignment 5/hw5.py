from distutils import log
import numpy as np
from scipy.optimize import minimize
import pandas as pd
import time


def softmax(parameters, X):
    theta = parameters.reshape((X.shape[1], -1))
    z = X @ theta
    e = np.exp(z)
    Y = e / np.sum(e, axis=1)[:, np.newaxis]
    return Y


def cost(parameters, X, y, lambda_):
    theta = parameters.reshape((X.shape[1], -1))
    P = softmax(parameters, X)
    Y = np.zeros_like(P) 
    Y[np.arange(len(y)), y] = 1 # matrika, kjer je v stolpcu, ki ustreza dejanskemu razredu primerka, 1, drugje 0
    C = - np.sum(Y*np.log(P)) / len(y) + lambda_ * np.sum(theta**2) / (2*len(y))
    return -C


def grad(parameters, X, y, lambda_):
    theta = parameters.reshape((X.shape[1], -1))
    P = softmax(parameters, X)
    Y = np.zeros_like(P) 
    Y[np.arange(len(y)), y] = 1 # matrika, kjer je v stolpcu, ki ustreza dejanskemu razredu primerka, 1, drugje 0
    grad_theta = - (X.T @ (Y - P) + lambda_ * theta) / len(y)
    return -grad_theta.flatten()



def bfgs(X, y, lambda_):
    # tukaj inicirajte parametere modela
    x0 = 0.001*np.random.rand(X.shape[1], np.max(y)+1).flatten()

    # preostanek funkcije pustite kot je
    res = minimize(lambda pars, X=X, y=y, lambda_=lambda_: -cost(pars, X, y, lambda_),
                   x0,
                   method='L-BFGS-B',
                   jac=lambda pars, X=X, y=y, lambda_=lambda_: -grad(pars, X, y, lambda_),
                   tol=0.00001)
    return res.x


class SoftMaxLearner:

    def __init__(self, lambda_=0, intercept=True):
        self.intercept = intercept
        self.lambda_ = lambda_

    def __call__(self, X, y):
        if self.intercept:
            X = np.hstack([np.ones((len(X), 1)), X])
        pars = bfgs(X, y, self.lambda_)
        return SoftMaxClassifier(pars, self.intercept)


class SoftMaxClassifier:

    def __init__(self, parameters, intercept):
        self.parameters = parameters
        self.intercept = intercept

    def __call__(self, X):
        if self.intercept:
            X = np.hstack([np.ones((len(X), 1)), X])
        ypred = softmax(self.parameters, X)
        return ypred


def test_learning(learner, X, y):
    """ vrne napovedi za iste primere, kot so bili uporabljeni pri učenju.
    To je napačen način ocenjevanja uspešnosti!

    Primer klica:
        res = test_learning(SoftMaxLearner(lambda_=0.0), X, y)
    """
    c = learner(X,y)
    results = c(X)
    return results


def test_cv(learner, X, y, k=5):
    """
    Primer klica:
        res = test_cv(SoftMaxLearner(lambda_=0.0), X, y)
    """
    
    # delitev v k enako velikih odsekov (če je možno)
    chunk_size = int(np.ceil(len(y) / k))
    chunk_index = np.array(list(range(0, k)) * chunk_size)[:len(y)]
    np.random.shuffle(chunk_index) # naključno premeša dodelitev v odseke

    res = np.zeros([len(y), np.max(y)+1])

    for i in range(k):
        model_i = learner(X[chunk_index != i, :], y[chunk_index != i]) # zgradi model na ostalih odsekih
        res[chunk_index == i, :] = model_i(X[chunk_index == i, :]) # napovedi za i-ti odsek

    return res


def CA(real, predictions):
    pred_y = np.argmax(predictions, axis=1) # razred z največjo napovedano verjetnostjo
    n_correct = np.sum(pred_y == real) # št. ujemanj z realnim razredom
    return n_correct / len(real)


def log_loss(real, predictions):
    Y = np.zeros_like(predictions) 
    Y[np.arange(len(real)), real] = 1 # matrika, kjer je v stolpcu, ki ustreza dejanskemu razredu primerka, 1, drugje 0
    return -np.sum(Y*np.log(predictions)) / len(real)


def open_file(filename, train=True):
    df = pd.read_csv(filename, sep=",")
    id = df["id"].to_numpy()

    y = None
    X = None
    if train:
        df["target"] = df["target"].apply(lambda x: x.split("_")[1])
        y = df["target"].astype(int).to_numpy()-1
        X = df.iloc[:, 1:-1].astype(float).to_numpy()
    else:
        X = df.iloc[:, 1:].astype(float).to_numpy()

    
    return id, X, y

def create_final_predictions():
    id, X_train, y_train = open_file("./train.csv.gz")
    '''for lam in [1, 2, 10, 100]:
        l = SoftMaxLearner(lambda_=lam)
        res = test_cv(l, X_train, y_train, k=3)
        print("lambda: {}  -  CA: {}, logloss: {}".format(lam, CA(y_train, res), log_loss(y_train, res)))'''
    '''for lam in range(2, 10, 2):
        l = SoftMaxLearner(lambda_=lam)
        res = test_cv(l, X_train, y_train, k=3)
        print("lambda: {}  -  CA: {}, logloss: {}".format(lam, CA(y_train, res), log_loss(y_train, res)))'''
    
    start = time.time()
    l = SoftMaxLearner(lambda_=4)
    c = l(X_train, y_train)
    print("čas gradnje modela je {:.2}s".format(time.time() - start))

    
    id, X_test, _ = open_file("./test.csv.gz", train=False)
    Y = c(X_test)
    df = pd.DataFrame(np.hstack((id[:, np.newaxis], Y)), columns=["id"] + ["Class_{}".format(i+1) for i in range(Y.shape[1])])
    df.id = df.id.astype(int)
    df.to_csv('final.txt', sep=',', index=False, mode='wt')


def test_softmax():
    # podatki, vzeti iz test_hw5.py
    mirisX = np.array([[5.1, 3.5, 1.4, 0.2],
                    [5.4, 3.9, 1.7, 0.4],
                    [5.4, 3.7, 1.5, 0.2],
                    [5.7, 4.4, 1.5, 0.4],
                    [5.4, 3.4, 1.7, 0.2],
                    [5. , 3. , 1.6, 0.2],
                    [4.8, 3.1, 1.6, 0.2],
                    [5. , 3.2, 1.2, 0.2],
                    [5. , 3.5, 1.3, 0.3],
                    [4.8, 3. , 1.4, 0.3],
                    [7. , 3.2, 4.7, 1.4],
                    [5.7, 2.8, 4.5, 1.3],
                    [5. , 2. , 3.5, 1. ],
                    [6.7, 3.1, 4.4, 1.4],
                    [5.9, 3.2, 4.8, 1.8],
                    [6.6, 3. , 4.4, 1.4],
                    [5.5, 2.4, 3.8, 1.1],
                    [6. , 3.4, 4.5, 1.6],
                    [5.5, 2.6, 4.4, 1.2],
                    [5.7, 3. , 4.2, 1.2],
                    [6.3, 3.3, 6. , 2.5],
                    [7.6, 3. , 6.6, 2.1],
                    [6.5, 3.2, 5.1, 2. ],
                    [6.4, 3.2, 5.3, 2.3],
                    [6.9, 3.2, 5.7, 2.3],
                    [7.2, 3.2, 6. , 1.8],
                    [7.4, 2.8, 6.1, 1.9],
                    [7.7, 3. , 6.1, 2.3],
                    [6.7, 3.1, 5.6, 2.4],
                    [6.7, 3. , 5.2, 2.3]])
    mirisY = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
                    2, 2, 2, 2, 2, 2, 2, 2])

    np.random.seed(666)

    print("====   Test podtkov miris, napoved na učni množici")
    for i in [0.001, 1, 10, 100]:
        l = SoftMaxLearner(lambda_=i)
        c = l(mirisX, mirisY)
        print("lambda: {}  -  CA: {}, logloss: {}".format(i, CA(mirisY, c(mirisX)), log_loss(mirisY, c(mirisX))))

        
    print("====   Test prevelike regularizacije na podatkih mirirs")
    i  = 10000000
    l = SoftMaxLearner(lambda_=i)
    c = l(mirisX, mirisY)
    print("lambda: {}  -  CA: {}, logloss: {}".format(i, CA(mirisY, c(mirisX)), log_loss(mirisY, c(mirisX))))

    print("====   Test podtkov miris, 3x prečno preverjanje")
    l = SoftMaxLearner(lambda_=0.001)
    res = test_cv(l, mirisX, mirisY, k=3)
    print("lambda: {}  -  CA: {}, logloss: {}".format(0.001, CA(mirisY, res), log_loss(mirisY, res)))


    print("====   Test na naključnih atributih, naključni razredi")
    X = np.random.randn(1000, 5)
    y = np.random.randint(0, 3, size=(1000,1))
    l = SoftMaxLearner(lambda_=0.01)
    c = l(X, y)
    print(c(np.random.randn(3, 5)))

    print("====   Test na podatkih, kjer so atributi one-hot encoding razreda")
    X = np.zeros((900, 3))
    X[:300, 0] = 1
    X[300:600, 1] = 1
    X[600:900, 2] = 1
    y = np.argmax(X, axis=1)
    l = SoftMaxLearner(lambda_=0.01)
    c = l(X, y)
    X_test = np.array([[1,0,0], [0, 1, 0], [0, 0, 1], [0,1,1], [1,1,1]])
    print("Testni podatki")
    print(X_test)
    print("verjetnost za razrede:")
    print(c(X_test))

if __name__ == "__main__":
    np.random.seed(678)
    create_final_predictions()
    #test_softmax()