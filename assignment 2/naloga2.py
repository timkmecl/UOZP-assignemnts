from os import listdir
from os.path import join

import numpy as np

from unidecode import unidecode
import itertools

import matplotlib.pyplot as plt
import sklearn.manifold

def terke(text, n):
    """
    Vrne slovar s preštetimi terkami dolžine n.
    """

    # predobdelava teksta
    text = unidecode(text)
    text = text.replace("\n\n", "\n")
    text = text.replace("\n \n", "\n")
    text = text.replace("\n", " ")
    text = text.lower()
    terke = {}

    # gre povrsti čez vsak znak besedila, ki je lahko začetek terke, poveča števec za trenutno terko v slovarju za 1 (oz. ga doda, če še ne obstaja v slovarju)
    for i in range(len(text) - n + 1):
        terke[text[i:i+n]] = terke.get(text[i:i+n], 0) + 1

    return terke


def read_data(n_terke):
    # Prosim, ne spreminjajte te funkcije. Vso potrebno obdelavo naredite
    # v funkciji terke.
    lds = {}
    for fn in listdir("jeziki"):
        if fn.lower().endswith(".txt"):
            with open(join("jeziki", fn), encoding="utf8") as f:
                text = f.read()
                nter = terke(text, n=n_terke)
                lds[fn] = nter
    return lds


def cosine_dist(d1, d2):
    """
    Vrne kosinusno razdaljo med slovarjema terk d1 in d2.
    """
    return 1 -  sum([d1[k] * d2[k] for k in set(d1.keys()).intersection(set(d2.keys()))]) / (np.sqrt(sum([v**2 for v in d1.values()])) * np.sqrt(sum([v**2 for v in d2.values()])))


def prepare_data_matrix(data_dict):
    """
    Return data in a matrix (2D numpy array), where each row contains triplets
    for a language. Columns should be the 100 most common triplets
    according to the idf (NOT the complete tf-idf) measure.
    """
    # create matrix X and list of languages
    # ...

    # združi terke vseh dokumentov v eno množico
    vse_terke = set(itertools.chain(*[doc.keys() for doc in data_dict.values()]))
    # generira seznam parov (št pojavitev terke povsod, terka) in ga sortira
    pogostost_terk = [[sum([doc.get(key, 0) for doc in data_dict.values()]), key] for key in vse_terke]
    pogostost_terk.sort(reverse=True)
    # vzame 100 najpogostejših terk
    top100 = [pogostost_terk[i][1] for i in range(100)]

    n = len(data_dict)
    X = np.zeros((n, 100))
    languages = []
    # gre čez vse dokumente
    for i, (key, d) in enumerate(data_dict.items()):
        # za vsakega v vrstico Xa zapiše frekvenco terk v njem
        for j in range(100):
            X[i, j] = d.get(top100[j], 0)
        languages.append(key)
    return X, languages


def power_iteration(X):
    """
    Compute the eigenvector with the greatest eigenvalue
    of the covariance matrix of X (a numpy array).

    Return two values:
    - the eigenvector (1D numpy array) and
    - the corresponding eigenvalue (a float)
    """
    cov = np.cov(X.T)

    n = X.shape[1]

    v = np.random.uniform(-1, 1, [n, 1])
    v = v / np.linalg.norm(v)

    tol = 10e-6
    vp = v.copy() + tol*2

    eigenval = 1

    while np.sum(np.abs(v - vp)) > tol:
        vp = v.copy()
        v = cov @ v
        eigenval = np.linalg.norm(v)
        v = v / eigenval

    return v.reshape(n), eigenval


def power_iteration_two_components(X):
    """
    Compute first two eigenvectors and eigenvalues with the power iteration method.
    This function should use the power_iteration function internally.

    Return two values:
    - the two eigenvectors (2D numpy array, each eigenvector in a row) and
    - the corresponding eigenvalues (a 1D numpy array)
    """

    # izračun prvega lastnega vektorja
    v1, l1 = power_iteration(X)

    n = X.shape[1]
    dolzina = X.shape[0]

    # projekcija na prvi lastni vektor
    Xp = project_to_eigenvectors(X, v1[np.newaxis, :]).reshape((dolzina, 1))

    # odšteje, kar je zastopano s prvim lastnim vektorjem, od podatkov
    X2 = X - Xp * v1.reshape([1, n])

    # določi drugi lastni vektor
    v2, l2 = power_iteration(X2)

    vecs = np.array([v1, v2])
    eigenvals = np.array([l1, l2])

    return vecs, eigenvals


def project_to_eigenvectors(X, vecs):
    """
    Project matrix X onto the space defined by eigenvectors.
    The output array should have as many rows as X and as many columns as there
    are vectors.
    """
    vecs = vecs.T
    #normira vektor in odšteje povprečje od podatkov
    vecs = vecs / np.linalg.norm(vecs, axis=0)
    X = X - X.mean(axis=0)

    return X @ vecs


def total_variance(X):
    """
    Total variance of the data matrix X. You will need to use for
    to compute the explained variance ratio.
    """
    return np.var(X, axis=0, ddof=1).sum()


def explained_variance_ratio(X, eigenvectors, eigenvalues):
    """
    Compute explained variance ratio.
    """

    Xp = project_to_eigenvectors(X, eigenvectors)
    proj_var = total_variance(Xp)
    total_var = total_variance(X)
    return proj_var / total_var

def distance_matrix(data_dict):
    keys = [key for key in data_dict.keys()]
    D = np.array([[cosine_dist(data_dict[k1], data_dict[k2]) for k2 in keys] for k1 in keys])
    return D
    

def plot_PCA():
    """
    Everything (opening files, computation, plotting) needed
    to produce a plot of PCA on languages data.
    """
    X, legend = prepare_data_matrix(read_data(3))
    # normiram vrstice, da dolžina dokumenta ne vpliva več na položaj v prostoru (popravi projekcijo PCA kot zahtevano na slacku)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    vecs, eigenvals = power_iteration_two_components(X)
    P = project_to_eigenvectors(X, vecs)

    expl_var = explained_variance_ratio(X, vecs, eigenvals)

    langs = [l.split(".")[0].split("_")[1] for l in legend]
    plt.scatter(P[:, 0], P[:, 1])
    for i, l in enumerate(langs):
        plt.annotate(langs[i], (P[i, 0], P[i, 1]))
    plt.title(f'PCA, expl. variance={expl_var:.3f}')
    plt.savefig('PCA.png')


def plot_MDS():
    """
    Everything (opening files, computation, plotting) needed
    to produce a plot of MDS on languages data.

    Use sklearn.manifold.MDS and explicitly run it with a distance
    matrix obtained with cosine distance on full triplets.
    """
    
    data_dict = read_data(3)
    D = distance_matrix(data_dict)
    em = sklearn.manifold.MDS(2)
    P = em.fit_transform(D)
    

    langs = [l.split(".")[0].split("_")[1] for l in data_dict.keys()]
    plt.scatter(P[:, 0], P[:, 1])
    for i, l in enumerate(langs):
        plt.annotate(langs[i], (P[i, 0], P[i, 1]))
    plt.title(f'MDS')
    plt.savefig('MDS.png')

def plot_tSNE():
    
    data_dict = read_data(3)
    D = distance_matrix(data_dict)
    em = sklearn.manifold.TSNE(2)
    P = em.fit_transform(D)
    

    langs = [l.split(".")[0].split("_")[1] for l in data_dict.keys()]
    plt.scatter(P[:, 0], P[:, 1])
    for i, l in enumerate(langs):
        plt.annotate(langs[i], (P[i, 0], P[i, 1]))
    plt.title('tSNE')
    plt.savefig('TSNE.png')


if __name__ == "__main__":
    # vse tri funkcije shranijo scatterplot v datoteko v CWD
    plot_MDS()
    plot_PCA()
    plot_tSNE()