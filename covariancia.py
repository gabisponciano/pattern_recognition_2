"""Métodos 1 a 4 de estimação da matriz de covariância (slides, X é p x N)."""
import numpy as np


def mcovar1(X):
    p, N = X.shape
    m = X.mean(axis=1, keepdims=True)
    soma = np.zeros((p, p))
    for j in range(N):
        aux = X[:, [j]] - m
        soma += aux @ aux.T
    C = soma / N
    return (C + C.T) / 2


def mcovar2(X):
    p, N = X.shape
    m = X.mean(axis=1, keepdims=True)
    aux = X - m
    C = aux @ aux.T / N
    return (C + C.T) / 2


def mcovar3(X):
    p, N = X.shape
    m = X.mean(axis=1, keepdims=True)
    R = np.zeros((p, p))
    for j in range(N):
        R += X[:, [j]] @ X[:, [j]].T
    C = R / N - m @ m.T
    return (C + C.T) / 2


def mcovar4(X):
    p, N = X.shape
    m = X.mean(axis=1, keepdims=True)
    R = X @ X.T / N
    C = R - m @ m.T
    return (C + C.T) / 2


def invcov(X):
    """Inversa da matriz de covariância via SVD (slides, Eqs. 170-172).

    Rotina invcov dos slides: C^{-1} = N U (ΣΣ^T)^{-1} U^T. O slide assume
    média nula, então centralizamos X para que o resultado seja exatamente
    inv(cov(X)). Requer posto cheio (todos os valores singulares > 0).
    """
    p, N = X.shape
    Xc = X - X.mean(axis=1, keepdims=True)
    U, s, _ = np.linalg.svd(Xc, full_matrices=False)  # Xc = U Σ V^T
    return N * (U * (1.0 / s**2)) @ U.T  # N U diag(1/σ²) U^T
