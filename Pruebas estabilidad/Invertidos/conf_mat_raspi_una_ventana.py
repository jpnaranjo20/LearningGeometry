import numpy as np
import pandas as pd
import itertools
from ast import literal_eval
from sklearn.metrics import confusion_matrix, accuracy_score

# Lista de nombres de fallas
fallas = ['NoFault', 'AB', 'BC', 'CA', 'ABC', 'AG', 'BG', 'CG']
idxNoFault = fallas.index('NoFault')
idxAB = fallas.index('AB')
idxBC = fallas.index('BC')
idxCA = fallas.index('CA')
idxABC = fallas.index('ABC')
idxAG = fallas.index('AG')
idxBG = fallas.index('BG')
idxCG = fallas.index('CG')

# Lista de distancias
distancias = [10, 20, 50, 75, 95]

# Creates column vector if x is row vector or list.
def col(x):
    return np.asarray(np.matrix(x)).T

# Calculates cosine distance between two vectors.
def cos_dist(u, v):
    return float(np.abs(np.dot(u, v)/(np.linalg.norm(u)*np.linalg.norm(v))))

# Creates m-by-2 matrix of data.
def create_x(alpha, beta):
    m = len(alpha)
    x = np.zeros((m, 2))

    for i in range(m):
        x[i, 0] = alpha[i]
        x[i, 1] = beta[i]

    return x

# Clarke Transform referred to phase A
def Clarke_ref_A(A, B, C):

    alpha = []
    beta = []
    z = []

    voltajes = np.zeros((3,1))
    mat_Clarke = np.array([[1, -1/2, -1/2], [0, np.sqrt(3)/2, -np.sqrt(3)/2], [1/np.sqrt(2), 1/np.sqrt(2), 1/np.sqrt(2)]])

    for i in range(len(A)):
        voltajes[0] = A[i]
        voltajes[1] = B[i]
        voltajes[2] = C[i]

        space_vec = np.sqrt(2/3)*mat_Clarke@voltajes # Si acaso cambiar por np.sqrt(2/3)

        alpha.append(space_vec[0])
        beta.append(space_vec[1])
        z.append(space_vec[2])

    return alpha, beta, z

# Clarke Transform referred to phase B
def Clarke_ref_B(B, C, A):

    alpha = []
    beta = []
    z = []

    voltajes = np.zeros((3,1))
    mat_Clarke = np.array([[1, -1/2, -1/2], [0, np.sqrt(3)/2, -np.sqrt(3)/2], [1/np.sqrt(2), 1/np.sqrt(2), 1/np.sqrt(2)]])

    for i in range(len(A)):
        voltajes[0] = B[i]
        voltajes[1] = C[i]
        voltajes[2] = A[i]

        space_vec = np.sqrt(2/3)*mat_Clarke@voltajes # Si acaso cambiar por np.sqrt(2/3)

        alpha.append(space_vec[0])
        beta.append(space_vec[1])
        z.append(space_vec[2])

    return alpha, beta, z

# Clarke Transform referred to phase C
def Clarke_ref_C(C, A, B):

    alpha = []
    beta = []
    z = []

    voltajes = np.zeros((3,1))
    mat_Clarke = np.array([[1, -1/2, -1/2], [0, np.sqrt(3)/2, -np.sqrt(3)/2], [1/np.sqrt(2), 1/np.sqrt(2), 1/np.sqrt(2)]])

    for i in range(len(A)):
        voltajes[0] = C[i]
        voltajes[1] = A[i]
        voltajes[2] = B[i]

        space_vec = np.sqrt(2/3)*mat_Clarke@voltajes # Si acaso cambiar por np.sqrt(2/3)

        alpha.append(space_vec[0])
        beta.append(space_vec[1])
        z.append(space_vec[2])

    return alpha, beta, z

# Function that fits an ellipse to a given set of data x, y. Returns a, b, c, d, e, f parameters.
def fit_ellipse(x, y):
    """
    Fit the coefficients a,b,c,d,e,f, representing an ellipse described by
    the formula F(x,y) = ax^2 + bxy + cy^2 + dx + ey + f = 0 to the provided
    arrays of data points x=[x1, x2, ..., xn] and y=[y1, y2, ..., yn].

    Based on the algorithm of Halir and Flusser, "Numerically stable direct
    least squares fitting of ellipses'.

    """

    D1 = np.vstack([x**2, x*y, y**2]).T
    D2 = np.vstack([x, y, np.ones(len(x))]).T
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.inv(S3) @ S2.T
    M = S1 + S2 @ T
    C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
    M = np.linalg.inv(C) @ M
    eigval, eigvec = np.linalg.eig(M)
    con = 4 * eigvec[0]* eigvec[2] - eigvec[1]**2
    ak = eigvec[:, np.nonzero(con > 0)[0]]
    
    return np.concatenate((ak, T @ ak)).ravel()

# Function to get Q, u and f parameters for each fault type
def get_ellipse_params(abcdef):

    a = abcdef[0]
    b = abcdef[1]
    c = abcdef[2]
    d = abcdef[3]
    e = abcdef[4]
    f = np.round(abcdef[5], 4)

    Q = np.round(np.array([[a, b], [b, c]]), 4)
    u = np.round(np.array([[d], [e]]), 4)

    return Q, u, f

def load_signals(path):
    datos = pd.read_excel(path)
    datos_dict = datos.to_dict()

    Va = list(datos_dict['Va'].values())
    Vb = list(datos_dict['Vb'].values())
    Vc = list(datos_dict['Vc'].values())

    return Va, Vb, Vc

########## IMPORTACIÓN DE VALORES Y VECTORES PROPIOS CARACTERÍSTICOS DE CADA FALLA ##########

# Valores propios
eigvals_10km = pd.read_excel('Pruebas estabilidad/Invertidos/eigvals_eigvects_invertidos/eigvals_10km_inv.xlsx')
eigvals_20km = pd.read_excel('Pruebas estabilidad/Invertidos/eigvals_eigvects_invertidos/eigvals_20km_inv.xlsx')
eigvals_50km = pd.read_excel('Pruebas estabilidad/Invertidos/eigvals_eigvects_invertidos/eigvals_50km_inv.xlsx')
eigvals_75km = pd.read_excel('Pruebas estabilidad/Invertidos/eigvals_eigvects_invertidos/eigvals_75km_inv.xlsx')
eigvals_95km = pd.read_excel('Pruebas estabilidad/Invertidos/eigvals_eigvects_invertidos/eigvals_95km_inv.xlsx')

dict_eigvals_10km = eigvals_10km.to_dict()
dict_eigvals_20km = eigvals_20km.to_dict()
dict_eigvals_50km = eigvals_50km.to_dict()
dict_eigvals_75km = eigvals_75km.to_dict()
dict_eigvals_95km = eigvals_95km.to_dict()

# Vectores propios
eigvects_10km = pd.read_excel('Pruebas estabilidad/Invertidos/eigvals_eigvects_invertidos/eigvects_10km_inv.xlsx')
eigvects_20km = pd.read_excel('Pruebas estabilidad/Invertidos/eigvals_eigvects_invertidos/eigvects_20km_inv.xlsx')
eigvects_50km = pd.read_excel('Pruebas estabilidad/Invertidos/eigvals_eigvects_invertidos/eigvects_50km_inv.xlsx')
eigvects_75km = pd.read_excel('Pruebas estabilidad/Invertidos/eigvals_eigvects_invertidos/eigvects_75km_inv.xlsx')
eigvects_95km = pd.read_excel('Pruebas estabilidad/Invertidos/eigvals_eigvects_invertidos/eigvects_95km_inv.xlsx')

dict_eigvects_10km = eigvects_10km.to_dict()
dict_eigvects_20km = eigvects_20km.to_dict()
dict_eigvects_50km = eigvects_50km.to_dict()
dict_eigvects_75km = eigvects_75km.to_dict()
dict_eigvects_95km = eigvects_95km.to_dict()

# These commented lines should only be ran once. They convert the array strings to actual arrays (lists).
for falla in fallas:
    for i in range(3):
        dict_eigvals_10km[falla][i] = literal_eval(dict_eigvals_10km[falla][i])
        dict_eigvals_20km[falla][i] = literal_eval(dict_eigvals_20km[falla][i])
        dict_eigvals_50km[falla][i] = literal_eval(dict_eigvals_50km[falla][i])
        dict_eigvals_75km[falla][i] = literal_eval(dict_eigvals_75km[falla][i])
        dict_eigvals_95km[falla][i] = literal_eval(dict_eigvals_95km[falla][i])
        dict_eigvects_10km[falla][i] = literal_eval(dict_eigvects_10km[falla][i])
        dict_eigvects_20km[falla][i] = literal_eval(dict_eigvects_20km[falla][i])
        dict_eigvects_50km[falla][i] = literal_eval(dict_eigvects_50km[falla][i])
        dict_eigvects_75km[falla][i] = literal_eval(dict_eigvects_75km[falla][i])
        dict_eigvects_95km[falla][i] = literal_eval(dict_eigvects_95km[falla][i])

# Creamos listas para guardar vectores característicos de cada falla. Cada lista es de longitud 8 (1 por cada falla).
vcts_carac_refA_10km = []
vcts_carac_refA_20km = []
vcts_carac_refA_50km = []
vcts_carac_refA_75km = []
vcts_carac_refA_95km = []

vcts_carac_refB_10km = []
vcts_carac_refB_20km = []
vcts_carac_refB_50km = []
vcts_carac_refB_75km = []
vcts_carac_refB_95km = []

vcts_carac_refC_10km = []
vcts_carac_refC_20km = []
vcts_carac_refC_50km = []
vcts_carac_refC_75km = []
vcts_carac_refC_95km = []

# Llenamos las listas de vectores propios característicos
for falla in fallas:
    # RefA, para todas las distancias
    v_carac_refA_10km = dict_eigvects_10km[falla][0]
    vcts_carac_refA_10km.append(v_carac_refA_10km)
    v_carac_refA_20km = dict_eigvects_20km[falla][0]
    vcts_carac_refA_20km.append(v_carac_refA_20km)
    v_carac_refA_50km = dict_eigvects_50km[falla][0]
    vcts_carac_refA_50km.append(v_carac_refA_50km)
    v_carac_refA_75km = dict_eigvects_75km[falla][0]
    vcts_carac_refA_75km.append(v_carac_refA_75km)
    v_carac_refA_95km = dict_eigvects_95km[falla][0]
    vcts_carac_refA_95km.append(v_carac_refA_95km)

    # RefB, para todas las distancias
    v_carac_refB_10km = dict_eigvects_10km[falla][1]
    vcts_carac_refB_10km.append(v_carac_refB_10km)
    v_carac_refB_20km = dict_eigvects_20km[falla][1]
    vcts_carac_refB_20km.append(v_carac_refB_20km)
    v_carac_refB_50km = dict_eigvects_50km[falla][1]
    vcts_carac_refB_50km.append(v_carac_refB_50km)
    v_carac_refB_75km = dict_eigvects_75km[falla][1]
    vcts_carac_refB_75km.append(v_carac_refB_75km)
    v_carac_refB_95km = dict_eigvects_95km[falla][1]
    vcts_carac_refB_95km.append(v_carac_refB_95km)

    # RefC, para todas las distancias
    v_carac_refC_10km = dict_eigvects_10km[falla][2]
    vcts_carac_refC_10km.append(v_carac_refC_10km)
    v_carac_refC_20km = dict_eigvects_20km[falla][2]
    vcts_carac_refC_20km.append(v_carac_refC_20km)
    v_carac_refC_50km = dict_eigvects_50km[falla][2]
    vcts_carac_refC_50km.append(v_carac_refC_50km)
    v_carac_refC_75km = dict_eigvects_75km[falla][2]
    vcts_carac_refC_75km.append(v_carac_refC_75km)
    v_carac_refC_95km = dict_eigvects_95km[falla][2]
    vcts_carac_refC_95km.append(v_carac_refC_95km)

# Creamos listas para guardar valores propios característicos. Cada lista es de longitud 40 (5 distancias por 8 fallas)
vals_carac_refA = []
vals_carac_refB = []
vals_carac_refC = []

# Llenamos las listas de valores propios característicos
for distancia in distancias:
    for falla in fallas:
        if distancia == 10:
            val_carac_refA = dict_eigvals_10km[falla][0]
            vals_carac_refA.append(val_carac_refA)
            val_carac_refB = dict_eigvals_10km[falla][1]
            vals_carac_refB.append(val_carac_refB)
            val_carac_refC = dict_eigvals_10km[falla][2]
            vals_carac_refC.append(val_carac_refC)
        elif distancia == 20:
            val_carac_refA = dict_eigvals_20km[falla][0]
            vals_carac_refA.append(val_carac_refA)
            val_carac_refB = dict_eigvals_20km[falla][1]
            vals_carac_refB.append(val_carac_refB)
            val_carac_refC = dict_eigvals_20km[falla][2]
            vals_carac_refC.append(val_carac_refC)
        elif distancia == 50:
            val_carac_refA = dict_eigvals_50km[falla][0]
            vals_carac_refA.append(val_carac_refA)
            val_carac_refB = dict_eigvals_50km[falla][1]
            vals_carac_refB.append(val_carac_refB)
            val_carac_refC = dict_eigvals_50km[falla][2]
            vals_carac_refC.append(val_carac_refC)
        elif distancia == 75:
            val_carac_refA = dict_eigvals_75km[falla][0]
            vals_carac_refA.append(val_carac_refA)
            val_carac_refB = dict_eigvals_75km[falla][1]
            vals_carac_refB.append(val_carac_refB)
            val_carac_refC = dict_eigvals_75km[falla][2]
            vals_carac_refC.append(val_carac_refC)
        elif distancia == 95:
            val_carac_refA = dict_eigvals_95km[falla][0]
            vals_carac_refA.append(val_carac_refA)
            val_carac_refB = dict_eigvals_95km[falla][1]
            vals_carac_refB.append(val_carac_refB)
            val_carac_refC = dict_eigvals_95km[falla][2]
            vals_carac_refC.append(val_carac_refC)

# Creamos una lista con los 40 índices
indices = [*range(40)]

# En cada una de estas listas están los índices de todas las distancias. Adentro de cada lista aplican los índices de cada falla creados en las
# primeras líneas del código.
idxs10km = indices[:8]
idxs20km = indices[8:16]
idxs50km = indices[16:24]
idxs75km = indices[24:32]
idxs95km = indices[32:40]

########### CLASIFICADOR #############

# Funciones para comparar valores propios de todas las referencias
def comparar_valores_refA(vals_refA):
    normas_L2_refA = []
    for par_valores in vals_carac_refA:
        normas_L2_refA.append(np.linalg.norm(vals_refA - par_valores)**2)
    
    idx_falla_elegida = np.argmin(normas_L2_refA)

    if idx_falla_elegida >= 0 and idx_falla_elegida <= 7:
        dist_elegida = 10
        falla_elegida = fallas[idxs10km.index(idx_falla_elegida)]
    elif idx_falla_elegida >= 8 and idx_falla_elegida <= 15:
        dist_elegida = 20
        falla_elegida = fallas[idxs20km.index(idx_falla_elegida)]
    elif idx_falla_elegida >= 16 and idx_falla_elegida <= 23:
        dist_elegida = 50
        falla_elegida = fallas[idxs50km.index(idx_falla_elegida)]
    elif idx_falla_elegida >= 24 and idx_falla_elegida <= 31:
        dist_elegida = 75
        falla_elegida = fallas[idxs75km.index(idx_falla_elegida)]
    elif idx_falla_elegida >= 32 and idx_falla_elegida <= 39:
        dist_elegida = 95
        falla_elegida = fallas[idxs95km.index(idx_falla_elegida)]

    return falla_elegida, dist_elegida

def comparar_valores_refB(vals_refB):
    normas_L2_refB = []
    for par_valores in vals_carac_refB:
        normas_L2_refB.append(np.linalg.norm(vals_refB - par_valores)**2)
    
    idx_falla_elegida = np.argmin(normas_L2_refB)

    if idx_falla_elegida >= 0 and idx_falla_elegida <= 7:
        dist_elegida = 10
        falla_elegida = fallas[idxs10km.index(idx_falla_elegida)]
    elif idx_falla_elegida >= 8 and idx_falla_elegida <= 15:
        dist_elegida = 20
        falla_elegida = fallas[idxs20km.index(idx_falla_elegida)]
    elif idx_falla_elegida >= 16 and idx_falla_elegida <= 23:
        dist_elegida = 50
        falla_elegida = fallas[idxs50km.index(idx_falla_elegida)]
    elif idx_falla_elegida >= 24 and idx_falla_elegida <= 31:
        dist_elegida = 75
        falla_elegida = fallas[idxs75km.index(idx_falla_elegida)]
    elif idx_falla_elegida >= 32 and idx_falla_elegida <= 39:
        dist_elegida = 95
        falla_elegida = fallas[idxs95km.index(idx_falla_elegida)]
        
    return falla_elegida, dist_elegida

def comparar_valores_refC(vals_refC):
    normas_L2_refC = []
    for par_valores in vals_carac_refC:
        normas_L2_refC.append(np.linalg.norm(vals_refC - par_valores)**2)
    
    idx_falla_elegida = np.argmin(normas_L2_refC)

    if idx_falla_elegida >= 0 and idx_falla_elegida <= 7:
        dist_elegida = 10
        falla_elegida = fallas[idxs10km.index(idx_falla_elegida)]
    elif idx_falla_elegida >= 8 and idx_falla_elegida <= 15:
        dist_elegida = 20
        falla_elegida = fallas[idxs20km.index(idx_falla_elegida)]
    elif idx_falla_elegida >= 16 and idx_falla_elegida <= 23:
        dist_elegida = 50
        falla_elegida = fallas[idxs50km.index(idx_falla_elegida)]
    elif idx_falla_elegida >= 24 and idx_falla_elegida <= 31:
        dist_elegida = 75
        falla_elegida = fallas[idxs75km.index(idx_falla_elegida)]
    elif idx_falla_elegida >= 32 and idx_falla_elegida <= 39:
        dist_elegida = 95
        falla_elegida = fallas[idxs95km.index(idx_falla_elegida)]

    return falla_elegida, dist_elegida

# Funciones para comparar vectores propios de todas las referencias
def comparar_vectores_refA(vect_refA):
    cos_dists_10km_refA = []
    cos_dists_20km_refA = []
    cos_dists_50km_refA = []
    cos_dists_75km_refA = []
    cos_dists_95km_refA = []
    
    idx_falla_elegida = 0
    
    for i in range(len(vcts_carac_refA_10km)):
        cos_dists_10km_refA.append(cos_dist(vect_refA, vcts_carac_refA_10km[i]))
        cos_dists_20km_refA.append(cos_dist(vect_refA, vcts_carac_refA_20km[i]))
        cos_dists_50km_refA.append(cos_dist(vect_refA, vcts_carac_refA_50km[i]))
        cos_dists_75km_refA.append(cos_dist(vect_refA, vcts_carac_refA_75km[i]))
        cos_dists_95km_refA.append(cos_dist(vect_refA, vcts_carac_refA_95km[i]))
    
    idx_falla_elegida_10km = np.argmax(cos_dists_10km_refA)
    idx_falla_elegida_20km = np.argmax(cos_dists_20km_refA)
    idx_falla_elegida_50km = np.argmax(cos_dists_50km_refA)
    idx_falla_elegida_75km = np.argmax(cos_dists_75km_refA)
    idx_falla_elegida_95km = np.argmax(cos_dists_95km_refA)

    indices = [idx_falla_elegida_10km, idx_falla_elegida_20km, idx_falla_elegida_50km, idx_falla_elegida_75km, idx_falla_elegida_95km]

    idx_falla_elegida = most_frequent(indices)
    
    if idx_falla_elegida==idxNoFault:
        falla_elegida = 'NoFault'
    elif idx_falla_elegida==idxAB:
        falla_elegida = 'AB'
    elif idx_falla_elegida==idxBC:
        falla_elegida = 'BC'
    elif idx_falla_elegida==idxCA:
        falla_elegida = 'CA'
    elif idx_falla_elegida==idxABC:
        falla_elegida = 'ABC'
    elif idx_falla_elegida==idxAG:
        falla_elegida = 'AG'
    elif idx_falla_elegida==idxBG:
        falla_elegida = 'BG'
    elif idx_falla_elegida==idxCG:
        falla_elegida = 'CG'

    return falla_elegida

def comparar_vectores_refB(vect_refB):
    cos_dists_10km_refB = []
    cos_dists_20km_refB = []
    cos_dists_50km_refB = []
    cos_dists_75km_refB = []
    cos_dists_95km_refB = []
    
    idx_falla_elegida = 0
    
    for i in range(len(vcts_carac_refB_10km)):
        cos_dists_10km_refB.append(cos_dist(vect_refB, vcts_carac_refB_10km[i]))
        cos_dists_20km_refB.append(cos_dist(vect_refB, vcts_carac_refB_20km[i]))
        cos_dists_50km_refB.append(cos_dist(vect_refB, vcts_carac_refB_50km[i]))
        cos_dists_75km_refB.append(cos_dist(vect_refB, vcts_carac_refB_75km[i]))
        cos_dists_95km_refB.append(cos_dist(vect_refB, vcts_carac_refB_95km[i]))
    
    idx_falla_elegida_10km = np.argmax(cos_dists_10km_refB)
    idx_falla_elegida_20km = np.argmax(cos_dists_20km_refB)
    idx_falla_elegida_50km = np.argmax(cos_dists_50km_refB)
    idx_falla_elegida_75km = np.argmax(cos_dists_75km_refB)
    idx_falla_elegida_95km = np.argmax(cos_dists_95km_refB)

    indices = [idx_falla_elegida_10km, idx_falla_elegida_20km, idx_falla_elegida_50km, idx_falla_elegida_75km, idx_falla_elegida_95km]

    idx_falla_elegida = most_frequent(indices)
    
    if idx_falla_elegida==idxNoFault:
        falla_elegida = 'NoFault'
    elif idx_falla_elegida==idxAB:
        falla_elegida = 'AB'
    elif idx_falla_elegida==idxBC:
        falla_elegida = 'BC'
    elif idx_falla_elegida==idxCA:
        falla_elegida = 'CA'
    elif idx_falla_elegida==idxABC:
        falla_elegida = 'ABC'
    elif idx_falla_elegida==idxAG:
        falla_elegida = 'AG'
    elif idx_falla_elegida==idxBG:
        falla_elegida = 'BG'
    elif idx_falla_elegida==idxCG:
        falla_elegida = 'CG'

    return falla_elegida

def comparar_vectores_refC(vect_refC):
    cos_dists_10km_refC = []
    cos_dists_20km_refC = []
    cos_dists_50km_refC = []
    cos_dists_75km_refC = []
    cos_dists_95km_refC = []
    
    idx_falla_elegida = 0
    
    for i in range(len(vcts_carac_refC_10km)):
        cos_dists_10km_refC.append(cos_dist(vect_refC, vcts_carac_refC_10km[i]))
        cos_dists_20km_refC.append(cos_dist(vect_refC, vcts_carac_refC_20km[i]))
        cos_dists_50km_refC.append(cos_dist(vect_refC, vcts_carac_refC_50km[i]))
        cos_dists_75km_refC.append(cos_dist(vect_refC, vcts_carac_refC_75km[i]))
        cos_dists_95km_refC.append(cos_dist(vect_refC, vcts_carac_refC_95km[i]))
    
    idx_falla_elegida_10km = np.argmax(cos_dists_10km_refC)
    idx_falla_elegida_20km = np.argmax(cos_dists_20km_refC)
    idx_falla_elegida_50km = np.argmax(cos_dists_50km_refC)
    idx_falla_elegida_75km = np.argmax(cos_dists_75km_refC)
    idx_falla_elegida_95km = np.argmax(cos_dists_95km_refC)

    indices = [idx_falla_elegida_10km, idx_falla_elegida_20km, idx_falla_elegida_50km, idx_falla_elegida_75km, idx_falla_elegida_95km]

    idx_falla_elegida = most_frequent(indices)
    
    if idx_falla_elegida==idxNoFault:
        falla_elegida = 'NoFault'
    elif idx_falla_elegida==idxAB:
        falla_elegida = 'AB'
    elif idx_falla_elegida==idxBC:
        falla_elegida = 'BC'
    elif idx_falla_elegida==idxCA:
        falla_elegida = 'CA'
    elif idx_falla_elegida==idxABC:
        falla_elegida = 'ABC'
    elif idx_falla_elegida==idxAG:
        falla_elegida = 'AG'
    elif idx_falla_elegida==idxBG:
        falla_elegida = 'BG'
    elif idx_falla_elegida==idxCG:
        falla_elegida = 'CG'

    return falla_elegida
# Función que hace el proceso de obtener las transformadas de Clarke en todas las referencias, hacer la regresión elíptica y sacar valores y
# vectores propios
def procesamiento(Va, Vb, Vc):
    
    # Transformadas de Clarke
    alpha_refA, beta_refA, z_refA = Clarke_ref_A(Va, Vb, Vc)
    alpha_refB, beta_refB, z_refB = Clarke_ref_B(Vb, Vc, Va)
    alpha_refC, beta_refC, z_refC = Clarke_ref_C(Vc, Va, Vb)

    # Crear matrices 2xm
    x_refA = create_x(alpha_refA, beta_refA)
    x_refB = create_x(alpha_refB, beta_refB)
    x_refC = create_x(alpha_refC, beta_refC)

    # Ellipse fitting con todos los datos
    abcdef_refA = fit_ellipse(x_refA[:,0], x_refA[:,1])
    abcdef_refB = fit_ellipse(x_refB[:,0], x_refB[:,1])
    abcdef_refC = fit_ellipse(x_refC[:,0], x_refC[:,1])

    # Construir matriz Q
    Q_refA, _, _ = get_ellipse_params(abcdef_refA)
    Q_refB, _, _ = get_ellipse_params(abcdef_refB)
    Q_refC, _, _ = get_ellipse_params(abcdef_refC)

    # Sacar valores y vectores propios
    valsA, vectsA = np.linalg.eig(Q_refA)
    valsB, vectsB = np.linalg.eig(Q_refB)
    valsC, vectsC = np.linalg.eig(Q_refC)

    return valsA, vectsA, valsB, vectsB, valsC, vectsC

def clasificar_eigvals(Va, Vb, Vc):

    # Procesamiento de las tres señales de voltaje.
    valsA, vectsA, valsB, vectsB, valsC, vectsC =  procesamiento(Va, Vb, Vc)

    # Miramos los valores propios de cada referencia y los comparamos con los valores propios característicos de cada falla y distancia.
    fallaA, distA = comparar_valores_refA(valsA)
    fallaB, distB = comparar_valores_refB(valsB)
    fallaC, distC = comparar_valores_refC(valsC)

    # Elegir distancia dependiendo de los votos de cada referencia de la transformada de Clarke.
    if distA == distB and distA != distC:
        return fallaA, distancias[distancias.index(distA)]
    elif distB == distC and distB != distA:
        return fallaB, distancias[distancias.index(distB)]
    elif distC == distA and distC != distB:
        return fallaC, distancias[distancias.index(distC)]
    elif distA == distB and distA == distC:
        return fallaA, distancias[distancias.index(distA)]
    else:
        return fallaA, distA

def clasificar_eigvects(Va, Vb, Vc):

    # Procesamiento de las tres señales de voltaje
    valsA, vectsA, valsB, vectsB, valsC, vectsC = procesamiento(Va, Vb, Vc)

    # Miramos los vectores propios de cada referencia y los comparamos con los vectores propios característicos de cada falla.
    falla_refA = comparar_vectores_refA(vectsA[:,0])
    falla_refB = comparar_vectores_refB(vectsB[:,0])
    falla_refC = comparar_vectores_refC(vectsC[:,0])

    # Elegir falla dependiendo de los votos de cada referencia de la transformada de Clarke.
    if falla_refA == falla_refB and falla_refA != falla_refC:
        return falla_refA
    elif falla_refB == falla_refC and falla_refB != falla_refA:
        return falla_refB
    elif falla_refC == falla_refA and falla_refC != falla_refB:
        return falla_refC
    elif falla_refA == falla_refB and falla_refA == falla_refC:
        return falla_refA
    else:
        return 'falla_indecisa'

# Función que llama las anteriores 9 funciones. Le entran los voltajes pertinentes, retorna tipo de falla y distancia.
def classify_faults(Va, Vb, Vc):
    """
    Entradas:
    Va -> vector de datos que representa la señal de voltaje en la fase A
    Vb -> vector de datos que representa la señal de voltaje en la fase B
    Vc -> vector de datos que representa la señal de voltaje en la fase C

    Salidas:
    falla -> resultado de la clasificación de la falla 
    distancia -> resultado de la clasificación de la distancia 
    """    

    falla_elegida1, dist_elegida = clasificar_eigvals(Va, Vb, Vc)
    falla_elegida = clasificar_eigvects(Va, Vb, Vc)

    return falla_elegida, dist_elegida

def most_frequent(List):
    counter = 0
    num = List[0]
     
    for i in List:
        curr_frequency = List.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i
 
    return num

# Tamaño de la ventana
m = 2000

NoFaults = ['NoFault' for x in range(5)]
ABs = ['AB' for x in range(5)]
BCs = ['BC' for x in range(5)]
CAs = ['CA' for x in range(5)]
ABCs = ['ABC' for x in range(5)]
AGs = ['AG' for x in range(5)]
BGs = ['BG' for x in range(5)]
CGs = ['CG' for x in range(5)]

fallas_true = [NoFaults, ABs, BCs, CAs, ABCs, AGs, BGs, CGs]
fallas_true = list(itertools.chain.from_iterable(fallas_true))

fallas_pred = []
dists_pred = []

enes = [x for x in range(5)]*8 # 40 archivos.

distancia = input("Distancia?")
dists_true = [int(distancia) for x in range(len(fallas_true))]

# Cuando se acabe este for, ya habremos leido los 5*8 = 40 archivos de TODAS las fallas y UNA distancia.
# Hay que tomar los datos así: NoFault*5, AB*5, BC*5,...
for j in range(len(enes)):
    i = enes[j]

    falla = fallas_true[j]

    path = f'Pruebas estabilidad/Invertidos/datos_valid/{falla}/voltajes_{falla}_{distancia}km_{i}.xlsx'
    Va_list, Vb_list, Vc_list = load_signals(path)

    fallas_elegidas_por_ventana = []
    dists_elegidas_por_ventana = []
    
    # Movimiento de la ventana
    for j in range(0, len(Va_list), m):

        Va_actual = Va_list[j:j+m]
        Vb_actual = Vb_list[j:j+m]
        Vc_actual = Vc_list[j:j+m]

        # Hacemos la clasificación en cada movimiento de la ventana.
        falla_elegida, dist_elegida = classify_faults(Va_actual, Vb_actual, Vc_actual)

        # Agregamos el resultado de la clasificación de esa ventana a la lista.
        fallas_elegidas_por_ventana.append(falla_elegida)
        dists_elegidas_por_ventana.append(dist_elegida)

    # La falla y distancia elegida final es la falla/distancia que más retornó el clasificador en cada ventana.
    falla_elegida_final = most_frequent(fallas_elegidas_por_ventana)
    dist_elegida_final = most_frequent(dists_elegidas_por_ventana)

    # Agregamos esas fallas/distancias finales a las listas de predicciones finales.
    fallas_pred.append(falla_elegida_final)
    dists_pred.append(dist_elegida_final)

conf_mat_fallas = confusion_matrix(fallas_true, fallas_pred, labels=fallas)
fila_conf_mat_dists = confusion_matrix(dists_true, dists_pred, labels=distancias)

accuracy_fallas = accuracy_score(fallas_true, fallas_pred)
correctas_fallas = accuracy_score(fallas_true, fallas_pred, normalize=False)
df_accuracy_fallas = pd.DataFrame([accuracy_fallas, correctas_fallas], index=['Percent', 'Correct'], columns=['Accuracy'])

df_conf_mat_fallas = pd.DataFrame(conf_mat_fallas, index=fallas, columns=fallas)
fila_df_conf_mat_dists = pd.DataFrame(fila_conf_mat_dists, index=distancias, columns=distancias)

df_conf_mat_fallas.to_excel(f'Pruebas estabilidad/Invertidos/una_ventana/matrices_conf_fallas/m{m}/mat_conf_fallas_{distancia}km.xlsx')
df_accuracy_fallas.to_excel(f'Pruebas estabilidad/Invertidos/una_ventana/matrices_conf_fallas/m{m}/accuracy_{distancia}km_m{m}.xlsx')
fila_df_conf_mat_dists.to_excel(f'Pruebas estabilidad/Invertidos/una_ventana/matrices_conf_dists/m{m}/fila{distancia}km.xlsx')