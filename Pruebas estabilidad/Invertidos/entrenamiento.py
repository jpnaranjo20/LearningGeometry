##### SCRIPT QUE LLEVA A CABO EL PROCESO DE ENTRENAMIENTO
from venv import create
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
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

# Listas de archivos de fallas simuladas en tiempo real
fallas_NoFault_10km = [f'dato_train/NoFault/voltajes_NoFault_10km_{i}.xlsx' for i in range(5)]
fallas_NoFault_20km = [f'dato_train/NoFault/voltajes_NoFault_20km_{i}.xlsx' for i in range(5)]
fallas_NoFault_50km = [f'dato_train/NoFault/voltajes_NoFault_50km_{i}.xlsx' for i in range(5)]
fallas_NoFault_75km = [f'dato_train/NoFault/voltajes_NoFault_75km_{i}.xlsx' for i in range(5)]
fallas_NoFault_95km = [f'dato_train/NoFault/voltajes_NoFault_95km_{i}.xlsx' for i in range(5)]

fallas_AB_10km = [f'dato_train/AB/voltajes_AB_10km_{i}.xlsx' for i in range(5)]
fallas_AB_20km = [f'dato_train/AB/voltajes_AB_20km_{i}.xlsx' for i in range(5)]
fallas_AB_50km = [f'dato_train/AB/voltajes_AB_50km_{i}.xlsx' for i in range(5)]
fallas_AB_75km = [f'dato_train/AB/voltajes_AB_75km_{i}.xlsx' for i in range(5)]
fallas_AB_95km = [f'dato_train/AB/voltajes_AB_95km_{i}.xlsx' for i in range(5)]

fallas_BC_10km = [f'dato_train/BC/voltajes_BC_10km_{i}.xlsx' for i in range(5)]
fallas_BC_20km = [f'dato_train/BC/voltajes_BC_20km_{i}.xlsx' for i in range(5)]
fallas_BC_50km = [f'dato_train/BC/voltajes_BC_50km_{i}.xlsx' for i in range(5)]
fallas_BC_75km = [f'dato_train/BC/voltajes_BC_75km_{i}.xlsx' for i in range(5)]
fallas_BC_95km = [f'dato_train/BC/voltajes_BC_95km_{i}.xlsx' for i in range(5)]

fallas_CA_10km = [f'dato_train/CA/voltajes_CA_10km_{i}.xlsx' for i in range(5)]
fallas_CA_20km = [f'dato_train/CA/voltajes_CA_20km_{i}.xlsx' for i in range(5)]
fallas_CA_50km = [f'dato_train/CA/voltajes_CA_50km_{i}.xlsx' for i in range(5)]
fallas_CA_75km = [f'dato_train/CA/voltajes_CA_75km_{i}.xlsx' for i in range(5)]
fallas_CA_95km = [f'dato_train/CA/voltajes_CA_95km_{i}.xlsx' for i in range(5)]

fallas_ABC_10km = [f'dato_train/ABC/voltajes_ABC_10km_{i}.xlsx' for i in range(5)]
fallas_ABC_20km = [f'dato_train/ABC/voltajes_ABC_20km_{i}.xlsx' for i in range(5)]
fallas_ABC_50km = [f'dato_train/ABC/voltajes_ABC_50km_{i}.xlsx' for i in range(5)]
fallas_ABC_75km = [f'dato_train/ABC/voltajes_ABC_75km_{i}.xlsx' for i in range(5)]
fallas_ABC_95km = [f'dato_train/ABC/voltajes_ABC_95km_{i}.xlsx' for i in range(5)]

fallas_AG_10km = [f'dato_train/AG/voltajes_AG_10km_{i}.xlsx' for i in range(5)]
fallas_AG_20km = [f'dato_train/AG/voltajes_AG_20km_{i}.xlsx' for i in range(5)]
fallas_AG_50km = [f'dato_train/AG/voltajes_AG_50km_{i}.xlsx' for i in range(5)]
fallas_AG_75km = [f'dato_train/AG/voltajes_AG_75km_{i}.xlsx' for i in range(5)]
fallas_AG_95km = [f'dato_train/AG/voltajes_AG_95km_{i}.xlsx' for i in range(5)]

fallas_BG_10km = [f'dato_train/BG/voltajes_BG_10km_{i}.xlsx' for i in range(5)]
fallas_BG_20km = [f'dato_train/BG/voltajes_BG_20km_{i}.xlsx' for i in range(5)]
fallas_BG_50km = [f'dato_train/BG/voltajes_BG_50km_{i}.xlsx' for i in range(5)]
fallas_BG_75km = [f'dato_train/BG/voltajes_BG_75km_{i}.xlsx' for i in range(5)]
fallas_BG_95km = [f'dato_train/BG/voltajes_BG_95km_{i}.xlsx' for i in range(5)]

fallas_CG_10km = [f'dato_train/CG/voltajes_CG_10km_{i}.xlsx' for i in range(5)]
fallas_CG_20km = [f'dato_train/CG/voltajes_CG_20km_{i}.xlsx' for i in range(5)]
fallas_CG_50km = [f'dato_train/CG/voltajes_CG_50km_{i}.xlsx' for i in range(5)]
fallas_CG_75km = [f'dato_train/CG/voltajes_CG_75km_{i}.xlsx' for i in range(5)]
fallas_CG_95km = [f'dato_train/CG/voltajes_CG_95km_{i}.xlsx' for i in range(5)]

fallas_NoFault = [fallas_NoFault_10km, fallas_NoFault_20km, fallas_NoFault_50km, fallas_NoFault_75km, fallas_NoFault_95km]
fallas_AB = [fallas_AB_10km, fallas_AB_20km, fallas_AB_50km, fallas_AB_75km, fallas_AB_95km]
fallas_BC = [fallas_BC_10km, fallas_BC_20km, fallas_BC_50km, fallas_BC_75km, fallas_BC_95km]
fallas_CA = [fallas_CA_10km, fallas_CA_20km, fallas_CA_50km, fallas_CA_75km, fallas_CA_95km]
fallas_ABC = [fallas_ABC_10km, fallas_ABC_20km, fallas_ABC_50km, fallas_ABC_75km, fallas_ABC_95km]
fallas_AG = [fallas_AG_10km, fallas_AG_20km, fallas_AG_50km, fallas_AG_75km, fallas_AG_95km]
fallas_BG = [fallas_BG_10km, fallas_BG_20km, fallas_BG_50km, fallas_BG_75km, fallas_BG_95km]
fallas_CG = [fallas_CG_10km, fallas_CG_20km, fallas_CG_50km, fallas_CG_75km, fallas_CG_95km]

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

# Función para cargar las señales de entrenamiento
def load_signals(path):
    datos = pd.read_excel(path)
    datos_dict = datos.to_dict()

    Va = list(datos_dict['Va'].values())
    Vb = list(datos_dict['Vb'].values())
    Vc = list(datos_dict['Vc'].values())

    return Va, Vb, Vc

# Número de datos de entrenamiento por falla.
n = 5
enes = [x for x in range(5)]*8 # 40 archivos.

NoFaults = ['NoFault' for x in range(5)]
ABs = ['AB' for x in range(5)]
BCs = ['BC' for x in range(5)]
CAs = ['CA' for x in range(5)]
ABCs = ['ABC' for x in range(5)]
AGs = ['AG' for x in range(5)]
BGs = ['BG' for x in range(5)]
CGs = ['CG' for x in range(5)]

# Lista de etiquetas.
fallas_true = [NoFaults, ABs, BCs, CAs, ABCs, AGs, BGs, CGs]
fallas_true = list(itertools.chain.from_iterable(fallas_true))

# Vectores para guardar las transformadas de Clarke de cada dato de entrenamiento. Se inicializan como vectores de ceros, para posteriormente eliminar esta primera fila.
refA = np.zeros((1,2))
refB = np.zeros((1,2))
refC = np.zeros((1,2))

# Valores propios de cada referencia
eigVals_refA = []
eigVals_refB = []
eigVals_refC = []

# Vectores propios de cada referencia
eigVects_refA = []
eigVects_refB = []
eigVects_refC = []

# El usuario ingresa la distancia de inserción y el tipo de falla para el cual va a realizar el entrenamiento.
# Para distancias se reciben valores númericos entre 10, 20, 50, 75 y 95.
# Para fallas se recibe uno de los siguientes strings: NoFault, AB, BC, CA, ABC, AG, BG o CG.
distancia = input('Distancia: ')
falla = input('Falla: ')
for j in range(n):
    # Se leen los datos de entrenamiento
    path = f'Pruebas estabilidad/Invertidos/datos_train/{falla}/voltajes_{falla}_{distancia}km_{j}.xlsx'
    Va_list, Vb_list, Vc_list = load_signals(path)

    # Se calcula la transformada de Clarke en cada referencia y el resultado se agrega a su vector correspondiente
    a_Va_refA, b_Vb_refA, z_Vc_refA = Clarke_ref_A(Va_list, Vb_list, Vc_list)
    x_refA = create_x(a_Va_refA, b_Vb_refA)
    refA = np.vstack([x_refA, refA]) # Función para guardar en el vector correspondiente.

    a_Va_refB, b_Vb_refB, z_Vc_refB = Clarke_ref_B(Vb_list, Vc_list, Va_list)
    x_refB = create_x(a_Va_refB, b_Vb_refB)
    refB = np.vstack([x_refB, refB]) # Función para guardar en el vector correspondiente.

    a_Va_refC, b_Vb_refC, z_Vc_refC = Clarke_ref_C(Vc_list, Va_list, Vb_list)
    x_refC = create_x(a_Va_refC, b_Vb_refC)
    refC = np.vstack([x_refC, refC]) # Función para guardar en el vector correspondiente.

# Cuando se sale de este for, se tienen tres vectores llenos de todas las 5 observaciones de la transformada de Clarke (uno por cada referencia)

# Se eliminan las primeras filas de las transformadas de Clarke de todas las referencias (se crearon por primera vez como un vector de ceros).
refA = np.delete(refA, 0, 0)
refB = np.delete(refB, 0, 0)
refC = np.delete(refC, 0, 0)

# Se lleva a cabo la regresión elíptica
abcdef_refA = fit_ellipse(refA[:,0], refA[:,1])
abcdef_refB = fit_ellipse(refB[:,0], refB[:,1])
abcdef_refC = fit_ellipse(refC[:,0], refC[:,1])

# Se construye la matriz Q de cada referencia
Q_refA, _, _ = get_ellipse_params(abcdef_refA)
Q_refB, _, _ = get_ellipse_params(abcdef_refB)
Q_refC, _, _ = get_ellipse_params(abcdef_refC)

# Se obtienen los valores y vectores propios de cada matriz Q y se agregan a las listas correspondientes.
valsQA, vectsQA = np.linalg.eig(Q_refA)
valsQB, vectsQB = np.linalg.eig(Q_refB)
valsQC, vectsQC = np.linalg.eig(Q_refC)
eigVects_refA.append(np.round(vectsQA[:,0], 4))
eigVects_refB.append(np.round(vectsQB[:,0], 4))
eigVects_refC.append(np.round(vectsQC[:,0], 4))
eigVals_refA.append(np.round(valsQA, 4))
eigVals_refB.append(np.round(valsQB, 4))
eigVals_refC.append(np.round(valsQC, 4))

eigvcs = [eigVects_refA, eigVects_refB, eigVects_refC]
eigvals = [eigVals_refA, eigVals_refB, eigVals_refC]

df_vects = pd.DataFrame(eigvcs, index=['eigVect (RefA)', 'eigVect (RefB)', 'eigVect (RefC)'], columns=[falla])
df_vals = pd.DataFrame(eigvals, index=['eigVals (RefA)', 'eigVals (RefB)', 'eigVals (RefC)'], columns=[falla])

# Se guardan los valores y vectores propios característicos de cada tipo de falla en un archivo de excel aparte.
# Después cada archivo de estos se leerá en otro script para construir una sola tabla de valores/vectores propios característicos para
# cada distancia de inserción.
df_vals.to_excel(f'Pruebas estabilidad/Invertidos/eigvals_eigvects_invertidos/eigvals_{distancia}km_{falla}.xlsx')
df_vects.to_excel(f'Pruebas estabilidad/Invertidos/eigvals_eigvects_invertidos/eigvects_{distancia}km_{falla}.xlsx')