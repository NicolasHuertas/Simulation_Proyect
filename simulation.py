import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import time

#Función de NAvier-Stokes
def navSt(u0, v0):
    # u0: Valores iniciales de la funcion U
    # v0: Valores iniciales de la función V
    l = int(np.sqrt(len(v0)))  # Longitud de la matriz
    u = np.zeros(len(u0))  # Vector de valores de la funcion F(U)
    v = u  # Vector de valores de la funcion F(V)
    vel0 = 1
    for i in range(len(u0)):
        if i == 0:  # Esquina (0,0)
            u[i] = u0[i] * (u0[i + 1] - vel0) / 2 + v0[i] * (u0[i + l] - vel0) / 2 - u0[i + 1] + 4 * u0[i] - u0[
                i + l] - 2 * vel0
        elif i < 4:  # Primera Fila (0,:)
            u[i] = u0[i] * (u0[i + 1] - u0[i - 1]) / 2 + v0[i] * (u0[i + l] - vel0) / 2 - u0[i + 1] - u0[i - 1] + 4 * \
                   u0[i] - u0[i + l] - vel0
        elif i == 4 or i == 5 or i == 11 or i == 26 or (i > 30 and i < 34):  # Vigas y puntos en contacto con ellas
            u[i] = 0
        elif i == l or i == 2 * l or i == 3 * l:  # Primera columna (:,0)
            u[i] = u0[i] * (u0[i + 1] - vel0) / 2 + v0[i] * (u0[i + l] - u0[i - l]) / 2 - u0[i + 1] + 4 * u0[i] - u0[
                i + l] - u0[i - l] - vel0
        elif i == 3 * l - 1 or i == 4 * l - 1 or i == 5 * l - 1:  # Ultima columna (:,l)
            u[i] = u0[i] * (-u0[i - 1]) / 2 + v0[i] * (u0[i + l] - u0[i - l]) / 2 - u0[i - 1] + 4 * u0[i] - u0[i + l] - \
                   u0[i - l]
        elif i == 30:  # Esquina (0,l)
            u[i] = u0[i] * (-vel0) / 2 + v0[i] * (-u0[i - l]) / 2 - vel0 + 4 * u0[i] - u0[i - l]
        elif i < 30:  # Valores intermedios
            u[i] = u0[i] * (u0[i + 1] - u0[i - 1]) / 2 + v0[i] * (u0[i + l] - u0[i - l]) / 2 - u0[i + 1] - u0[
                i - 1] + 4 * u0[i] - u0[i + l] - u0[i - l]
        elif i < len(u0) - 1:  # Ultima fila (l,:)
            u[i] = u0[i] * (u0[i + 1] - u0[i - 1]) / 2 + v0[i] * (-u0[i - l]) / 2 - u0[i + 1] - u0[i - 1] + 4 * u0[i] - \
                   u0[i - l]
        elif i == len(u0) - 1:  # Esquina (l,l)
            u[i] = u0[i] * (-u0[i - 1]) / 2 + v0[i] * (-u0[i - l]) / 2 - u0[i - 1] + 4 * u0[i] - u0[i - l]

    for i in range(len(v0)):
        if i == 0:  # Esquina (0,0)
            v[i] = v0[i] * v0[i + l] / 2 + u0[i] * v0[i + 1] / 2 - v0[i + 1] + 4 * v0[i] - v0[i + l]
        elif i < 4:  # Primera Fila (0,:)
            v[i] = v0[i] * (v0[i + l]) / 2 + u0[i] * (v0[i + 1] - v0[i - 1]) / 2 - v0[i + 1] - v0[i - 1] + 4 * v0[i] - \
                   v0[i + l]
        elif i == 4:  # En contacto con I
            v[i] = -2 * (u[i] - u[i + l])
        elif i == 5 or i == 32:  # Vigas
            v[i] = 0
        elif i == l or i == 2 * l or i == 3 * l:  # Primera columna (:,0)
            v[i] = u0[i] * (v0[i + 1]) / 2 + v0[i] * (v0[i + l] - v0[i - l]) / 2 - v0[i + 1] + 4 * v0[i] - v0[i + l] - \
                   v0[i - l]
        elif i == 11:  # En contacto con J
            v[i] = -2 * (-u0[i])
        elif i == 26:  # En contacto con C
            v[i] = -2 * (u0[i + 1] - u0[i])
        elif i == 31:  # En contacto con B
            v[i] = -2 * (u0[i - l] - u0[i])
        elif i == 33:  # En contacto con D
            v[i] = -2 * (-u0[i])
        elif i == 3 * l - 1 or i == 4 * l - 1 or i == 5 * l - 1:  # Ultima columna (:,l)
            v[i] = u0[i] * (-v0[i - 1]) / 2 + v0[i] * (v0[i + l] - v0[i - l]) / 2 - v0[i - 1] + 4 * v0[i] - v0[i + l] - \
                   v0[i - l]
        elif i < 30:  # Valores intermedios
            v[i] = v0[i] * (v0[i + l] - v0[i - l]) / 2 + u0[i] * (v0[i + 1] - v0[i - 1]) / 2 - v0[i + 1] - v0[
                i - 1] + 4 * v0[i] - v0[i + l] - v0[i - l]
        elif i < len(v0) - 1:  # Ultima Fila (l,:)
            v[i] = v0[i] * (-v0[i - l]) / 2 + u0[i] * (v0[i + 1] - v0[i - 1]) / 2 - v0[i + 1] - v0[i - 1] + 4 * v0[i] - \
                   v0[i - l]
        elif i == len(v0) - 1:  # Esquina (l,l)
            v[i] = v0[i] * (-v0[i - l]) / 2 + u0[i] * (-v0[i - 1]) / 2 - v0[i - 1] + 4 * v0[i] - v0[i - l]

    return [u, v]

#Cálculo del Jacobiano
def jacobiano(u0, v0):
    l = int(np.sqrt(len(v0)))
    vel0 = 1
    JU = np.zeros([len(u0), len(u0)])  # Inicializacion del Jacobiano de U
    JV = JU  # Inicializacion del Jacobiano de V
    for i in range(len(u0)):
        if i < l:  # Primera fila (0,:)
            JU[i, i + 1] = u0[i] / 2 - 1
            JU[i, i + l] = v0[i] / 2 - 1

            JV[i, i + 1] = JU[i, i + 1]
            JV[i, i + l] = JU[i, i + l]

            if i == 0:  # Esquina (0,0)
                JU[i, i] = (u0[i + 1] - vel0) / 2 + 4
                JV[i, i] = (v0[i + 1] - vel0) / 2 + 4
            elif i == 5:  # Esquina (0,l)
                JU[i, i] = -u0[i - 1] / 2 + 4
                JV[i, i] = -v0[i - 1] / 2 + 4
            else:  # Valores Intermedios
                JU[i, i] = (u0[i + 1] - u0[i - 1]) / 2 + 4
                JV[i, i] = (v0[i + 1] - v0[i - 1]) / 2 + 4

        elif i % l == 0:  # Primera columna (:,0)
            JU[i, i] = (u0[i + 1] - vel0) / 2 + 4
            JU[i, i + 1] = u0[i] / 2 - 1
            JU[i, i - l] = -v0[i] / 2 - 1

            JV[i, i] = (v0[i + 1] - vel0) / 2 + 4
            JV[i, i + 1] = u0[i] / 2 - 1
            JV[i, i - l] = -v0[i] / 2 - 1

            if i != 30:  # No esquina inferior
                JU[i, i + l] = v0[i] / 2 - 1
                JV[i, i + l] = v0[i] / 2 - 1

        elif i % l == l - 1:  # Ultima columna (:,l)
            JU[i, i] = -u0[i - 1] / 2 + 4
            JU[i, i - 1] = -u0[i] / 2 - 1
            JU[i, i - l] = -v0[i] / 2 - 1

            JV[i, i] = -v0[i - 1] / 2 + 4
            JV[i, i - 1] = -u0[i] / 2 - 1
            JV[i, i - l] = -v0[i] / 2 - 1

            if i != 35:  # No esquina inferior
                JU[i, i + l] = v0[i] / 2 - 1
                JV[i, i + l] = v0[i] / 2 - 1

        elif i > l * l - l:  # Ultima fila (l,:)
            JU[i, i] = (u0[i + 1] - u0[i - 1]) / 2 + 4
            JU[i, i + 1] = u0[i] - 1
            JU[i, i - 1] = -u0[i] - 1
            JU[i, i - l] = -v0[i] / 2 - 1

            JV[i, i] = (v0[i + 1] - v0[i - 1]) / 2 + 4
            JV[i, i + 1] = u0[i] - 1
            JV[i, i - 1] = -u0[i] - 1
            JV[i, i - l] = -v0[i] / 2 - 1

        else:  # Valores intermedios
            JU[i, i] = (u0[i + 1] - u0[i - 1]) / 2 + 4
            JU[i, i + 1] = u0[i] - 1
            JU[i, i - 1] = -u0[i] - 1
            JU[i, i - l] = -v0[i] / 2 - 1
            JU[i, i + l] = v0[i] / 2 - 1

            JV[i, i] = (v0[i + 1] - v0[i - 1]) / 2 + 4
            JV[i, i + 1] = u0[i] - 1
            JV[i, i - 1] = -u0[i] - 1
            JV[i, i - l] = -v0[i] / 2 - 1
            JV[i, i + l] = v0[i] / 2 - 1

    return [JU, JV]

#Transformación del vector solución a forma matricial
def matriz(x0):
    l = int(np.sqrt(len(x0)))
    m = np.zeros([l, l])
    for i in range(l):
        m[i, :] = x0[i * l:(1 + i) * l]
    return m

#Método de Richardson para la solución de sistemas de ecuaciones lineales
def richardson(A, b, x0, iter):
    for i in range(iter):
        x = np.matmul((np.identity(len(x0)) - A), x0) + b
        print(np.abs(np.linalg.norm(x - x0)))
        x0 = x
    return x0

#Método de Jacobi para la solución de sistemas de ecuaciones lineales
def jacobi(A, b, x0, iter):
    Q = np.zeros([len(x0), len(x0)])
    for i in range(len(x0)):
        Q[i, i] = A[i, i]
    for j in range(iter):
        x = np.matmul((np.identity(len(x0)) - np.matmul(np.linalg.inv(Q), A)), x0) + np.matmul(np.linalg.inv(Q), b)
        x0 = x
    return x0

#Método de Gauss-Seidel para la solución de sistemas de ecuaciones lineales
def gaussSeidel(A, b, x0, iter):
    Q = A
    for i in range(len(x0)):
        for j in range(len(x0)):
            if i > j:
                Q[i, j] = 0
    for j in range(iter):
        x = np.matmul((np.identity(len(x0)) - np.matmul(np.linalg.inv(Q), A)), x0) + np.matmul(np.linalg.inv(Q), b)
        x0 = x
    return x0

#Solución del Sistema usando el método de Newton
def solNS(fun, u0, v0, iter, iterml, tipo):
    # Fun: Funcion a evaluar
    # u0: Valores iniciales de u
    # v0: Valores iniciales de v
    # iter: Numero de iteraciones
    # Tipo de criterio -> 0=Newton, 1=Richardson, 2=Jacobi
    errU = np.zeros(iter)
    errV = errU
    l = int(np.sqrt(len(u0)))
    for i in range(iter):
        F0 = fun(u0, v0)  # Funcion evaluada en los valores iniciales

        J = jacobiano(u0, v0)

        if tipo == 0:  # Newton
            u = u0 + np.linalg.solve(J[0], -F0[0])
            v = v0 + np.linalg.solve(J[1], -F0[1])
            # u = u0 - np.matmul(np.linalg.inv(J[0]), F0[0])
            # v = v0 - np.matmul(np.linalg.inv(J[1]), F0[1])
        if tipo == 1:  # Richardson
            du = richardson(J[0], -F0[0], u0, iterml)
            dv = richardson(J[1], -F0[1], v0, iterml)
            u = u0 + du
            v = v0 + dv
            # print(np.linalg.norm(np.identity(len(v0))-J[1]))
        if tipo == 2:  # Jacobi
            du = jacobi(J[0], -F0[0], u0, iterml)
            dv = jacobi(J[1], -F0[1], v0, iterml)
            u = u0 + du
            v = v0 + dv

        if tipo == 3:  # Gauss-Seidel
            du = gaussSeidel(J[0], -F0[0], u0, iterml)
            dv = gaussSeidel(J[1], -F0[1], v0, iterml)
            u = u0 + du
            v = v0 + dv

        for j in range(4, len(u)):
            if j == 4 or j == 5 or j == 11 or j == 26 or (j > 30 and j < 34):
                u[j] = 0

        for k in range(4, len(v)):
            if k == 5 or k == 32:
                v[k] = 0
            # if k==4:
            #     v[k] = -2 * (u[k]-u[k+l])
            # elif k==10:
            #     v[k] = -2*(-u[k])
            # elif k==26:
            #     v[k]= -2*(u[k+1]-u[k])
            # elif k==31:
            #     v[k] = -2*(u[k-l]-u[k])
            # elif k==33:
            #     v[k] = -2*(-u[k])

        errU[i] = np.abs(np.linalg.norm(u - u0) / np.linalg.norm(u0))
        errV[i] = np.abs(np.linalg.norm(v - v0) / np.linalg.norm(v0))

        u0 = u
        v0 = v

    return [matriz(u0), matriz(v0), errU, errV]

#Función de interpolación con polinomios de Lagrange para un punto (x,y)
def lagrange(x, y, malla):
    xValues = np.arange(0, malla.shape[0])
    yValues = np.arange(0, malla.shape[1])
    lx = np.ones(len(xValues))
    ly = np.ones(len(xValues))
    z = 0

    for i in range(len(xValues)):
        for j in range(len(yValues)):
            if i != j:
                lx[i] *= ((x - xValues[j]) / (xValues[i] - xValues[j]))
                ly[i] *= ((y - yValues[j]) / (yValues[i] - yValues[j]))

    for i in range(len(lx)):
        for j in range(len(ly)):
            z += lx[i] * ly[j] * malla[i, j]

    return z

#Interpolación de splines cúbicos para un sistema con una sola variable
def bicubic_interpolation(x, a, t, ox):
    n = len(a) - 1
    b = np.zeros(n + 1)
    c = np.zeros(n + 1)
    d = np.zeros(n + 1)
    alpha = np.zeros(n + 1)
    l = np.zeros(n + 1)
    mu = np.zeros(n + 1)
    z = np.zeros(n + 1)
    h = np.zeros(n + 1)
    # Colocar valores de h
    for i in range(n):
        h[i] = x[i + 1] - x[i]
    for i in range(1,n):
        alpha[i] = 3 * ((a[i + 1] - a[i]) / h[i] - (a[i] - a[i - 1]) / h[i - 1])
    l[0] = 1
    mu[0] = 0
    z[0] = 0
    # Reduccion de Matriz
    for i in range(1, n):
        l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

    l[n] = 1
    z[n] = 0
    c[n] = 0
    for i in range(n):
        j = n - i - 1
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (a[j + 1] - a[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
        d[j] = (c[j + 1] - c[j]) / 3 * h[j]
    n2 = len(x)
    nt = (t) * (n2 - 1) + 1
    y = np.ones(nt)
    for i in range(n2):
        y[i * t] = a[i]
    for i in range(n2-1):
        index = i * t
        for j in range(index, index + t):
            y[j] = d[i] * ((ox[j] - ox[index]) ** 3) + c[i] * ((ox[j] - ox[index]) ** 2) + b[i] * (ox[j] - ox[index]) + a[i]
    return y

#Cálculo de Rendimiento de las funciones de interpolación
def calc_tiempos(ns, malla):
    print(ns)
    pasot = 1 / ns
    xit = np.arange(0, malla.shape[0])
    yit = np.arange(0, malla.shape[0])
    xt = np.arange(0, malla.shape[0] - 1 + pasot, pasot)
    yt = np.arange(0, malla.shape[0] - 1 + pasot, pasot)
    L = np.zeros((len(xt), len(yt)))
    RBS = np.zeros((len(xt), len(yt)))
    BIaux = np.zeros((len(xt), len(yit)))
    BI = np.zeros((len(xt), len(yt)))
    inter = interpolate.RectBivariateSpline(xit, yit, malla)
    start1 = time.time()
    for i in range(len(xt)):
        for k in range(len(yt)):
            L[i, k] = lagrange(xt[i], yt[k], malla)
    end1 = time.time()
    tl = end1 - start1

    start2 = time.time()
    for n in range(len(xt)):
        for m in range(len(yt)):
            RBS[n, m] = inter(xt[n], yt[m])
    end2 = time.time()
    trbs = end2 - start2

    start3 = time.time()
    for i in range(malla.shape[0]):
        BIaux[:, i] = bicubic_interpolation(xit, malla[:, i], int(1 / pasot), xt)
    for i in range(BIaux.shape[0]):
        BI[i, :] = bicubic_interpolation(xit, BIaux[i, :], int(1 / pasot), xt)
    end3 = time.time()
    tbi = end3 - start3

    return [tl, tbi, trbs]

#Valores iniciales de U y V
u0 = np.ones(36)

for j in range(4, len(u0)):
    if j == 4 or j == 5 or j == 11 or j == 26 or (j > 30 and j < 34):
        u0[j] = 0
v0 = np.ones(36)
for j in range(4, len(v0)):
    if j == 5 or j == 32:
        v0[j] = 0

tipo = 3 #Tipo de método lineal para la solución
tipos = ['linalg solve', 'Richardson', 'Jacobi', 'Gauss-Seidel']

NR = solNS(navSt, u0, v0, 5, 10, tipo)
matrix=NR[0] #Matriz a interpolar
paso = 1/10 #Distancia entre los nuevos valores de x y y a interpolar
xi = np.arange(0, matrix.shape[0]) #Rango original de x
yi = np.arange(0, matrix.shape[0]) #Rango original de y
x = np.arange(0, matrix.shape[0] - 1 + paso, paso) #Nuevos valores de x
y = np.arange(0, matrix.shape[0] - 1 + paso, paso) #Nuevos valores de y
#Polinomio de interpolación con el método de splines bicubicos generalizado
interp = interpolate.RectBivariateSpline(xi, yi, matrix)
# print(bicubic_interpolation(xi,yi,NR[0],x,y))
zL = np.zeros((len(x), len(y))) #Matriz resultante con polinomios de LAgrange
zRBS = np.zeros((len(x), len(y))) #Matriz resultante con el método de splines bicubicos generalizado
zBIaux = np.zeros((len(x), len(yi))) #Matriz auxiliar para la aproximacion por splines bicubios
zBI = np.zeros((len(x), len(y))) #Matriz resultante con la aproximacion por splines bicubios

#Interpolacion de valores con polinomios de lagrange y el método de splines bicubicos generalizado
for i in range(len(x)):
    for k in range(len(y)):
        zL[i, k] = lagrange(x[i], y[k], matrix)
        zRBS[i, k] = interp(x[i], y[k])

#Interpolacion con con la aproximacion por splines bicubios
for i in range(matrix.shape[0]):
    zBIaux[:, i] = bicubic_interpolation(xi, matrix[:, i], int(1 / paso), x)
for i in range(zBIaux.shape[0]):
    zBI[i, :] = bicubic_interpolation(xi, zBIaux[i, :], int(1 / paso), x)

# #Calculo de Rendimiento
# new_size = [2, 4, 5, 10, 20, 40, 50, 80, 100]
# tL = np.zeros(len(new_size))
# tBI = np.zeros(len(new_size))
# tRBS = np.zeros(len(new_size))
#
# for j in range(len(new_size)):
#     t = calc_tiempos(new_size[j], NR[0])
#     tL[j] = t[0]
#     tBI[j] = t[1]
#     tRBS[j] = t[2]

figure, axis = plt.subplots(2, 2)
axis[0, 0].imshow(matrix)
axis[0, 0].set_title('Malla original')
axis[0, 1].imshow(zL)
axis[0, 1].set_title('Lagrange')
axis[1, 1].imshow(zRBS)
axis[1, 1].set_title('scipy.RectBivariateSpline')
axis[1, 0].imshow(zBI)
axis[1, 0].set_title('Bicubic Splines')
#
# axis[2].plot(NR[2])
# print(NR[1])
# plt.imshow(z)
# plt.colorbar()
plt.show()

# plt.plot(NR[2])
# plt.title('Error con '+tipos[tipo])
# plt.ylabel('Error en U')
# plt.xlabel('Número de Iteraciones de Newton-Raphson')

plt.show()
z = [zL, zBI, zRBS]
zTitles = ['Lagrange', 'Bicubic Splines', 'Scipy.RectBivariateSpline']
X, Y = np.meshgrid(x, y)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        ax.scatter(i, j, matrix[j, i], color="black")

ax.plot_surface(X, Y, z[1], color='orange')
plt.title('Superficie usando '+zTitles[1])
plt.show()

# fig2, ax2 = plt.subplots()
# plt.title('Rendimiento de las Funciones')
# plt.xlabel('Tamaño de paso (1/h)')
# plt.ylabel('Tiempo transcurrido (s)')
# ax2.plot(new_size, tL, label='Lagrange')
# ax2.plot(new_size, tRBS, label='Scipy Bivariate Spline')
# ax2.plot(new_size, tBI, label='Bicubic Spline')
# plt.grid()
# plt.legend()
# plt.autoscale(enable=None, axis="x", tight=True)
# plt.show()
