#!/usr/bin/env python
# coding: utf-8

imprimir="phi"

import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.chebyshev as pch

N=18
M=int(N/2)
N0=N
r_m1=0.1
r_m2=2.5
r_m2m=5.0
r_m3=25.0

#sigmax=25.
#sigmax=64.

sigmax=520

phi0=2.23e-18

datafile="g-10.00_h20.00/lapse_5.53e-01.dat"



pi=3.14159265359
x=np.zeros(N+1)
w=np.zeros(N+1)

for i in range(N+1):
    x[i]=np.cos(pi*i/N)
    if i==0 or i==N:
        w[i]=pi/(2.*N)
    else:
        w[i]=pi/N

#Interpolante (ojo: TODOS en N+1 coeficientes)
# de f(x)

def I_f(f):
    sum1=np.zeros(N+1)
    sum2=np.zeros(N+1)
    for nn in range(N+1):
        a=np.zeros(N+1)
        a[nn]=1
        for ii in range(N+1):
            sum1[nn]=f(x[ii])*pch.Chebyshev(a)(x[ii])*w[ii]+sum1[nn]
            sum2[nn]=pch.Chebyshev(a)(x[ii])*pch.Chebyshev(a)(x[ii])*w[ii]+sum2[nn]
    return sum1/sum2


g=np.loadtxt(datafile)
M0=int(N0/2)
u10=np.zeros(N0+1)
for ii in np.arange(M0+1):
    u10[2*ii]=g[ii]
u20=g[M0+1:M0+1+N0+1]
u20m=g[M0+1+N0+1:M0+1+2*N0+2]
u30=g[M0+1+2*N0+2:M0+1+3*N0+3]
u40=g[M0+1+3*N0+3:M0+1+4*N0+4]
v10=np.zeros(N0+1)
for ii in np.arange(M0+1):
    v10[2*ii]=g[ii+M0+1+4*N0+4]
v20=g[2*M0+2+4*N0+4:2*M0+2+5*N0+5]
v20m=g[2*M0+2+5*N0+5:2*M0+2+6*N0+6]
v30=g[2*M0+2+6*N0+6:2*M0+2+7*N0+7]
v40=g[2*M0+2+7*N0+7:2*M0+2+8*N0+8]
w10=np.zeros(N0+1)
for ii in np.arange(M0+1):
    w10[2*ii]=g[ii+2*M0+2+8*N0+8]
w20=g[3*M0+3+8*N0+8:3*M0+3+9*N0+9]
w20m=g[3*M0+3+9*N0+9:3*M0+3+10*N0+10]
w30=g[3*M0+3+10*N0+10:3*M0+3+11*N0+11]
w40=g[3*M0+3+11*N0+11:3*M0+3+12*N0+12]
 
xx=np.linspace(-1,1,1000)

#Masa ADM

#Masa ADM
if imprimir=="M":
    MADM=-4.*r_m3*pch.Chebyshev.deriv(pch.Chebyshev(w40), m=1)(xx)
    MKomar=2.*r_m3*pch.Chebyshev.deriv(pch.Chebyshev(v40), m=1)(xx)
    plt.plot(xx,MADM, color="k", label=r"$M_{ADM}=-\frac{1}{2\pi}\oint_{S}\partial_rNr^2\sin\theta d\theta d\varphi$")
    #plt.plot(xx,MKomar, color="r", linestyle="--", label=r"$M_{Komar}=\frac{1}{4\pi}\oint_{S}\partial_r\Psi r^2\sin\theta d\theta d\varphi$")
    #plt.text(0.9,+0.63, r"$\infty$", fontsize=16)
    plt.legend(loc="lower right", fontsize=14)
    plt.xlabel(r"$x_2$", fontsize=16)
    plt.ylabel(r"$M$", fontsize=16)
    plt.xlim(-1,1)

#Campo
if imprimir=="phi":
    campo1=pch.Chebyshev(u10)(xx)
    campo2=pch.Chebyshev(u20)(xx)
    campo2m=pch.Chebyshev(u20m)(xx)
    campo3=pch.Chebyshev(u30)(xx)
    campo4=pch.Chebyshev(u40)(xx)

    epsilon=5.0e-10
    r1=lambda x:r_m1*x
    r2=lambda x:0.5*(r_m2-r_m1)*x+0.5*(r_m2+r_m1)
    r2m=lambda x:0.5*(r_m2m-r_m2)*x+0.5*(r_m2m+r_m2)
    r3=lambda x:0.5*(r_m3-r_m2m)*x+0.5*(r_m3+r_m2m)
    r4=lambda x:2.0*r_m3/(1.0-x+epsilon)

    plt.plot(r_m1*xx,campo1, color="k")
    plt.plot(0.5*(r_m2-r_m1)*xx+0.5*(r_m2+r_m1),campo2, color="k", linestyle="--")
    plt.plot(0.5*(r_m2m-r_m2)*xx+0.5*(r_m2m+r_m2),campo2m, color="orange", linestyle="--")
    plt.plot(0.5*(r_m3-r_m2m)*xx+0.5*(r_m3+r_m2m),campo3, color="k", linestyle="-.")
    plt.plot(2.*r_m3/(1.0-xx+epsilon),campo4, color="k", linestyle=":")

    
    plt.xlim(0,20)
    #plt.xlim(135,145.0)
    #plt.ylim(0,0.002)
    #plt.yscale("log")
    #plt.ylim(0.8,1.0)

    #plt.plot(xx,campo4, color="k")
plt.savefig('w.png', dpi=400)
