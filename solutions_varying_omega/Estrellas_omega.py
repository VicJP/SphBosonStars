#!/usr/bin/env python
print(" ")
print("**********************************************")
print("***   Solucion numerica                    ***")
print("***   para las estrellas de bosones        ***")
print("***   utilizando metodos espectrales       ***")
print("***   en la base de Chebyshev              ***")
print("***                                        ***")
print("***   Victor Jaramillo: Septiembre 2020    ***")
print("***         X O C H I M I L C O    MX      ***")
print("**********************************************")
print(" ")

import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.chebyshev as pch

import os

N=18
gg = -10.0          #THIS IS g
hh = 10.0          #THIS IS h 


#Ns=np.linspace(9.529482481577983277e-01,0.99,40)
#Ns=np.linspace(0.995,0.0,200)
Ns=np.linspace(0.998,0.0,400)


#Colocar N0!=N si queremos construir un dato inicial a partir de otro con N menor
N0=N
# HERE SUBSTITUTE " " with an specific file, to construct family of solutions starting from it
#datafile_N0=" "
#datafile_N0="start_point/omega_9.90e-01.dat"
datafile_N0="start_point/omega_9.95e-01.dat"






#

phi0_semilla=0.02/np.sqrt(2) * np.sqrt(8*3.14159265)   #gg=0
solution_folder="g"+"{:.2f}".format(gg)+"_h"+"{:.2f}".format(hh)
os.mkdir(solution_folder)

omega_inicial=0.95
r_m1=0.1
r_m2=2.5
r_m2m=5.0
r_m3=25.0

dV=1.0e-10   #ojo antes era 1e-10
tolf=5.0e-11
epsilon=5e-10  #antes 1e-10
M=int(N/2)

sigmax =64  #gg=0



uc_a=np.zeros(len(Ns))
omega_a=np.zeros(len(Ns))
MADM_a=np.zeros(len(Ns))
MKomar_a=np.zeros(len(Ns))
R99_a=np.zeros(len(Ns))
C99_a=np.zeros(len(Ns))
Rtest_a=np.zeros(len(Ns))
Ctest_a=np.zeros(len(Ns))

for i in np.arange(len(Ns)):
    if i==0 and datafile_N0==" ":
        dato_inicial="gaussiana"
        omegamanual=omega_inicial
    if i==0 and datafile_N0!=" ":
        dato_inicial="archivo"
        omegamanual=0.0
    if i!=0:
        dato_inicial="archivo"
        datafile_N0=" "
        omegamanual=0.0


    datafile=solution_folder+"/omega_"+"{:.2e}".format(Ns[i-1])+".dat"

    N00=Ns[i]


    #Numero de coeficientes, puntos de colocacion y pesos:
    pi=3.14159265359
    x=np.zeros(N+1)
    w=np.zeros(N+1)

    ##Gauss-Lobato##

    for ii in range(N+1):
        x[ii]=np.cos(pi*ii/N)
        if ii==0 or ii==N:
            w[ii]=pi/(2.*N)
        else:
            w[ii]=pi/N

    #Interpolante (ojo: TODOS en N+1 coeficientes)
    # de f(x)

    def I_f(f):
        sum1=np.zeros(N+1)
        sum2=np.zeros(N+1)
        for nn in range(N+1):
            a=np.zeros(N+1)
            a[nn]=1
            for ii in range(N+1):
                cheba=pch.Chebyshev(a)(x[ii])
                wii=w[ii]
                sum1[nn]=f[ii]*cheba*wii+sum1[nn]
                sum2[nn]=cheba**2*wii+sum2[nn]
        return sum1/sum2

    ##
    ##Formulas exactas en el espacio de coeficientes
    ##
    #1/x * funcion impar# #mejorado#
    def rm1f(un):
        d=np.zeros(N+1)
        d[N-2]=2*un[N-1]
        for ii in N-2-np.arange(len(un)-3):
            if ii!=1 and ii%2!=0:
                d[ii-1]=2.*un[ii]-d[ii+1]
            if ii==1:
                d[ii-1]=0.5*(2.*un[ii]-d[ii+1])
        return d
        
    ##
    ##Formulas aproximadas en el espacio de funciones
    ##
    #Derivada
    def fd(un):
        H=np.append(pch.Chebyshev.deriv(pch.Chebyshev(un), m=1).coef,0)
        return H
    #Segunda derivada
    def fdd(un):
        H=np.append(pch.Chebyshev.deriv(pch.Chebyshev(un), m=2).coef,[0,0])
        return H


    #############################
    #############################

    #Representacion espectral del "lado derecho" de las ecuaciones para, Estrellas de Bosones

    #DOMINIO 1
    def Psi_d_1(un,vn,wn,omega):
        phi=pch.Chebyshev(un)(x)
        NN=pch.Chebyshev(vn)(x)
        Psi=pch.Chebyshev(wn)(x)
        dphi=pch.Chebyshev.deriv(pch.Chebyshev(un), m=1)(x)
        sigma=0.25*Psi**5*((omega*phi/NN)**2+dphi**2/(r_m1**2*Psi**4)+phi**2+0.5*gg*phi**4+(1.0/3.0)*hh*phi**6)
        return sigma

    def N_d_1(un,vn,wn,omega):
        phi=pch.Chebyshev(un)(x)
        NN=pch.Chebyshev(vn)(x)
        Psi=pch.Chebyshev(wn)(x)
        dphi=pch.Chebyshev.deriv(pch.Chebyshev(un), m=1)(x)
        dNN=pch.Chebyshev.deriv(pch.Chebyshev(vn), m=1)(x)
        dPsi=pch.Chebyshev.deriv(pch.Chebyshev(wn), m=1)(x)
        sigma= 2.*dNN*dPsi/(r_m1**2*Psi)-NN*Psi**4*(2.0*omega**2*phi**2/NN**2-phi**2-0.5*gg*phi**4-(1.0/3.0)*hh*phi**6)
        return sigma

    def phi_d_1(un,vn,wn,omega):
        phi=pch.Chebyshev(un)(x)
        NN=pch.Chebyshev(vn)(x)
        Psi=pch.Chebyshev(wn)(x)
        dphi=pch.Chebyshev.deriv(pch.Chebyshev(un), m=1)(x)
        dNN=pch.Chebyshev.deriv(pch.Chebyshev(vn), m=1)(x)
        dPsi=pch.Chebyshev.deriv(pch.Chebyshev(wn), m=1)(x)
        sigma=-Psi**4*(1.0+gg*phi**2+hh*phi**4-omega**2/NN**2)*phi+dphi*dNN/(r_m1**2*NN)+2.0*dphi*dPsi/(r_m1**2*Psi)
        return sigma
        
    #DOMINIO 2
    rr2=0.5*(r_m2-r_m1)*x+0.5*(r_m2+r_m1)
    rr2d=0.5*(r_m2-r_m1)
    def DOS(un,vn,wn,omega):
        phi=pch.Chebyshev(un)(x)
        NN=pch.Chebyshev(vn)(x)
        Psi=pch.Chebyshev(wn)(x)
        dphi=(pch.Chebyshev.deriv(pch.Chebyshev(un), m=1)(x))
        dNN=pch.Chebyshev.deriv(pch.Chebyshev(vn), m=1)(x)
        dPsi=(pch.Chebyshev.deriv(pch.Chebyshev(wn), m=1)(x))
        ddphi=pch.Chebyshev.deriv(pch.Chebyshev(un), m=2)(x)
        ddNN=pch.Chebyshev.deriv(pch.Chebyshev(vn), m=2)(x)
        ddPsi=(pch.Chebyshev.deriv(pch.Chebyshev(wn), m=2)(x))
        sigma1=ddPsi/rr2d**2+2.0*dPsi/(rr2*rr2d)+0.25*Psi**5*((omega*phi/NN)**2+(dphi/rr2d)**2/(Psi**4)+(1.0+0.5*gg*phi**2+(1.0/3.0)*hh*phi**4)*phi**2)
        sigma2=ddNN/rr2d**2+2.0*dNN/(rr2*rr2d)+2.*dNN*dPsi/(rr2d**2*Psi)-NN*Psi**4*(2.0*omega**2*phi**2/NN**2-phi**2-0.5*gg*phi**4-(1.0/3.0)*hh*phi**6)
        sigma3=ddphi/rr2d**2+2.0*dphi/(rr2*rr2d)-Psi**4*(1.0+gg*phi**2+hh*phi**4-omega**2/NN**2)*phi+dphi*dNN/(rr2d**2*NN)+2.0*dphi*dPsi/(rr2d**2*Psi)
        return sigma3, sigma2, sigma1
        
    #DOMINIO 2m
    rr2m=0.5*(r_m2m-r_m2)*x+0.5*(r_m2m+r_m2)
    rr2dm=0.5*(r_m2m-r_m2)
    def DOSm(un,vn,wn,omega):
        phi=pch.Chebyshev(un)(x)
        NN=pch.Chebyshev(vn)(x)
        Psi=pch.Chebyshev(wn)(x)
        dphi=(pch.Chebyshev.deriv(pch.Chebyshev(un), m=1)(x))
        dNN=pch.Chebyshev.deriv(pch.Chebyshev(vn), m=1)(x)
        dPsi=(pch.Chebyshev.deriv(pch.Chebyshev(wn), m=1)(x))
        ddphi=pch.Chebyshev.deriv(pch.Chebyshev(un), m=2)(x)
        ddNN=pch.Chebyshev.deriv(pch.Chebyshev(vn), m=2)(x)
        ddPsi=(pch.Chebyshev.deriv(pch.Chebyshev(wn), m=2)(x))
        sigma1=ddPsi/rr2dm**2+2.0*dPsi/(rr2m*rr2dm)+0.25*Psi**5*((omega*phi/NN)**2+(dphi/rr2dm)**2/(Psi**4)+(1.0+0.5*gg*phi**2+(1.0/3.0)*hh*phi**4)*phi**2)
        sigma2=ddNN/rr2dm**2+2.0*dNN/(rr2m*rr2dm)+2.*dNN*dPsi/(rr2dm**2*Psi)-NN*Psi**4*(2.0*omega**2*phi**2/NN**2-phi**2-0.5*gg*phi**4-(1.0/3.0)*hh*phi**6)
        sigma3=ddphi/rr2dm**2+2.0*dphi/(rr2m*rr2dm)-Psi**4*(1.0+gg*phi**2+hh*phi**4-omega**2/NN**2)*phi+dphi*dNN/(rr2dm**2*NN)+2.0*dphi*dPsi/(rr2dm**2*Psi)
        return sigma3, sigma2, sigma1
        
    #DOMINIO 3
    rr3=0.5*(r_m3-r_m2m)*x+0.5*(r_m3+r_m2m)
    rr3d=0.5*(r_m3-r_m2m)
    def TRES(un,vn,wn,omega):
        phi=pch.Chebyshev(un)(x)
        NN=pch.Chebyshev(vn)(x)
        Psi=pch.Chebyshev(wn)(x)
        dphi=(pch.Chebyshev.deriv(pch.Chebyshev(un), m=1)(x))
        dNN=pch.Chebyshev.deriv(pch.Chebyshev(vn), m=1)(x)
        dPsi=(pch.Chebyshev.deriv(pch.Chebyshev(wn), m=1)(x))
        ddphi=pch.Chebyshev.deriv(pch.Chebyshev(un), m=2)(x)
        ddNN=pch.Chebyshev.deriv(pch.Chebyshev(vn), m=2)(x)
        ddPsi=(pch.Chebyshev.deriv(pch.Chebyshev(wn), m=2)(x))
        sigma1=ddPsi/rr3d**2+2.0*dPsi/(rr3*rr3d)+0.25*Psi**5*((omega*phi/NN)**2+(dphi/rr3d)**2/(Psi**4)+(1.0+0.5*gg*phi**2+(1.0/3.0)*hh*phi**4)*phi**2)
        sigma2=ddNN/rr3d**2+2.0*dNN/(rr3*rr3d)+2.*dNN*dPsi/(rr3d**2*Psi)-NN*Psi**4*(2.0*omega**2*phi**2/NN**2-phi**2-0.5*gg*phi**4-(1.0/3.0)*hh*phi**6)
        sigma3=ddphi/rr3d**2+2.0*dphi/(rr3*rr3d)-Psi**4*(1.0+gg*phi**2+hh*phi**4-omega**2/NN**2)*phi+dphi*dNN/(rr3d**2*NN)+2.0*dphi*dPsi/(rr3d**2*Psi)
        return sigma3, sigma2, sigma1

    #DOMINIO4
    srr=(1.0-x)/(2.0*r_m3)
    
    def CUATRO(un,vn,wn,omega):
        phi=pch.Chebyshev(un)(x)
        NN=pch.Chebyshev(vn)(x)
        Psi=pch.Chebyshev(wn)(x)
        dphi=pch.Chebyshev.deriv(pch.Chebyshev(un), m=1)(x)
        dNN=pch.Chebyshev.deriv(pch.Chebyshev(vn), m=1)(x)
        dPsi=pch.Chebyshev.deriv(pch.Chebyshev(wn), m=1)(x)
        ddPsi=pch.Chebyshev.deriv(pch.Chebyshev(wn), m=2)(x)
        ddNN=pch.Chebyshev.deriv(pch.Chebyshev(vn), m=2)(x)
        ddphi=pch.Chebyshev.deriv(pch.Chebyshev(un), m=2)(x)
        sigma1=0.25*(1.0-x)**4*ddPsi/r_m3**2+0.25*Psi**5*((omega*phi/NN)**2+srr**(2)*((1.0-x)*dphi)**2/(Psi**4)+(1.0+0.5*gg*phi**2+(1.0/3.0)*hh*phi**4)*phi**2)
        sigma2=0.25*(1.0-x)**4*ddNN/r_m3**2+0.5*(1.0-x)**4*dNN*dPsi/(Psi*r_m3**2)-NN*Psi**4*(2.0*omega**2*phi**2/NN**2-phi**2-0.5*gg*phi**4-(1.0/3.0)*hh*phi**6)
        sigma3=0.25*(1.0-x)**4*ddphi/r_m3**2-Psi**4*(1.0+gg*phi**2+hh*phi**4-omega**2/NN**2)*phi+0.25*(1.0-x)**4*dphi*dNN/(r_m3**2*NN)+0.5*(1.0-x)**4*dphi*dPsi/(r_m3**2*Psi)
        return sigma3, sigma2, sigma1

    #############################
    #############################

    #############################
    #############################
    def R(o):
        un1=o[0:M+1]
        un2=o[M+1:M+1+N+1]
        un2m=o[M+1+N+1:M+1+2*N+2]
        un3=o[M+1+2*N+2:M+1+3*N+3]
        un4=o[M+1+3*N+3:M+1+4*N+4]
        vn1=o[M+1+4*N+4:2*M+2+4*N+4]
        vn2=o[2*M+2+4*N+4:2*M+2+5*N+5]
        vn2m=o[2*M+2+5*N+5:2*M+2+6*N+6]
        vn3=o[2*M+2+6*N+6:2*M+2+7*N+7]
        vn4=o[2*M+2+7*N+7:2*M+2+8*N+8]
        wn1=o[2*M+2+8*N+8:3*M+3+8*N+8]
        wn2=o[3*M+3+8*N+8:3*M+3+9*N+9]
        wn2m=o[3*M+3+9*N+9:3*M+3+10*N+10]
        wn3=o[3*M+3+10*N+10:3*M+3+11*N+11]
        wn4=o[3*M+3+11*N+11:3*M+3+12*N+12]

        omega=o[3*M+3+12*N+12]
        
        un1_f=np.zeros(N+1)
        vn1_f=np.zeros(N+1)
        wn1_f=np.zeros(N+1)
        for ii in np.arange(M+1):
            un1_f[2*ii]=un1[ii]
            vn1_f[2*ii]=vn1[ii]
            wn1_f[2*ii]=wn1[ii]
            
        #definicion de coeficientes para terminos mezclados que involucran a 1/r
        phi_d_2,N_d_2,Psi_d_2=DOS(un2,vn2,wn2,omega)
        phi_d_2m,N_d_2m,Psi_d_2m=DOSm(un2m,vn2m,wn2m,omega)
        phi_d_3,N_d_3,Psi_d_3=TRES(un3,vn3,wn3,omega)
        phi_d_4,N_d_4,Psi_d_4=CUATRO(un4,vn4,wn4,omega)

        Lu1_f=pch.Chebyshev(fdd(un1_f)/r_m1**2+2.0*rm1f(fd(un1_f))/r_m1**2)(x)+phi_d_1(un1_f,vn1_f,wn1_f,omega)
        Lu2=phi_d_2
        Lu2m=phi_d_2m
        Lu3=phi_d_3
        Lu4=phi_d_4
        Lv1_f=pch.Chebyshev(fdd(vn1_f)/r_m1**2+2.0*rm1f(fd(vn1_f))/r_m1**2)(x)+N_d_1(un1_f,vn1_f,wn1_f,omega)
        Lv2=N_d_2
        Lv2m=N_d_2m
        Lv3=N_d_3
        Lv4=N_d_4
        Lw1_f=pch.Chebyshev(fdd(wn1_f)/r_m1**2+2.0*rm1f(fd(wn1_f))/r_m1**2)(x)+Psi_d_1(un1_f,vn1_f,wn1_f,omega)
        Lw2=Psi_d_2
        Lw2m=Psi_d_2m
        Lw3=Psi_d_3
        Lw4=Psi_d_4

        
        Lu1=np.zeros(M+1)
        Lv1=np.zeros(M+1)
        Lw1=np.zeros(M+1)
        for ii in np.arange(M+1):
            Lu1[ii]=Lu1_f[M+ii]
            Lv1[ii]=Lv1_f[M+ii]
            Lw1[ii]=Lw1_f[M+ii]
            
        
        residuou1=Lu1
        residuou2=Lu2
        residuou2m=Lu2m
        residuou3=Lu3
        residuou4=Lu4
        residuov1=Lv1
        residuov2=Lv2
        residuov2m=Lv2m
        residuov3=Lv3
        residuov4=Lv4
        residuow1=Lw1
        residuow2=Lw2
        residuow2m=Lw2m
        residuow3=Lw3
        residuow4=Lw4
    #############################
    #############################
        
        
    #############################
    #############################
        residuou1[M]=pch.Chebyshev(un1_f)(+1)-pch.Chebyshev(un2)(-1)
        residuou2[0]=pch.Chebyshev.deriv(pch.Chebyshev(un1_f), m=1)(+1)-(2.*r_m1/(r_m2-r_m1))*pch.Chebyshev.deriv(pch.Chebyshev(un2), m=1)(-1)
        residuou2[N]=pch.Chebyshev(un2)(+1)-pch.Chebyshev(un2m)(-1)
        residuou2m[0]=(1./(r_m2-r_m1))*pch.Chebyshev.deriv(pch.Chebyshev(un2), m=1)(+1)-(1./(r_m2m-r_m2))*pch.Chebyshev.deriv(pch.Chebyshev(un2m), m=1)(-1)
        residuou2m[N]=pch.Chebyshev(un2m)(+1)-pch.Chebyshev(un3)(-1)
        residuou3[0]=(1./(r_m2m-r_m2))*pch.Chebyshev.deriv(pch.Chebyshev(un2m), m=1)(+1)-(1./(r_m3-r_m2m))*pch.Chebyshev.deriv(pch.Chebyshev(un3), m=1)(-1)
        residuou3[N]=pch.Chebyshev(un3)(+1)-pch.Chebyshev(un4)(-1)
        residuou4[0]=pch.Chebyshev.deriv(pch.Chebyshev(un3), m=1)(+1)-((r_m3-r_m2m)/r_m3)*pch.Chebyshev.deriv(pch.Chebyshev(un4), m=1)(-1)
        residuou4[N]=pch.Chebyshev(un4)(+1)
        
        residuov1[M]=pch.Chebyshev(vn1_f)(+1)-pch.Chebyshev(vn2)(-1)
        residuov2[0]=pch.Chebyshev.deriv(pch.Chebyshev(vn1_f), m=1)(+1)-(2.*r_m1/(r_m2-r_m1))*pch.Chebyshev.deriv(pch.Chebyshev(vn2), m=1)(-1)
        residuov2[N]=pch.Chebyshev(vn2)(+1)-pch.Chebyshev(vn2m)(-1)
        residuov2m[0]=(1./(r_m2-r_m1))*pch.Chebyshev.deriv(pch.Chebyshev(vn2), m=1)(+1)-(1./(r_m2m-r_m2))*pch.Chebyshev.deriv(pch.Chebyshev(vn2m), m=1)(-1)
        residuov2m[N]=pch.Chebyshev(vn2m)(+1)-pch.Chebyshev(vn3)(-1)
        residuov3[0]=(1./(r_m2m-r_m2))*pch.Chebyshev.deriv(pch.Chebyshev(vn2m), m=1)(+1)-(1./(r_m3-r_m2m))*pch.Chebyshev.deriv(pch.Chebyshev(vn3), m=1)(-1)
        residuov3[N]=pch.Chebyshev(vn3)(+1)-pch.Chebyshev(vn4)(-1)
        residuov4[0]=pch.Chebyshev.deriv(pch.Chebyshev(vn3), m=1)(+1)-((r_m3-r_m2m)/r_m3)*pch.Chebyshev.deriv(pch.Chebyshev(vn4), m=1)(-1)
        residuov4[N]=pch.Chebyshev(vn4)(+1)-1.0
        
        residuow1[M]=pch.Chebyshev(wn1_f)(+1)-pch.Chebyshev(wn2)(-1)
        residuow2[0]=pch.Chebyshev.deriv(pch.Chebyshev(wn1_f), m=1)(+1)-(2.*r_m1/(r_m2-r_m1))*pch.Chebyshev.deriv(pch.Chebyshev(wn2), m=1)(-1)
        residuow2[N]=pch.Chebyshev(wn2)(+1)-pch.Chebyshev(wn2m)(-1)
        residuow2m[0]=(1./(r_m2-r_m1))*pch.Chebyshev.deriv(pch.Chebyshev(wn2), m=1)(+1)-(1./(r_m2m-r_m2))*pch.Chebyshev.deriv(pch.Chebyshev(wn2m), m=1)(-1)
        residuow2m[N]=pch.Chebyshev(wn2m)(+1)-pch.Chebyshev(wn3)(-1)
        residuow3[0]=(1./(r_m2m-r_m2))*pch.Chebyshev.deriv(pch.Chebyshev(wn2m), m=1)(+1)-(1./(r_m3-r_m2m))*pch.Chebyshev.deriv(pch.Chebyshev(wn3), m=1)(-1)
        residuow3[N]=pch.Chebyshev(wn3)(+1)-pch.Chebyshev(wn4)(-1)
        residuow4[0]=pch.Chebyshev.deriv(pch.Chebyshev(wn3), m=1)(+1)-((r_m3-r_m2m)/r_m3)*pch.Chebyshev.deriv(pch.Chebyshev(wn4), m=1)(-1)
        residuow4[N]=pch.Chebyshev(wn4)(+1)-1.0
        
        residuoomega=omega-N00
        
        residuo=np.concatenate((residuou1,residuou2,residuou2m,residuou3,residuou4,residuov1,residuov2,residuov2m,residuov3,residuov4,residuow1,residuow2,residuow2m,residuow3,residuow4,residuoomega), axis=None)
        return residuo
    #############################
    #############################

    def NR(f,x0, dV,tolf):
        T=12*(N+1)+3*(M+1)+1
        #T=M+1
        J=np.zeros((T,T))
        y=x0
        errf=sum(abs(f(y)))
        i=1
        while errf>tolf:
            print(" --------------------------------------------")
            print("| i="+str(i)+" | sum(|R_n|)="+"{:.2e}".format(errf)+" | omega_i="+"{:.4f}".format(y[-1])+" |")
            fy=f(y)
            for jj in np.arange(T):
                dy=np.zeros(T)
                dy[jj]=dV
                fy_dy=f(y+dy)
                for ii in np.arange(T):
                    J[ii,jj]=(fy_dy[ii]-fy[ii])/dV
            Jinv=np.linalg.inv(J)
            deltay=np.zeros(T)
            for ii in np.arange(T):
                for jj in np.arange(T):
                    deltay[ii]=deltay[ii]-Jinv[ii,jj]*fy[jj]
            y=y+deltay
            errf2=sum(abs(f(y)))
            i=i+1
            if errf2>errf:
                print("no parece estar convergiendo")
                #break
            else:
                errf=errf2
        return y




    #Construccion de estimacion inicial a partir de cero
    if dato_inicial=="gaussiana":
        #############################
        #############################
        r1=r_m1*x
        r2=0.5*(r_m2-r_m1)*x+0.5*(r_m2+r_m1)
        r2m=0.5*(r_m2m-r_m2)*x+0.5*(r_m2m+r_m2)
        r3=0.5*(r_m3-r_m2m)*x+0.5*(r_m3+r_m2m)
        r4=2.0*r_m3/(1.0-x+epsilon)
        u_ini1=phi0_semilla*np.exp(-r1**2/sigmax)
        v_ini1=-(0.0)*np.exp(-r1**2)+1.0
        w_ini1=0.0*np.exp(-r1**2/sigmax)+1.0
        u_ini2=phi0_semilla*np.exp(-r2**2/sigmax)
        v_ini2=-(0.0)*np.exp(-r2**2)+1.0
        w_ini2=0.0*np.exp(-r2**2/sigmax)+1.0
        u_ini2m=phi0_semilla*np.exp(-r2m**2/sigmax)
        v_ini2m=-(0.0)*np.exp(-r2m**2)+1.0
        w_ini2m=0.0*np.exp(-r2m**2/sigmax)+1.0
        u_ini3=phi0_semilla*np.exp(-r3**2/sigmax)
        v_ini3=-(0.0)*np.exp(-r3**2)+1.0
        w_ini3=0.0*np.exp(-r3**2/sigmax)+1.0
        u_ini4=phi0_semilla*np.exp(-r4**2/sigmax)
        v_ini4=-(0.0)*np.exp(-r4**2)+1.0
        w_ini4=0.0*np.exp(-r4**2/sigmax)+1.0
        #############################
        #############################

    #Construccion de estimacion inicial a partir de otra solucion
    if dato_inicial=="archivo" and datafile_N0==" ":
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

        #Cambio de N
        #############################
        #############################
        u_ini1=pch.Chebyshev(u10)(x)
        v_ini1=pch.Chebyshev(v10)(x)
        w_ini1=pch.Chebyshev(w10)(x)
        u_ini2=pch.Chebyshev(u20)(x)
        v_ini2=pch.Chebyshev(v20)(x)
        w_ini2=pch.Chebyshev(w20)(x)
        u_ini2m=pch.Chebyshev(u20m)(x)
        v_ini2m=pch.Chebyshev(v20m)(x)
        w_ini2m=pch.Chebyshev(w20m)(x)
        u_ini3=pch.Chebyshev(u30)(x)
        v_ini3=pch.Chebyshev(v30)(x)
        w_ini3=pch.Chebyshev(w30)(x)
        u_ini4=pch.Chebyshev(u40)(x)
        v_ini4=pch.Chebyshev(v40)(x)
        w_ini4=pch.Chebyshev(w40)(x)
        #############################
        #############################
        
        
    #Construccion de estimacion inicial a partir de otra solucion con menor N
    #SOLO FUNCIONA PARESSSS OJO
    if dato_inicial=="archivo" and datafile_N0!=" ":
        g=np.loadtxt(datafile_N0)

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

        #Cambio de N
        #############################
        #############################
        u_ini1=pch.Chebyshev(u10)(x)
        v_ini1=pch.Chebyshev(v10)(x)
        w_ini1=pch.Chebyshev(w10)(x)
        u_ini2=pch.Chebyshev(u20)(x)
        v_ini2=pch.Chebyshev(v20)(x)
        w_ini2=pch.Chebyshev(w20)(x)
        u_ini2m=pch.Chebyshev(u20m)(x)
        v_ini2m=pch.Chebyshev(v20m)(x)
        w_ini2m=pch.Chebyshev(w20m)(x)
        u_ini3=pch.Chebyshev(u30)(x)
        v_ini3=pch.Chebyshev(v30)(x)
        w_ini3=pch.Chebyshev(w30)(x)
        u_ini4=pch.Chebyshev(u40)(x)
        v_ini4=pch.Chebyshev(v40)(x)
        w_ini4=pch.Chebyshev(w40)(x)
        #############################
        #############################
        

    u10_f=I_f(u_ini1)
    v10_f=I_f(v_ini1)
    w10_f=I_f(w_ini1)
        
    u10=np.zeros(M+1)
    v10=np.zeros(M+1)
    w10=np.zeros(M+1)
    for ii in np.arange(M+1):
        u10[ii]=u10_f[2*ii]
        v10[ii]=v10_f[2*ii]
        w10[ii]=w10_f[2*ii]
    #Dominio 2
    u20=I_f(u_ini2)
    v20=I_f(v_ini2)
    w20=I_f(w_ini2)
    
    #Dominio 2m
    u20m=I_f(u_ini2m)
    v20m=I_f(v_ini2m)
    w20m=I_f(w_ini2m)
    
    #Dominio 3
    u30=I_f(u_ini3)
    v30=I_f(v_ini3)
    w30=I_f(w_ini3)
    
    #Dominio 4
    u40=I_f(u_ini4)
    v40=I_f(v_ini4)
    w40=I_f(w_ini4)

    if omegamanual!=0:
        omega0=omegamanual
    else:
        omega0=g[3*M0+3+12*N0+12]
    N0=N  # Para asegurar que despues del primer paso, lea la omega de la iteracion anterior

    x0=np.concatenate((u10,u20,u20m,u30,u40,v10,v20,v20m,v30,v40,w10,w20,w20m,w30,w40,omega0), axis=None)
    #x0=Un
    print(" ")

    Un=NR(R,x0,dV, tolf)

    print(" ")
    print("omega: "+str(Un[-1]))

    np.savetxt(solution_folder+"/omega_"+"{:.2e}".format(N00)+".dat", Un)


    #### output ####

    u1=np.zeros(N+1)
    for ii in np.arange(M+1):
        u1[2*ii]=Un[ii]
        
    u2=Un[M+1:M+1+N+1]
    u2m=Un[M+1+N+1:M+1+2*N+2]
    u3=Un[M+1+2*N+2:M+1+3*N+3]
    u4=Un[M+1+3*N+3:M+1+4*N+4]

    v1=np.zeros(N+1)
    for ii in np.arange(M+1):
        v1[2*ii]=Un[ii+M+1+4*N+4]
        
    v2=Un[2*M+2+4*N+4:2*M+2+5*N+5]
    v2m=Un[2*M+2+5*N+5:2*M+2+6*N+6]
    v3=Un[2*M+2+6*N+6:2*M+2+7*N+7]
    v4=Un[2*M+2+7*N+7:2*M+2+8*N+8]

    w1=np.zeros(N+1)
    for ii in np.arange(M+1):
        w1[2*ii]=Un[ii+2*M+2+8*N+8]
        
    w2=Un[3*M+3+8*N+8:3*M+3+9*N+9]
    w2m=Un[3*M+3+9*N+9:3*M+3+10*N+10]
    w3=Un[3*M+3+10*N+10:3*M+3+11*N+11]
    w4=Un[3*M+3+11*N+11:3*M+3+12*N+12]

    xx=np.linspace(-1,1,1000)
    
    ################################
    #Masas ADM y Komar
    omega_a[i]=Un[-1]
    uc_a[i]=pch.Chebyshev.deriv(pch.Chebyshev(u1), m=0)(0.0)
    MADM_a[i]=-4.*r_m3*pch.Chebyshev.deriv(pch.Chebyshev(w4), m=1)(xx)[-1]
    MKomar_a[i]=2.*r_m3*pch.Chebyshev.deriv(pch.Chebyshev(v4), m=1)(xx)[-1]
    
    #Radios y Compacidades
    rr2=0.5*(r_m2-r_m1)*xx+0.5*(r_m2+r_m1)
    rr2m=0.5*(r_m2m-r_m2)*xx+0.5*(r_m2m+r_m2)
    rr3=0.5*(r_m3-r_m2m)*xx+0.5*(r_m3+r_m2m)
    rr4=2.0*r_m3/(1.0-xx+epsilon)
    rr=np.concatenate((rr2,rr2m,rr3,rr4), axis=None)
    Psi2=pch.Chebyshev(w2)(xx)
    Psi2m=pch.Chebyshev(w2m)(xx)
    Psi3=pch.Chebyshev(w3)(xx)
    Psi4=pch.Chebyshev(w4)(xx)
    Psirr=np.concatenate((Psi2,Psi2m,Psi3,Psi4), axis=None)
    N2=pch.Chebyshev(v2)(xx)
    N2m=pch.Chebyshev(v2m)(xx)
    N3=pch.Chebyshev(v3)(xx)
    N4=pch.Chebyshev(v4)(xx)
    Nrr=np.concatenate((N2,N2m,N3,N4), axis=None)
    dPsi2=(2.0/(r_m2-r_m1))*pch.Chebyshev.deriv(pch.Chebyshev(w2), m=1)(xx)
    dPsi2m=(2.0/(r_m2m-r_m2))*pch.Chebyshev.deriv(pch.Chebyshev(w2m), m=1)(xx)
    dPsi3=(2.0/(r_m3-r_m2m))*pch.Chebyshev.deriv(pch.Chebyshev(w3), m=1)(xx)
    dPsi4=((1.-xx)**2/(2.*r_m3))*pch.Chebyshev.deriv(pch.Chebyshev(w4), m=1)(xx)
    dPsirr=np.concatenate((dPsi2,dPsi2m,dPsi3,dPsi4), axis=None)
    
    Mms=-2.*rr**2*dPsirr*(Psirr+rr*dPsirr)
    
    noventaynueve=Mms-0.99*MADM_a[i]
    ii99=np.argmin(abs(noventaynueve))
    R99_a[i]=rr[ii99]*Psirr[ii99]**2
    C99_a[i]=MADM_a[i]/R99_a[i]
    
    #Mmsrmax=Mms/(rr*Psirr**2)
    #iimax=np.argmax(Mmsrmax)
    #Rtest_a[i]=rr[iimax]*Psirr[iimax]**2
    #Ctest_a[i]=Mmsrmax[iimax]
    ###################################
    
    #if i!=0 and Un[-1]>omega0:
    #    break
    print(" ")
    print("rel. err. MADM MKomar= "+str(abs(MADM_a[i]-MKomar_a[i])/(MADM_a[i]+MKomar_a[i])))
    print("(the number above should be << 1)")
    print(" ")
    print("phi(r=0) = "+str(uc_a[i]))
    print("MADM = "+str(MADM_a[i]))
    #print("MKom = "+str(MKomar_a[i]))
    print("R99 = "+str(R99_a[i]))
    #print("C99 = "+str(C99_a[i]))
    #print("Rtest = "+str(Rtest_a[i]))
    #print("Ctest = "+str(Ctest_a[i]))
    #print(" ")
    #if i!=0 and MADM_a[i-1]>MADM_a[i]:
    #    break
    if i!=0 and omega_a[i-1]<omega_a[i]:
        break

uc_a_i=uc_a[0:i+1]
omega_a_i=omega_a[0:i+1]
MADM_a_i=MADM_a[0:i+1]
MKomar_a_i=MKomar_a[0:i+1]
R99_a_i=R99_a[0:i+1]
C99_a_i=C99_a[0:i+1]
#Rtest_a_i=Rtest_a[0:i+1]
#Ctest_a_i=Ctest_a[0:i+1]


np.savetxt(solution_folder+"/phi_0.dat", uc_a_i)
np.savetxt(solution_folder+"/omega.dat", omega_a_i)
np.savetxt(solution_folder+"/MADM.dat", MADM_a_i)
np.savetxt(solution_folder+"/MKomar.dat", MKomar_a_i)
np.savetxt(solution_folder+"/R99.dat", R99_a_i)
np.savetxt(solution_folder+"/C99.dat", C99_a_i)
#np.savetxt("Rtest.dat", Rtest_a_i)
#np.savetxt("Ctest.dat", Ctest_a_i)

plt.plot(omega_a_i,MADM_a_i, color="k")
plt.plot(omega_a_i,MKomar_a_i, color="g")
plt.savefig(solution_folder+'/omega_vs_M.png', dpi=400)

print(" ")
print(" listo ")
print(" ")
