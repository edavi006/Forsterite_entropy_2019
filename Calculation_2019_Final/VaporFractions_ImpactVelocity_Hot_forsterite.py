# Calculations for vapour fractions and critical impact velocities.
# Uses two Equations of State
# They can be input in the first few lines of the code.

#This case uses M-ANEOS forsterite model EOS and the Measured Forsterite Hugoniot
#
#Requires input of vapor domes and hugoniots
#This case also uses the Dunite M-ANEOS vapor dome, for both. Should another dome
#be used at the same time, another dome would have to be defined below.'

#Monte Carlo perturbation uncertainty propagation is also performed on the
# calculations here. 

import matplotlib as mpl
import pylab as py
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import matplotlib.cm as cm

import matplotlib.gridspec as gridspec
import scipy.integrate as integrate
import scipy.interpolate as interpolate

#Triple Point Pressure
Fo_tp_P=5.2*(10**-5)*100000 #Pa, 2163C melting point
#Entropy of fusion, solid -> liquid
ef=464.118 #Richet 1993
efe=ef*0.03
#Constants
R=8.314 # J/K/mol
G=6.67408*10**(-11) #m^3/kg/s^2

#Monte Carlo Steps
steps=100000
#Set up classes

class hugoniot: # mks
    def __init__(h):
        h.note= ''
        h.r0 = 0 # initial density kg/m3
        h.v0 = 0 # initial specific volume = 1/r0
        h.c = 0 # linear eos intercept m/s
        h.s = 0 # linear eos slope
        h.p = np.zeros(nup) # pressure Pa
        h.v = np.zeros(nup) # specific volume m3/kg
        h.e = np.zeros(nup) # specific energy J/kg
        h.us = np.zeros(nup) # shock velocity m/s
        h.t=np.zeros(nup)
        h.se=np.zeros(nup)

class dome:
    def __init__(d):
        d.note=''
        d.t=0
        d.rl=0
        d.rv=0
        d.p=0
        d.sl=0
        d.sv=0

# function to make a Hugoniot structure for a single material
# eos=[rho0,c,s]
def makehugoniot(eos,up,note):
    h = hugoniot()
    h.note = note
    h.r0 = eos[er]
    h.v0 = 1/eos[er]
    h.c  = eos[ec]
    h.s  = eos[es]
    h.us = eos[ec]+eos[es]*up 
    h.p = eos[er]*up*h.us 
    h.v = (1/eos[er]) * (1-up/h.us) 
    h.e = 0.5 * h.p * ((1/eos[er])-h.v)
    h.t=np.zeros(nup)
    h.se=np.zeros(nup)
    return h

#Loading Vapor Dome data
# forsterite
fovcfile = 'PulledHugoniots/L_VDome_STS_f.csv'
fovc = dome()
fovc.note='M-ANEOS Forsterite Melosh'
fovc.t = np.loadtxt(fovcfile,skiprows=3,usecols=[0],delimiter=',')*(10**np.loadtxt(fovcfile,skiprows=3,usecols=[1],delimiter=','))
fovc.rl = np.loadtxt(fovcfile,skiprows=3,usecols=[2],delimiter=',')*(10**np.loadtxt(fovcfile,skiprows=3,usecols=[3],delimiter=','))
fovc.rv = np.loadtxt(fovcfile,skiprows=3,usecols=[4],delimiter=',')*(10**np.loadtxt(fovcfile,skiprows=3,usecols=[5],delimiter=','))
fovc.p = np.loadtxt(fovcfile,skiprows=3,usecols=[6],delimiter=',')*(10**np.loadtxt(fovcfile,skiprows=3,usecols=[7],delimiter=','))
fovc.sl = np.loadtxt(fovcfile,skiprows=3,usecols=[14],delimiter=',')*(10**np.loadtxt(fovcfile,skiprows=3,usecols=[15],delimiter=','))
fovc.sv = np.loadtxt(fovcfile,skiprows=3,usecols=[16],delimiter=',')*(10**np.loadtxt(fovcfile,skiprows=3,usecols=[17],delimiter=','))

#Maneos hugoniot curves
xm=np.loadtxt('forsterite-maneos_py.txt',skiprows=5,usecols=[2])*10**9 #pressure
ym=np.loadtxt('forsterite-maneos_py.txt',skiprows=5,usecols=[1]) #temperature
sm=np.loadtxt('forsterite-maneos_py.txt',skiprows=5,usecols=[5]) #entropy
dm=np.loadtxt('forsterite-maneos_py.txt',skiprows=5,usecols=[0]) #density


#########CALC THE ENTROPY OF HEATING TO 1200k##########

cpa=-402.753
cpb=74.29
cpc=87.588
cpd=-25.913
cpe=25.374
ma=140.6931 #(g/mol) molecular weight of forsterite
ma=ma/1000 # putting the previous into kg
T_amb=298.15 # K, ambient initial temperature
Tm_mc=1200 # K, heating temperature


temp1=-(10**8)*cpe/(3*Tm_mc**3)-500000*cpd/(Tm_mc**2)-1000*cpc/Tm_mc+cpa*np.log(Tm_mc)+.5*cpb*np.log(Tm_mc)**2
temp2=-(10**8)*cpe/(3*T_amb**3)-500000*cpd/(T_amb**2)-1000*cpc/T_amb+cpa*np.log(T_amb)+.5*cpb*np.log(T_amb)**2          
refS_HEAT=(temp1-temp2)/ma
print('heating entropy, J/K/kg', refS_HEAT)



########INPUT FORSTERITE DATA#############

zfile = 'Z_Hugoniot.txt'
skip=150
ZP = np.genfromtxt(zfile,dtype=float,delimiter=',',skip_header=skip,usecols=0)
ZT = np.genfromtxt(zfile,dtype=float,delimiter=',',skip_header=skip,usecols=1)
ZPe= np.genfromtxt(zfile,dtype=float,delimiter=',',skip_header=skip,usecols=4)
ZTe= np.genfromtxt(zfile,dtype=float,delimiter=',',skip_header=skip,usecols=5)


def sFit(xx,A,B,C,D):
    #return A*xx**1+ B*xx**2+C*np.log(xx*D)
    return A*xx**(-0.5)+B*xx**(0.5)+C*xx**1.5+D#*xx+2000

a1=-28196.1283044 
b1= 159.403207274
c1=-0.0239572180975
d1= 3391.20648387

param_cov=[[  9.13835599e+06 ,  2.92139412e+04,  -6.75973437e+00 ,  7.64221286e+05],
 [  2.92139412e+04,   9.34550297e+01 , -2.16596932e-02 ,  2.38728996e+03],
 [ -6.75973437e+00 , -2.16596932e-02 ,  5.04075125e-06 , -5.18693657e-01],
 [  7.64221286e+05 ,  2.38728996e+03 , -5.18693657e-01 ,  1.19628080e+05]]
temp_s=np.zeros((np.size(ZP),steps))
lmat=sp.linalg.cholesky(param_cov,lower=True)
for j in range(0,steps):
    temp_mat=sp.randn(4,1)
    bmat=np.matmul(lmat,temp_mat)    
    afit1=a1+bmat[0,0]
    bfit1=b1+bmat[1,0]
    cfit1=c1+bmat[2,0]
    dfit1=d1+bmat[3,0]
    temp_s[:,j]=sFit(ZP/(10**9),afit1,bfit1,cfit1,dfit1)
SS_mean=np.zeros(np.size(ZP))
SS_std=np.zeros(np.size(ZP))
for i in range(0,np.size(ZP)):
    SS_mean[i]=np.mean(temp_s[i,:])
    SS_std[i]=np.std(temp_s[i,:])

ZS = SS_mean+refS_HEAT
ZSe= SS_std

#Interpolations for input data. For Entropies and pressure

fo_p_s_l=interpolate.interp1d(fovc.p,fovc.sl) # getting entropies
fo_p_s_v=interpolate.interp1d(fovc.p,fovc.sv)

fo_sl_tp1=3474.46894092#entropy for triple point liquid side - incipient vapor
fo_sl_tpe=196.883316564
fo_sv_tp=11759#entropy for triple point vapour side -CV
fo_sol_tp1=3010.34116866# No solid-liquid transition in ANEOS, create my own, #premelt
fo_sol_tpe=196.837462729
#assume relatively constant offset.

fo_sl_atm1=4270.86182461 #Same as above 3 but for atmo pressure IC
fo_sl_atme=257.8038
fo_sv_atm=9000 #CV # from StS rework of ANEOS
fo_sol_atm1=3474.46894092 #Complete melting
fo_sol_atme=196.883316564

#Forsterite entropy of melting 1 bar (incipient, complete)
print(fo_sl_tp1,fo_sv_tp,fo_sl_atm1,fo_sv_atm) #Prints vapour and liquid entropies
print('Complete Melt:', ' 1 Bar - ', fo_sol_atm1, ' TP - ',fo_sl_tp1)
print('Incipient Vapor:', ' 1 Bar - ', fo_sl_atm1,' TP - ',fo_sl_tp1)
print('50% Vapor:', ' 1 Bar - ', (0.5*(fo_sv_atm-fo_sl_atm1))+fo_sl_atm1,' TP - ',(0.5*(fo_sv_tp-fo_sl_tp1))+fo_sl_tp1)

def lever(f,S1,S2):
    result=f*S1+(1-f)*S2
    return result

#Begin monte carlo set up and loop



#Universal Hugoniot Forsterite fit (Root et al 2018)
def hugoniot(a,b,c,d,x):
    return a + b*x - c * x *np.exp(-d*x)
    
#parameters
au1=6.86
bu1=1.23
cu1=1.66
du1=0.43
#uncertainties
au2=0.33
bu2=0.03
cu2=0.34
du2=0.08

#Array set up
atm_man_pressures=sp.zeros((6,steps))#pressures, atm, maneos
tp_man_pressures=sp.zeros((6,steps))#pressures, triple point, maneos
atm_z_pressures=sp.zeros((6,steps))#pressures, atm, z
tp_z_pressures=sp.zeros((6,steps))#pressures, triple point, z
civ_tp_z=sp.zeros((6,steps)) #Critical impact velocities, triple point, z
civ_atm_z=sp.zeros((6,steps)) #Critical impact velocities, atm, z
civ_tp_man=sp.zeros((6,steps)) #Critical impact velocities, triple point, maneos
civ_atm_man=sp.zeros((6,steps)) #Critical impact velocities, atm, maneos

up=np.linspace(0,20,1000) #Array of particle velocities (km/s)

for j in range(0,steps):
    fo_sl_atm=fo_sl_atm1+fo_sl_atme*sp.randn()
    fo_sol_atm=fo_sol_atm1+fo_sol_atme*sp.randn()
    fo_sl_tp=fo_sl_tp1+fo_sl_tpe*sp.randn()
    fo_sol_tp=fo_sol_tp1+fo_sol_tpe*sp.randn()
    au=au1+au2*sp.randn() #Perturbations on hugoniot parameters
    bu=bu1+bu2*sp.randn()
    cu=cu1+cu2*sp.randn()
    du=du1+du2*sp.randn()
    ZP1=ZP+ZPe*sp.randn() #Pressure for z experiment
    ZS1=ZS+ZSe*sp.randn()
    ZT1=ZT+ZTe*sp.randn()
    rho=3220+3220*0.003*sp.randn() #Density uncertainty

    #Interpolate between the perturbed points and M-ANEOS points
    fo_man=interpolate.interp1d(sm,xm,bounds_error=False) # getting the pressures, maneos
    fo_z=interpolate.interp1d(ZS1,ZP1,bounds_error=False) # getting the pressures, z

    #MANEOS critical pressures
    atm_man_pressures[0,j]=fo_man(fo_sol_atm)*10**-9 #complete melt
    atm_man_pressures[1,j]=fo_man(fo_sl_atm)*10**-9 #incipient vapor
    atm_man_pressures[2,j]=fo_man(lever(0.95,fo_sl_atm,fo_sv_atm))*10**-9 # 5%
    atm_man_pressures[3,j]=fo_man(lever(0.9,fo_sl_atm,fo_sv_atm))*10**-9 # 10%
    atm_man_pressures[4,j]=fo_man(lever(0.75,fo_sl_atm,fo_sv_atm))*10**-9 # 25%
    atm_man_pressures[5,j]=fo_man(lever(0.5,fo_sl_atm,fo_sv_atm))*10**-9 # 50%

    tp_man_pressures[0,j]=fo_man(fo_sl_tp)*10**-9 #complete melt
    tp_man_pressures[1,j]=fo_man(fo_sl_tp)*10**-9 #incipient vapor
    tp_man_pressures[2,j]=fo_man(lever(0.95,fo_sl_tp,fo_sv_tp))*10**-9 # 5%
    tp_man_pressures[3,j]=fo_man(lever(0.9,fo_sl_tp,fo_sv_tp))*10**-9 # 10%
    tp_man_pressures[4,j]=fo_man(lever(0.75,fo_sl_tp,fo_sv_tp))*10**-9 # 25%
    tp_man_pressures[5,j]=fo_man(lever(0.5,fo_sl_tp,fo_sv_tp))*10**-9 # 50%

    #Z critical pressures
    atm_z_pressures[0,j]=fo_z(fo_sol_atm)*10**-9 #complete melt
    atm_z_pressures[1,j]=fo_z(fo_sl_atm)*10**-9 #incipient vapor
    atm_z_pressures[2,j]=fo_z(lever(0.95,fo_sl_atm,fo_sv_atm))*10**-9 # 5%
    atm_z_pressures[3,j]=fo_z(lever(0.9,fo_sl_atm,fo_sv_atm))*10**-9 # 10%
    atm_z_pressures[4,j]=fo_z(lever(0.75,fo_sl_atm,fo_sv_atm))*10**-9 # 25%
    atm_z_pressures[5,j]=fo_z(lever(0.5,fo_sl_atm,fo_sv_atm))*10**-9 # 50%

    tp_z_pressures[0,j]=fo_z(fo_sl_tp)*10**-9 #complete melt
    tp_z_pressures[1,j]=fo_z(fo_sl_tp)*10**-9#incipient vapor
    tp_z_pressures[2,j]=fo_z(lever(0.95,fo_sl_tp,fo_sv_tp))*10**-9 # 5%
    tp_z_pressures[3,j]=fo_z(lever(0.9,fo_sl_tp,fo_sv_tp))*10**-9 # 10%
    tp_z_pressures[4,j]=fo_z(lever(0.75,fo_sl_tp,fo_sv_tp))*10**-9 # 25%
    tp_z_pressures[5,j]=fo_z(lever(0.5,fo_sl_tp,fo_sv_tp))*10**-9 # 50%

    #Calculating Critical impact velocities for all of these.
    Fo_us=hugoniot(au,bu,cu,du,up)
    Fo_P=rho*up*Fo_us/1000
    #Pressure particle velocity function
    crit_up=interpolate.interp1d(Fo_P,up,bounds_error=False)

    #assuming rock on rock impacts. impact v is 2*up.
    civ_atm_man[0,j]=crit_up(atm_man_pressures[0,j])*2
    civ_atm_man[1,j]=crit_up(atm_man_pressures[1,j])*2
    civ_atm_man[2,j]=crit_up(atm_man_pressures[2,j])*2
    civ_atm_man[3,j]=crit_up(atm_man_pressures[3,j])*2
    civ_atm_man[4,j]=crit_up(atm_man_pressures[4,j])*2
    civ_atm_man[5,j]=crit_up(atm_man_pressures[5,j])*2

    civ_tp_man[0,j]=crit_up(tp_man_pressures[0,j])*2
    civ_tp_man[1,j]=crit_up(tp_man_pressures[1,j])*2
    civ_tp_man[2,j]=crit_up(tp_man_pressures[2,j])*2
    civ_tp_man[3,j]=crit_up(tp_man_pressures[3,j])*2
    civ_tp_man[4,j]=crit_up(tp_man_pressures[4,j])*2
    civ_tp_man[5,j]=crit_up(tp_man_pressures[5,j])*2

    civ_atm_z[0,j]=crit_up(atm_z_pressures[0,j])*2
    civ_atm_z[1,j]=crit_up(atm_z_pressures[1,j])*2
    civ_atm_z[2,j]=crit_up(atm_z_pressures[2,j])*2
    civ_atm_z[3,j]=crit_up(atm_z_pressures[3,j])*2
    civ_atm_z[4,j]=crit_up(atm_z_pressures[4,j])*2
    civ_atm_z[5,j]=crit_up(atm_z_pressures[5,j])*2

    civ_tp_z[0,j]=crit_up(tp_z_pressures[0,j])*2
    civ_tp_z[1,j]=crit_up(tp_z_pressures[1,j])*2
    civ_tp_z[2,j]=crit_up(tp_z_pressures[2,j])*2
    civ_tp_z[3,j]=crit_up(tp_z_pressures[3,j])*2
    civ_tp_z[4,j]=crit_up(tp_z_pressures[4,j])*2
    civ_tp_z[5,j]=crit_up(tp_z_pressures[5,j])*2

atm_man_pressures_final=sp.zeros((6,2))#pressures, atm, maneos
tp_man_pressures_final=sp.zeros((6,2))#pressures, triple point, maneos
atm_z_pressures_final=sp.zeros((6,2))#pressures, atm, z
tp_z_pressures_final=sp.zeros((6,2))#pressures, triple point, z
civ_tp_z_final=sp.zeros((6,2)) #Critical impact velocities, triple point, z
civ_atm_z_final=sp.zeros((6,2)) #Critical impact velocities, atm, z
civ_tp_man_final=sp.zeros((6,2)) #Critical impact velocities, triple point, maneos
civ_atm_man_final=sp.zeros((6,2)) #Critical impact velocities, atm, maneos

#Means of the data clouds, and standard deviations
for i in range(0,6):
    atm_man_pressures_final[i,0]=np.nanmedian(atm_man_pressures[i,:])
    atm_man_pressures_final[i,1]=np.nanstd(atm_man_pressures[i,:])
    tp_man_pressures_final[i,0]=np.nanmedian(tp_man_pressures[i,:])
    tp_man_pressures_final[i,1]=np.nanstd(tp_man_pressures[i,:])
    atm_z_pressures_final[i,0]=np.nanmedian(atm_z_pressures[i,:])
    atm_z_pressures_final[i,1]=np.nanstd(atm_z_pressures[i,:])
    tp_z_pressures_final[i,0]=np.nanmedian(tp_z_pressures[i,:])
    tp_z_pressures_final[i,1]=np.nanstd(tp_z_pressures[i,:])
    
    civ_atm_man_final[i,0]=np.nanmedian(civ_atm_man[i,:])
    civ_atm_man_final[i,1]=np.nanstd(civ_atm_man[i,:])
    civ_tp_man_final[i,0]=np.nanmedian(civ_tp_man[i,:])
    civ_tp_man_final[i,1]=np.nanstd(civ_tp_man[i,:])
    civ_atm_z_final[i,0]=np.nanmedian(civ_atm_z[i,:])
    civ_atm_z_final[i,1]=np.nanstd(civ_atm_z[i,:])
    civ_tp_z_final[i,0]=np.nanmedian(civ_tp_z[i,:])
    civ_tp_z_final[i,1]=np.nanstd(civ_tp_z[i,:])

print('MANEOS Atm release pressures',atm_man_pressures_final[:,0],'Uncertainty',atm_man_pressures_final[:,1])
print('MANEOS tp release pressures',tp_man_pressures_final[:,0],'Uncertainty',tp_man_pressures_final[:,1])
print('Z Atm release pressures',atm_z_pressures_final[:,0],'Uncertainty',atm_z_pressures_final[:,1])
print('Z tp release pressures',tp_z_pressures_final[:,0],'Uncertainty',tp_z_pressures_final[:,1])
print('MANEOS Atm Crit V',civ_atm_man_final[:,0],'Uncertainty',civ_atm_man_final[:,1])
print('MANEOS tp Crit V',civ_tp_man_final[:,0],'Uncertainty',civ_tp_man_final[:,1])
print('Z Atm Crit V',civ_atm_z_final[:,0],'Uncertainty',civ_atm_z_final[:,1])
print('Z tp Crit V',civ_tp_z_final[:,0],'Uncertainty',civ_tp_z_final[:,1])

#writing to files
with open('VaporFractions_MANEOS_1200.txt','w') as f:
    f.write('1200K, CM, CMe, IV, IVe, 5, 5e, 10, 10e, 25, 25e, 50, 50e \r\n')
    f.write('atm Pressures')
    f.write(',')
    for i in range(np.size(civ_tp_z_final[:,0])):      
        f.write(str(atm_man_pressures_final[i,0]))
        f.write(',')
        f.write(str(atm_man_pressures_final[i,1]))
        f.write(',')
    f.write('\r\n')
    f.write('atm Velocity')
    f.write(',')
    for i in range(np.size(civ_tp_z_final[:,0])):      
        f.write(str(civ_atm_man_final[i,0]))
        f.write(',')
        f.write(str(civ_atm_man_final[i,1]))
        f.write(',')
    f.write('\r\n')
    f.write('tp Pressures')
    f.write(',')
    for i in range(np.size(civ_tp_z_final[:,0])):      
        f.write(str(tp_man_pressures_final[i,0]))
        f.write(',')
        f.write(str(tp_man_pressures_final[i,1]))
        f.write(',')
    f.write('\r\n')
    f.write('tp Velocity')
    f.write(',')
    for i in range(np.size(civ_tp_z_final[:,0])):      
        f.write(str(civ_tp_man_final[i,0]))
        f.write(',')
        f.write(str(civ_tp_man_final[i,1]))
        f.write(',')
    f.write('\r\n')

with open('VaporFractions_Forsterite_1200.txt','w') as f:
    f.write('1200K, CM, CMe, IV, IVe, 5, 5e, 10, 10e, 25, 25e, 50, 50e \r\n')
    f.write('atm Pressures')
    f.write(',')
    for i in range(np.size(civ_tp_z_final[:,0])):      
        f.write(str(atm_z_pressures_final[i,0]))
        f.write(',')
        f.write(str(atm_z_pressures_final[i,1]))
        f.write(',')
    f.write('\r\n')
    f.write('atm Velocity')
    f.write(',')
    for i in range(np.size(civ_tp_z_final[:,0])):      
        f.write(str(civ_atm_z_final[i,0]))
        f.write(',')
        f.write(str(civ_atm_z_final[i,1]))
        f.write(',')
    f.write('\r\n')
    f.write('tp Pressures')
    f.write(',')
    for i in range(np.size(civ_tp_z_final[:,0])):      
        f.write(str(tp_z_pressures_final[i,0]))
        f.write(',')
        f.write(str(tp_z_pressures_final[i,1]))
        f.write(',')
    f.write('\r\n')
    f.write('tp Velocity')
    f.write(',')
    for i in range(np.size(civ_tp_z_final[:,0])):      
        f.write(str(civ_tp_z_final[i,0]))
        f.write(',')
        f.write(str(civ_tp_z_final[i,1]))
        f.write(',')
    f.write('\r\n')






plt.figure()
plt.hist(tp_z_pressures[0,~np.isnan(tp_z_pressures[0,:])],bins=100)


plt.figure()
plt.hist(tp_z_pressures[1,~np.isnan(tp_z_pressures[1,:])],bins=100)


plt.figure()
plt.hist(tp_z_pressures[2,~np.isnan(tp_z_pressures[2,:])],bins=100)


plt.figure()
plt.hist(tp_z_pressures[3,~np.isnan(tp_z_pressures[3,:])],bins=100)

plt.figure()
plt.plot(ZS,ZP)
plt.plot(sm,xm)
plt.show()

