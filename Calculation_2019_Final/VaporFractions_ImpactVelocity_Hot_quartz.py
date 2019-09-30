# Calculations for vapour fractions and critical impact velocities.
# Uses two Equations of State
# They can be input in the first few lines of the code.

#This case uses M-ANEOS quartz model EOS and the Measured quartz Hugoniot
#
#Requires input of vapor domes and hugoniots
#. Should another dome
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
Fo_tp_P=2.68#Pa, 2163C melting point
#Entropy of fusion, solid -> liquid
#ef=253 #estimation based on Rick Kraus' 2012 paper, stishovite
#ef= 100.8 # hemingway 1987.

#Constants
R=8.314 # J/K/mol
G=6.67408*10**(-11) #m^3/kg/s^2
mm=60.08/1000 #kg/mole, molar mass of silica
hot_ent=68.454-41.46 # J/(mole*K) 500K 116 ,quartz
ef=159.662-154.136 # hemingway 1987.
IncMelt=154.136/mm
hot_ent=hot_ent/mm
ef=ef/mm
efe=ef*0.03

#Monte Carlo Steps
steps=1000
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
fovcfile = 'Silica-dome.txt'
fovc = dome()
fovc.note='M-ANEOS Kraus Correction'
fovc.t = np.genfromtxt(fovcfile,dtype=float,skip_header=1,usecols=0)
fovc.rl = np.genfromtxt(fovcfile,dtype=float,skip_header=1,usecols=4)
fovc.rv = np.genfromtxt(fovcfile,dtype=float,skip_header=1,usecols=5)
fovc.p = np.genfromtxt(fovcfile,dtype=float,skip_header=1,usecols=1)*10**9
fovc.sl = np.genfromtxt(fovcfile,dtype=float,skip_header=1,usecols=2)
fovc.sv = np.genfromtxt(fovcfile,dtype=float,skip_header=1,usecols=3)


#Maneos hugoniot curves
xm=np.loadtxt('forsterite-maneos_py.txt',skiprows=5,usecols=[2])*10**9 #pressure
ym=np.loadtxt('forsterite-maneos_py.txt',skiprows=5,usecols=[1]) #temperature
sm=np.loadtxt('forsterite-maneos_py.txt',skiprows=5,usecols=[5]) #entropy
dm=np.loadtxt('forsterite-maneos_py.txt',skiprows=5,usecols=[0]) #density

########INPUT Quartz DATA############# Taken from Kraus 2012, interpolated for points in between.


ZP = np.array([0,116.5,126.6,136,172,244,277,307,318])
ZS = np.array([0,3848,4000,4130,4584,5267,5525,5728,5798])
ZT = np.array([0,3890,4150,3756,4470,4717,4823,4839,4961])
ZPe= np.array([0,2,2.2,8,10,12,14,17,18])
ZTe= np.array([0,110,20,519,536,527,531,540,559])
ZSe= np.array([0,168,168,169,175,191,196,206,206])

ZP=ZP*10**9
ZPe=ZPe*10**9
#Interpolations for input data. For Entropies and pressure

fo_p_s_l=interpolate.interp1d(fovc.p,fovc.sl) # getting entropies
fo_p_s_v=interpolate.interp1d(fovc.p,fovc.sv)

fo_sl_tp=fo_p_s_l(Fo_tp_P)#entropy for triple point liquid side
fo_sv_tp=fo_p_s_v(Fo_tp_P)#entropy for triple point vapour side
fo_sol_tp=fo_sl_tp-ef# No solid-liquid transition in ANEOS, create my own,
#assume relatively constant offset.

fo_sl_atm=fo_p_s_l(100000) #Same as above 3 but for atmo pressure
fo_sv_atm=fo_p_s_v(100000)
fo_sol_atm=fo_sol_tp

#Forsterite entropy of melting 1 bar (incipient, complete)
#HnrvrrNcwlv, Bnucr S. "Quartz: Heat capacities from 340 to
#1000 K and revised values for the thermodynamic properties."
#American Mineralogist 72 (1987): 273-279.
Fo_s_atm=[fo_sl_atm-ef, fo_sl_atm] #J/K/kg
print(fo_sl_tp,fo_sv_tp,fo_sl_atm,fo_sv_atm) #Prints vapour and liquid entropies
print(Fo_s_atm)
print('Complete Melt:', ' 1 Bar - ', Fo_s_atm[1], ' TP - ',fo_sl_tp)
print('Incipient Vapor:', ' 1 Bar - ', fo_sl_atm,' TP - ',fo_sl_tp)
print('50% Vapor:', ' 1 Bar - ', (0.5*(fo_sv_atm-fo_sl_atm))+fo_sl_atm,' TP - ',(0.5*(fo_sv_tp-fo_sl_tp))+fo_sl_tp)
def lever(f,S1,S2):
    result=f*S1+(1-f)*S2
    return result
def lever(f,S1,S2):
    result=f*S1+(1-f)*S2
    return result


##def entropy(a,b,c,d,e,x):
##    return a + b*(x*(10**-9)) - c*(x*(10**-9))**2 + d*(x*(10**-9))**3 - e*(x*(10**-9))**4
def entropy(a,b,c,d,x):
    return a*x**(-0.5)+b*x**(0.5)+c*x+d
def entropy_hot(a,b,c,d,x):
    return a*x**(-0.5)+b*x**(0.5)+c*x**1.5+d

##def entropy_uncert(a,b,c,x):
##    return a + b*(x*(10**-9)) - c*(x*(10**-9))**2

#Q S parameters
##aq=1793
##bq=22.4
##cq=0.0426
##dq=4.319*(10**-5)
##eq=1.742*(10**-8)

#Entropy hot params
##aq=-1290.51475361
##bq=315.186775927
##cq=-0.0992570710016
##dq=791.965847975
##aqe=161.714676241
##bqe=1.72934312977
##cqe=0.000969977997525
##dqe=33.8174848666


#medium Params
aq=2820
bq=468.3
cq=-6.68
dq=-593
aqe=106
bqe=1.8
cqe=.037
dqe=25.2
#High
aqh=1406
bqh=472
cqh=-6.713
dqh=-354.7
aqhe=150
bqhe=1.8
cqhe=.031
dqhe=25.3
#low
aql=4280
bql=461.4
cql=-6.571
dql=-803.3
aqle=150
bqle=2.5
cqle=.052
dqle=25.4


##def sFit(xx,A,B,C,D):
##    return A*xx+ B*xx**2+C*np.log(xx*D)
###Fit of kraus's fit
##aq=4.16714646e-09 
##bq=-2.66448616e-21
##cq=1.17854895e+03
##dq=1.71353154e-10
##aqe=(6.18788731e-21)**(1/2)
##bqe=(2.56387348e-45)**(1/2)
##cqe=(1.30244857e+02)**(1/2)
##dqe=(1.43168531e-23)**(1/2)


#Uncert parameters
aqu=127.3
bqu=0.328
cqu=0.000193


#Universal Hugoniot quartz fit #knudson
def hugoniot(a,b,c,d,x):
    return a + b*x - c * x *np.exp(-d*x)
    
#HUG parameters
au1=6.26
bu1=1.2
cu1=2.56
du1=0.37
#uncertainties
au2=0.35
bu2=0.02
cu2=0.15
du2=0.02

#Begin monte carlo set up and loop

#Array set up
atm_man_pressures=sp.zeros((6,steps))#pressures, atm, maneos
tp_man_pressures=sp.zeros((6,steps))#pressures, triple point, maneos
atm_z_pressures=sp.zeros((6,steps))#pressures, atm, z
tp_z_pressures=sp.zeros((6,steps))#pressures, triple point, z
civ_tp_z=sp.zeros((6,steps)) #Critical impact velocities, triple point, z
civ_atm_z=sp.zeros((6,steps)) #Critical impact velocities, atm, z
civ_tp_man=sp.zeros((6,steps)) #Critical impact velocities, triple point, maneos
civ_atm_man=sp.zeros((6,steps)) #Critical impact velocities, atm, maneos

up=np.linspace(0,15,10000) #Array of particle velocities (km/s)
P=np.linspace(0,10**12,10000) #Pressure array

for j in range(0,steps):
    au=au1+au2*sp.randn() #Perturbations on hugoniot parameters
    bu=bu1+bu2*sp.randn()
    cu=cu1+cu2*sp.randn()
    du=du1+du2*sp.randn()
    ZP1=P#Pressure quartz
    aq1=aq+aqe*sp.randn()
    bq1=bq+bqe*sp.randn()
    cq1=cq+cqe*sp.randn()
    dq1=dq+dqe*sp.randn()
    aqh1=aqh+aqhe*sp.randn()
    bqh1=bqh+bqhe*sp.randn()
    cqh1=cqh+cqhe*sp.randn()
    dqh1=dqh+dqhe*sp.randn()
    aql1=aql+aqle*sp.randn()
    bql1=bql+bqle*sp.randn()
    cql1=cql+cqle*sp.randn()
    dql1=dql+dqle*sp.randn()
#    ZS1=sFit(P,aq1,bq1,cq1,dq1) + hot_ent
#    ZS1=entropy_hot(aq1,bq1,cq1,dq1,P/(10**9)) + hot_ent
    ZSh=entropy(aqh1,bqh1,cqh1,dqh1,P/(10**9))
    ZSl=entropy(aql1,bql1,cql1,dql1,P/(10**9))
    ZS1=entropy(aq1,bq1,cq1,dq1,P/(10**9))+ hot_ent#+entropy_uncert(aqu,bqu,cqu,P)*sp.randn()
    rho=2651+2651*0.03*sp.randn() #Density uncertainty

    #Interpolate between the perturbed points and M-ANEOS points
    fo_man=interpolate.interp1d(sm,xm,bounds_error=False) # getting the pressures, maneos
    fo_z=interpolate.interp1d(ZS1,ZP1,kind='linear',bounds_error=False) # getting the pressures, z

    #MANEOS critical pressures
    atm_man_pressures[0,j]=fo_man(Fo_s_atm[0])*10**-9 #incipient melt
    atm_man_pressures[1,j]=fo_man(lever(0.5,Fo_s_atm[0],Fo_s_atm[1]))*10**-9 #50% melt
    atm_man_pressures[2,j]=fo_man(Fo_s_atm[1])*10**-9 #complete melt
    atm_man_pressures[3,j]=fo_man(fo_sl_atm)*10**-9 #incipient vapor
    atm_man_pressures[4,j]=fo_man(lever(0.5,fo_sl_atm,fo_sv_atm))*10**-9 # 50%
    atm_man_pressures[5,j]=fo_man(fo_sv_atm)*10**-9 #complete vapor

    tp_man_pressures[0,j]=fo_man(fo_sol_tp)*10**-9 #incipient melt
    tp_man_pressures[1,j]=fo_man(lever(0.5,fo_sol_tp,fo_sl_tp))*10**-9 #50% melt
    tp_man_pressures[2,j]=fo_man(fo_sl_tp)*10**-9 #complete melt
    tp_man_pressures[3,j]=fo_man(fo_sl_tp)*10**-9 #incipient vapor
    tp_man_pressures[4,j]=fo_man(lever(0.5,fo_sl_tp,fo_sv_tp))*10**-9 # 50%
    tp_man_pressures[5,j]=fo_man(fo_sv_tp)*10**-9 #complete vapor

    #Z critical pressures
    atm_z_pressures[0,j]=fo_z(Fo_s_atm[1])*10**-9 #complete melt
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
    atm_man_pressures_final[i,0]=np.nanmean(atm_man_pressures[i,:])
    atm_man_pressures_final[i,1]=np.nanstd(atm_man_pressures[i,:])
    tp_man_pressures_final[i,0]=np.nanmean(tp_man_pressures[i,:])
    tp_man_pressures_final[i,1]=np.nanstd(tp_man_pressures[i,:])
    atm_z_pressures_final[i,0]=np.nanmean(atm_z_pressures[i,:])
    atm_z_pressures_final[i,1]=np.nanstd(atm_z_pressures[i,:])
    tp_z_pressures_final[i,0]=np.nanmean(tp_z_pressures[i,:])
    tp_z_pressures_final[i,1]=np.nanstd(tp_z_pressures[i,:])
    
    civ_atm_man_final[i,0]=np.nanmean(civ_atm_man[i,:])
    civ_atm_man_final[i,1]=np.nanstd(civ_atm_man[i,:])
    civ_tp_man_final[i,0]=np.nanmean(civ_tp_man[i,:])
    civ_tp_man_final[i,1]=np.nanstd(civ_tp_man[i,:])
    civ_atm_z_final[i,0]=np.nanmean(civ_atm_z[i,:])
    civ_atm_z_final[i,1]=np.nanstd(civ_atm_z[i,:])
    civ_tp_z_final[i,0]=np.nanmean(civ_tp_z[i,:])
    civ_tp_z_final[i,1]=np.nanstd(civ_tp_z[i,:])

print('MANEOS Atm release pressures',atm_man_pressures_final[:,0],'Uncertainty',atm_man_pressures_final[:,1])
print('MANEOS tp release pressures',tp_man_pressures_final[:,0],'Uncertainty',tp_man_pressures_final[:,1])
print('Z Atm release pressures',atm_z_pressures_final[:,0],'Uncertainty',atm_z_pressures_final[:,1])
print('Z tp release pressures',tp_z_pressures_final[:,0],'Uncertainty',tp_z_pressures_final[:,1])
print('MANEOS Atm Crit V',civ_atm_man_final[:,0],'Uncertainty',civ_atm_man_final[:,1])
print('MANEOS tp Crit V',civ_tp_man_final[:,0],'Uncertainty',civ_tp_man_final[:,1])
print('Z Atm Crit V',civ_atm_z_final[:,0],'Uncertainty',civ_atm_z_final[:,1])
print('Z tp Crit V',civ_tp_z_final[:,0],'Uncertainty',civ_tp_z_final[:,1])

with open('VaporFractions_Quartz_1200.txt','w') as f:
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
plt.plot(ZS1,P,label='Quartz')
plt.plot(ZSh,P,label='Quartz high')
plt.plot(ZSl,P,label='Quartz low')
plt.plot(sm,xm,label='Fo')
plt.legend()
plt.show()  


