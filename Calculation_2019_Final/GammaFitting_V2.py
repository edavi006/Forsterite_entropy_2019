"""

Fitting Forsterite Gruneisen Parameters

@author: Erik Davies

Takes data from our own shallow release experiments and from the Asimow group
and creates a continuous fit.

Reports a functions that fits the data using a density dependence only.

Requires data file from the shallow release experiment, asimow gamma is
input with specific densities chosen from experiment shock densities.

This script implements alternative methods of fitting to better fit the data.
This script does not work!

"""

import pylab as py
import numpy as np
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.integrate as integrate
import sys
import scipy as sp
import statistics as stat
from scipy.optimize import curve_fit
from scipy import interpolate
from matplotlib import rc

########Plot Parameters begin############3
#These control font size in plots.
params = {'legend.fontsize': 10,
         'axes.labelsize': 10,
         'axes.titlesize':10,
         'xtick.labelsize':10,
         'ytick.labelsize':10}
from cycler import cycler
plt.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')
plt.rcParams.update(params)
plt.rcParams['xtick.major.size'] = 4
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['xtick.minor.size'] = 2
plt.rcParams['xtick.minor.width'] = 0.5
plt.rcParams['ytick.major.size'] = 4
plt.rcParams['ytick.major.width'] = 0.5
plt.rcParams['ytick.minor.size'] = 2
plt.rcParams['ytick.minor.width'] = 0.5
plt.rcParams['axes.linewidth']= 1

plt.rcParams['lines.linewidth'] = 1.0
plt.rcParams['lines.dashed_pattern'] = [6, 6]
plt.rcParams['lines.dashdot_pattern'] = [3, 5, 1, 5]
plt.rcParams['lines.dotted_pattern'] = [1, 3]
plt.rcParams['errorbar.capsize'] = 3
plt.rcParams['lines.scale_dashes'] = False
plt.rcParams['legend.fancybox'] = False
plt.rcParams['legend.framealpha'] = None
plt.rcParams['legend.edgecolor'] = 'inherit'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100

plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['font.family']='Times New Roman'
plt.rcParams['figure.figsize']=5,4

########Plot Parameters finish############3

#some Constants
size=100 #array size for created arrays
steps=100000 #array size for created arrays
rho_init_hp=5433

gamma_rho=2597 #Initial Liquid Volume

#Asimow gamma and q, 2018
gamma_asi=.396
gamma_a_e=0
q_asi=-2.02
q_a_e=1.03

def gamma_fit(x,a,b,c,d,e): #fitting function for gamma 444444444444444444values
    #return a * (x **(-b))
    #return a*(x**b) + c*(x**d) 
    #return a+ b*x**(1)+ c*x**(2)
    #return 2/3+(0.39-2/3)*np.power((2597/x),a) + b*np.exp(-np.power((x-5200)/c,2))
    return 2/3 + (a - 2/3)*(2597/x)**b + c*np.exp((-(x-d)**2)/(e**2))
    #return a*x**(0)+ b*x**(1)+ c*x**(2) + d*x**(3) + e*x**(4)
    #return a+b*np.sin(1*x)+c*np.cos(1*x)+d*np.sin(2*x)+e*np.cos(2*x)
    #return a+b*np.exp(-c*((x-d)**2))

def gamma_e_func(x):
    return 0.32*x

file='Gamma_Release_Data_gau.txt'
#Load in Shallow Release Densities and Gammas
SR_rho=np.loadtxt(file,delimiter=',',skiprows=1,usecols=[2])
SR_rho_e=np.loadtxt(file,delimiter=',',skiprows=1,usecols=[3])
SR_gamma=np.loadtxt(file,delimiter=',',skiprows=1,usecols=[0])
SR_gamma_e=np.loadtxt(file,delimiter=',',skiprows=1,usecols=[1])

#print(SR_rho)

#Creating data set from Asimow
#Input shock densities

asi_rho=np.array([2597,3740.0,3335.9,4691.0,4088.0,4347.0,4681.180])
asi_rho_e=np.array([11,0.01*1000,0.03*1000,0.009*1000,0.17*1000,0.05*1000,0.08*1000])

asi_rho2=np.array([2597,3740.0,3335.9,4088.0])# Additional lower points to note that those are probably more constrained
asi_rho_e2=np.array([11,0.01*1000,0.03*1000,0.17*1000])


#Center points for asimow
asi_g_cent=gamma_asi*((gamma_rho/asi_rho)**q_asi)

asi_rho_mc=np.zeros((np.size(asi_rho),steps))#mc
SR_rho_mc=np.zeros((np.size(SR_rho),steps))#mc
asi_gamma_mc=np.zeros((np.size(asi_rho),steps))#mc
SR_gamma_mc=np.zeros((np.size(SR_rho),steps))#mc
A=np.zeros(steps)
B=np.zeros(steps)
C=np.zeros(steps)
D=np.zeros(steps)
E=np.zeros(steps)
rho_index=np.linspace(gamma_rho-500, 7000, size)
gamma_fitted_mc=np.zeros((size,steps))
for i in range(0,steps):
    
    gamma_asi2=gamma_asi+gamma_a_e*sp.randn(np.size(gamma_asi))#Additional for weighting
    
    q_asi2=q_asi+q_a_e*sp.randn(np.size(q_asi))#Additional for weighting
    SR_rho_mc[:,i]=SR_rho+SR_rho_e*sp.randn(np.size(SR_rho_e))
    asi_rho_mc[:,i]=asi_rho+asi_rho_e*sp.randn(np.size(asi_rho_e)) #
    asi_rho_mc2=asi_rho+asi_rho_e*sp.randn(np.size(asi_rho_e)) #Additional for weighting
    SR_gamma_mc[:,i]=SR_gamma+SR_gamma_e*sp.randn(np.size(SR_gamma_e))

    rho_init_l=gamma_rho+11*sp.randn()#For asimow gamma
    #q_asi1=q_asi+q_a_e*sp.randn()
    for j in range(0,np.size(asi_rho)):
        q_asi1=q_asi+q_a_e*sp.randn()
        gamma_asi1=gamma_asi+gamma_a_e*sp.randn()
        asi_gamma_mc[j,i]=gamma_asi1*((rho_init_l/asi_rho_mc[j,i])**q_asi1)
    #asi_gamma_mc2=gamma_asi2*((rho_init_l/asi_rho_mc2)**q_asi2)#Additional for weighting
sorted_asi_gamma_mc=np.zeros((np.size(asi_rho),steps))#mc
asi_gamma=sp.zeros(np.size(asi_rho))
asi_gamma_std=sp.zeros(np.size(asi_rho))
asi_gamma_upe=sp.zeros(np.size(asi_rho))
asi_gamma_lowe=sp.zeros(np.size(asi_rho))



for i in range(np.size(asi_rho)):
    sorted_asi_gamma_mc[i,:]=np.sort(asi_gamma_mc[i,:])
    asi_gamma[i]=stat.median(asi_gamma_mc[i,:])
 #   asi_gamma[i]=np.mean(asi_gamma_mc[i,:])
    asi_gamma_std[i]=np.std(asi_gamma_mc[i,:])
    asi_gamma_lowe[i]=-sorted_asi_gamma_mc[i,int(0.841*steps)]+asi_gamma[i] #picking out the standard deviation
    asi_gamma_upe[i]=sorted_asi_gamma_mc[i,int(0.159*steps)]-asi_gamma[i]
asi_gammae=[asi_gamma_lowe,asi_gamma_upe]
#print(asi_gamma,asi_gamma_lowe,asi_gamma_upe)
    
final_rho=[]
final_gamma=[]
final_rho_e=[]
final_gamma_e=[]

final_rho.extend(SR_rho[:])
final_gamma.extend(SR_gamma[:])
final_rho.extend(asi_rho[:])
final_gamma.extend(asi_gamma[:])

final_rho_e.extend(SR_rho_e[:])
final_gamma_e.extend(SR_gamma_e[:])
final_rho_e.extend(asi_rho_e[:])
final_gamma_e.extend(asi_gamma_std[:])

#Weighting the lone ambient point
#final_rho.extend(np.array([2300]))
#final_gamma.extend(np.array([0.35]))
#final_rho_e.extend(np.array([11]))
#final_gamma_e.extend(np.array([0.1]))


    #final_rho.extend(asi_rho_mc2)
    #final_gamma.extend(asi_gamma_mc2)
#for k in range(2):
        #final_gamma.extend(np.array([0.5 + 0.3*sp.randn()]))
        #final_rho.extend(np.array([8000]))
 #   final_gamma.extend(np.array([0.396]))
 #   final_rho.extend(np.array([2597]))
    #print(final_rho)


    
temp1, temp2 =curve_fit(gamma_fit, final_rho,final_gamma,p0=[0.2,0.3,0.7,5000,1200],
                        #bounds=[[0,0,0,3000,0],[1,1,2,7000,4000]],absolute_sigma=True,max_nfev=20000)
                        absolute_sigma=True,maxfev=20000)
A=np.mean(temp1[0])
B=np.mean(temp1[1])
C=np.mean(temp1[2])
D=np.mean(temp1[3])
E=np.mean(temp1[4])

#print(temp1)
#p0=np.asarray([0.39, 0.33, 0.9, 5200,1200]

A_mean=np.mean(A)
#A_std=np.std(A)
B_mean=np.mean(B)
#B_std=np.std(B)
C_mean=np.mean(C)
#C_std=np.std(C)
D_mean=np.mean(D)
#D_std=np.std(D)
E_mean=np.mean(E)
#E_std=np.std(E)

#X=[]
#X.append(A)
#X.append(B)
#X.append(C)
#X.append(D)
#X.append(E)
#Covar=np.cov(X)
#E_mean=np.mean(E)
#E_std=np.std(E)
Covar=temp2

print("A,B,C,D,E",A_mean,B_mean,C_mean,D_mean,E_mean)
#print("Ae,Be,Ce,De,Ee",A_std,B_std,C_std, D_std)#, E_std)
print(Covar)


#plt.figure()
#plt.hist(A,100)
#plt.figure()
#plt.hist(B,100)
#plt.figure()
#plt.hist(C,100)
#plt.figure()
#plt.hist(D,100)

#plt.hist(asi_gamma_mc[4,:],100)

plt.hist(asi_gamma_mc[5,:],100)
plt.plot(asi_gamma[5], 800, 'o')
plt.plot(asi_gamma[5]+asi_gamma_upe[5], 800, 'o')
plt.plot(asi_gamma[5]-asi_gamma_lowe[5], 800, 'o')

##Final_gammas=[asi_gamma,SR_gamma]
##Final_gammas_e=[asi_gamma_e,SR_gamma_e]
##Final_rho=[asi_rho,SR_rho]
##final_rho_e=[asi_rho_e,SR_rho_e]

#Propagate errors to relative density relationship
#asi_rel_rho=(gamma_rho/(asi_rho[:]**2))*asi_rho_e[:]
#SR_rel_rho=(gamma_rho/(SR_rho[:]**2))*SR_rho_e[:]

lmat=sp.linalg.cholesky(Covar,lower=True)
########
rho_index=np.linspace(gamma_rho-100, 7000, size)
##gamma_fitted_mc=np.zeros((size,steps))
##for i in range(steps):
##    temp_mat=sp.randn(5,1)
##    bmat=np.matmul(lmat,temp_mat)
##    a1=A_mean+bmat[0,0]
##    b1=B_mean+bmat[1,0]
##    c1=C_mean+bmat[2,0]
##    d1=D_mean+bmat[3,0]
##    e1=E_mean+bmat[4,0]
##    gamma_fitted_mc[:,i]=gamma_fit(rho_index,a1,b1,c1,d1,e1)
###Gamma_eye=gamma_fit(rho_index,2.04092*10**(-9),3.686659,-2.03996931745*10**(-9),3.68671127744)
##gamma_fitted_mean=np.median(gamma_fitted_mc, axis=1)
##gamma_fit_std=np.std(gamma_fitted_mc, axis=1)

gamma_center=gamma_fit(rho_index,A_mean,B_mean,C_mean,D_mean,E_mean)
gamma_center_asi =gamma_asi*((rho_init_l/rho_index)**q_asi)

gamma_center_e=gamma_e_func(gamma_center)

#Now for each, find percentage of MC'd parameters fit in the fit. These are already generated with the following names
#asi_rho_mc=np.zeros((np.size(asi_rho),steps))#mc
#SR_rho_mc=np.zeros((np.size(SR_rho),steps))#mc
#asi_gamma_mc=np.zeros((np.size(asi_rho),steps))#mc
#SR_gamma_mc=np.zeros((np.size(SR_rho),steps))#mc

#First, get percentage at each rho that fits into the error bands, then sum percentage from all
gamma_percent=[]
gamma_sum=0
gamma_cent_check=0
count=0 #Count of total samples

#First Asimow stuff
for i in range(np.size(asi_rho)):
    gamma_cent_check=gamma_fit(asi_rho[i],A_mean,B_mean,C_mean,D_mean,E_mean)
    gamma_error=gamma_e_func(gamma_cent_check)
    temp_sort=np.sort(asi_gamma_mc[i,:])
    p_check_min=min(min(sp.where(temp_sort > gamma_cent_check-gamma_error)))
    p_check_max=max(min(sp.where(temp_sort < gamma_cent_check+gamma_error)))
    #print(p_check_min)
   # print(p_check_max)
    temp_percent=((p_check_max-p_check_min)/steps)*100
    gamma_percent.extend(np.array([temp_percent]))

    gamma_sum=gamma_sum+( p_check_max-p_check_min)

    count=count+1

for i in range(np.size(SR_rho)):
    gamma_cent_check=gamma_fit(SR_rho[i],A_mean,B_mean,C_mean,D_mean,E_mean)
    gamma_error=gamma_e_func(gamma_cent_check)
    temp_sort=np.sort(SR_gamma_mc[i,:])
    p_check_min=min(min(sp.where(temp_sort > gamma_cent_check-gamma_error)))
    p_check_max=max(min(sp.where(temp_sort < gamma_cent_check+gamma_error)))
    #print(p_check_min)
   # print(p_check_max)
    temp_percent=((p_check_max-p_check_min)/steps)*100
    gamma_percent.extend(np.array([temp_percent]))

    gamma_sum=gamma_sum+( p_check_max-p_check_min)

    count=count+1
gamma_total_percent=(gamma_sum/(count*steps))*100

print(gamma_total_percent)

##sorted_gamma_fit=np.zeros((size,steps))
##gamma_fitted_center=np.zeros(size)
##gamma_fitted_std=np.zeros(size)
##gamma_fitted_upe=np.zeros(size)
##gamma_fitted_lowe=np.zeros(size)

##for i in range(size):
##    sorted_gamma_fit[i,:]=np.sort(gamma_fitted_mc[i,:])
##    gamma_fitted_center[i]=stat.median(gamma_fitted_mc[i,:])
## #   asi_gamma[i]=np.mean(asi_gamma_mc[i,:])
##    gamma_fitted_std[i]=np.std(sorted_gamma_fit[i,:])
##    gamma_fitted_upe[i]=sorted_gamma_fit[i,int(0.841*steps)]- gamma_fitted_center[i] #picking out the standard deviation
##    gamma_fitted_lowe[i]=-sorted_gamma_fit[i,int(0.159*steps)]+ gamma_fitted_center[i]
##gamma_fitted_e=[gamma_fitted_upe,gamma_fitted_lowe]

#DFT data
#gamma_dft=np.loadtxt('ForsteriteHugoniot_FINAL.txt',skiprows=4,usecols=[4])
#dens_dft=np.loadtxt('ForsteriteHugoniot_FINAL.txt',skiprows=4,usecols=[0])*1000


#Sarah-Kraus fit
ginf = 2./3.
g00=0.4
beta = 0.33
gee = .9
rsig=1200
ree=5200
krausgammaarr = ginf+(g00-ginf)*np.power((2597/rho_index),beta) + gee*np.exp(-np.power((rho_index-ree)/rsig,2))

#Figures
##########Gamma - Rho###########3
plt.figure()
plt.rc('grid',color='black', linestyle=':', linewidth=0.5,alpha=0.25)
#plt.grid()
plt.plot(rho_index, gamma_center,color='blue',label='Fit, This Work')
plt.fill_between(rho_index,gamma_center+gamma_center_e,gamma_center-gamma_center_e, alpha=0.5, color='blue')
plt.errorbar(asi_rho,asi_gamma,yerr=asi_gammae, xerr=asi_rho_e, fmt='o',label='Thomas et al. 2013',color='red')
#plt.plot(asi_rho,asi_g_cent, 'o')
plt.errorbar(SR_rho,SR_gamma,yerr=SR_gamma_e, xerr=SR_rho_e, fmt='o', label ='This Work', color='green')
#plt.plot(rho_index, gamma_fitted_mean, label='Fit, This Work', color='blue')
#plt.fill_between(rho_index,gamma_fitted_mean+gamma_fit_std,gamma_fitted_mean-gamma_fit_std, alpha=0.5, color='blue')
#plt.plot(rho_index, gamma_fitted_center, label='Fit, This Work', color='blue')
#plt.fill_between(rho_index,gamma_fitted_center+gamma_fitted_upe,gamma_fitted_center-gamma_fitted_lowe, alpha=0.5, color='blue')
#plt.plot(rho_index, krausgammaarr, label='Kraus Fit, no Uncertainty', color='brown')
#plt.plot(rho_index, Gamma_eye, label='Gamma eyeball Fit')
plt.ylim(0,2)
plt.xlabel('Density (kg/m$^3$)')
plt.ylabel('Grüneisen parameter')
plt.legend(loc='best', fontsize='x-small',numpoints=1,scatterpoints=1)
plt.savefig('Gamma_fit_V-gamma_Swift2.pdf', format='pdf')
plt.savefig('Gamma_fit_V-gamma_swift2.png', format='png', dpi=1000)

#Figures
##########Gamma - Rho###########3
##plt.figure()
##plt.rc('grid',color='black', linestyle=':', linewidth=0.5,alpha=0.25)
###plt.grid()
###for i in range(steps):
###    plt.plot(rho_index,gamma_fitted_mc[:,i], color='black', alpha=0.1)
##plt.plot(rho_index, gamma_center,color='red')
##plt.fill_between(rho_index,gamma_center+gamma_center_e,gamma_center-gamma_center_e, alpha=0.5, color='red')
##
##plt.plot(rho_index, gamma_center_asi,color='blue')
##
##plt.errorbar(asi_rho,asi_gamma,yerr=asi_gammae, xerr=asi_rho_e, fmt='o',label='Thomas et al. 2013')
###plt.plot(asi_rho,asi_g_cent, 'o')
##plt.errorbar(SR_rho,SR_gamma,yerr=SR_gamma_e, xerr=SR_rho_e, fmt='o', label ='This Work', color='green')
###plt.plot(rho_index, gamma_fitted_mean, label='Fit, This Work', color='blue')
###plt.fill_between(rho_index,gamma_fitted_mean+gamma_fit_std,gamma_fitted_mean-gamma_fit_std, alpha=0.5, color='blue')
###plt.plot(rho_index, gamma_fitted_center, label='Fit, This Work', color='blue')
###plt.fill_between(rho_index,gamma_fitted_center+gamma_fitted_upe,gamma_fitted_center-gamma_fitted_lowe, alpha=0.5, color='blue')
###plt.plot(rho_index, Gamma_eye, label='Gamma eyeball Fit')
###plt.plot(dens_dft,gamma_dft,'o',label='QMD')
##plt.plot(rho_index, krausgammaarr, label='Kraus Fit, no Uncertainty', color='brown')
##plt.ylim(0,2)
##plt.xlabel('Density (kg/m$^3$)')
##plt.ylabel('Grüneisen parameter')
##plt.legend(loc='best', fontsize='x-small',numpoints=1,scatterpoints=1)
##plt.savefig('Gamma_fit_V-gammaDFT_swift2.pdf', format='pdf')
##plt.savefig('Gamma_fit_V-gammaDFT_swift2.png', format='png', dpi=1000)


plt.show()
