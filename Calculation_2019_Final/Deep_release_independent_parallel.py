"""

Trying to calculate the Gruneisen parameter from deep release data.

@author: Erik Davies

The principles from which this works. The experiment has a flyer impact a sample
backed by a lower imedance window in which to release into. By definition, the
pressure and particle velocity in the window are the same as that reflected into
sample.

From this I have P-V-up at the shock state, and P-up at the release state. 
A Riemann integral transformation gives us P-V on release in a seperate code.
Once we have P-V points, we fit an isentrope.
We do this for each initial shock state, so we will
have several isentropes. We then calculate the gruneisen paramter by the
gruneisen parameter approximation for volume dependence.




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
#from Parallel_test import Mie_Grun_iterator
from Parallel_MieGrun import Mie_Grun_iterator
from scipy.optimize import curve_fit
from scipy import interpolate
from matplotlib import rc
from joblib import Parallel, delayed

#These control font size in plots.
params = {'legend.fontsize': 10,
         'axes.labelsize': 10,
         'axes.titlesize':10,
         'xtick.labelsize':10,
         'ytick.labelsize':10}
plt.rcParams.update(params)
plt.rcParams['font.family']='Times New Roman'
plt.rcParams['figure.figsize']=5,4

#Some Constants
steps1=10000 #monte carlo steps for at iterator
steps2=10000 #Inital MC for Vales
size=1000 #array size for created arrays
rho0s=3220#3327 #kg/m3 initial density in solid #Forsterite
rho0se=rho0s*0.003

ma=140.6931 #(g/mol) molecular weight of forsterite
ma=ma/1000 # putting the previous into kg
gamma_rho=1000000/(52.36/ma)

class state: #
    def __init__(s):
        s.note=''
        s.window=''
        s.shot=0
        s.flyer=0
        s.flyer_e=0
        s.win_us=0
        s.win_us_e=0
        s.sam_us=0
        s.sam_us_e=0
        s.sam_up=0
        s.sam_up_e=0
        s.sam_P=0
        s.sam_P_e=0
        s.sam_dens=0
        s.sam_dens_e=0
        s.rel_P=0
        s.rel_P_e=0
        s.rel_dens=0
        s.rel_dens_e=0
        s.p_flag=0

def make_params(note):
    h=state()
    h.note=note
    return h

fo=make_params('Forsterite shock states and released states.')
print(fo.note)
file='Release_VandP_Data.txt'
#load the data

tem_sam_P=np.loadtxt(file,dtype='float',delimiter=',',skiprows=1,usecols=[4])
tem_sam_P_e=np.loadtxt(file,dtype='float',delimiter=',',skiprows=1,usecols=[5])
tem_sam_dens=np.loadtxt(file,dtype='float',delimiter=',',skiprows=1,usecols=[6])
tem_sam_dens_e=np.loadtxt(file,dtype='float',delimiter=',',skiprows=1,usecols=[7])
tem_rel_P=np.loadtxt(file,dtype='float',delimiter=',',skiprows=1,usecols=[0])
tem_rel_P_e=np.loadtxt(file,dtype='float',delimiter=',',skiprows=1,usecols=[1])
tem_rel_dens=np.loadtxt(file,dtype='float',delimiter=',',skiprows=1,usecols=[2])
tem_rel_dens_e=np.loadtxt(file,dtype='float',delimiter=',',skiprows=1,usecols=[3])

index=np.where(tem_rel_dens > 5400)
fo.sam_P=tem_sam_P[index]
fo.sam_P_e=tem_sam_P_e[index]
fo.sam_dens=tem_sam_dens[index]
fo.sam_dens_e=tem_sam_dens_e[index]
fo.rel_P=tem_rel_P[index]
fo.rel_P_e=tem_rel_P_e[index]
fo.rel_dens=tem_rel_dens[index]
fo.rel_dens_e=tem_rel_dens_e[index]

#Forsterite Hugoniot set up
up=np.linspace(3,14,size) #defining particle velocity
def fo_hugoniot(a,b,c,d,x):
        #return a + b*x - c * x *np.exp(-d*x)
    return a + b*x +c*x**2+d*x**3
#fo Hugoniot fit information, cubic is just easiest to use and propagate error
a1=4.63200 # 
b1=1.45495
c1=0.00429
d1=-7.843708417433112e-04
covparamh=[[   1.831017240698122,  -0.652611265951972 ,  0.073185789070778,  -0.002609417024651],
 [ -0.652611265951972,   0.236606823014799,  -0.026846129171457,   0.000967124009405],
 [  0.073185789070778,  -0.026846129171457,   0.003083556207129,  -0.000112248130990],
 [ -0.002609417024651 ,  0.000967124009405,  -0.000112248130990,   0.000004124931939]]
lmath=sp.linalg.cholesky(covparamh,lower=False)

#Arrays
#forsterite
PHmc=sp.zeros((size,steps2))#mc
rhoH=sp.zeros((size,steps2))#mc
P=sp.zeros(size)#fin
Pe=sp.zeros(size)#fin
rho=sp.zeros(size)#fin
rhoe=sp.zeros(size)#fin

for j in range(0,steps2):
    #Making a Forsterite Hugoniot
    bmath=np.matmul(sp.randn(1,4), lmath) #For covariance calculation on the hugoniot
    ah=a1+bmath[0,0]
    bh=b1+bmath[0,1]
    ch=c1+bmath[0,2]
    dh=d1+bmath[0,3]
    rho_init_s=rho0s+rho0se*sp.randn() #initial density #forsterite
    us=fo_hugoniot(ah,bh,ch,dh,up)#getting shock velocity
    rhoH[:,j]=rho_init_s*us/(us-up) #getting density array
    PHmc[:,j]=rho_init_s*us*up # Getting pressure from rankine-hugoniot, in MPa
    PHmc[:,j]=PHmc[:,j]*(10**6) # putting pressure into Pa
#Averaging things
for i in range(0,size):
    P[i]=np.mean(PHmc[i,:])
    Pe[i]=np.std(PHmc[i,:])
    rho[i]=np.mean(rhoH[i,:])
    rhoe[i]=np.std(rhoH[i,:])

#Begin

#Isentrope MC arrays
P_isen=sp.zeros((np.size(fo.sam_P),steps1,size)) #data points, mc steps, density index size
E_isen=sp.zeros((np.size(fo.sam_P),steps1,size))
gamma_isen=sp.zeros((np.size(fo.sam_P),steps1,size))
P_isen_f=sp.zeros((np.size(fo.sam_P),size)) #data points, density index size
E_isen_f=sp.zeros((np.size(fo.sam_P),size))
P_isen_fu=sp.zeros((np.size(fo.sam_P),size)) #data points, density index size
E_isen_fu=sp.zeros((np.size(fo.sam_P),size))
gamma_isen_f=sp.zeros((np.size(fo.sam_P),size))
gamma_isen_fu=sp.zeros((np.size(fo.sam_P),size)) #data points, density index size
gamma_rel=sp.zeros((np.size(fo.sam_P),steps1))
q_rel=sp.zeros((np.size(fo.sam_P),steps1))
gamma_rel_f=sp.zeros((np.size(fo.sam_P)))
q_rel_f=sp.zeros((np.size(fo.sam_P)))
gamma_rel_fu=sp.zeros((np.size(fo.sam_P)))
q_rel_fu=sp.zeros((np.size(fo.sam_P)))


#Isentrope in parallel arrays
P_isen_p1=sp.zeros((np.size(fo.sam_P),size)) #data points, density index size
E_isen_p1=sp.zeros((np.size(fo.sam_P),size))
gamma_isen_p1=sp.zeros((np.size(fo.sam_P),size))
q_rel_p1=sp.zeros((np.size(fo.sam_P)))
gamma_rel_p1=sp.zeros((np.size(fo.sam_P)))


#Asimow gamma and q, 2018
gamma_asi1=1.2#.396#1
gamma_a_e=.5
#q_asi1=1.5# for linear or exponential#
#q_asi1=-10**-3 # for non-normalized
q_asi1=-2.1*10**-6 # for non-normalized gaussian
#q_a_e=.05 #iterative step size for gaussian
q_a_e=-.1#-10**-5 #iterative step size
hp_dens=5000#7033
hp_dens_e=500
liq_den=2597
liq_den_e=11
print('Got to Isentrope Iteration Section')
#Iterate to gamma and q
with Parallel(n_jobs=-2,require='sharedmem') as parallel:
#for j in range(0,steps1):
#    P_isen[:,j,:], E_isen[:,j,:], gamma_isen[:,j,:], gamma_rel[:,j], q_rel[:,j] =Mie_Grun_iterator(
 #                     lmath,rho0s,rho0se,liq_den,liq_den_e,hp_dens,hp_dens_e,
  #                    fo.sam_P,fo.sam_P_e,fo.sam_dens,fo.sam_dens_e,fo.rel_P,
   #                   fo.rel_P_e,fo.rel_dens,fo.rel_dens_e,gamma_asi1,gamma_a_e,q_asi1,q_a_e,up,P_isen_p1,
    #                  E_isen_p1,gamma_isen_p1,q_rel_p1,gamma_rel_p1,a1,b1,c1)
    parallel(delayed(Mie_Grun_iterator)(
                      lmath,rho0s,rho0se,liq_den,liq_den_e,hp_dens,hp_dens_e,
                      fo.sam_P,fo.sam_P_e,fo.sam_dens,fo.sam_dens_e,fo.rel_P,
                      fo.rel_P_e,fo.rel_dens,fo.rel_dens_e,gamma_asi1,gamma_a_e,q_asi1,q_a_e,up,P_isen,
                      E_isen,gamma_isen,q_rel,gamma_rel,a1,b1,c1,d1,j) for j in range(0,steps1))
    
    #Reset arrays to zeros to re-fill
##    P_isen_p1=sp.zeros((np.size(fo.sam_P),size)) #data points, density index size
##    E_isen_p1=sp.zeros((np.size(fo.sam_P),size))
##    gamma_isen_p1=sp.zeros((np.size(fo.sam_P),size))
##    q_rel_p1=sp.zeros((np.size(fo.sam_P)))
##    gamma_rel_p1=sp.zeros((np.size(fo.sam_P)))

    #print(gamma_rel_p)
#    if j == int(0.5*steps1):
 #       print('Halfway Done')
    #print(np.nanmean(E_isen[3,j,:]))                
#Average things for plotting
for i in range(0,np.size(fo.sam_P)):
    gamma_rel_f[i]=np.nanmedian(gamma_rel[i,:])
    q_rel_f[i]=np.nanmedian(q_rel[i,:])
    gamma_rel_fu[i]=np.nanstd(gamma_rel[i,:])
    q_rel_fu[i]=np.nanstd(q_rel[i,:])
    P_isen_f[i,:]=np.nanmedian(P_isen[i,:,:],axis=0)
    P_isen_fu[i,:]=np.nanstd(P_isen[i,:,:],axis=0)
    E_isen_f[i,:]=np.nanmedian(E_isen[i,:,:],axis=0)
    E_isen_fu[i,:]=np.nanstd(E_isen[i,:,:],axis=0)
    gamma_isen_f[i,:]=np.nanmedian(gamma_isen[i,:,:],axis=0)
    gamma_isen_fu[i,:]=np.nanstd(gamma_isen[i,:,:],axis=0)

##Write Gamma, density, and uncertainties to file.
with open('Gamma_Release_Data_gau.txt', 'w+') as f:
    f.write('Gamma, G_e, Density, rho_e , q, q_e \r\n')
    for i in range(np.size(fo.sam_P)):
        f.write(str(gamma_rel_f[i]))
        f.write(',')
        f.write(str(gamma_rel_fu[i]))
        f.write(',')
        f.write(str(fo.rel_dens[i]))
        f.write(',')
        f.write(str(fo.rel_dens_e[i]))
        f.write(',')
        f.write(str(q_rel_f[i]))
        f.write(',')
        f.write(str(q_rel_fu[i]))
        f.write('\r\n')
f.closed

print('Calculated variables for experiments.')
print('Release Pressure, P, P_e: ',fo.rel_P,fo.rel_P_e)
print('Release Density, rho, rho_e: ',fo.rel_dens, fo.rel_dens_e)
print('Release Gamma, gamma, gamma_e: ',gamma_rel_f,gamma_rel_fu) 


#Writing V-P paths to file - include E for posterity.########
##Write Gamma, density, and uncertainties to file.
with open('P_V_E_releasePaths_gau.txt', 'w+') as f:
    f.write('(P (Pa), Pe, E (J/kg), Ee) All Paths, rho (kg/m^3, index) \r\n')
    for j in range(np.size(rho)):
        for i in range(np.size(fo.sam_P)):
            f.write(str(P_isen_f[i,j]))
            f.write(',')
            f.write(str(P_isen_fu[i,j]))
            f.write(',')
            f.write(str(E_isen_f[i,j]))
            f.write(',')
            f.write(str(E_isen_fu[i,j]))
            f.write(',')
        f.write(str(rho[j]))
        f.write('\r\n')
f.closed
################3

#PLOTTING
#########################Rho-P############################
plt.figure()
plt.rc('grid',color='black', linestyle=':', linewidth=0.5,alpha=0.25)
plt.grid()

plt.plot(rho,P_isen_f[0,:]*(10**-9), color='red', label='Calculated Isentropes')
for i in range(1,np.size(fo.sam_P)):
    #index=min(np.where(P_isen_f[i,:]>2))
    plt.plot(rho,P_isen_f[i,:]*(10**-9), color='red')
    plt.fill_between(rho,(P_isen_f[i,:]-P_isen_fu[i,:])*(10**-9),(P_isen_f[i,:]+P_isen_fu[i,:])*(10**-9),alpha=0.25, color='red')

plt.plot(rho,P*(10**-9),color='blue', label='Fo, Root et al. 2018')
plt.fill_between(rho,(P-Pe)*(10**-9),(P+Pe)*(10**-9),color='blue',alpha=0.4)
#plt.fill_betweenx(P*(10**-9),(rho-rhoe),(rho+rhoe),color='blue',alpha=0.4)

plt.errorbar(fo.sam_dens, fo.sam_P, yerr=fo.sam_P_e, xerr=fo.sam_dens_e, fmt='o', label='Shock States', color='blue')
plt.errorbar(fo.rel_dens, fo.rel_P, yerr=fo.rel_P_e, xerr=fo.rel_dens_e, fmt='o', label='Release States', color='green')

    
                                                                          
plt.ylim(100,800)
plt.xlim(5250,7500)
plt.xlabel('Density (kg/m$^3$)')
plt.ylabel('Pressure (GPa)')
plt.legend(loc='best', fontsize='x-small',numpoints=1,scatterpoints=1)
plt.savefig('Gamma_calc_V-P_gau.pdf', format='pdf', dpi=1000)
###########################################################
#########################Rho-E############################
plt.figure()
plt.rc('grid',color='black', linestyle=':', linewidth=0.5,alpha=0.25)
plt.grid()

for i in range(0,np.size(fo.sam_P)):
    #index=min(np.where(P_isen_f[i,:]>2))
    plt.plot(rho,E_isen_f[i,:])
    plt.fill_between(rho[:],(E_isen_f[i,:]-E_isen_fu[i,:]),(E_isen_f[i,:]+E_isen_fu[i,:]),alpha=0.25)
    
                                                                          

plt.xlabel('Density (kg/m$^3$)')
plt.ylabel('Internal Energy (J/kg)')
#plt.legend(loc='best', fontsize='x-small',numpoints=1,scatterpoints=1)
plt.savefig('Gamma_calc_V-E_gau.pdf', format='pdf', dpi=1000)
###########################################################
#########################Rho-gamma############################
plt.figure()
plt.rc('grid',color='black', linestyle=':', linewidth=0.5,alpha=0.25)
plt.grid()

for i in range(0,np.size(fo.sam_P)):
    index=min(np.where(P_isen_f[i,:]>2))
    plt.plot(rho[index],gamma_isen_f[i, index])
    plt.fill_between(rho[ index],(gamma_isen_f[i, index]-gamma_isen_fu[i, index]),(gamma_isen_f[i, index]+gamma_isen_fu[i, index]),alpha=0.25)
for i in range(0,np.size(fo.sam_P)):
    plt.errorbar(fo.rel_dens[i],gamma_rel_f[i],yerr=gamma_rel_fu[i], xerr=fo.rel_dens_e[i], fmt='o')     
                                                                          
plt.ylim(0,3)
plt.xlabel('Densities (kg/m$^3$)')
plt.ylabel('Grüneisen parameter')
#plt.legend(loc='best', fontsize='x-small',numpoints=1,scatterpoints=1)
plt.savefig('Gamma_calc_V-gamma_gau.pdf', format='pdf', dpi=1000)
###########################################################
#####################GAMMA#################################

plt.figure()
plt.rc('grid',color='black', linestyle=':', linewidth=0.5,alpha=0.25)
plt.grid()

for i in range(0,np.size(fo.sam_P)):
    plt.errorbar(fo.rel_dens[i],gamma_rel_f[i],yerr=gamma_rel_fu[i], xerr=fo.rel_dens_e[i], fmt='o')                                                                    

plt.xlabel('Densities (kg/m$^3$)')
plt.ylabel('Grüneisen parameter')
plt.legend(loc='best', fontsize='x-small',numpoints=1,scatterpoints=1)
plt.savefig('Gamma_calc_V-gamma_gau.pdf', format='pdf', dpi=1000)

plt.show()
         









