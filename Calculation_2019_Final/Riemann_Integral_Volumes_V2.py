"""

Trying to calculate the Gruneisen parameter from deep release data.

@author: Erik Davies

The principles from which this works. The experiment has a flyer impact a sample
backed by a lower imedance window in which to release into. By definition, the
pressure and particle velocity in the window are the same as that reflected into
sample.

From this I have P-V-up at the shock state, and P-up at the release state. 
A Riemann integral transformation gives us P-V on release. Once we have P-V
points, we fit an isentrope. We do this for each initial shock state, so we will
have several isentropes. We then calculate the gruneisen paramter by the
gruneisen parameter approximation for volume dependence.

Version that doesn't use impedance matching, only independent measurement is
shock velocity, and uses known Hugoniots to get Up-P.


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

#Some Constants
steps1=10000 #monte carlo steps for at iterator
steps2=10000#Inital MC for Vales
size=1000 #array size for created arrays
rho0s=3220#3327 #kg/m3 initial density in solid #Forsterite
rho0se=rho0s*0.003

ma=140.6931 #(g/mol) molecular weight of forsterite
ma=ma/1000 # putting the previous into kg
gamma_rho=1000000/(52.36/ma)


rho_qtz_i=2651
rho_qtz_ie=rho_qtz_i*0.003

rho_tpx_i=833
rho_tpx_ie=rho_tpx_i*0.003

rho_fs_i=2200
rho_fs_ie=rho_fs_i*0.003

#First thing, class for properties
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
file='Z_Shot_Record-Deep Release-FO.csv'
#load the data

fo.window=np.loadtxt(file,dtype='S',delimiter=',',skiprows=1,usecols=[4])
fo.shot=np.loadtxt(file,dtype='float',delimiter=',',skiprows=1,usecols=[5])
fo.flyer=np.loadtxt(file,dtype='float',delimiter=',',skiprows=1,usecols=[6])
fo.flyer_e=np.loadtxt(file,dtype='float',delimiter=',',skiprows=1,usecols=[7])
fo.win_us=np.loadtxt(file,dtype='float',delimiter=',',skiprows=1,usecols=[8])
fo.win_us_e=np.loadtxt(file,dtype='float',delimiter=',',skiprows=1,usecols=[9])
fo.sam_us=np.loadtxt(file,dtype='float',delimiter=',',skiprows=1,usecols=[10])
fo.sam_us_e=np.loadtxt(file,dtype='float',delimiter=',',skiprows=1,usecols=[11])
fo.sam_up=np.loadtxt(file,dtype='float',delimiter=',',skiprows=1,usecols=[12])
fo.sam_up_e=np.loadtxt(file,dtype='float',delimiter=',',skiprows=1,usecols=[13])
fo.sam_P=np.loadtxt(file,dtype='float',delimiter=',',skiprows=1,usecols=[14])
fo.sam_P_e=np.loadtxt(file,dtype='float',delimiter=',',skiprows=1,usecols=[15])
fo.sam_dens=np.loadtxt(file,dtype='float',delimiter=',',skiprows=1,usecols=[16])
fo.sam_dens_e=np.loadtxt(file,dtype='float',delimiter=',',skiprows=1,usecols=[17])
#fo.rel_P=np.loadtxt('Z_Shot_Record-Deep Release-FO.csv',dtype='float',delimiter=',',skiprows=1,usecols=[18])
#fo.rel_P_e=np.loadtxt('Z_Shot_Record-Deep Release-FO.csv',dtype='float',delimiter=',',skiprows=1,usecols=[19])

#For this data set, flags are increasing integer values. This determines which
# shots will be grouped together to give more data for each release path
fo.p_flag=sp.zeros(np.size(fo.sam_P))-1
fo.rel_dens=sp.zeros(np.size(fo.sam_P))
fo.rel_dens_e=sp.zeros(np.size(fo.sam_P))
temp_f=0   #GROUP FLAG (1 = NO GROUP, 0 = GROUPING)
P_sens=1.5 #percentile of pressure sensitivity to inclue in single isentrope fit, default is 1.5%
for i in range(0,np.size(fo.sam_P)-1):
    if fo.p_flag[i] == -1:
        for j in range(i+1,np.size(fo.sam_us)):
            temp_flag=abs(fo.sam_us[i]-fo.sam_us[j])
            #print(temp_flag)
            if temp_flag/fo.sam_us[i] < P_sens/100:
                fo.p_flag[i]=temp_f
                fo.p_flag[j]=temp_f
            if fo.p_flag[i] == -1:
                fo.p_flag[i]=temp_f
            if j == np.size(fo.sam_P)-1:
                temp_f=temp_f+1

#print(fo.sam_P, fo.p_flag)
#Fitting up-P.
def P_UP_fit(x,a,b):
    #return a * (x **(-b))
    return a * (x **(-2)) + b

def UP_P_fit(x,a,b):
    return ((x-b)/a)**(-1.0/2)

#Forsterite Hugoniot set up

def fo_hugoniot(a,b,c,x):
    return a * x ** 2 + b * x + c
#fo Hugoniot fit information, quadratic is just easiest to use and propagate error
a1=4.63200 # 
b1=1.45495
c1=0.00429
d1=-7.843708417433112e-04
covparamh=[[   1.831017240698122,  -0.652611265951972 ,  0.073185789070778,  -0.002609417024651],
 [ -0.652611265951972,   0.236606823014799,  -0.026846129171457,   0.000967124009405],
 [  0.073185789070778,  -0.026846129171457,   0.003083556207129,  -0.000112248130990],
 [ -0.002609417024651 ,  0.000967124009405,  -0.000112248130990,   0.000004124931939]]
lmath=sp.linalg.cholesky(covparamh,lower=False)
#Quartz Hugoniot
#Universal Hugoniot quartz fit #knudson, also TPX 
def qtz_hugoniot(a,b,c,d,x):
    return a + b*x - c * x *np.exp(-d*x)
    #return a + b*x +c*x**2+d*x**3
    
#qtz hug params knudson 2013 (cubic), had to increase the a_3^2 term to 1.441 from 1.438 to force positive definite. rounding issue
au1=1.754
bu1=1.862
cu1=-3.364*(10**(-2))
du1=5.666*(10**(-4))
#uncertainties
covparamqtz=[[2.0970*(10.0**(-2.0)),-6.1590*(10.0**(-3.0)),5.5660*(10.0**(-4.0)),-1.5720*(10.0**(-5.0))],
             [-6.1590*(10.0**(-3.0)),1.8770*(10.0**(-3.0)),-1.7420*(10.0**(-4.0)),5.0170*(10.0**(-6.0))],
             [5.5660*(10.0**(-4.0)),-1.7420*(10.0**(-4.0)),1.6500*(10.0**(-5.0)),-4.8340*(10.0**(-7.0))],
             [-1.5720*(10.0**(-5.0)),5.0170*(10.0**(-6.0)),-4.8340*(10.0**(-7.0)),1.441*(10.0**(-8.0))]]
#print(sp.linalg.eigvalsh(covparamqtz))
lmath_qtz=sp.linalg.cholesky(covparamqtz,lower=False)

#TPX Hugoniot
at1=1.795
bt1=1.357
ct1=-0.694
dt1=0.273
at2=0.018
bt2=0.003
ct2=0.027
dt2=0.011

def fs_hugoniot(a,b,c,d,x):
    #return a + b*x - c * x *np.exp(-d*x)
    return a + b*x +c*x**2+d*x**3
#Fusedsilica - cubic from mccoy 2016
#af1=.798
#bf1=1.880
#cf1=-3.541*(10**(-2))
#df1=6.504*(10**(-4))
#Fusedsilica - cubic from root 2019 pending
af1=1.386
bf1=1.647
cf1=-0.01146
df1=-0.6052*(10**(-4))

covparamfs=[[4.857*(10**(-2)), -1.343*(10**(-2)), 1.17*(10**(-3)), -3.2366*(10**(-5))],
             [-1.343*(10**(-2)), 3.84*(10**(-3)), -3.4252*(10**(-4)), 9.6408*(10**(-6))],
             [1.17*(10**(-3)), -3.4252*(10**(-4)), 3.1126*(10**(-5)), -8.886*(10**(-7))],
             [-3.2366*(10**(-5)), 9.6408*(10**(-6)), -8.886*(10**(-7)), 2.5662*(10**(-8))]]

#covparamfs=[[9.056*(10**(-2)),-2.487*(10**(-2)),2.014*(10**(-3)),-4.974*(10**(-5))],
#             [-2.487*(10**(-2)),7.174*(10**(-3)),-5.997*(10**(-4)),1.515*(10**(-5))],
#             [2.014*(10**(-3)),-5.997*(10**(-4)),5.145*(10**(-5)),-1.327*(10**(-6))],
#             [-4.974*(10**(-5)),1.515*(10**(-5)),-1.327*(10**(-6)),3.486*(10**(-8))]]

lmath_fs=sp.linalg.cholesky(covparamfs,lower=False)

#Arrays
#calculating pressure of windows

fo_P_remc=sp.zeros((np.size(fo.sam_P),steps2))#mc release pressure
fo_up_remc=sp.zeros((np.size(fo.sam_P),steps2))#mc release pressure
fo_P_smc=sp.zeros((np.size(fo.sam_P),steps2))#mc release pressure
fo_up_smc=sp.zeros((np.size(fo.sam_P),steps2))#mc release pressure
fo_dens_smc=sp.zeros((np.size(fo.sam_P),steps2))#mc release pressure
#fo_P_s=sp.zeros(np.size(fo.sam_P))# fine
#fo_P_se=sp.zeros(np.size(fo.sam_P))# 
#print(np.size(fo.sam_P))


#forsterite
PHmc=sp.zeros((size,steps2))#mc
rhoH=sp.zeros((size,steps2))#mc
P=sp.zeros(size)#fin
Pe=sp.zeros(size)#fin
rho=sp.zeros(size)#fin
rhoe=sp.zeros(size)#fin

#quartz
PHmc_qtz=sp.zeros((size,steps2))#mc
rhoH_qtz=sp.zeros((size,steps2))#mc
P_qtz=sp.zeros(size)#fin
Pe_qtz=sp.zeros(size)#fin
rho_qtz=sp.zeros(size)#fin
rhoe_qtz=sp.zeros(size)#fin

#tpx
PHmc_tpx=sp.zeros((size,steps2))#mc
rhoH_tpx=sp.zeros((size,steps2))#mc
P_tpx=sp.zeros(size)#fin
Pe_tpx=sp.zeros(size)#fin
rho_tpx=sp.zeros(size)#fin
rhoe_tpx=sp.zeros(size)#fin

#fs
PHmc_fs=sp.zeros((size,steps2))#mc
rhoH_fs=sp.zeros((size,steps2))#mc
P_fs=sp.zeros(size)#fin
Pe_fs=sp.zeros(size)#fin
rho_fs=sp.zeros(size)#fin
rhoe_fs=sp.zeros(size)#fin

print('Calculating Pressures from Shock Velocity')
for j in range(0,steps2):
    if j == int(0.5*steps2):
        print('Halfway done')
    bmath=np.matmul(sp.randn(1,4), lmath) #For covariance calculation on the hugoniot
    ah=a1+bmath[0,0]
    bh=b1+bmath[0,1]
    ch=c1+bmath[0,2]
    dh=d1+bmath[0,3]
    qmath=np.matmul(sp.randn(1,4), lmath_qtz) #For covariance calculation on the hugoniot
    aqtz=au1+qmath[0,0]
    bqtz=bu1+qmath[0,1]
    cqtz=cu1+qmath[0,2]
    dqtz=du1+qmath[0,3]
    atpx=at1+at2*sp.randn()
    btpx=bt1+bt2*sp.randn()
    ctpx=ct1+ct2*sp.randn()
    dtpx=dt1+dt2*sp.randn()
    fsmath=np.matmul(sp.randn(1,4), lmath_fs)
    afs=af1+fsmath[0,0]
    bfs=bf1+fsmath[0,1]
    cfs=cf1+fsmath[0,2]
    dfs=df1+fsmath[0,3]

    #Window params
    win_us=fo.win_us+fo.win_us_e*sp.randn(np.size(fo.win_us))
    sam_us=fo.sam_us+fo.sam_us_e*sp.randn(np.size(fo.sam_us))
    win_ident=fo.window
    
    #Initial densities
    rho_init_s=rho0s+rho0se*sp.randn() #initial density #forsterite
    rho_qtz_init=rho_qtz_i+rho_qtz_ie*sp.randn() #quartz
    rho_tpx_init=rho_tpx_i+rho_tpx_ie*sp.randn() #tpx
    rho_fs_init=rho_fs_i+rho_fs_ie*sp.randn() #fused silica

    #Making a Forsterite Hugoniot
    up=np.linspace(3,18,size) #defining particle velocity
    us=fs_hugoniot(ah,bh,ch,dh,up)#getting shock velocity
    rhoH[:,j]=rho_init_s*us/(us-up) #getting density array
    PHmc[:,j]=rho_init_s*us*up # Getting pressure from rankine-hugoniot, in MPa
    PHmc[:,j]=PHmc[:,j]*(10**6) # putting pressure into Pa
    
    us_qtz=fs_hugoniot(aqtz,bqtz,cqtz,dqtz,up) #getting shock velocity
    rhoH_qtz[:,j]=rho_qtz_init*us_qtz/(us_qtz-up) #getting density array
    PHmc_qtz[:,j]=rho_qtz_init*us_qtz*up # Getting pressure from rankine-hugoniot, in MPa
    PHmc_qtz[:,j]=PHmc_qtz[:,j]*(10**6) # putting pressure into Pa

    us_tpx=qtz_hugoniot(atpx,btpx,ctpx,dtpx,up) #getting shock velocity
    rhoH_tpx[:,j]=rho_tpx_init*us_tpx/(us_tpx-up) #getting density array
    PHmc_tpx[:,j]=rho_tpx_init*us_tpx*up # Getting pressure from rankine-hugoniot, in MPa
    PHmc_tpx[:,j]=PHmc_tpx[:,j]*(10**6) # putting pressure into Pa

    us_fs=fs_hugoniot(afs,bfs,cfs,dfs,up) #getting shock velocity
    rhoH_fs[:,j]=rho_fs_init*us_fs/(us_fs-up) #getting density array
    PHmc_fs[:,j]=rho_fs_init*us_fs*up # Getting pressure from rankine-hugoniot, in MPa
    PHmc_fs[:,j]=PHmc_fs[:,j]*(10**6) # putting pressure into Pa

    #Calculate the pressures # this method of point finding for pressure introduces
    #uncertainty of less than 1 percent (about 0.1 percent depending) this is managable
    for i in range(0,np.size(fo.sam_P)):
        index_s=min(np.where(us > sam_us[i]))
        #print(index_s)
        fo_P_smc[i,j]=PHmc[index_s[0],j]
        fo_up_smc[i,j]=up[index_s[0]]
        fo_dens_smc[i,j]=rho_init_s*sam_us[i]/(sam_us[i]-fo_up_smc[i,j])
        if win_ident[i]==b'Quartz':
            index=min(np.where(us_qtz > win_us[i]))
            fo_P_remc[i,j]=PHmc_qtz[index[0],j]
            fo_up_remc[i,j]=up[index[0]]
            #print(np.size(index))
            #print(us_qtz[index[0]],win_us[i])
            #print('Quartz')           
        elif win_ident[i]==b'Fused Silica':
            index=min(np.where(us_fs > win_us[i]))
            fo_P_remc[i,j]=PHmc_fs[index[0],j]
            fo_up_remc[i,j]=up[index[0]]
            #print(us_fs[index[0]],win_us[i])
            #print('Fused Silica')
        elif win_ident[i]==b'TPX':
            #print(win_us[i], us_tpx[-1])
            index=min(np.where(us_tpx > win_us[i]))
            fo_P_remc[i,j]=PHmc_tpx[index[0],j]
            fo_up_remc[i,j]=up[index[0]]
            #print(us_tpx[index[0]],win_us[i])
            #print('TPX')
        else:
            print('error, no matching string')
            break
#Averaging things
for i in range(0,size):
    P[i]=np.mean(PHmc[i,:])
    Pe[i]=np.std(PHmc[i,:])
    rho[i]=np.mean(rhoH[i,:])
    rhoe[i]=np.std(rhoH[i,:])
    
    P_qtz[i]=np.mean(PHmc_qtz[i,:])
    Pe_qtz[i]=np.std(PHmc_qtz[i,:])
    rho_qtz[i]=np.mean(rhoH_qtz[i,:])
    rhoe_qtz[i]=np.std(rhoH_qtz[i,:])

    P_tpx[i]=np.mean(PHmc_tpx[i,:])
    Pe_tpx[i]=np.std(PHmc_tpx[i,:])
    rho_tpx[i]=np.mean(rhoH_tpx[i,:])
    rhoe_tpx[i]=np.std(rhoH_tpx[i,:])
    
    P_fs[i]=np.mean(PHmc_fs[i,:])
    Pe_fs[i]=np.std(PHmc_fs[i,:])
    rho_fs[i]=np.mean(rhoH_fs[i,:])
    rhoe_fs[i]=np.std(rhoH_fs[i,:])

#fill in Pressure
fo.rel_P=sp.zeros(np.size(fo.sam_P))
fo.rel_P_e=sp.zeros(np.size(fo.sam_P))
win_up=sp.zeros(np.size(fo.sam_P))
win_up_e=sp.zeros(np.size(fo.sam_P))
#for i in range(0,np.size(fo.sam_P)):
fo.rel_P=np.mean(fo_P_remc[:,:],axis=1)*(10**-9)
fo.rel_P_e=np.std(fo_P_remc[:,:],axis=1)*(10**-9)
fo.sam_P=np.mean(fo_P_smc[:,:],axis=1)*(10**-9)
fo.sam_P_e=np.std(fo_P_smc[:,:],axis=1)*(10**-9)
fo.sam_dens=np.mean(fo_dens_smc[:,:],axis=1)
fo.sam_dens_e=np.std(fo_dens_smc[:,:],axis=1)
fo.sam_up=np.mean(fo_up_smc[:,:],axis=1)
fo.sam_up_e=np.std(fo_up_smc[:,:],axis=1)
win_up=np.mean(fo_up_remc[:,:],axis=1)
win_up_e=np.std(fo_up_remc[:,:],axis=1)


###Fit Up-P points - single release point version
###first each individually.
###Function for fitting P-up release curves

##temp_p=[fo.rel_P[8],fo.sam_P[8]]
##temp_pe=[fo.rel_P_e[8],fo.sam_P_e[8]]
##temp_up=[win_up[8],fo.sam_up[8]]
###print(temp_p, temp_up)
##temp1, temp2 = curve_fit(P_UP_fit, temp_up,temp_p, sigma=temp_pe, absolute_sigma=True)#, bounds=[[-100,50],[0,1000]])
##fitP=P_UP_fit(temp1[0],temp1[1],up)
##
###print(temp1)
##temp_fitp=np.zeros((size,steps))
##lmat=sp.linalg.cholesky(temp2,lower=True)
##for j in range(0,steps):
##    temp_mat=sp.rand(2,1)
##    bmat=np.matmul(lmat,temp_mat)    
##    afit1=temp1[0]+bmat[0,0]#sp.randn()*(temp2[0,0]**(1/2))#bmat[0,0]
##    bfit1=temp1[1]+bmat[1,0]#sp.randn()*(temp2[1,1]**(1/2))#bmat[1,0]
##    temp_fitp[:,j]=P_UP_fit(up,afit1,bfit1)
##    
##P_mean=np.zeros(size)
##P_std=np.zeros(size)
##for i in range(0,size):
##    P_mean[i]=np.mean(temp_fitp[i,:])
##    P_std[i]=np.std(temp_fitp[i,:])
print('Begin Riemann Integral')
#Here group like pressures together, where statements, and fit up-P, then integrate for volume between states.
#Initialize needed arrays
P_mean=np.zeros((size,temp_f))
P_std=np.zeros((size,temp_f))
P_isentrope=np.zeros((size,np.size(fo.sam_dens)))
rho_isentrope=np.zeros((size,np.size(fo.sam_dens)))

for i in range(0, temp_f):
    index=min(np.where(fo.p_flag == i))
    temp_p=np.append(fo.rel_P[index],fo.sam_P[index]) #populate arrays
    temp_pe=np.append(fo.rel_P_e[index],fo.sam_P_e[index])
    temp_up=np.append(win_up[index],fo.sam_up[index])
    temp_upe=np.append(win_up_e[index],fo.sam_up_e[index])
    #print(temp_p, temp_up)
    #Fit the pressure-up points
    A=np.zeros(steps1)
    B=np.zeros(steps1)
   #C=np.zeros(steps)
    temp_fitp=np.zeros((size,steps1))
    V_temp=np.zeros((np.size(index),steps1))
    for j in range(steps1): #Cycle through Monte carlo steps for fit
        temp_p_mc=temp_p+temp_pe*sp.randn(np.size(temp_pe))
        temp_up_mc=temp_up+temp_upe*sp.randn(np.size(temp_upe))
        temp1, temp2 = curve_fit(P_UP_fit, temp_up_mc,temp_p_mc, bounds=[[0,-1000],[1000000,1000]])
        A[j]=temp1[0]
        B[j]=temp1[1]
        #C[j]=temp1[2]
        temp_fitp[:,j]=P_UP_fit(up,A[j],B[j])#,cfit1)
        for l in range(0,np.size(index)): #cycle though grouped points
            
            dens_temp=fo.sam_dens[index[l]]+fo.sam_dens_e[index[l]]*sp.randn() #shock density
            upr_temp1=win_up[index[l]]+win_up_e[index[l]]*sp.randn() #release particle velocity
            
            #Calculate Residual from the fit and add it to the releast particle velocity fit
            P_temp=fo.rel_P[index[l]]+fo.rel_P_e[index[l]]*sp.randn()# get pressure of release
            up_release_fitted=UP_P_fit(P_temp,A[j],B[j]) #get up of release predicted by fit
            up_diff=abs(up_release_fitted-upr_temp1) #calc the difference
            upr_temp=upr_temp1+up_diff*sp.randn() #Adding the difference into the calculated release particle velocity in an normal distribution
            #print(P_temp,upr_temp1,up_release_fitted,up_diff,upr_temp)
            ups_temp=fo.sam_up[index[l]]+fo.sam_up_e[index[l]]*sp.randn() #shock particle velocity
            # Calculating volume of release state from definite integral of P_UP_fit function
            #temp_int = ((1/(afit1 * bfit1 *(bfit1+2)))*(ups_temp**(bfit1+2) -upr_temp**(bfit1+2)))
            temp_int =  (1/(A[j] * 8)) * (ups_temp**(4) -upr_temp**(4)) #begin calc for volume, see ahrens 1969 feldspars
            #temp_int =  (1/(afit1 * 15)) * (ups_temp**(5) -upr_temp**(5)) #Different fitting function
            #print(1/dens_temp,temp_int)
            V_temp[l,j] = (1/dens_temp) - (temp_int * 10**(-3)) #end calc for volume
            #print(V_temp[l,j])
    #fitP=P_UP_fit(temp1[0],temp1[1],up)
    Amean=np.mean(A)
    Bmean=np.mean(B)
    #Cmean=np.mean(C)
    X=[]
    X.append(A)
    X.append(B)
    #X.append(C)
    Covar=np.cov(X)
    print(temp1)
    print(Covar)
    #print(temp1)
    lmat=sp.linalg.cholesky(Covar,lower=True)
##    for j in range(0,steps1): #Cycle through monte carlo steps to calc volume
##        temp_mat=sp.randn(2,1)
##        bmat=np.matmul(lmat,temp_mat)    
##        afit1=Amean+bmat[0,0]
##        bfit1=Bmean+bmat[1,0]
##        #cfit1=Cmean+bmat[2,0]
##        temp_fitp[:,j]=P_UP_fit(up,afit1,bfit1)#,cfit1)
##        #integration for volume

    volume_isen=np.zeros((np.size(index),size)) # Arrays for calculating Volume over the entire fit.
    P_ise=np.zeros((np.size(index),size))# Arrays for calculating Volume over the entire fit.
    for l in range(0,np.size(index)): #cycle though grouped points #Second system to cycle and calculate volume change over the fits
        dens_temp=fo.sam_dens[index[l]] #shock density
        #upr_temp=win_up[index[l]]+win_up_e[index[l]]*sp.randn() #release particle velocity
        P_ise[l,:]=P_UP_fit(up,Amean,Bmean)
        ups_temp=fo.sam_up[index[l]] #shock particle velocity
        # Calculating volume of release state from definite integral of P_UP_fit function
        #temp_int = ((1/(afit1 * bfit1 *(bfit1+2)))*(ups_temp**(bfit1+2) -upr_temp**(bfit1+2)))
        for k in range(size):
            upr_temp=up[k]
            temp_int =  (1/(Amean * 8)) * (ups_temp**(4) -upr_temp**(4)) #begin calc for volume, see ahrens 1969 feldspars
        #temp_int =  (1/(afit1 * 15)) * (ups_temp**(5) -upr_temp**(5)) #Different fitting function
        #print(1/dens_temp,temp_int)
            volume_isen[l,k] = (1/dens_temp) - (temp_int * 10**(-3)) #end calc for volume
        #print(V_temp[l,j])
    for l in range(0,np.size(index)):
        fo.rel_dens[index[l]]=np.nanmean(1/(V_temp[l,:]))
        fo.rel_dens_e[index[l]]=np.nanstd(1/(V_temp[l,:]))
        P_isentrope[:,index[l]]=P_ise[l,:]
        rho_isentrope[:,index[l]]=1/volume_isen[l,:]
                                                                          
    for k in range(0,size):
        P_mean[k,i]=np.mean(temp_fitp[k,:])
        P_std[k,i]=np.std(temp_fitp[k,:])    
print('End Riemann Integral')


print('Calculated variables for experiments.')
print('Release Pressure, P, P_e: ',fo.rel_P,fo.rel_P_e)
print('Release Density, rho, rho_e: ',fo.rel_dens, fo.rel_dens_e)
#print('Release Gamma, gamma, gamma_e: ',gamma_rel_f,gamma_rel_fu) 

##Write pressure, density, and uncertainties to file.
with open('Release_VandP_Data.txt', 'w+') as f:
    f.write(' Pressure_R, P_e Density_R, rho_e, Pressure_S, P_e Density_S, rho_e\r\n')
    for i in range(np.size(fo.sam_P)):
        f.write(str(fo.rel_P[i]))
        f.write(',')
        f.write(str(fo.rel_P_e[i]))
        f.write(',')
        f.write(str(fo.rel_dens[i]))
        f.write(',')
        f.write(str(fo.rel_dens_e[i]))
        f.write(',')
        f.write(str(fo.sam_P[i]))
        f.write(',')
        f.write(str(fo.sam_P_e[i]))
        f.write(',')
        f.write(str(fo.sam_dens[i]))
        f.write(',')
        f.write(str(fo.sam_dens_e[i]))        
        f.write('\r\n')
f.closed

#PLOTTING
#########################Up-P############################
plt.figure()
plt.rc('grid',color='black', linestyle=':', linewidth=0.5,alpha=0.25)
#plt.grid()

plt.plot(up,P*(10**-9),color='blue', label='Fo, Root et al. 2018')
plt.fill_between(up,(P-Pe)*(10**-9),(P+Pe)*(10**-9),color='blue',alpha=0.4)

plt.plot(up,P_qtz*(10**-9),color='green', label='Qtz, Knudson et al. 2009')
plt.fill_between(up,(P_qtz-Pe_qtz)*(10**-9),(P_qtz+Pe_qtz)*(10**-9),color='green',alpha=0.4)

plt.plot(up,P_tpx*(10**-9),color='purple', label='TPX, Root et al. 2015')
plt.fill_between(up,(P_tpx-Pe_tpx)*(10**-9),(P_tpx+Pe_tpx)*(10**-9),color='purple',alpha=0.4)

plt.plot(up,P_fs*(10**-9),color='orange', label='FS, Root et al. 2019')
plt.fill_between(up,(P_fs-Pe_fs)*(10**-9),(P_fs+Pe_fs)*(10**-9),color='orange',alpha=0.4)

plt.errorbar(fo.sam_up, fo.sam_P, yerr=fo.sam_P_e, xerr=fo.sam_up_e, fmt='o', label='Shock States')
plt.errorbar(win_up, fo.rel_P, yerr=fo.rel_P_e, xerr=win_up_e, fmt='o', label='Window States')
plt.plot(up,P_mean[:,0], color='red',label='Isentrope Fits')
plt.fill_between(up,(P_mean[:,0]-P_std[:,0]),(P_mean[:,0]+P_std[:,0]),alpha=0.3, color='red')
for i in range(1, temp_f):
    plt.plot(up,P_mean[:,i], color='red')
    plt.fill_between(up,(P_mean[:,i]-P_std[:,i]),(P_mean[:,i]+P_std[:,i]),alpha=0.3, color='red')

plt.gcf().subplots_adjust(bottom=0.15)
plt.xlabel('Up (km/s)')
plt.ylim(0,1400)
plt.ylabel('Pressure (GPa)')
plt.legend(loc='upper left', fontsize='x-small',numpoints=1,scatterpoints=1)
plt.savefig('Gamma_calc_Us-P_example.pdf', format='pdf', dpi=1000)

#########################Rho-P############################
plt.figure()
plt.rc('grid',color='black', linestyle=':', linewidth=0.5,alpha=0.25)
#plt.grid()

#for i in range(0,np.size(fo.sam_P)):
#    index=min(np.where(P_isen_f[i,:]>2))
#    plt.plot(rhoH[index],P_isen_f[i,index]*(10**-9))
#    plt.fill_between(rhoH[index],(P_isen_f[i,index]-P_isen_fu[i,index])*(10**-9),(P_isen_f[i,index]+P_isen_fu[i,index])*(10**-9),alpha=0.4)

plt.plot(rho,P*(10**-9),color='blue', label='Fo, Root et al. 2018')
plt.fill_between(rho,(P-Pe)*(10**-9),(P+Pe)*(10**-9),color='blue',alpha=0.4)
plt.fill_betweenx(P*(10**-9),(rho-rhoe),(rho+rhoe),color='blue',alpha=0.4)

plt.errorbar(fo.sam_dens, fo.sam_P, yerr=fo.sam_P_e, xerr=fo.sam_dens_e, fmt='o', label='Shock States')
plt.errorbar(fo.rel_dens, fo.rel_P, yerr=fo.rel_P_e, xerr=fo.rel_dens_e, fmt='o', label='Release States')

    
plt.gcf().subplots_adjust(bottom=0.15)                                                                         
plt.ylim(50,800)
plt.xlim(4500,7500)
plt.xlabel('Density (kg/m$^3$)')
plt.ylabel('Pressure (GPa)')
plt.legend(loc='best', fontsize='x-small',numpoints=1,scatterpoints=1)
plt.savefig('Gamma_calc_V-P.pdf', format='pdf', dpi=1000)
###########################################################
plt.figure()
plt.rc('grid',color='black', linestyle=':', linewidth=0.5,alpha=0.25)
#plt.grid()
plt.plot(rho_isentrope[:,1],P_isentrope[:,1],linestyle='-', label='Isentropes')
for i in range(1, np.size(fo.sam_dens)):
    plt.plot(rho_isentrope[:,i],P_isentrope[:,i],linestyle='-')

    
plt.plot(rho,P*(10**-9),color='blue', label='Fo, Root et al. 2018')
plt.errorbar(fo.sam_dens, fo.sam_P, yerr=fo.sam_P_e, xerr=fo.sam_dens_e, fmt='o', label='Shock States')
plt.errorbar(fo.rel_dens, fo.rel_P, yerr=fo.rel_P_e, xerr=fo.rel_dens_e, fmt='o', label='Release States')
plt.gcf().subplots_adjust(bottom=0.15)
plt.ylim(50,800)
plt.xlim(4500,7500)
plt.xlabel('Density (kg/m$^3$)')
plt.ylabel('Pressure (GPa)')
plt.legend(loc='best', fontsize='x-small',numpoints=1,scatterpoints=1)
plt.savefig('Gamma_calc_V-P_fits.pdf', format='pdf', dpi=1000)
plt.show()
