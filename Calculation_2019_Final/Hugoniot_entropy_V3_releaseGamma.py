"""
Isentrope and Entropy calculation for a liquid isentrope in Forsterite

@author: Erik Davies

This is the second major version of the code to streamline the calculation and
to implement a proper monte carlo calculation. This calculation determines the
entropy at some intersection between an isentrope and a defined hugoniot.
Parameters should be default set for Forsterite, but changes can be made to
deal with additional materials. Furthermore, only one isentrope is calculated,
in the liquid part of the phase diagram. The thermodynamic path 5is as follows:

Starting at STP conditions (T=298.15K, P=1bar)
Isobaric heating to the melting curve (T=2163K, P=1bar)
Melting (T=2163K, P=1bar)
Isochoric heating to an isentrope (T=Tref, P=Ptherm)
Isentropic compression to the Hugoniot

Entropy is then calculated on the Hugoniot using the first law of thermodynamics.
This requires a temperature-Us relation to be defined. Heat capacity is also
calculated using this data.

The isentrope foot must be defined. This code should not be altered to calculate
additional paths, in the interest of having a streamlined and efficient code.
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

#These control font size in plots.
params = {'legend.fontsize': 10,
         'axes.labelsize': 10,
         'axes.titlesize':10,
         'xtick.labelsize':10,
         'ytick.labelsize':10}
plt.rcParams.update(params)
plt.rcParams['font.family']='Times New Roman'
plt.rcParams['figure.figsize']=5,4

#Reference Values and step values (Most edits should happen here)
Tref=3000 #K, Reference value for the isentrope. Acts as the foot. It is set here
# so no uncertainty is needed
steps=10000 #Number of monte carlo steps, 10000 typically is enough for convergance
size=1200 #number of steps in isentrope, more is better

#Isentrope parameters for forsterite liquid
K0=19E9 #Pa, from dekoker and stixrude 2009, isentropic bulk modulus
K0e=3E9 # Pa, uncertainty in K0
K1=5.9 #dimensionless parameter, also from dekoker
K1e=1
K2=-0.063E-9 # second derivative of bulk modulus, check dekoker for confirmation
K2e=K2*0.03
gamma0=0.64 #Ambient gruneisen
gamma0e=.06
q0=-1.2 #Gruneisen fitting parameter
q0e=0.2
q2=-1.2#1.3 #1.3 WAS THE BEST FIT
q2e=3#.5

#Asimow gamma and q, 2018
gamma_asi1=.396
gamma_a_e=0
q_asi1=-2.03
q_a_e=1.03

#constants
n=7 # Number of atoms in formula unit
R=8.314 # kg m^2/(s*K*mol) Gas constant
ma=140.6931 #(g/mol) molecular weight of forsterite
ma=ma/1000 # putting the previous into kg

#Known entropies
S_stp=94.11 #J/(mol*K)
dS0=65.3 #Entropy of melting
dS0e=0.6

#heat capacities
#cvc=4.4*n*R/ma #Gives SI unit heat capacity J/K/Kg, constant in liquid,
cvc=1737.36 #From thomas and asimow 2013
print(cvc)
cvce=0.05 # 
cvc_d=0.68# If we need to integrate over large pressure ranges at some point, no use here
cvc_de=0.16# heat capacity derivative
#Defining heat capacity function, for use in solid only. Also will only work in
# a entropy integration, as the Cp is divided by temperature


#initial conditions
rho0l=2597 #kg/m3 initial density in liquid, thomas 2013
#rho0l=1000000/(57.8/ma)
gamma_rho=2597
print(rho0l,gamma_rho)
#print(rho0l)
rho0le=11
gamma_rhoe=gamma_rho*0.03
rho0s=3220#3327 #kg/m3 initial density in solid
#rho0s=3118#3327 #kg/m3 1200K density
rho0se=rho0s*0.003
T_amb=298.15 # K, ambient initial temperature
Tm=2174 # K, melting temperature
Tme=100
Pbar=100000 #pa, atmospheric pressure

#Reference Hugoniot Point
TH1=6901.3 # K, reference hugoniot temperature
TH1e=52 # K uncertainty
PH1=266E9 # Pa
PH1e=1.8E9 # uncertainty
rhoH1=5904.8 # kg/m3 Hugoniot density reference
rhoH1e=62.7

#Hugoniot fit information
#a1=-0.0212858071 # Us=c+b*x+a*x^2
#b1=1.71672141
#c1=3.8432255

#Hugoniot fit information
a1=4.63200 # Us=c+b*x+a*x^2
b1=1.45495
c1=0.00429
d1=-7.843708417433112e-04


#temperature relation information
a2r=-179.5692 # T=a*x+b*x**2+c*x**3, for this us in km/s
b2r=15.129
c2r=2.7999

a2re=190/3 #Quartz error bars
b2re=.52/3
c2re=.046/3

#a2r=196.465883499# T=a+b*us^c, for this us in km/s
#b2r=1.4922701338 
#c2r=3.24580688905

#covariance matrix for the hugoniot and T fit, respectively


#covparamh=[[   0.220055553677958,   -0.047674485073788,   0.002430817072725],
#          [   -0.047674485073788,   0.010501268839752, -0.000541933825686],
#         [   0.002430817072725,   -0.000541933825686,    0.000028259351523]]
covparamh=[[   1.831017240698122,  -0.652611265951972 ,  0.073185789070778,  -0.002609417024651],
 [ -0.652611265951972,   0.236606823014799,  -0.026846129171457,   0.000967124009405],
 [  0.073185789070778,  -0.026846129171457,   0.003083556207129,  -0.000112248130990],
 [ -0.002609417024651 ,  0.000967124009405,  -0.000112248130990,   0.000004124931939]]

covparam=[[  1.61439802e+05,  -1.97446645e+04,   5.83435139e+02],
         [ -1.97446645e+04,   2.46208982e+03,  -7.42868010e+01],
         [  5.83435139e+02,  -7.42868010e+01,   2.29747618e+00]]


lmath=sp.linalg.cholesky(covparamh,lower=False)
lmat=sp.linalg.cholesky(covparam,lower=False)

#Useful hugoniot fits
#def hugoniot(a,b,c,x):
#    return a * x ** 2 + b * x + c
def hugoniot(x,a,b,c,d):
    return a + b * (x**1) + c * (x**2) + d * (x**3)
    

def temperature(a2,b2,c2,x2):
    #return a2+b2*x2+c2*x2**2
    return a2 * x2 + b2 * x2**2 + c2 * x2**3

#def gamma_fit(x,a,b,c): #fitting function for gamma values
    #return a * (x **(-b))
    #return a*(x**b) + c*(x**d) 
 #   return a*x**(0)+ b*x**(1)+ c*x**(2)
    #return a + b* x + c* x**d
def gamma_fit(x,a,b,c,d,e): #fitting function for gamma values
    return 2/3 + (a - 2/3)*(2597/x)**b + c*np.exp((-(x-d)**2)/(e**2))

#Gamma Paramters
A_mean=0.464372154854
B_mean=0.645891883744
C_mean=1.01114044541
D_mean=5128.97644885
E_mean=1324.10397093
           
Covar=[[  6.01748973e-02,  -5.25514538e-02 ,  8.52968507e-02 , -4.74607187e+01 , -1.36117685e+02],
 [ -5.25514538e-02,   2.12491057e-01,  -9.84335454e-02,   2.23774711e+01 ,  8.49695247e+01],
 [  8.52968507e-02,  -9.84335454e-02,   2.44553745e-01,  -1.68689128e+02,  -1.65557887e+02],
 [ -4.74607187e+01,   2.23774711e+01,  -1.68689128e+02,   3.16948640e+05,   1.05691286e+05],
 [ -1.36117685e+02,   8.49695247e+01,  -1.65557887e+02,   1.05691286e+05,   6.00820780e+05]]
lmat_G=sp.linalg.cholesky(Covar,lower=False)

# Initialize all needed arrays, we need monte carlo arrays, global arrays, and
# temp arrays for calculations in loops. do temp arrays right before loops inside

#gamma_dft=np.loadtxt('ForsteriteHugoniot_FINAL.txt',skiprows=4,usecols=[4])
#dens_dft=np.loadtxt('ForsteriteHugoniot_FINAL.txt',skiprows=3,usecols=[0])*1000

#gamma_dft=np.insert(gamma_dft,0,.396)
#dens_dft[0]=gamma_rho
#print(gamma_dft, dens_dft)
#gamma_dft_int=interpolate.interp1d(dens_dft,gamma_dft,bounds_error=False)

#Final arrays
T=sp.zeros(size)
Te=sp.zeros(size)
P=sp.zeros(size)
Pe=sp.zeros(size)
SH=sp.zeros(size)
SHe=sp.zeros(size)
CvH=sp.zeros(size)
CvHe=sp.zeros(size)
Cvnum=sp.zeros(size)
Cvnume=sp.zeros(size)
TH=sp.zeros(size)
THe=sp.zeros(size)
PH=sp.zeros(size)
PHe=sp.zeros(size)
rho=sp.zeros(size)
rhoe=sp.zeros(size)
gamma_m=sp.zeros(size)
gamma_me=sp.zeros(size)
gamma_m_a=sp.zeros(size)
gamma_me_a=sp.zeros(size)

#Monte carlo arrays
Tmc=sp.zeros((size,steps))
Pmc=sp.zeros((size,steps))
SHmc=sp.zeros((size,steps))
CvHmc=sp.zeros((size,steps))
Cv_int=sp.zeros((size,steps))

THmc=sp.zeros((size,steps))
PHmc=sp.zeros((size,steps))
rhoH=sp.zeros((size,steps))
gamma=sp.zeros((size,steps))
gamma_asi=sp.zeros((size,steps))

refSa=sp.zeros(steps)
refSb=sp.zeros(steps)
refSc=sp.zeros(steps)
refSd=sp.zeros(steps)
refSe=sp.zeros(steps)
Ptherm=sp.zeros((size,steps))
crossing=sp.zeros(steps)
crossingP=sp.zeros(steps)
Trec=sp.zeros(steps) #Indice tracking
THrec=sp.zeros(steps) # '' ''
Tcross=sp.zeros(steps) # These four track the actual crossing value
Pcross=sp.zeros(steps) #
RhoCross=sp.zeros(steps) #
Scross=sp.zeros(steps) #
K0a=sp.zeros(steps) #Tracking isentrope parameters
K1a=sp.zeros(steps)
K2a=sp.zeros(steps)
gamma_a=sp.zeros(steps)
gamma_index=sp.zeros(steps)
qa=sp.zeros(steps)
cvc_mc=sp.zeros(size)
 #################################################################     
# begin monte carlo, only one, encompasses all calculations.
j=0
while j < steps:
    #Calculate perturbations for everything here.
    K0s=K0+K0e*sp.randn()
    K0a[j]=K0s
    K1s=K1+K1e*sp.randn()
    K1a[j]=K1s
    K2s=K2+K2e*sp.randn()
    K2a[j]=K2s
    gamma_i=gamma0+gamma0e*sp.randn()
    gamma_a[j]=gamma_i
    rho_gamma=gamma_rho+gamma_rhoe*sp.randn()
    q=q0+q0e*sp.randn()
    qa[j]=q
    q2_mc=q2+q2e*sp.randn()
    #rho_init_l=rho0l+rho0le*sp.randn()
    rho_init_l=rho0l+11*sp.randn()#For asimow version
    rho_init_s=rho0s+rho0se*sp.randn()
    dSm=dS0+dS0e*sp.randn()
    cvc1=cvc+cvce*sp.randn()
    cvc_d1=cvc_d+cvc_de*sp.randn()
    bmath=np.matmul(sp.randn(1,4), lmath) #For covariance calculation on the hugoniot
    ah=a1+bmath[0,0]
    bh=b1+bmath[0,1]
    ch=c1+bmath[0,2]
    dh=d1+bmath[0,3]
    bmat=np.matmul(sp.randn(1,3), lmat) #For covariance calculation on the temperature
    at=a2r+bmat[0,0]
    bt=b2r+bmat[0,1]
    ct=c2r+bmat[0,2]
    Tm_mc=Tm+Tme*sp.randn()
    cpa=-402.753-402.753*0.02*sp.randn()
    cpb=74.29+74.29*0.02*sp.randn()
    cpc=87.588+87.588*0.02*sp.randn()
    cpd=-25.913-25.913*0.02*sp.randn()
    cpe=25.374+25.374*0.02*sp.randn()
    #Gamma Fit stuff
    temp_mat=sp.randn(1,5)
    bmat=np.matmul(temp_mat,lmat_G)
    AG1=A_mean+bmat[0,0]
    BG1=B_mean+bmat[0,1]
    CG1=C_mean+bmat[0,2]
    DG1=D_mean+bmat[0,3]
    EG1=E_mean+bmat[0,4] 
    S_stp1=S_stp+.1*sp.randn()

    #gruneisen plot
    gamma_asi2=gamma_asi1+gamma_a_e*sp.randn()
    q_asi2=q_asi1+q_a_e*sp.randn()
    #End perturbations
    begin_up=0
    #Define hugoniot and reference volume density arrays
    up=np.linspace(begin_up,16,size) #defining particle velocity]
    us=hugoniot(up,ah,bh,ch,dh) #getting shock velocity
    rhoH[:,j]=rho_init_s*us/(us-up) #getting density array
    up_temp=min(np.where(up<4))
    rhoH[up_temp,j]=np.linspace(rho0l-100,rhoH[up_temp[-1]+1,j],len(up_temp))

    
    PHmc[:,j]=rho_init_s*us*up # Getting pressure from rankine-hugoniot, in MPa
    PHmc[:,j]=PHmc[:,j]*(10**6) # putting pressure into Pa
    #Getting hugoniot temperature
    THmc[:,j]=temperature(at,bt,ct,us)


    


    #Make sure we start pressure and temperature isentrope calculations at
    # the reference initial liquid density
    ll=min(sp.where(rhoH[:,j]>rho_init_l))
    #ll=min(sp.where(rhoH[:,j]>rho_init_s))
    #print(rhoH[:,j])
    #print(ll[0])
    Tmc[ll[0],j]=Tref

    #set gruneisen for the array
    gamma_asi[:,j]=gamma_asi2*((rho_init_l/rhoH[:,j])**q_asi2)
    gamma[:,j]=gamma_fit(rhoH[:,j],AG1,BG1,CG1,DG1,EG1)
    #gamma[:,j]=gamma_i+q*((rho_gamma/rhoH[:,j])-1)
    #print(gamma)
#for setting custom gammas that decrease later on
 #   gamma_set=.52#1.20
 #   gamma_index=min(min(sp.where((rho_gamma/rhoH[:,j])<gamma_set)))
#    for i in range(0,size):
#        if (rho_gamma/rhoH[i,j]) < gamma_set:
#            gamma[i,j]=gamma[gamma_index,j] + (q2_mc)*((rhoH[gamma_index,j]/rhoH[i,j])-1)
            #gamma_e[i,j]=gamma_e[gamma_index,j]-(q1-.075)*((rhoH[gamma_index,j]/rhoH[i,j])-1)
            #gamma[i,j]=2
            #gamma_e[i,j]=.666667#2
    
    #Isentrope temperature
    for i in range(ll[0]+1,size):
        #temperautre only depends on gamma and v
        #d ln(T) = gamma * d ln(rho)
        Tmc[i,j]=np.exp(gamma[i,j] * (np.log(rhoH[i,j]) - np.log(rhoH[i-1,j])) + np.log(Tmc[i-1,j]))
    for i in range(ll[0]-1,0,-1):
        #temperautre only depends on gamma and v
        #d ln(T) = gamma * d ln(rho)
        Tmc[i,j]=np.exp(gamma[i,j] * (np.log(rhoH[i,j]) - np.log(rhoH[i+1,j])) + np.log(Tmc[i+1,j]))

    #Calculate thermal pressure from isobaric heating, because we isobarically heated the forsterite, it is indexed to the same initial volume
        # however, there is now a thermal pressure which must be accounted for. this is why Ptherm is in ref to the melting temperature
    # in this case, Tm_mc is a reference melting temperature
    cvc_mc[:]=cvc1+cvc_d1*((rho_gamma/rhoH[:,j])-1)
    Ptherm[:,j]=gamma[:,j]*rhoH[:,j]*cvc_mc*(Tmc[:,j]-Tm_mc)
       
    # Find crossing


    #print(Tcross[j],RhoCross[j],Pcross[j])

    #print('Temperature crossing indice:', max(tmp))
    #print('Temperature of Isentrope Surrounding Tie:',Tmc[max(tmp),j],Tmc[max(tmp)+1,j])
    #print('Temperature of Hugoniot Surrounding Tie:',THmc[max(tmp),j],THmc[max(tmp)+1,j])
    #print(crossing[j])
    #Isentrope pressure, 3rd or 4th Birch-Murnaghan
    f=((rhoH[:,j]/rho_init_l)**(2/3)-1)/2 # eulerian strain
    Piso=3* K0s * f * ((2*f+1)**(5/2))*(1+(3/2)*(K1s-4)*f)#+(3/2)*(K0s*K2s+K1s*(K1s-7)+(143/9))*f**2)#
    Pmc[:,j]=Piso#+Ptherm[:,j]

    tmp=min(sp.where(Tmc[:,j]-THmc[:,j]>0))
    #print(tmp)
    #print("Got to Crossing")
    try:
        crossing[j]= (int(tmp[-1]))
    except IndexError:
        continue
    cross=int(crossing[j])
    Trec[j]=Tmc[cross,j]
    THrec[j]=THmc[cross,j]
    TempT=(Tmc[cross,j]+THmc[cross,j])/2
    Tcross[j]=THmc[cross,j]#np.interp(TempT,THmc[:,j],THmc[:,j])
    RhoCross[j]=rhoH[cross,j]#np.interp(TempT,THmc[:,j],rhoH[:,j])
    Pcross[j]=PHmc[cross,j]#np.interp(TempT,THmc[:,j],PHmc[:,j])

    #Set the crossing point in the hugoniot
    PHmc[cross,j]=Pcross[j]
    THmc[cross,j]=Tcross[j]
    rhoH[cross,j]=RhoCross[j]

    #print(Pmc[:,j],PHmc[:,j])
    #print('Pressure crossing indice:', max(tmpP))
    #print('Pressure of Isentrope Surrounding Tie:',Pmc[max(tmp),j]*10**(-9),Pmc[max(tmp)+1,j]*10**(-9))
    #print('Pressure of Hugoniot Surrounding Tie:',PHmc[max(tmp),j]*10**(-9),PHmc[max(tmp)+1,j]*10**(-9))
    #print(Piso,Ptherm[j])
    
    #Calculate Entropies

    # Reference Point A: STP Conditions
    refSa[j]=S_stp1/ma

    # Reference Point B: Isobaric heating
    #temp=integrate.quad(solid_Cp, T_amb, Tm_mc)
    #def solid_Cp(x,a,b,c,d,e):
    #return (a+b*np.log(x)+c*(10**3)/(x)+d*(10**6)/(x**(2))+e*(10**8)/(x**(3)))/x
    temp1=-(10**8)*cpe/(3*Tm_mc**3)-500000*cpd/(Tm_mc**2)-1000*cpc/Tm_mc+cpa*np.log(Tm_mc)+.5*cpb*np.log(Tm_mc)**2
    temp2=-(10**8)*cpe/(3*T_amb**3)-500000*cpd/(T_amb**2)-1000*cpc/T_amb+cpa*np.log(T_amb)+.5*cpb*np.log(T_amb)**2          
    refSb[j]=(temp1-temp2)/ma
    #print(refSb[j])

    # Reference Point C: Melting
    refSc[j]=dSm/ma
    #print(refSc[j])
    # Reference Point D: Isochric heating
    refSd[j]=cvc1*np.log(Tref/Tm_mc)
    #print(refSd[j])
    #print(cvc1)
    # Reference Point E: Total entropy
    refSe[j]=refSa[j]+refSb[j]+refSc[j]+refSd[j]

    #Calculate energy on the Hugoniot, change in entropy, not total
    eh=-0.5*PHmc[:,j]*((1/rhoH[:,j])-1/rho_init_s)

    #print(Tcross[j],RhoCross[j],Pcross[j])
    #Calculate Entropy on the Hugoniot
    Scross[j]=refSe[j]
    SHmc[cross,j]=refSe[j]
    
    for i in range(int(crossing[j])+1,size):
        SHmc[i,j]=SHmc[i-1,j]+(eh[i]-eh[i-1])/THmc[i,j]+PHmc[i,j]*((1/rhoH[i,j])-(1/rhoH[i-1,j]))/THmc[i,j]
    for i in range(int(crossing[j])-1,0,-1):
        SHmc[i,j]=SHmc[i+1,j]+(eh[i]-eh[i+1])/THmc[i,j]+PHmc[i,j]*((1/rhoH[i,j])-(1/rhoH[i+1,j]))/THmc[i,j]

    # Calculate semi-empirical isochoric specific heat capacity along hugoniot ## Hicks 2006#
        #Requires the gruneisen praramter for the required pressure.

    #CVL=cvc1*((rho_init_l/rhoH[:,j])**cvc_d)
    for i in range(0,size):
        Cv_int[i,j] = ((eh[i]-eh[i-1])/((1/rhoH[i,j])-(1/rhoH[i-1,j]))+PHmc[i,j])/((THmc[i,j]-THmc[i-1,j])/((1/rhoH[i,j])-(1/rhoH[i-1,j]))+gamma[i,j]*THmc[i,j]/(1/rhoH[i,j])) #Orig
        #CvHmc[i,j] = ((eh[i]-eh[i-1])/((1/rhoH[i,j])-(1/rhoH[i-1,j]))+Pmc[i,j])/((THmc[i,j]-THmc[i-1,j])/((1/rhoH[i,j])-(1/rhoH[i-1,j]))+gamma[i,j]*Tmc[i,j]/(1/rhoH[i,j])) #Using isentrop temp and pres, I can't think of  a basis for this
        #CvHmc[i,j] = ((eh[i]-eh[i-1])/((1/rhoH[i,j])-(1/rhoH[i-1,j]))+PHmc[i,j])/((THmc[i,j]-THmc[i-1,j])/((1/rhoH[i,j])-(1/rhoH[i-1,j]))+gamma_dft_int(rhoH[i,j])*THmc[i,j]/(1/rhoH[i,j])) #dft gamma
        #Cv_int[i,j]=(eh[i]-eh[i-1])/(THmc[i,j]-THmc[i-1,j])
    j=j+1
#Get all means and std
for i in range(0,size):
    T[i]=np.median(Tmc[i,:])
    Te[i]=np.std(Tmc[i,:])
    P[i]=np.median(Pmc[i,:])
    Pe[i]=np.std(Pmc[i,:])    
    SH[i]=np.median(SHmc[i,:])#+1528)
    SHe[i]=np.std(SHmc[i,:])
    CvH[i]=np.median(CvHmc[i,:])
    CvHe[i]=np.std(CvHmc[i,:])
    Cvnum[i]=np.median(Cv_int[i,:])
    Cvnume[i]=np.std(Cv_int[i,:])
    TH[i]=np.median(THmc[i,:])#+1200)
    THe[i]=np.std(THmc[i,:])
    PH[i]=np.median(PHmc[i,:])
    PHe[i]=np.std(PHmc[i,:])
    rho[i]=np.median(rhoH[i,:])
    rhoe[i]=np.std(rhoH[i,:])
    gamma_m[i]=np.median(gamma[i,:])
    gamma_me[i]=np.std(gamma[i,:])
    gamma_m_a[i]=np.median(gamma_asi[i,:])
    gamma_me_a[i]=np.std(gamma_asi[i,:])


#gamma_dft_print=gamma_dft_int(rho)

Sa=np.median(refSa)
Sae=np.std(refSa)
Sb=np.median(refSb)
Sbe=np.std(refSb)
Sc2=np.median(refSc)
Sc2e=np.std(refSc)
Sd=np.median(refSd)
Sde=np.std(refSd)
Se=np.median(refSe)
See=(Sae**2+Sbe**2+Sc2e**2+Sde**2)**(1/2)
Pth=np.median(Ptherm)
Pthe=np.std(Ptherm)
cross=np.int(np.median(crossing))
Tc=np.median(Tcross)
Tce=np.std(Tcross)
Pc=np.median(Pcross)
Pce=np.std(Pcross)
Rhoc=np.median(RhoCross)
Rhoce=np.std(RhoCross)
Sc=np.median(Scross)
Sce=np.std(Scross)
gamma_ind=int(np.median(gamma_index))
print('volume ration of gamma part',rho[gamma_ind]/gamma_rho)


cross_temp=np.sort(Tcross)
Tc_up=-cross_temp[int(0.841*steps)]+Tc
Tc_low=cross_temp[int(0.159*steps)]-Tc
print("T alt uncert",Tc_up,Tc_low)
cross_temp=np.sort(Pcross)
Pc_up=-cross_temp[int(0.841*steps)]+Pc
Pc_low=cross_temp[int(0.159*steps)]-Pc
print("P alt uncert",Pc_up,Pc_low)
cross_temp=np.sort(RhoCross)
Rhoc_up=-cross_temp[int(0.841*steps)]+Rhoc
Rhoc_low=cross_temp[int(0.159*steps)]-Rhoc
print("Rho alt uncert",Rhoc_up,Rhoc_low)
cross_temp=np.sort(Scross)
Sc_up=-cross_temp[int(0.841*steps)]+Sc
Sc_low=cross_temp[int(0.159*steps)]-Sc
print("S alt uncert",Sc_up,Sc_low)

##plt.figure()
##plt.hist(K0a,100)
##plt.figure()
##plt.hist(Tcross,100)
##plt.figure()
##plt.hist(K1a,100)
##plt.figure()
##plt.scatter(K0a,K1a)
##plt.figure()
##plt.scatter(K0a,Tcross)
##plt.figure()
##plt.scatter(K1a,Tcross)
##plt.show()
#print(cross)
#print(crossing)

##Plotting###
#maneos
xm=np.loadtxt('forsterite-maneos_py.txt',skiprows=15,usecols=[2]) #pressure
ym=np.loadtxt('forsterite-maneos_py.txt',skiprows=15,usecols=[1]) #temperature
sm=np.loadtxt('forsterite-maneos_py.txt',skiprows=15,usecols=[5]) #entropy
dm=np.loadtxt('forsterite-maneos_py.txt',skiprows=15,usecols=[0]) #density
vm=1/dm
xm=xm*(10**-3)
############################################L-V Dome############################################
lvdT=np.loadtxt('dunite-aneos-vapor-curve2.txt',skiprows=5,usecols=[0]) #Temperatre
lvdTp=np.loadtxt('dunite-aneos-vapor-curve2.txt',skiprows=5,usecols=[1]) #Temperature power
lvdT=lvdT*10**lvdTp
lvdDensL=np.loadtxt('dunite-aneos-vapor-curve2.txt',skiprows=5,usecols=[2]) # density liquid
lvdDensLp=np.loadtxt('dunite-aneos-vapor-curve2.txt',skiprows=5,usecols=[3])
lvdDensL=lvdDensL*10**lvdDensLp
lvdDensV=np.loadtxt('dunite-aneos-vapor-curve2.txt',skiprows=5,usecols=[4]) # density vapor
lvdDensVp=np.loadtxt('dunite-aneos-vapor-curve2.txt',skiprows=5,usecols=[5])
lvdDensV=lvdDensV*10**lvdDensVp
lvdPL=np.loadtxt('dunite-aneos-vapor-curve2.txt',skiprows=5,usecols=[6]) #Pressure liquid
lvdPLp=np.loadtxt('dunite-aneos-vapor-curve2.txt',skiprows=5,usecols=[7]) #Pressure power
lvdPL=lvdPL*10**lvdPLp
lvdPL=lvdPL*10**-3
lvdPV=np.loadtxt('dunite-aneos-vapor-curve2.txt',skiprows=5,usecols=[8]) #Pressure vapor
lvdPVp=np.loadtxt('dunite-aneos-vapor-curve2.txt',skiprows=5,usecols=[9]) #Pressure power
lvdPV=lvdPV*10**lvdPVp
lvdPV=lvdPV*10**-3
lvdSL=np.loadtxt('dunite-aneos-vapor-curve2.txt',skiprows=5,usecols=[14]) #entropy liquid
lvdSLp=np.loadtxt('dunite-aneos-vapor-curve2.txt',skiprows=5,usecols=[15]) #power
lvdSL=lvdSL*10**lvdSLp
lvdSV=np.loadtxt('dunite-aneos-vapor-curve2.txt',skiprows=5,usecols=[16]) #entropy liquid
lvdSVp=np.loadtxt('dunite-aneos-vapor-curve2.txt',skiprows=5,usecols=[17]) #power
lvdSV=lvdSV*10**lvdSVp

#############Adding DFT in as well##########
#dftT=np.loadtxt('ForsteriteHugoniot_FINAL.txt',skiprows=4,usecols=[3])
#dftP=np.loadtxt('ForsteriteHugoniot_FINAL.txt',skiprows=4,usecols=[2])
#dftP=dftP/1000
##dftCV=np.loadtxt('ForsteriteHugoniot_FINAL.txt',skiprows=4,usecols=[5])*1000

#################################Sekine########################################
#xmg4l=np.loadtxt('Sekine_Mg2SiO4_Liquidus.txt',delimiter=',',skiprows=1,usecols=[0])
#ymg4l=np.loadtxt('Sekine_Mg2SiO4_Liquidus.txt',delimiter=',',skiprows=1,usecols=[1])

#ymg4l=ymg4l*10**3
#xmg4l=xmg4l*10**-3

###########################Z_data#######################################
##ZP=np.loadtxt('FoPT.csv',delimiter=',',skiprows=3,usecols=[0])
##ZPe=np.loadtxt('FoPT.csv',delimiter=',',skiprows=3,usecols=[1])
##ZP=ZP/1000
##ZPe=ZPe/1000
##
##ZTr=np.loadtxt('FoPT.csv',delimiter=',',skiprows=3,usecols=[2])
##ZTer=np.loadtxt('FoPT.csv',delimiter=',',skiprows=3,usecols=[3])
##ZT=np.loadtxt('FoPT.csv',delimiter=',',skiprows=3,usecols=[6])
##ZTe=np.loadtxt('FoPT.csv',delimiter=',',skiprows=3,usecols=[7])
##Zrho=np.loadtxt('FoPT.csv',delimiter=',',skiprows=3,usecols=[10])
##Zrhoe=np.loadtxt('FoPT.csv',delimiter=',',skiprows=3,usecols=[11])
##
###Luo et al. 2004, no error bars
##xluo=np.loadtxt('Luo_etal_2004.txt',delimiter=',',skiprows=1,usecols=[0])
##yluo=np.loadtxt('Luo_etal_2004.txt',delimiter=',',skiprows=1,usecols=[1])
##
##yluo=yluo*10**3
##xluo=xluo*10**-3
##
#############################Mosenfelder#######################################
##mosP=np.loadtxt('mosenfelder.txt',delimiter=',',skiprows=2,usecols=[0])
##mosT=np.loadtxt('mosenfelder.txt',delimiter=',',skiprows=2,usecols=[1])
##mosP=mosP/1000

#####Most Recent Data###
##P_z=[.4507,.5173]
##Pe_z=[.01,0.02]
##T_z=[13880,16770]
##Te_z=[790,1270]
##
###Release Z, T
##S_z=[5212,5212]
##Se_z=[250,250]
##Tr_z=[5.040,13.880]
##Tre_z=[.207,.790]
###Release Z, T
##S_z2=[5483,5483]
##Se_z2=[250,250]
##Tr_z2=[5.177,16.770]
##Tre_z2=[.207,.790]

#Crossing for lowest liquid density
ld=min(min(sp.where(rho>rho_init_l)))

######PLOTTING####################################################################
S_tp=np.linspace(3306/1000,10000/1000,10)
S_at=np.linspace(3829/1000,8024/1000,10)
P_tp=5.2*10**-9 + S_tp*0
P_at=100000*10**-9 + S_at*0
isen1=np.linspace(265.8,P_tp[0],10)
isen1s=4.150+0*isen1
isen2=np.linspace(325.5,P_tp[0],10)
isen2s=4.560+0*isen1
isen3=np.linspace(368.5,P_tp[0],10)
isen3s=4.808+0*isen1
isen4=np.linspace(450.7,P_tp[0],10)
isen4s=5.212+0*isen1


plt.figure()
plt.rc('grid',color='black', linestyle=':', linewidth=0.5,alpha=0.25)
tmp3=np.where((TH-3000)>0)
####TEMPERATURE PRESSURE#########
plt.plot(xm*(10**3),ym*(10**-3), label="M-ANEOS Hugoniot", color='purple', linewidth=2,alpha=0.5,linestyle='--')
plt.plot(PH[min(tmp3[0]):]*(10**-9),TH[min(tmp3[0]):]*(10**-3),color='blue')
#plt.fill_between(PH[min(tmp3[0]):]*(10**(-12)),TH[min(tmp3[0]):]-THe[min(tmp3[0]):],TH[min(tmp3[0]):]+THe[min(tmp3[0]):],label="Z Hugoniot",color='blue', alpha=0.4)
plt.fill_between(PH[min(tmp3[0]):]*(10**(-9)),(TH[min(tmp3[0]):]-THe[min(tmp3[0]):])*(10**-3),(TH[min(tmp3[0]):]+THe[min(tmp3[0]):])*(10**-3),label="Root et al. 2018",color='blue', alpha=0.4)
plt.plot(P*(10**(-9)),T*(10**-3),color='black')
plt.fill_between(P[ld:]*(10**(-9)),(T[ld:]-Te[ld:])*(10**-3),(T[ld:]+Te[ld:])*(10**-3),label="Liquid Isentrope",color='black',alpha=0.4)
#plt.errorbar(ZP*(10**3),ZTr*(10**-3),yerr=ZTer*(10**-3), xerr=ZPe*(10**3), fmt='o',linewidth=2, color='black',label='Z Hugoniot Points, Root et al. 2018')
#plt.errorbar(P_z,T_z,yerr=Te_z, xerr=Pe_z, fmt='o',linewidth=2, color='brown',label='Z Hugoniot Points, Most Recent')
#print(Ts1[:5])
#plt.scatter(Pth*(10**(-9)),Tref*(10**-3), label='Isentrope Foot', color='black')
plt.errorbar(Pc*(10**-9),Tc*(10**-3),yerr=Tce*(10**-3), xerr=Pce*(10**-9), fmt='o',linewidth=2, color='red',label='Isentrope-Hugoniot Intersection')
#plt.plot(xmg4l,ymg4l,linestyle='--',label="Mg2SiO4 Liquidus", linewidth=2,alpha=0.8)
plt.plot(mosP*(10**3),mosT*(10**-3),linestyle='--',label="Mg$_2$SiO$_4$ Liquidus, Mosenfelder et al. 2009", linewidth=2)
#plt.plot(dftP*(10**3),dftT*(10**-3),'o', label="DFT-MD Root et al. 2018", color='red')
#####################LUO#####################################
#plt.plot(xluo*1000,yluo/1000,'o', label="Luo et al. 2004", color='brown')
plt.grid()
#plt.figtext(0.35,0.11,'Shots Z2792, Z2868, Z2879, Z3033', fontsize='xx-small')

#plt.title('Forsterite T - P')
plt.xlabel('Pressure (GPa)')
plt.ylabel('Temperature (1000 K)')
plt.xlim(0,800)
plt.ylim(0,30)
plt.legend(loc='upper left', fontsize='x-small',numpoints=1,scatterpoints=1)
plt.savefig('Hugoniot_T_P.pdf', format='pdf', dpi=1000)

#####TEMP ENTROPY###################

T_tp=2121*10**-3+S_tp*0
T_at=3355*10**-3+S_at*0
plt.figure()
plt.rc('grid',color='black', linestyle=':', linewidth=0.5,alpha=0.25)
plt.plot(sm/1000,ym*10**-3, label="M-ANEOS Hugoniot", color='purple', linewidth=2,alpha=0.5,linestyle='--')
plt.plot(lvdSL/1000,lvdT*10**-3, color='purple', label="M-ANEOS Liquid-Vapor Curve", linewidth=2,linestyle=':')
plt.plot(lvdSV/1000,lvdT*10**-3,color='purple', linewidth=2,linestyle=':')
#plt.plot(Sa,298.15,'o',label='Path A, STP')
#plt.plot(Sa+Sb,Tm,'o',label='Path B, Isobaric Heating to melting')
#plt.plot(Sa+Sb+Sc2,Tm,'o',label='Path C, Entropy of Fusion')
#plt.plot(Sa+Sb+Sc2+Sd,Tref,'o',label='Path D, Isochoric Heating to Isentrope')
Tisen=np.linspace(Tref,T[cross],size)
Ss=np.linspace(Se/1000,Se/1000,size)
#plt.plot(Ss,Tisen,label='Path E, Isentrope',linestyle=':', linewidth=2)
#plt.errorbar(S_z[0]/1000,Tr_z[0],yerr=Tre_z[0], xerr=Se_z[0]/1000, fmt='o', color='brown',label='Release')
#plt.errorbar(S_z2[0]/1000,Tr_z2[0],yerr=Tre_z2[0], xerr=Se_z2[0]/1000, fmt='o', color='brown')
#plt.errorbar(S_z[1]/1000,Tr_z[1],yerr=Tre_z[1], xerr=Se_z[1]/1000, fmt='o', color='red',label='Shocked state')
#plt.errorbar(S_z2[1]/1000,Tr_z2[1],yerr=Tre_z2[1], xerr=Se_z2[1]/1000, fmt='o', color='red')



plt.plot(SH[min(tmp3[0]):]/1000,TH[min(tmp3[0]):]*10**-3, color='blue')
#plt.fill_between(SH[min(tmp3[0]):],TH[min(tmp3[0]):]-THe[min(tmp3[0]):],TH[min(tmp3[0]):]+THe[min(tmp3[0]):],label="Z Hugoniot",color='blue', alpha=0.4)
plt.fill_betweenx(TH[min(tmp3[0]):]*10**-3,SH[min(tmp3[0]):]/1000-SHe[min(tmp3[0]):]/1000, SH[min(tmp3[0]):]/1000+SHe[min(tmp3[0]):]/1000,label="Hugoniot, this Work",color='blue', alpha=0.4)
plt.errorbar(Sc/1000,Tc*10**-3,yerr=Tce*10**-3, xerr=Sce/1000, fmt='o',linewidth=2, color='red',label='Isentrope-Hugoniot Intersection')
plt.plot(S_tp,T_tp,color='black', linewidth=2, label='Triple Point Pressure', linestyle=':')
plt.plot(S_at,T_at,color='Green', linewidth=2, label='1 Bar', linestyle=':')
plt.grid()
#plt.figtext(0.35,0.11,'Shots Z2792, Z2868, Z2879, Z3033', fontsize='xx-small')
plt.xlabel('Specific Entropy (kJ/K/kg)')
plt.ylabel('Temperature (1000 K)')
plt.xlim(1,8)
plt.ylim(0,30)
plt.legend(loc='upper left', fontsize='x-small',numpoints=1,scatterpoints=1)
plt.savefig('Hugoniot_T_S.pdf', format='pdf', dpi=1000)

########Entropy Pressure#########

plt.figure()
plt.rc('grid',color='black', linestyle=':', linewidth=0.5,alpha=0.25)
axMain = plt.subplot(111)
axMain.plot(sm/1000,xm*1000, label="MANEOS Hugoniot", color='purple', linewidth=2,alpha=0.5,linestyle='--')
axMain.plot(lvdSL/1000,lvdPL*1000, label="MANEOS Liquid-Vapor Curve", color='purple', linewidth=2,alpha=0.5,linestyle='--')
axMain.plot(SH[min(tmp3[0]):]/1000,PH[min(tmp3[0]):]*10**-9, color='blue')
axMain.fill_between(SH[min(tmp3[0]):]/1000,PH[min(tmp3[0]):]*10**-9-PHe[min(tmp3[0]):]*10**-9,PH[min(tmp3[0]):]*10**-9+PHe[min(tmp3[0]):]*10**-9,label="This Work",color='blue', alpha=0.4)
#axMain.plot(isen1s,isen1, color='red')
#axMain.plot(isen2s,isen2, color='orange')
#axMain.plot(isen3s,isen3, color='cyan')
#axMain.plot(isen4s,isen4, color='purple')
axMain.plot(lvdSV/1000,lvdPV*1000,color='purple', linewidth=2,alpha=0.5, linestyle='--')
axMain.set_yscale('log')
axMain.set_ylim(10**-8,10)
#axMain.spines["top"].set_visible(False)
divider = make_axes_locatable(axMain)
axLin = divider.append_axes("top", size=2.25, sharex=axMain)
axLin.xaxis.set_visible(True)
axMain.set_yticks(axMain.get_yticks()[::3])

plt.plot(sm/1000,xm*1000, label="MANEOS Hugoniot and Liquid-Vapor Curve", color='purple', linewidth=2,alpha=0.5,linestyle='--')
axMain.plot(S_tp,P_tp,color='Black', linewidth=2, label='Triple Point', linestyle=':')
axMain.plot(S_at,P_at,color='Green', linewidth=2, label='1 Bar', linestyle=':')
plt.plot(lvdSL/1000,lvdPL*1000, color='purple', linewidth=2,linestyle=':')
plt.plot(SH[min(tmp3[0]):]/1000,PH[min(tmp3[0]):]*10**-9, color='blue')
#plt.fill_between(SH/1000,PH*10**-9-PHe*10**-9,PH*10**-9+PHe*10**-9,label="Hugoniot",color='blue', alpha=0.4)
plt.fill_betweenx(PH[min(tmp3[0]):]*10**-9,SH[min(tmp3[0]):]/1000-SHe[min(tmp3[0]):]/1000, SH[min(tmp3[0]):]/1000+SHe[min(tmp3[0]):]/1000,label="This Work",color='blue', alpha=0.4)
#plt.errorbar(SH,PH*10**-12,yerr=PHe*10**-12, xerr=SHe, alpha=0.2, linestyle='-', capthick=0,linewidth=2)

plt.plot(lvdSV/1000,lvdPV,color='purple', linewidth=2,alpha=0.5)
#plt.plot(isen1s,isen1, color='red')
#plt.plot(isen2s,isen2, color='orange')
#plt.plot(isen3s,isen3, color='cyan')
#plt.plot(isen4s,isen4, color='purple')
axLin.set_xscale('linear')
axLin.set_ylim((0.01, .75))
tmp2= np.where((TH1-TH)<0)
plt.errorbar(Sc/1000, Pc*(10**-9),xerr=Sce/1000, yerr=Pce*(10**-9), fmt='o',linewidth=2, color='red',label='Intersection')
#plt.figtext(0.35,0.11,'Shots Z2792, Z2868, Z2879, Z3033', fontsize='xx-small')
axLin.spines["bottom"].set_visible(False)
#axLin.spines["top"].set_visible(False)
axMain.spines["top"].set_visible(False)
axMain.get_xaxis().tick_bottom()
axLin.get_xaxis().tick_top()
#axLin.get_xaxis().tick_bottom()
plt.setp(axLin.get_xticklabels(), visible=False)

#plt.title('Forsterite P - S')
#plt.grid()
axMain.grid()
axLin.grid()
axMain.set_xlabel('Specific Entropy (kJ/K/kg)')
axLin.set_ylabel('Pressure (GPa)')
plt.xlim(2,10)
plt.ylim(10,1000)



#plt.legend(loc='upper left', fontsize='x-small',numpoints=1,scatterpoints=1)
plt.savefig('Hugoniot_S_P.pdf', format='pdf', dpi=1000)

########Entropy Pressure#########


plt.figure()
plt.rc('grid',color='black', linestyle=':', linewidth=0.5,alpha=0.25)
plt.plot(sm,xm*10**3, label="MANEOS Hugoniot and Liquid-Vapor Curve", color='purple', linewidth=2,alpha=0.5,linestyle='--')
plt.plot(lvdSL,lvdPL*10**3, color='purple', linewidth=2,linestyle=':')
plt.plot(SH[min(tmp3[0]):]+1528,PH[min(tmp3[0]):]*10**-9, color='blue')
#plt.fill_between(SH,PH*10**-12-PHe*10**-12,PH*10**-12+PHe*10**-12,label="Hugoniot",color='blue', alpha=0.4)
plt.fill_betweenx(PH[min(tmp3[0]):]*10**-9,SH[min(tmp3[0]):]-SHe[min(tmp3[0]):], SH[min(tmp3[0]):]+SHe[min(tmp3[0]):],label="This Work",color='blue', alpha=0.4)
#plt.errorbar(SH,PH*10**-12,yerr=PHe*10**-12, xerr=SHe, alpha=0.2, linestyle='-', capthick=0,linewidth=2)
plt.plot(lvdSV,lvdPV*10**3,color='purple', linewidth=2,linestyle=':')
plt.errorbar(Sc, Pc*(10**-9),xerr=Sce, yerr=Pce*(10**-9), fmt='o',linewidth=2, color='red',label='Intersection')
#plt.figtext(0.35,0.11,'Shots Z2792, Z2868, Z2879, Z3033', fontsize='xx-small')
#plt.title('Forsterite P - S')
plt.plot(S_tp,P_tp,color='black', linewidth=2, label='Triple Point')
plt.xlabel('Specific Entropy (J/K/kg)')
plt.ylabel('Pressure (GPa)')
plt.xlim(2000,10000)
#plt.semilogy()
plt.xlim(2000,8000)
plt.grid()


#plt.legend(loc='upper right', fontsize='x-small',numpoints=1,scatterpoints=1)
plt.savefig('Hugoniot_S_P_log.pdf', format='pdf', dpi=1000)
############DENSITY PRESSURE#########
plt.figure()
plt.rc('grid',color='black', linestyle=':', linewidth=0.5,alpha=0.25)
plt.plot(dm,xm, label="MANEOS Hugoniot", color='purple', linewidth=2,alpha=0.5,linestyle='--')
plt.plot(rho[min(tmp3[0]):],PH[min(tmp3[0]):]*10**-12, color='blue')
plt.fill_betweenx(PH[min(tmp3[0]):]*10**-12,rho[min(tmp3[0]):]-rhoe[min(tmp3[0]):],rho[min(tmp3[0]):]+rhoe[min(tmp3[0]):],label="This Work",color='blue', alpha=0.4)
plt.plot(rho[ld:],P[ld:]*(10**(-12)),color='black')
plt.fill_between(rho[ld:], P[ld:]*(10**(-12))-Pe[ld:]*(10**(-12)),P[ld:]*(10**(-12))+Pe[ld:]*(10**(-12)),label="Liquid Isentrope",color='black',alpha=0.4)
#plt.scatter(rho_init_l,Pth[ll[0]]*(10**(-12)), label='Isentrope Foot', color='black')
#plt.errorbar(rhoH1,PH1*(10**-12),xerr=rhoH1e, yerr=PH1e*(10**-12), fmt='o',linewidth=2, color='black',label='Reference Hugoniot Point')
plt.errorbar(Rhoc,Pc*(10**-12),yerr=Pce*(10**-12), xerr=Rhoce, fmt='o',linewidth=2, color='red',label='Isentrope-Hugoniot Intersection')
plt.errorbar(Zrho,ZP,yerr=ZPe, xerr=Zrhoe, fmt='o',linewidth=2, color='black',label='Z Hugoniot Points, Omega Reflectivity')

plt.grid()
#plt.figtext(0.35,0.11,'Shots Z2792, Z2868, Z2879, Z3033', fontsize='xx-small')

#plt.title('Forsterite Density - P')
plt.ylabel('Pressure (TPa)')
plt.xlabel('Density (kg/m$^3$)')
plt.ylim(0,1)
plt.xlim(2000,8000)
plt.legend(loc='upper left', fontsize='x-small',numpoints=1,scatterpoints=1)
plt.savefig('Hugoniot_Rho_P.pdf', format='pdf', dpi=1000)

######Density Temperature####################
plt.figure()
plt.rc('grid',color='black', linestyle=':', linewidth=0.5,alpha=0.25)
#plt.plot(dm,ym/1000, label="M-ANEOS Hugoniot", color='purple', linewidth=2,alpha=0.5,linestyle='--')
plt.plot(rho[min(tmp3[0]):],TH[min(tmp3[0]):]/1000, color='blue')
#plt.fill_betweenx(TH[min(tmp3[0]):]/1000,rho[min(tmp3[0]):]-rhoe[min(tmp3[0]):],rho[min(tmp3[0]):]+rhoe[min(tmp3[0]):],label="Root et al. 2018",color='blue', alpha=0.4)
plt.fill_between(rho[min(tmp3[0]):],TH[min(tmp3[0]):]/1000-THe[min(tmp3[0]):]/1000,TH[min(tmp3[0]):]/1000+THe[min(tmp3[0]):]/1000,label="Fo Hugoniot Fit",color='blue', alpha=0.4)
#plt.plot(lvdDensL,lvdT/1000, label="MANEOS Liquid-Vapor Dome", color='purple', linewidth=2,linestyle=':')
#plt.plot(lvdDensV,lvdT/1000, color='purple', linewidth=2,linestyle=':')
plt.plot(rho[ld:],T[ld:]/1000,color='black')
plt.fill_between(rho[ld:],T[ld:]/1000-Te[ld:]/1000,T[ld:]/1000+Te[ld:]/1000,label="Liquid Isentrope",color='black',alpha=0.4)
#plt.errorbar(rhoH1,TH1,yerr=TH1e, xerr=rhoH1e, fmt='o',linewidth=2, color='black',label='Reference Hugoniot Point')
#plt.scatter(rho_init_l,Tref, label='Isentrope Foot', color='black')
plt.errorbar(Rhoc,Tc/1000,yerr=Tce/1000, xerr=Rhoce, fmt='o',linewidth=2, color='red',label='Isentrope-Hugoniot Intersection')
#plt.errorbar(Zrho,ZTr/1000,yerr=ZTer/1000, xerr=Zrhoe, fmt='o',linewidth=2, color='black',label='Z Hugoniot Points, Root et al. 2018')
#plt.errorbar(Zrho,ZT,yerr=ZTe, xerr=Zrhoe, fmt='o',linewidth=2, color='red',label='Z Hugoniot Points, 0 Reflectivity')
#plt.grid()
#plt.figtext(0.35,0.11,'Shots Z2792, Z2868, Z2879, Z3033', fontsize='xx-small')

#plt.title('Forsterite T - P')
plt.ylabel('Temperature (1000 K)')
plt.xlabel('Density (kg/m$^3$)')
#plt.semilogx()
plt.xlim(2000,8000)
plt.ylim(0,20)
plt.legend(loc='best', fontsize='x-small',numpoints=1,scatterpoints=1)
plt.savefig('Hugoniot_Rho_T.pdf', format='pdf', dpi=1000)

##############Cv############33
plt.figure()
plt.rc('grid',color='black', linestyle=':', linewidth=0.5,alpha=0.25)
tmp4=np.where((TH-4000)>0)
CvH=CvH*ma/(R*7)
CvHe=CvHe*ma/(R*7)
Cvnum=Cvnum*ma/(R*7)
Cvnume=Cvnume*ma/(R*7)
dftCV=dftCV*ma/(R*7)
#plt.plot(TH[tmp4[0]]*(10**-3),CvH[tmp4[0]],color='blue')
#plt.fill_between(TH[tmp4[0]]*(10**-3),CvH[tmp4[0]]-CvHe[tmp4[0]],CvH[tmp4[0]]+CvHe[tmp4[0]],label= 'DFT $\gamma$ Slope Method',color='blue',alpha=0.4)
plt.plot(TH[tmp4[0]]*(10**-3),Cvnum[tmp4[0]],color='green')
plt.fill_between(TH[tmp4[0]]*(10**-3),Cvnum[tmp4[0]]-Cvnume[tmp4[0]],Cvnum[tmp4[0]]+Cvnume[tmp4[0]],label= 'Linear $\gamma$ Slope Method',color='green',alpha=0.4)
plt.plot(dftT*(10**-3),dftCV,'o', label='DFT-MD')
plt.xlabel('Temperature (1000 K)')
plt.ylabel('Cv/Nk$_B$')#(J/K/kg)')
plt.ylim(3,7)
plt.xlim(2000*(10**-3),40000*(10**-3))
plt.legend(loc='upper left', fontsize='x-small',numpoints=1,scatterpoints=1)
plt.savefig('Hugoniot_Cv.pdf', format='pdf', dpi=1000)

##############Gamma - volume############
plt.figure()
plt.rc('grid',color='black', linestyle=':', linewidth=0.5,alpha=0.25)
plt.plot(rho_gamma/rho,gamma_m,color='blue',label='de Koker et al. 2008')
plt.fill_between(rho_gamma/rho,gamma_m-gamma_me,gamma_m+gamma_me,color='blue',alpha=0.4)
plt.plot(rho_gamma/dens_dft[1:],gamma_dft[1:],'o',color='green',label='DFT')
#plt.plot(rho_gamma/rho,gamma_m_a,color='purple',label='Asimow 2018')
#plt.fill_between(rho_gamma/rho,gamma_m_a-gamma_me_a,gamma_m_a+gamma_me_a,color='purple',alpha=0.4)
plt.plot(rho_gamma/5437, 2.1, 'o', color='red', label='Brown et al. 1987')
plt.xlabel('V/V_0')
plt.ylabel('$\gamma$')#(J/K/kg)')
#plt.ylim(3,7)
plt.xlim(.3,1)
plt.legend(loc='upper left', fontsize='x-small',numpoints=1,scatterpoints=1)
plt.savefig('gamma_volume.pdf', format='pdf', dpi=1000)

print('Density ( kg/m3) ABCDE=',rho0s, '3000.0', rho0l, rho0l, Rhoc)
print('Density uncertainty ( kg/m3) ABCDE=',rho0se, '3000.0', rho0le, rho0le, Rhoce)
print('Volume(m3/kg) ABCDE=', 1/rho0s, 1/3000, 1/rho0l, 1/rho0l,1/Rhoc)
print('Volume uncertainty (m3/kg) ABCDE=', 1/rho0se, 1/3000, 1/rho0le, 1/rho0le,1/Rhoce)
print('Pressure (GPa) ABCDE=', 0.001, 0.001, 0.001, 2.3, Pc*10**-9)
print('Pressure uncertainty (GPa) ABCDE=', 0.001, 0.001, 0.001, .4, Pce*10**-9)
print('Temperature (K) ABCDE=', 298.15, Tm, Tm, Tref, Tc)
print('Temperature uncertainty (K) ABCDE=', 298.15, Tme, Tme, Tref, Tce)
print('Entropy (J/K/kg) ABCDE=', Sa,Sb,Sc2,Sd,Se)
print('Entropy uncertainty (J/K/kg) ABCDE=', Sae,Sbe,Sc2e,Sde,See)
print('Intersection Point:',' P=',Pc*10**-9,'+/-',Pce*10**-9,'GPa')
print('T=',Tc,'+/-',Tce,'K')
print('Rho=',Rhoc,'+/-',Rhoce,'kg/m^3')
print('S=',Sc,'+/-',Sce,'J/K/kg')


#####P-S Fit
##def tFit(xx,A,B,C,D):
##    #return A+
##    #return A*xx+ B*xx**2+C*xx**3
##    #return A + B*xx + C*xx**2 + D/xx
##    #return A*xx+ B*xx**2+C*np.log(xx*D)
##    return A*xx**(-0.5)+B*xx**(0.5)+C*xx**1.5+D
##tmp5=np.where((TH-1500)>0)
##A=np.zeros(steps)
##B=np.zeros(steps)
##C=np.zeros(steps)
##D=np.zeros(steps)
##for i in range(0,steps):
##    pressure = PH[tmp5[0]] + PHe[tmp5[0]]*sp.randn()
##    entropy = SH[tmp5[0]] + SHe[tmp5[0]]*sp.randn()
##                                                 
##    temp1, temp2 = curve_fit(tFit, pressure*10**-9,entropy, absolute_sigma=True)
##    A[i]=np.mean(temp1[0])
##    B[i]=np.mean(temp1[1])
##    C[i]=np.mean(temp1[2])
##    D[i]=np.mean(temp1[3])
##
##
##A_mean=np.mean(A)
##A_std=np.std(A)
##B_mean=np.mean(B)
##B_std=np.std(B)
##C_mean=np.mean(C)
##C_std=np.std(C)
##D_mean=np.mean(D)
##D_std=np.std(D)
##
##X=[]
##X.append(A)
##X.append(B)
##X.append(C)
##X.append(D)
##Covar=np.cov(X)
##
##print('A','B','C','D', A_mean,B_mean,C_mean,D_mean)
##print("Entropy fit Covariance matrix = ",Covar)
##
##temp_s=np.zeros((size,steps))
##lmat=sp.linalg.cholesky(temp2,lower=True)
##for j in range(0,steps):
##    temp_mat=sp.randn(4,1)
##    bmat=np.matmul(lmat,temp_mat)
##    a1=A_mean+bmat[0,0]
##    b1=B_mean+bmat[1,0]
##    c1=C_mean+bmat[2,0]
##    d1=D_mean+bmat[3,0]
##    temp_s[:,j]=tFit(PH*10**-9,a1,b1,c1,d1)
##SS_mean=np.zeros(size)
##SS_std=np.zeros(size)
##for i in range(0,size):
##    SS_mean[i]=np.mean(temp_s[i,:])
##    SS_std[i]=np.std(temp_s[i,:])
##
###print(temp1[0],temp1[1],temp1[2],temp1[3])
###print(temp2[0,0]**(1/2),temp2[1,1]**(1/2),temp2[2,2]**(1/2),temp2[3,3]**(1/2))
##
##
##
##
##plt.figure()
##plt.plot(SH[tmp5[0]],PH[tmp5[0]]*10**-9)
##plt.fill_betweenx(PH[min(tmp5[0]):]*10**-9,SH[min(tmp5[0]):]-SHe[min(tmp5[0]):], SH[min(tmp5[0]):]+SHe[min(tmp5[0]):],label="Z Hugoniot",color='blue', alpha=0.4)
##plt.plot(SS_mean,PH*10**-9,color='red')
##plt.fill_betweenx(PH*10**-9,SS_mean-SS_std,SS_mean+SS_std,color='red',alpha=0.4, label='Quartz refit')
##
##plt.xlabel('Specific Entropy (J/K/kg)')
##plt.ylabel('Pressure (GPa)')

with open('Z_Hugoniot.txt','w+') as f:
    f.write('P (Pa), T (K), S (J/K/kg), rho (kg/m^3), Pe, Te, Se, rho_e  \r\n')
    for i in range(np.size(PH)):
        f.write(str(PH[i]))
        f.write(',')
        f.write(str(TH[i]))
        f.write(',')
        f.write(str(SH[i]))
        f.write(',')
        f.write(str(rho[i]))
        f.write(',')
        f.write(str(PHe[i]))
        f.write(',')
        f.write(str(THe[i]))
        f.write(',')
        f.write(str(SHe[i]))
        f.write(',')
        f.write(str(rhoe[i]))
        f.write(',')
        f.write('\r\n')
f.closed


up=np.linspace(begin_up,14,size) #defining particle velocity
us=hugoniot(up,a1,b1,c1,d1) #getting shock velocity
with open('Z_Hugoniot_us-up.txt','w+') as f:
    f.write('P (GPa), T (K), S (J/K/kg),rho(kg/m^3), gamma, Cv/NKb, us(km/s), up(km/s) \r\n')
    for i in range(np.size(PH)):
        f.write(str(PH[i]*10**(-9)))
        f.write(',')
        f.write(str(TH[i]))
        f.write(',')
        f.write(str(SH[i]))
        f.write(',')
        f.write(str(rho[i]))
        f.write(',')
        f.write(str(gamma_dft_print[i]))
        f.write(',')
        f.write(str(CvH[i]))
        f.write(',')
        f.write(str(us[i]))
        f.write(',')
        f.write(str(up[i]))
        f.write('\r\n')
f.closed



             
plt.show()
