"""
Created April 25th, 2019
@author: Erik Davies

This function calculates an isentrope between two pressure-volume states, using
Mie-Gruneisen EOS referenced to a known Hugoniot. Gamma is assumed to be
unknown, so a formulation must be provided in this code. The calculation
iterates gamma until pressure and volume are correct.

Implemented here is also the structure for a monte carlo calculation, that,
ideally, would be parallelized. This has many inputs, probably the only
use case for this function is my already existing Deep_release codes.
"""

import numpy as np
import scipy as sp

def fo_hugoniot(a,b,c,d,x):
    return a + b*x +c*x**2+d*x**3

def Mie_Grun_iterator(lmath,rho0s,rho0se,liq_den,liq_den_e,hp_dens,hp_dens_e,
                      fo_sam_P,fo_sam_P_e,fo_sam_dens,fo_sam_dens_e,fo_rel_P,
                      fo_rel_P_e,fo_rel_dens,fo_rel_dens_e,gamma_asi1,gamma_a_e,q_asi1,q_a_e,up,P_isen,
                      E_isen,gamma_isen,q_rel,gamma_rel,a1,b1,c1,d1,j):

    #if j == int(0.5*steps1):
        #print('Halfway done')
    #print(j)
    bmath=np.matmul(sp.randn(1,4), lmath) #For covariance calculation on the hugoniot
    ah=a1+bmath[0,0]
    bh=b1+bmath[0,1]
    ch=c1+bmath[0,2]
    dh=d1+bmath[0,3]
    rho_init_s=rho0s+rho0se*sp.randn() #initial density for hugoniot
    rho_init_l=liq_den+liq_den_e*sp.randn()#For asimow gamma
    rho_init_hp=hp_dens+hp_dens_e*sp.rand() #5433 from shock hugoniot at 168 GPa, see Brown 1987, # for gaussian version 5633

    sam_p=(fo_sam_P+fo_sam_P_e*sp.randn(np.size(fo_sam_P)))*(10**9)
    #sam_p=(fo_sam_P)*(10**9)
    sam_dens=fo_sam_dens+fo_sam_dens_e*sp.randn(np.size(fo_sam_P))
    rel_p=(fo_rel_P+fo_rel_P_e*sp.randn(np.size(fo_sam_P)))*(10**9)
    #rel_p=(fo_rel_P)*(10**9)
    rel_dens=fo_rel_dens+fo_rel_dens_e*sp.randn(np.size(fo_sam_P))
    ####GAMMA####
    gamma_asi2=gamma_asi1+gamma_a_e*sp.rand()
    #q_asi2=q_asi1
    #Making a Forsterite Hugoniot


    us=fo_hugoniot(ah,bh,ch,dh,up)#getting shock velocity
    rhoH=rho_init_s*us/(us-up) #getting density array
    PHmc=rho_init_s*us*up # Getting pressure from rankine-hugoniot, in MPa
    PHmc=PHmc*(10**6) # putting pressure into Pa

    #Calculate energy on the Hugoniot, change in entropy, not total
    eh=-0.5*PHmc*((1/rhoH)-1/rho_init_s)
    #cycle through measurements
    for k in range(0,np.size(fo_sam_P)):
        count=0
        rho_init=sam_dens[k]
        rho_final=rel_dens[k]
        while rho_init < rho_final: #if densities are unphysical
            sam_dens[k]=fo_sam_dens[k]+fo_sam_dens_e[k]*sp.randn()
            rel_dens[k]=fo_rel_dens[k]+fo_rel_dens_e[k]*sp.randn()
            rho_init=sam_dens[k]
            rho_final=rel_dens[k]
        index_init=min(min(np.where(rhoH>rho_init)))
        index_fin=min(min(np.where(rhoH>rho_final)))
##        if PHmc[index_init] < rel_p[k]:
##            gamma_rel[k,j]=float("NaN")
##            P_isen[k,j,:]=float("NaN")
##            E_isen[k,j,:]=float("NaN")
##            gamma_isen[k,j,:]=float("NaN")
##            break
        temp=eh[index_init] #Ref energy at peak pressure intersection
        e_hug=eh-temp
        #print(index_init,index_fin)
        #####GAMMA#####
        q_asi2=q_asi1
        #gamma=gamma_asi2*((rho_init_l/rhoH)**q_asi2) #exponential
        #gamma=gamma_asi2+((rho_init_l/rhoH)**q_asi2) #exponential + lower limit
        #gamma=gamma_asi2+((rho_init_l/rhoH - 1)*q_asi2) #linear
        #gamma=gamma_asi2 + (q_asi2)*((rho_init_hp/rhoH)-1) #negative linear slope, high pressure initial volume
        #gamma=.6+(gamma_asi2-.6)*np.exp(-q_asi2*(rho_init_l/rhoH-rho_init_l/rho_init_hp)**2) # gaussian version, peak at likely intersection
        #gamma = gamma_asi2 + (q_asi2)*(rhoH-rho_init_hp) # Non normalized linear
        #gamma=0.66+(gamma_asi2-.66)*np.exp(q_asi2*((rhoH-rho_init_hp)**2)) #non-normalized gaussian
        gamma=0.5+(gamma_asi2-.5)*np.exp(q_asi2*((rhoH-rho_init_hp)**2)) #non-normalized gaussian

        gamma_rel[k,j]=gamma[index_fin]
        q_rel[k,j]=q_asi2
        P_isen[k,j,:]=0
        E_isen[k,j,:]=0
        #######INITIAL VALUES######
        while abs(P_isen[k,j,index_fin]-rel_p[k] ) > .005*rel_p[k]: #Iterating to called sensitivity
            
            P_isen[k,j,:]=float("NaN")
            E_isen[k,j,:]=float("NaN")
            P_isen[k,j,index_init]=PHmc[index_init]
            E_isen[k,j,index_init] = e_hug[index_init]
            #print(index_init)
     #       P_isen[k,j,index_init-1]=(PHmc[index_init-1]- (e_hug[index_init-1]- E_isen[k,j,index_init] + P_isen[k,j,index_init]*((1/rhoH[index_init-1]-1/rhoH[index_init])/2))*(
    #            rhoH[index_init-1] * gamma[index_init-1]) )/(1+ ((1/rhoH[index_init-1]-1/rhoH[index_init])/2)*(rhoH[index_init-1] * gamma[index_init-1]))
    #        E_isen[k,j,index_init-1]=E_isen[k,j,index_init] + ((P_isen[k,j,index_init]+P_isen[k,j,index_init-1])/2)*(1/rhoH[index_init-1]-1/rhoH[index_init])
            #print(P_isen[k,j,index_init-1],PHmc[index_init-1])
            for l in range(index_init,index_fin-1,-1): #calculating iterating isentrope pressure and energy
                #print(l,index_init)
                #if sam_p[k]>900*10**9:
 #                   print(P_isen[k,j,l],E_isen[k,j,l])
                P_isen[k,j,l-1]=(PHmc[l-1]- (e_hug[l-1]- E_isen[k,j,l] + P_isen[k,j,l]*((1/rhoH[l-1]-1/rhoH[l])/2))*(
                    rhoH[l-1] * gamma[l-1]) )/(1+ ((1/rhoH[l-1]-1/rhoH[l])/2)*(rhoH[l-1] * gamma[l-1]))
                E_isen[k,j,l-1]=E_isen[k,j,l] - ((P_isen[k,j,l]+P_isen[k,j,l-1])/2)*(1/rhoH[l-1]-1/rhoH[l])
            gamma_isen[k,j,:]=gamma
            #print(rel_p[k],P_isen[k,j,index_fin],P_isen[k,j,index_init],gamma[index_init],gamma[index_fin])
            #print(index_init,index_fin)
                ##switch the greater/lesser than symbols for negative slope
            count=1+count # count of the while
            if P_isen[k,j,index_fin]-rel_p[k] > .005*rel_p[k]:
                #print(q_asi2) 
                #temp1=q_asi2+0.1#alter "canonical" versions by +0.1, gaussian by +5
                #temp1=q_asi2-q_a_e #non normalized
                temp1=q_asi2*(1+q_a_e) #non normalized, geometric step
                q_asi2=temp1
                #print(q_asi2)
                #gamma=gamma_asi2*((rho_init_l/rhoH)**q_asi2) #exponential
                #gamma=gamma_asi2+((rho_init_l/rhoH)**q_asi2) #exponential + lower limit
                #gamma=gamma_asi2+((rho_init_l/rhoH - 1)*q_asi2) #linear
                #gamma=gamma_asi2 + (q_asi2)*((rho_init_hp/rhoH)-1) #negative linear slope
                #gamma = gamma_asi2 + (q_asi2)*(rhoH-rho_init_hp) # Non normalized linear
                #gamma=0.66+(gamma_asi2-.66)*np.exp(q_asi2*((rhoH-rho_init_hp)**2)) #non-normalized gaussian
                gamma=0.5+(gamma_asi2-.5)*np.exp(q_asi2*((rhoH-rho_init_hp)**2)) #non-normalized gaussian

                #gamma=.66+(gamma_asi2-.66)*np.exp(-q_asi2*(rho_init_l/rhoH-rho_init_l/rho_init_hp)**2) # gaussian version, peak at likely intersection
                gamma_isen[k,j,:]=gamma
                gamma_rel[k,j]=gamma[index_fin]
                q_rel[k,j]=q_asi2
            if P_isen[k,j,index_fin]-rel_p[k] < .005*rel_p[k]:
                #temp2=q_asi2-.1#alter "canonical" versions by -0.1, gaussian by-5 q_a_e
                #temp2=q_asi2+q_a_e #non normalized
                temp2=q_asi2*(1-q_a_e) #non normalized, geometric step
                q_asi2=temp2
                #gamma=gamma_asi2*((rho_init_l/rhoH)**q_asi2) #exponential
                #gamma=gamma_asi2+((rho_init_l/rhoH)**q_asi2) #exponential + lower limit
                #gamma=gamma_asi2+((rho_init_l/rhoH - 1)*q_asi2) #linear
                #gamma=gamma_asi2 + (q_asi2)*((rho_init_hp/rhoH)-1) #negative linear slope
                #gamma = gamma_asi2 + (q_asi2)*(rhoH-rho_init_hp) # Non normalized linear
                #gamma=.66+(gamma_asi2-.66)*np.exp(q_asi2*((rhoH-rho_init_hp)**2)) #non-normalized gaussian
                gamma=0.5+(gamma_asi2-.5)*np.exp(q_asi2*((rhoH-rho_init_hp)**2)) #non-normalized gaussian

                gamma_isen[k,j,:]=gamma
                #gamma=.66+(gamma_asi2-.66)*np.exp(-q_asi2*(rho_init_l/rhoH-rho_init_l/rho_init_hp)**2) # gaussian version, peak at likely intersection
                gamma_rel[k,j]=gamma[index_fin]
                q_rel[k,j]=q_asi2
            if count > 300: #if there are excessive iterations
                print('reached three hundred counts, output, then set NaN',k,q_asi2,gamma_rel[k,j], index_init, index_fin,j ) #letting me know
                gamma_rel[k,j]=float("NaN")
                P_isen[k,j,:]=float("NaN")
                E_isen[k,j,:]=float("NaN")
                gamma_isen[k,j,:]=float("NaN")
                break
            if gamma[index_fin] < 0: #negative gamma is wrong as well
                print('Negative Gamma, no convergance, set everything to NaN',j)
                gamma_rel[k,j]=float("NaN")
                P_isen[k,j,:]=float("NaN")
                E_isen[k,j,:]=float("NaN")
                break
                
