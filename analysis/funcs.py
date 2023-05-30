# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 13:01:07 2022

@author: ehn268
"""


import matplotlib.pyplot as plt 
import numpy as np
from SMART import SMART_Funcs as SF
import pandas as pd 

class analysis_funcs: 
    
    
    # some vars 
    
    colors = ['red','#0089cf','#559834', '#810d70','orange','#eb6734','#14691f', '#47a654','#0d4714']  # 0red, 1blue, 2green, 3purple,4yellow-orange,5orange, 6darkgreen, 7lightgreen, 8darkergreen     #['red', '#0089cf', '#306a30','orange','purple']
    colors3 = ['#164150', '#65A4B8', '#A6C9C5'] # blues of the attentional capture paper 

    krnSize = 25
    minTime = 150
    maxTime = 451
    stepTime = 1
    nPerm = 1   # this need to increase if we do it for real obviously 
    sigLevel = 0.01        
    
    
    
    def make_standard_plot(self, res, color1, color2): 
        
        
        # plot the first dv here 
        plt.fill_between(res.timeVect,res.weighDv1Average - res.conf95, res.weighDv1Average + res.conf95, color=color1, alpha=0.25,zorder = 2)
        plt.plot(res.timeVect, res.weighDv1Average, color = color1,zorder = 2, linewidth = 2)
        
        # and the second one here 
        plt.fill_between(res.timeVect,res.weighDv2Average - res.conf95, res.weighDv2Average + res.conf95, color= color2, alpha=0.25,zorder = 2)
        plt.plot(res.timeVect, res.weighDv2Average, color = color2,zorder = 2, linewidth = 2)
        
        # add stats if sig difference 
        c = 0 
        linepos= 0
        for ind, i in enumerate(res.sigCL):
            if res.sumTvals[ind] >= res.sigThres:  #only if it is really significant 
                    x = res.timeVect[res.sigCL[ind]]
                    y  = [linepos + c*-0.05]* len(x)
                    c+=1 # use this instead of ind, becuase it will only count the sig iterations 
                    plt.plot(x,y,color1, linewidth = 5)
                    plt.plot(x,y,color2,linestyle = '--', linewidth = 5, dashes = [2,2])
                    print('There is a sig difference here')
      
        
        # adjust axis 
        plt.ylim(-0.5, 1.1)
        
      
        # Plot kernel density estimation KDE
        ax1 = plt.gca()
        sTimes1, unqT1, countT1 = SF.getKDE(np.hstack(res.data[res.t1]),res.timeVect, res.krnSize) 
        sTimes2, unqT2, countT2 = SF.getKDE(np.hstack(res.data[res.t2]),res.timeVect, res.krnSize)
        #countT1 = res.smooth(countT1,5)
        #countT2 = res.smooth(countT2,5)
        ax1_1 = ax1.twinx()
        ax1_1.plot(res.timeVect, sTimes1, ls = '--', alpha = 0.6, color = color1)
        ax1_1.plot(res.timeVect, sTimes2, ls=  '-', alpha = 0.3,color = color2)
        ax1_1.bar(unqT1, countT1, alpha = 0.25, color = 'black')
        ax1_1.bar(unqT2, countT2, alpha = 0.25, color = 'black')
        ax1_1.set_ylim(0,30)
        ax1_1.set_xlim(res.timeMin,res.timeMax)
        ax1_1.spines['top'].set_visible(False)
        ax1_1.spines['right'].set_visible(False)
        ax1_1.set_yticks(np.linspace(0,np.max(np.hstack([sTimes1, sTimes2])),2, dtype=int))

    def permute(self,df_new, nPerm):
        sum_t_vals = [] # saving the biggest t val for every permutation 
        for p in range(nPerm): 
          
            
            groups_list = []
            for name, group in df_new.groupby([df_new.index]): 
                np.random.shuffle(group['condition'].values)
                groups_list.append(group) # put together all subjects again
            
            df_shuf = pd.concat(groups_list) # make a new data frame with labels shuffled for each subject
            
            # select new permuted conditions
            cond1_df = df_shuf[df_shuf['condition']==1]
            cond2_df = df_shuf[df_shuf['condition']==2]
            
            # convert to 2d numpy array
            cond1 = np.vstack(cond1_df.diffs_dv)
            cond2 = np.vstack(cond2_df.diffs_dv)
            weight1 = np.vstack(cond1_df.sum_weights)
            weight2 = np.vstack(cond2_df.sum_weights)

            # now put in t-test for comparisions between conditions 
            [t_vals,p_vals] = SF.weighted_ttest_rel(cond1,cond2, weight1, weight2) 
         
            # extract biggest cluster 
            clusters, indx = SF.getCluster(p_vals < 0.05) # I think using this function is OK
            # calculate sum of t-vals (only for those clusters where it is lower than 0.05 )
            sigCl = [indx[i] for i in range(len(clusters)) if clusters[i][0] == True]
            sums_t = [np.sum(abs(t_vals[i])) for i in sigCl]
            # see which one is the biggest, and add it here 
            if len(sums_t) ==0: 
                sum_t_vals.append(np.max(t_vals))

            else: 
                sum_t_vals.append(np.sort(sums_t)[-1])
                
                
                
        return sum_t_vals  
    
    
    
    def custom_legend(self, labels,fontsize = 12, alpha = 0.5, no_dist = False): 
        import matplotlib.patches as mpatches
        plt.rcParams['hatch.linewidth'] = 4  # previous pdf hatch linewidth
        handles = []
        for l,label in enumerate(labels):
            handles.append(mpatches.Patch(color=self.colors[l], label=label, alpha = alpha))
        plt.legend(handles=handles, fontsize = fontsize, framealpha =1 ,frameon = False)  
            