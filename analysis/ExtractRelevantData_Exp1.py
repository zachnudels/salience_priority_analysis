# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:51:57 2019

@author: User1
"""

# =============================================================================
# Import moduels
# =============================================================================
import pandas as pd
import numpy as np
import os


def extractData(params, excluded_participants):
    fl = params['directories']['clean_data']
    saveLoc = params['directories']['root']
    ppList = params['participants']
    for excluded_participant in excluded_participants:
        ppList.remove(excluded_participant)
    print(ppList)
    SOAs = params['experiment_settings']['SOAs']  # data.DV_ISI

    # =============================================================================
    # Extract the correct data for each participant
    # =============================================================================
    # Store temporary values
    for dep in params['dependent_variables']:
        DataDict = {}
        DataDict['ppNr'] = ppList

        ## Create dictionary keys
        for soa in SOAs:
            # To store temp values
            DataDict['High_RT_{}'.format(soa)] = []
            DataDict['High_Corr_{}'.format(soa)] = []
            DataDict['Low_RT_{}'.format(soa)] = []
            DataDict['Low_Corr_{}'.format(soa)] = []


        for idx, pp in enumerate(ppList):
            # Load data, and remove excluded trials then reset dataframe index
            fileName = os.path.join(fl, f"sub_{pp}.p")
            data = pd.read_pickle(fileName)
            data = data.loc[data.DK_includedTrial.values,:]
            # data = data.loc[data.DV_TargetHit.values == 1,:] #target only latencies
            # data = data.loc[data.DV_TargetHit.values == 0,:] #distractor only latencies
            data = data.reset_index()

            # Make booleans
            highDep = data.DV_target_salience.values == 'high'
            lowDep = data.DV_target_salience.values == 'low'

            for soa in SOAs:
                soaBool = data.DV_ISI.values == soa
                high = np.logical_and(highDep, soaBool)
                low = np.logical_and(lowDep, soaBool)

                # Extract RT and Correct or incorrect hits for high salience trials
                DataDict['High_RT_{}'.format(soa)].append(data.DV_RT[high].values)
                DataDict['High_Corr_{}'.format(soa)].append(data.DV_TargetHit[high].values)

                # Extract RT and Correct or incorrect hits for low salience trials
                if dep == 'salience':
                    DataDict['Low_RT_{}'.format(soa)].append(data.DV_RT[low].values)
                    DataDict['Low_Corr_{}'.format(soa)].append(data.DV_TargetHit[low].values)
                elif dep == 'relevance':
                    DataDict['Low_RT_{}'.format(soa)].append(data.DV_RT[low].values)
                    DataDict['Low_Corr_{}'.format(soa)].append(abs(data.DV_TargetHit[low].values -1))
                else:
                    raise ValueError
                #create variable with DV as saccade to Most salient singleton
                # DataDict['Low_Corr_{}'.format(soa)].append(abs(data.DV_TargetHit[low].values -1))

        # Put the temp dict in the pandas dataframe and save it
        smartData = pd.DataFrame(DataDict)

        #smartData = pd.DataFrame.from_dict(DataDict, orient='index')  # this one does work, but I'm still not completey sure why it throws the error

        smartData.to_pickle(os.path.join(saveLoc, f"{dep}_SMARTdata.p"))

    return ppList

