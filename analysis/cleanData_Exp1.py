# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 14:18:26 2019

@author: User1
"""

# =============================================================================
# Import moduels
# =============================================================================
import pandas as pd
import numpy as np
import os
from math import radians as to_radians




# =============================================================================
# Set varibales
# =============================================================================


# =============================================================================
# Helper functions ( I am Lazy)
# =============================================================================
def pix_per_deg(h_res, screen_size, distance):
    pix_per_mm = h_res / screen_size
    pix = 2 * (distance * (np.tan(to_radians(0.5)))) * pix_per_mm
    return pix

def centerToTopLeft(pointXY, screenXY, flipY = False):
    """
    Switch between screen coordinate systems.
    
    Switches from (0,0) as center to (0,0) as top left
    Assumes negative y values are up and positve values are down
    if flip = True, assumes negative y values are down and positive up
    """
    newX = pointXY[0] + (screenXY[0]/2)
    if flipY == False:
        newY = pointXY[1] + (screenXY[1]/2)
    else:
        newY = (pointXY[1]*-1) + (screenXY[1]/2)
    return (newX, newY)

def distBetweenPoints(point1, point2):
    """
    Return the euclidian distance between 2 points.

    Parameters
    ----------
    point1 : tuple
        point1.
    point2 : tuple
        point2.

    Returns
    -------
    float
        The euclidian distance between the two points.

    """
    return np.sqrt( (point1[0]-point2[0])**2 + (point1[1] - point2[1])**2 )

# =============================================================================
# Itterate over participants
# =============================================================================


def cleanData(params):
    dataDir = params['directories']['parsed_data']
    cleanDir = params['directories']['clean_data']
    ANOVADir = params['directories']['ANOVA']

    # Exclusion criteria
    screenSize = params['experiment_settings']['screenRes']
    pxPerDeg = pix_per_deg(screenSize[0],
                           params['experiment_settings']['screenSize'][0],
                           params['experiment_settings']['viewingDistance'])
    maxDist = params['exclusion_criteria']['maxDist']  # Visual Degrees
    minRT = params['exclusion_criteria']['minRT']
    maxRT = params['exclusion_criteria']['maxRT']

    center = [int(x) / 2. for x in screenSize]

    excluded_subjects = []

    for pp in params['participants']:
        filePath = os.path.join(dataDir, f"sub_{pp}Parsed.p")
        data = pd.read_pickle(filePath)

    # =============================================================================
    # Prealocate lists for storing data
    # =============================================================================
        data['DV_targXtl'] = np.zeros(len(data))
        data['DV_targYtl'] = np.zeros(len(data))
        data['DV_distrXtl'] = np.zeros(len(data))
        data['DV_distrYtl'] = np.zeros(len(data))
        data['DV_RT'] = np.zeros(len(data))
        data['DV_saccIdx'] = np.zeros(len(data))
        data['DV_ExclReason'] = 'Included'
        data['DV_fixSDist'] = np.zeros(len(data))
        data['DV_fixEDist'] = np.zeros(len(data))
        data['DV_targDist'] = np.zeros(len(data))
        data['DV_distrDist'] = np.zeros(len(data))
        data['DV_TargetHit'] = np.zeros(len(data))

    # =============================================================================
    # Itterate over trials
    # =============================================================================
        for tNr in range(len(data)):

            # Check if practice trial
            if data.DV_practice[tNr] == 'practice':  # change this back to practice
                data.loc[tNr, 'DK_includedTrial'] = False
                data.loc[tNr,'DV_ExclReason'] = 'PracticeTrial'
                continue

            # Recode target and distractor to top left coordinates
            targx, targy = centerToTopLeft((data.DV_target_co_x[tNr], data.DV_target_co_y[tNr]) , screenSize,flipY = True)
            distrx, distry = centerToTopLeft((data.DV_distractor_co_x[tNr], data.DV_distractor_co_y[tNr]) , screenSize, flipY = True)
            data.loc[tNr, 'DV_targXtl'] = targx
            data.loc[tNr, 'DV_targYtl'] = targy
            data.loc[tNr, 'DV_distrXtl'] = distrx
            data.loc[tNr, 'DV_distrYtl'] = distry

            # Extract all relevant saccade events and times
            allSsaccX = data.DK_ssaccX[tNr] # X pos start
            allSsaccY = data.DK_ssaccY[tNr] # Y pos start
            allESaccX = data.DK_esaccX[tNr] # X pos end
            allESaccY = data.DK_esaccY[tNr] # Y pos end
            allSsacc = data.DK_ssacc[tNr] # Start time

            # Extract all relevant display events and latencies
            targOn = data.DV_target_display[tNr]
            latencies = allSsacc - targOn

            ### Get distanes for all saccades
            # Distance between start of saccade and center of the screen
            fixSDist = distBetweenPoints(center,(allSsaccX, allSsaccY)) / pxPerDeg

            # Distance between end of saccade and center of the screen
            fixEDist = distBetweenPoints(center,(allESaccX, allESaccY)) / pxPerDeg

            # Distance between the end of the saccade and the target and distractor
            targDist = distBetweenPoints((targx, targy), (allESaccX, allESaccY)) / pxPerDeg
            distrDist = distBetweenPoints((distrx, distry),(allESaccX, allESaccY)) / pxPerDeg

            ### Itterate through saccade and exclude or include trial
            # First check if there are any saccades starting after target onset
            if sum(latencies > 0) == 0:
                data.loc[tNr, 'DK_includedTrial'] = False
                data.loc[tNr, 'DV_ExclReason'] = 'No_saccade_after_target'
            else:
                nSacc = len(allSsacc)
                for sacc in range(nSacc):
                    # We check the start and landing positions and RT
                    corrStart = fixSDist[sacc] <= maxDist
                    corrEnd = targDist[sacc] <= maxDist or distrDist[sacc] <= maxDist
                    corrRT = latencies[sacc] >= minRT and latencies[sacc] <= maxRT
                    if corrStart and corrEnd and corrRT:
                        data.loc[tNr, 'DV_fixSDist'] = fixSDist[sacc]
                        data.loc[tNr, 'DV_fixEDist'] = fixEDist[sacc]
                        data.loc[tNr, 'DV_targDist'] = targDist[sacc]
                        data.loc[tNr, 'DV_distrDist'] = distrDist[sacc]
                        data.loc[tNr, 'DV_RT'] = latencies[sacc]
                        data.loc[tNr, 'DV_saccIdx'] = sacc
                        if targDist[sacc] <= maxDist:
                            data.loc[tNr, 'DV_TargetHit'] = 1
                        break
                    elif latencies[sacc] < 0:
                        continue
                    elif fixSDist[sacc] > maxDist:
                        data.loc[tNr, 'DK_includedTrial'] = False
                        data.loc[tNr,'DV_ExclReason'] = 'Incorrect_Fixation'
                        break
                    # Check if saccade stated and ended near fixation (micro saccade o corrective saccade)
                    elif fixSDist[sacc] < maxDist and fixEDist[sacc] < maxDist:
                        continue
                    # Check if saccades landed far away from both target and distractor
                    elif targDist[sacc] > maxDist and distrDist[sacc] > maxDist:
                        data.loc[tNr, 'DK_includedTrial'] = False
                        data.loc[tNr,'DV_ExclReason'] = 'Incorrect_saccade_landing'
                        break
                    # Saccade shoulf not be to fast
                    elif latencies[sacc] < minRT or latencies[sacc] > maxRT:
                        data.loc[tNr, 'DK_includedTrial'] = False
                        data.loc[tNr,'DV_ExclReason'] = 'Saccade_to_fast_or_slow'
                        break
                    # This section should only be reached if no saccades are found
                    elif sacc+1 == nSacc:
                        data.loc[tNr, 'DK_includedTrial'] = False
                        data.loc[tNr,'DV_ExclReason'] = 'Unexpected_saccade_pattern'

    # =============================================================================
    # Save the cleaned data
    # =============================================================================
        if data.DK_includedTrial.sum() / len(data) >= 0.5:
            filePath = os.path.join(cleanDir, f"sub_{pp}.p")
            data.to_pickle(filePath)

            filePath = os.path.join(ANOVADir, f"sub_{pp}.csv")
            df_ANOVA = data[['DV_target_salience', 'DV_TargetHit', 'DV_ISI', 'DV_RT', 'DK_includedTrial']]
            df_ANOVA.to_csv(filePath)


        else:
            excluded_subjects.append(pp)


    return excluded_subjects
