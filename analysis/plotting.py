import os
import matplotlib.pyplot as plt
from SMART.SMARTClass import SMART
from SMART import SMART_Funcs as SF
import numpy as np
from . import funcs
import pandas as pd
from itertools import combinations


def plot(params):
    e = funcs.analysis_funcs()
    data_path = params['directories']['clean_data']
    fig_path = params['directories']['figs']
    os.chdir(data_path)
    exp = 2

    # colors of the main plots
    color1 = params['plotting']['color1']
    color2 = params['plotting']['color2']

    SOAs = params['experiment_settings']['SOAs']
    fName = os.path.join(params['directories']['root'], "SMARTdata.p")

    # I changed this..
    plt.close('all')

    ####################################################
    ######### make the first figure with all the SOAs
    #######################################################
    all_conditions = {}
    for i, SOA in enumerate(SOAs):
        plt.subplot(3, 2, i + 2)
        plt.title('SOA: ' + str(SOA))
        timeVar1 = 'High_RT_{}'.format(SOA)  # The name of the column with he time variables in dataFile
        depVar1 = 'High_Corr_{}'.format(SOA)  # The name of the column with he depVar variables in dataFile

        # Variable of interest 2 (Low salience)
        timeVar2 = 'Low_RT_{}'.format(SOA)  # The name of the column with he time variables in dataFile
        depVar2 = 'Low_Corr_{}'.format(SOA)

        res = SMART(fName, depVar1, timeVar1, depVar2, timeVar2)
        res.runSmooth(e.krnSize, e.minTime, e.maxTime, e.stepTime)

        # already get the difference for the next plot
        diffs_dv = res.smooth_dv1 - res.smooth_dv2
        sum_weights = res.weights_dv1 + res.weights_dv2

        # these are for plotting , not used in analysis perse, but are incorporated in the t-test
        weighDv1_diff = SF.weighArraysByColumn(diffs_dv, sum_weights)
        weighDv1Average_diff = np.nansum(weighDv1_diff, axis=0)
        # conf95 = SF.weighConfOneSample95(diffs_dv, sum_weights)

        # saving some information for later
        all_conditions['diffs' + str(SOA)] = diffs_dv
        all_conditions['sum_weights' + str(SOA)] = sum_weights
        all_conditions['weighted_average' + str(SOA)] = weighDv1Average_diff

        print(all_conditions)

        res.runPermutations(e.nPerm)
        res.runStats(e.sigLevel)
        e.make_standard_plot(res, color1, color2)

    # now put a custom legend in the first space
    plt.subplot(3, 2, 1)
    from matplotlib.lines import Line2D
    salient = Line2D([0], [0], color=color1, lw=4, label=f"{params['dependent_variable']} target")
    non_salient = Line2D([0], [0], color=color2, lw=4, label=f"Non-{params['dependent_variable']} target")
    salient_KDE = Line2D([0], [0], color=color1, lw=2, label=f"KDE {params['dependent_variable']} target",
                         linestyle='dashed')
    non_salient_KDE = Line2D([0], [0], color=color2, lw=2, label=f"KDE non-{params['dependent_variable']} target",
                             linestyle='dashed')
    plt.legend(handles=[salient, non_salient, salient_KDE, non_salient_KDE], fontsize=12, frameon=False, ncol=1)
    ax = plt.gca()
    ax.set_yticks([])
    ax.set_xticks([])

    # now remove those lines as well
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, f"fig1_exp_{str(exp)}_{params['dependent_variable']}.png"))
    plt.close()

    #################################
    ### make the differences figure #
    ##################################
    combs = [list((comb)) for comb in combinations(SOAs, 2)]
    # MAKE FILE WITH THE INFO WE NEED
    # prepare figure
    plt.plot()
    linepos = -0.2
    saved_CI = {}
    # make a color dictionary
    this_col = {}
    for s, SOA in enumerate(SOAs):
        saved_CI[str(SOA)] = []
        this_col[str(SOA)] = e.colors[s]
    for plt_nr, comb in enumerate(combs):  # loop over the different combinations
        # now do custom permutation routine
        # get the sum of t_values of the biggest cluster for each permutation

        df1 = pd.DataFrame({'diffs_dv': list(all_conditions['diffs' + str(comb[0])]),
                            'sum_weights': list(all_conditions['sum_weights' + str(comb[0])]),
                            'condition': np.ones(res.nPP)})
        df2 = pd.DataFrame({'diffs_dv': list(all_conditions['diffs' + str(comb[1])]),
                            'sum_weights': list(all_conditions['sum_weights' + str(comb[1])]),
                            'condition': 2 * np.ones(res.nPP)})
        df_new = pd.concat([df1, df2])

        permDistr = e.permute(df_new, e.nPerm)

        # determine the threshold
        sigThres = np.percentile(permDistr, 95)
        ##########
        # 'real'data
        #######
        # do weighted sampled t-test
        [t_values, p_values] = SF.weighted_ttest_rel(all_conditions['diffs' + str(comb[0])],
                                                     all_conditions['diffs' + str(comb[1])],
                                                     all_conditions['sum_weights' + str(comb[0])], all_conditions[
                                                         'sum_weights' + str(
                                                             comb[1])])  # cond1, cond2, weights1, weights2
        # get cluster
        clusters, indx = SF.getCluster(p_values < 0.05)  # I think using this function is OK
        # caculate sum of t-vals (only for those clusters where it is lower than 0.05 )
        sigCL = [indx[i] for i in range(len(clusters)) if clusters[i][0] == True]
        sumTvals = [np.sum(abs(t_values[i])) for i in sigCL]

        # plotting
        conf95 = SF.weighPairedConf95(all_conditions['diffs' + str(comb[0])], all_conditions['diffs' + str(comb[1])],
                                      all_conditions['sum_weights' + str(comb[0])],
                                      all_conditions['sum_weights' + str(comb[1])])
        # save this to use at the end
        saved_CI[str(comb[0])].append(conf95)
        saved_CI[str(comb[1])].append(conf95)

        # plot the first one, and see where they are different from zero
        y1 = all_conditions['weighted_average' + str(comb[0])]
        plt.plot(res.timeVect, y1, this_col[str(comb[0])])

        # plot the second one
        y2 = all_conditions['weighted_average' + str(comb[1])]
        plt.plot(res.timeVect, y2, this_col[str(comb[1])])

        # add stats
        for ind, i in enumerate(sigCL):
            if sumTvals[ind] >= sigThres:  # only if it is really significant
                x = res.timeVect[sigCL[ind]]
                y = [linepos] * len(x)
                plt.plot(x, y, this_col[str(comb[0])], linewidth=5)
                plt.plot(x, y, this_col[str(comb[1])], linestyle='--', linewidth=5, dashes=[2, 2])

            linepos -= 0.025  # decrease for the next contrast

    # now add the CIs
    for s, SOA in enumerate(SOAs):
        y1 = all_conditions['weighted_average' + str(SOA)]
        conf95 = np.mean(saved_CI[str(SOA)], axis=0)
        plt.fill_between(res.timeVect, y1 - conf95, y1 + conf95, color=this_col[str(SOA)], alpha=0.5)

    e.custom_legend(['SOA: ' + str(i) + ' ms' for i in SOAs])

    ylabel = 'Salience effect'
    xlabel = 'Saccade latency (ms)'
    plt.xlim(np.min(res.timeVect), np.max(res.timeVect))
    ax1 = plt.gca()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.tick_params(axis='both', which='minor', labelsize=12)
    ax1.set_xlabel(xlabel, fontsize=18)
    ax1.set_ylabel(ylabel, size=18)
    plt.savefig(os.path.join(fig_path, f"fig2_exp_{str(exp)}_{params['dependent_variable']}.png"))
    # plt.show()
    # # saving the figure
    figure = plt.gcf()  # get current figure
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    figure.set_size_inches(10, 6)
    # plt.savefig(os.path.join(fig_path, f"fig2_exp_{str(exp)}.png"))

    plt.close()
