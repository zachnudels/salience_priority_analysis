import os
import yaml
import warnings

import pandas as pd

from analysis.cleanData_Exp1 import cleanData
from analysis.ExtractRelevantData_Exp1 import extractData
from analysis.parse import parseEyeTrackerData
from analysis.plotting import plot
from utils import unzipData, find_files_with_extension, make_dir


def construct_balance_tables():
    cleaned_data = []

    for filename in find_files_with_extension(params['directories']['clean_data'], '.p'):
        cleaned_data.append(pd.read_pickle(filename))

    salience_df = pd.DataFrame(columns=cleaned_data[0].DV_target_salience.value_counts().sort_index().index)
    SOA_df = pd.DataFrame(columns=cleaned_data[0].DV_ISI.value_counts().sort_index().index)

    for i in range(len(cleaned_data)):
        df = cleaned_data[i]
        salience_df = pd.concat(
            [salience_df, df[df.DK_includedTrial].DV_target_salience.value_counts().sort_index().to_frame().T],
            ignore_index=True)
        SOA_df = pd.concat([SOA_df, df[df.DK_includedTrial].DV_ISI.value_counts().sort_index().to_frame().T],
                           ignore_index=True)

    salience_df.to_csv(os.path.join(params['directories']['tables'], "salience_balances.csv"))
    SOA_df.to_csv(os.path.join(params['directories']['tables'], "SOA_balances.csv"))


def main(params):
    subjects = unzipData(params['directories']['root'])
    params['participants'] = subjects
    parseEyeTrackerData(params['directories']['root'], params['directories']['parsed_data'])
    excluded_participants = cleanData(params)
    extractData(params, excluded_participants)
    plot(params)
    construct_balance_tables()


    if len(excluded_participants) > 0:
        warnings.warn(f" \n\n WARNING!!! \n The following participants were excluded for having less "
                      f"than 50% valid trials: \n{excluded_participants}"
                      f"\n WARNING!!! \n\n", category=Warning)




if __name__ == '__main__':
    with open(os.path.join(os.getcwd(), 'analysis_settings.yml'), 'r') as f_in:
        params = yaml.safe_load(f_in)

    data_dirs = ['parsed_data', 'clean_data', 'figs', 'ANOVA', 'tables']
    for data_dir in data_dirs:
        params['directories'][data_dir] = os.path.join(params['directories']['root'], data_dir)
        make_dir(params['directories'][data_dir])

    main(params)
