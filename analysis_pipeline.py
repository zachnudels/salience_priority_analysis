from Parser.EyeParser import readFile, sortDict
from Parser.parseFuncs import eyeLinkDataParser
import os
import multiprocessing
import yaml
from cleanData_Exp1 import cleanData
from ExtractRelevantData_Exp1 import extractData
from plotting import plot
from utils import extractTARS, find_files_with_extension, make_dir
import pandas as pd
import warnings

PROCESSES = 8


def parse_file(filename, parsed_dir):
    settings = readFile('eyetracker_parser_settings.json')
    eyelink = sortDict(settings['Eyelink']['par'])
    result = eyeLinkDataParser(filename, **eyelink)
    return result[0], result[1], parsed_dir


def saveParsedData(parsedData):
    last_slash_index = parsedData[0].rfind("/")+1
    saveFilePath = os.path.join(parsedData[2], (parsedData[0][last_slash_index:-4] + 'Parsed.p'))
    parsedData[1].to_pickle(saveFilePath)


def parseEyeTrackerData(raw_data_dir, parsed_dir):
    make_dir(parsed_dir)

    with multiprocessing.Pool(PROCESSES) as pool:
        results = []
        for filename in find_files_with_extension(raw_data_dir, '.asc'):
            print(filename)
            results.append(pool.apply_async(parse_file,
                                            args=(filename, parsed_dir,),
                                            callback=saveParsedData))
        for r in results:
            r.wait()


def main(params):
    subjects = extractTARS(params['directories']['root'])
    parseEyeTrackerData(params['directories']['root'], params['directories']['parsed_data'])
    excluded_participants = cleanData(params)
    final_participants = extractData(params, excluded_participants)
    plot(params)

    included_subjects = subjects
    for excluded_participant in excluded_participants:
        included_subjects.remove(excluded_participant)

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

    if included_subjects != final_participants:
        warnings.warn(f" \n\n WARNING!!! \nThe specified participants to use in analysis_settings does not correspond"
                      f" with the files found in the specified root folder."
                      f"\nIn the directory: {included_subjects}. \nIn the settings: {final_participants}"
                      f"\nPlease make sure you are using the correct participants in analysis_settings.yml."
                      f"\n WARNING!!! \n\n", category=Warning)

    if len(excluded_participants) > 0:
        warnings.warn(f" \n\n WARNING!!! \n The following participants were excluded for having less "
                      f"than 50% valid trials: \n{excluded_participants}"
                      f"\n WARNING!!! \n\n", category=Warning)


if __name__ == '__main__':
    maxCores = multiprocessing.cpu_count()
    if int(PROCESSES) > maxCores:
        PROCESSES = int(maxCores)

    with open(os.path.join(os.getcwd(), 'analysis_settings.yml'), 'r') as f_in:
        params = yaml.safe_load(f_in)

    data_dirs = ['parsed_data', 'clean_data', 'figs', 'ANOVA', 'tables']
    for data_dir in data_dirs:
        params['directories'][data_dir] = os.path.join(params['directories']['root'], data_dir)
        make_dir(params['directories'][data_dir])

    main(params)
