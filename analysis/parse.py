import multiprocessing
import os

from Parser.EyeParser import readFile, sortDict
from Parser.parseFuncs import eyeLinkDataParser
from utils import find_files_with_extension, make_dir


def parse_file(filename, parsed_dir, eyelink):
    result = eyeLinkDataParser(filename, **eyelink)
    return result[0], result[1], parsed_dir


def saveParsedData(parsedData):
    last_slash_index = parsedData[0].rfind(os.sep)+1
    saveFilePath = os.path.join(parsedData[2], (parsedData[0][last_slash_index:-4] + 'Parsed.p'))
    parsedData[1].to_pickle(saveFilePath)


def parseEyeTrackerData(raw_data_dir, parsed_dir):
    make_dir(parsed_dir)
    settings = readFile('eyetracker_parser_settings.json')
    eyelink = sortDict(settings['Eyelink']['par'])

    processes = settings["Processes"]
    maxCores = multiprocessing.cpu_count()
    if int(processes) > maxCores:
        processes = int(maxCores)

    with multiprocessing.Pool(processes) as pool:
        results = []
        for filename in find_files_with_extension(raw_data_dir, '.asc'):
            if filename[filename.rfind(os.sep)+1] == ".":
                continue
            print(filename)
            results.append(pool.apply_async(parse_file,
                                            args=(filename, parsed_dir, eyelink),
                                            callback=saveParsedData))
        for r in results:
            r.wait()
