import os
import json


def get_folder(instrument,root_path):
    #instrument must be a string including "cello" or "violin"
    if type(instrument)!=str or instrument.lower() not in ['cello','violin']:
        raise TypeError('The given variable "instrument" must be a string including "cello" or "violin"')
    else:
        folder_names = []
        file_names = os.listdir(root_path)
        for file in file_names:
            if os.path.isdir(root_path + os.sep + file):
                folder_names.append(file)
        folder_names.sort()
    return folder_names


def get_inform(folder_name,root_path):
    summary_jsonpath = os.path.abspath(f'{root_path}{os.sep}{folder_name}{os.sep}{folder_name}_summary.json')
    with open(summary_jsonpath,'r') as f:
        summary = json.load(f) 
    return summary,summary_jsonpath
