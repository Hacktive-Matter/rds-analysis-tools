import quantify_analysis as qa
import os, re
import pandas as pd

def read_folder(folder, substring, substring2, code_function = None, condition = None):
    files = []
    for dir, _, filenames in os.walk(folder):
        if not substring in dir:
            continue
        for filename in filenames:
            if code_function and condition:
                match code_function:
                    case 1: 
                        options = find_code(condition)
                    case 2: 
                        options = find_code_2(condition)[0]
                if substring2 in filename and any([code in dir for code in options]):
                    files.append(os.path.join(dir, filename))
            else:
                if substring2 in filename:
                    files.append(os.path.join(dir, filename))
    return files


def find_code(condition):
    dfo = pd.read_excel('names.xlsx', sheet_name='filenames classes speeds')
    df = dfo[dfo['Class'] == condition]
    video_list = df['Video'].to_list()
    video_list = [str(int(video))[:-2] + '_' + str(int(video))[-2:] for video in video_list if str(video) != 'nan']
    return video_list

def find_code_2(condition):
    def check_critical_connected(xi1, xi2):
        if xi2 > 300: return "Critically Connected"
        elif xi1 > 1000: return "Global Contraction"
        else: return "Local Contraction"
    def check_name(input_branch):
        return input_branch.split('\\')[0]

    dfo = pd.read_excel('JoseSamples.xlsx', sheet_name='Sheet1 (2)')
    cleaned_dfo = dfo[dfo["xlink_type"] == "fasc"]
    cleaned_dfo = cleaned_dfo[cleaned_dfo["myo_t"] == "1/100"]
    cleaned_dfo = cleaned_dfo.dropna(axis=0, subset="xi")
    cleaned_dfo = cleaned_dfo.dropna(axis=0, subset="chi")
    cleaned_dfo = cleaned_dfo.dropna(axis=0, subset=["um_per_pixel", "seconds_per_frame"]) # Remove folders with no um-pixel ratio
    cleaned_dfo['ctype'] = cleaned_dfo.apply(lambda x: check_critical_connected(x['xi'], x['chi']), axis = 1)
    cleaned_dfo['new_id'] = cleaned_dfo["input_branch_myosin"].apply(check_name)
    final_dfo = cleaned_dfo[cleaned_dfo['ctype'] == condition]

    return final_dfo['new_id'].to_list(), final_dfo['um_per_pixel'].to_list(), final_dfo['seconds_per_frame'].to_list()


codes = ["5_13+kn+my_002", "10_16_M-M+kn+my_006", "4_10_A-A+kn_001"]