import os
import itertools
import json
import numpy as np


def product_dic(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    a = {}
    for num, item in enumerate(itertools.product(*vals)):
        a[num] = dict(zip(keys, item))
    return a


def merge_dicts(dic1, dic2):
    z = dic1.copy()
    z.update(dic2)
    return z

def write_json_file(dic, pth, filename):
    """ WRITE TO JSON FILE RETURN FILE PATH"""
    with open(os.path.join(pth, filename), 'w') as file:
        json.dump(dic, file, indent=4)
    return os.path.join(pth, filename)

def load_json_file(j_file):
    with open(j_file) as json_file:
        return json.load(json_file)

def write_text_file(data=None, root=os.getcwd(), filename=None):
    """ WRITE TO JSON FILE RETURN FILE PATH"""
    with open(os.path.join(root, filename), 'w') as file:
        for item in data:
            file.write('%s\n'%(item))
    return os.path.join(root, filename)

def read_text_file(root=os.getcwd(), filename=None):
    """ READ TEXT FILE LINE BY LINE"""
    with open(os.path.join(root, filename), 'r') as file:
        output = file.readlines()
        return [i.splitlines()[0] for i in output]

def new_dir_add_dic(dic=None,
                    key=None,
                    path=None,
                    dir_name=None,
                    exist_ok=False):
    """
    CREATE A NEW DIRECTORY AND ADD TO PATH DICTIONARY RETURN UPDATED DICTIONARY

    :param dic: dictionary of existing paths
    :param key: KEY TO BE ASSIGNED TO THE NEW FOLDER
    :param path: PATH WHERE FOLDER IS TO BE LOCATED
    :param dir_name: NAME OF THE NEW DIRECTORY
    :param exist_ok: OVERWRITE EXISITNG FILE? DEFAULT IS FALSE
    :return: UPDATED DICTIONARY OF PATHS
    """

    full_path = os.path.join(path, dir_name.upper())
    # ##MAKE THE DIRECTORY
    try:
        os.makedirs(full_path, exist_ok=exist_ok)
    except FileExistsError:
        name = file_exists_error(full_path, path)
        full_path = os.path.join(path, name)
        os.makedirs(full_path, exist_ok=True)
    # ##ADD THE PATH TO PATH DIC
    dic.update({key: full_path})
    return dic


def file_exists_error(path, root):
    """ IF DIRECTORY ALREADY EXISTS ADD A NUMBER"""
    # ##CHECK IF DATE FOLDER IS EMPTY
    files = os.listdir(path)
    # ## IF FILES WE NEED NEW DIRECTORY
    if files:
        # ##GET FOLDER NAME (CURRENT DATE)
        base_name = path[path.rfind('\\') + 1:]
        # ##GET ALL FOLDERS IN THE ROOT DIRECTORY (ONE LEVEL UP FROM DATE FOLDER)
        folder_list = [d for r, d, f in os.walk(root) if base_name in d][0]
        # ##SEARCH FOR GREATEST EXISTING INDEXER
        index_list = [int(f[f.rfind('_') + 1:]) for f in folder_list if '_' in f and base_name in f]
        if index_list:
            max_index = max(index_list)
            # ## LATEST FOLDER
            name = base_name + '_' + str(max_index)
            latest_folder = os.path.join(root, name)
            # ##CHECK FOR FILES IN THE LATEST FOLDER (I.E. HIGHEST _X FOLDER)
            files = os.listdir(latest_folder)
            # ##IF THERE ARE FILES WE NEED A NEW FOLDER
            if files:
                val = max_index + 1
                name = base_name + '_' + str(val)
        else:
            # ##NEW NAME
            name = base_name + '_1'
    else:
        name = path[path.rfind('\\') + 1:]
    return name

def bounds_select_features(bounds_path=os.getcwd(),
                           feature_list=None):
    """ FUNCTION TO RETURN MODIFIED BOUNDS
    BASED ON FEATURE SELECTION.
    1. ONLY SELECTED FEATURES WILL ITERATE
    2. UNSELECTED FEATURES WILL RETURN MEAN VALUE
    """
    pbounds = load_json_file(bounds_path)
    stable = {}
    ndic={}
    for k, v in pbounds.items():
        if k in feature_list:
            ndic[k] = v
        else:
            stable[k] = np.round(sum(v)/len(v), 4)
    return ndic, stable

def percentage_dic(percentage=None, **dic):
    """ FUNCTION TO ITERATE DICTIONARY
    KEYS AND MODIFY BOUNDS TO BE PERCENTAGE OF MEAN VALUE"""
    new_dic={}
    for k, v in dic.items():
        # ##GET MEAN VALUE
        mean = np.mean(v)
        # ##FIND THE PERCENTAGE MOVEMENT
        perc = mean * (percentage/100)
        # ## GET NEW MIN MAX ADD TO DIC
        new_dic[k] = (mean - perc, mean + perc)
    return new_dic