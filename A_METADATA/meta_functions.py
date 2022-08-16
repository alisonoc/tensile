import os
import sys
import datetime

# ##RETURNS THE WORK DIRECTORY SPECIFIED IN PROJECT RUN CONFIGURATION
root = os.getcwd()
# ##GET TODAYS DATE
today = datetime.date.today()
today = today.strftime('%Y%m%d')


def define_meta_tree(root, avoid_list):
    """ FUNCTION TO RECURSIVELY MOVE THROUGH DIRECTORY TREE
    REMOVING ANY DIRECTORIES THAT ARE IN AVOID LIST
    ADDING OTHER DIRECTORIES TO DICTIONARY"""
    dic = {}
    for r, d, f in os.walk(root):
        if not any(av in r for av in avoid_list):
            dic[r] = {'DIRECTORY': r[r.rfind('\\') + 1:],
                      'FILES': f}
    return dic


def readlines_from_file(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
    file.close()
    return lines


def meta_from_template(root, today, dir_name, gen_file, filenames_file, filelist):
    # ##OPEN NEW FILE
    with open(os.path.join(root, 'METADATA_' + today + '.txt'), 'w') as file:
        # ##ADD GENERAL DETAILS
        for ind, line in enumerate(gen_file):
            if '%s' in line:
                if (ind == 0) or (ind == 23):
                    line = line % (today)
                elif ind == 7:
                    line = line % (dir_name)
                file.write(line)
            else:
                file.write(line)
        # ##ADD FILE DETAILS
        for numfile, cfile in enumerate(filelist):
            # ##FILES THAT DO NOT NEED TO BE ADDED
            if not 'METADATA' in cfile:
                for ind, line in enumerate(filenames_file):
                    if '%s' in line:
                        line = line % (cfile)
                        file.write(line)
                    else:
                        file.write(line)
    file.close()


def closest_date_from_filename(list):
    dates = []
    for f in list:
        # ##IF FILENAME INCLUDES METADATA BUT NOT TODAYS DATE THEN ADD TO LIST
        if ('METADATA_' in f) and not(datetime.datetime.today().strftime('%Y%m%d') in f):
            dates.append(datetime.datetime.strptime(f[f.rfind('_') + 1:-4], '%Y%m%d').date())
    # ##GET THE MAXIMUM DATE (i.e. the last generated metadata file
    mdate = max(dates)
    # ##RETURN THE RELEVANT FILENAME
    filename = [f for f in list if mdate.strftime('%Y%m%d') in f][0]
    return filename, mdate


def check_four_prows(check, linenum):
    # ##DEFINE OUR SEARCH PARAM
    search = [
        '########################################################################################################\n',
        'DATA-SPECIFIC INFORMATION FOR:\n',
        '########################################################################################################\n']
    if check == search:
        return linenum
    else:
        return None


def meta_from_previous(root, today, existing_file, pdate, filelist, generic_data):
    # ##CREATE STRINGS FROM DATES
    str_pdate = pdate.strftime('%Y%m%d')
    # ##OPEN NEW FILE
    with open(os.path.join(root, 'METADATA_' + today + '.txt'), 'w') as file:
        # ##FOR EVERY LINE IN THE EXISTING FILE
        for line, text in enumerate(existing_file):
            # ##REPLACE THE DATE
            if str_pdate in text:
                file.write(text.replace(str_pdate, today))
            # ##MAKE NO CHANGES
            else:
                file.write(text)
            # ##IDENTIFY RELEVANT AREA TO ADD NEW
            if line >= 4:
                prows = existing_file[line - 4:line]
                dataline = check_four_prows(prows, line)
                if dataline:
                    red_list = [f for f in filelist if not 'METADATA' in f]
                    for fname in red_list:
                        for line2, text2 in enumerate(generic_data):
                            if '%s' in text2:
                                file.write(fname)
                            else:
                                file.write(text2)
                    # ##BREAK OUT OF LOOP
                    break
    file.close()
