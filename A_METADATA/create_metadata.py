import os
import sys
from datetime import datetime
from meta_functions import *

''' PYTHON SCRIPT TO ADD METADATA TEXT FILES TO EVERY DIRECTORY IN THE PROJECT.
	WHERE NO METADATA EXISTS TEMPLATE WILL BE USED TO CREATE THE FILE. 
	WHERE PREVIOUS METADATA EXISTS THE EXISTING FILE WILL BE COPIED AND MODIFIED'''

# ##RETURNS THE WORK DIRECTORY SPECIFIED IN PROJECT RUN CONFIGURATION
root = os.getcwd()
# ##FOLDERS THAT DON'T REQUIRE METADATA
avoid = ['.git', '.idea', '__pycache__', 'LATEX', 'A_METADATA']
# ##NEED META
need_meta = define_meta_tree(root=root, avoid_list=avoid)
# ##READ TEMPLATE FILE
temp_file = readlines_from_file(os.path.join(root, 'A_METADATA/metadata_template.txt'))
# ##READ IN DATA SPECIFIC TEMPLATE
data_text = readlines_from_file(os.path.join(root, 'A_METADATA/data_specific_info.txt'))
# ##GET TODAYS DATE
today = datetime.date.today()
today = today.strftime('%Y%m%d')

# ##FOR EVERY KEY (ROOT) IN NEED META_DIC CHECK IF META ALREADY EXISTS
for k, v in need_meta.items():
	files = v['FILES']
	# ##METADATA EXISTS ALREADY
	if any('METADATA_' in f for f in files):
		# ##GET LAST GENERATED FILENAME
		prev_filename, prev_date = closest_date_from_filename(files)
		# ##READ THIS FILE
		prev_file = readlines_from_file(os.path.join(k, prev_filename))
		# ##NEW FILE BASED ON PREVIOUS
		meta_from_previous(root=k,
						   today=today,
						   existing_file=prev_file,
						   pdate=prev_date,
						   filelist=files,
						   generic_data=data_text)

	# ##NO EXISTING METADATA
	else:
		# ##WRITE FILE FROM TEMPLATE
		meta_from_template(root=k,
						   today=today,
						   dir_name=v['DIRECTORY'],
						   gen_file=temp_file,
						   filenames_file=data_text,
						   filelist=files)