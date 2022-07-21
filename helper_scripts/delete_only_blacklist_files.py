import os
import pathlib
from os import listdir, path
from os.path import isfile, join

white_list = ['.py']

cur_path = pathlib.Path().absolute()
files_in_folder = [f for f in listdir(cur_path) if isfile(join(cur_path, f))]
import pdb; pdb.set_trace()
for file in files_in_folder:
    _, file_extension = path.splitext(file)
    if file_extension not in white_list:
        os.remove(file)