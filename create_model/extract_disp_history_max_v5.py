from odbAccess import openOdb
import sys
# from odbAccess import *

# import pathlib    # cannot be used on desktop pc since there python 2 is used in Abaqus
from os import listdir, path, getcwd
from os.path import isfile, join, dirname, realpath

import numpy as np
import csv
import traceback

'''
Script extracts displacement information from samling nodes and writes them into .csv file 

To execute this script on the remote desktop, cd into the directory of this script and the corresponding .odb file,
then type in
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
abaqus python extract_disp_history_max_v5.py
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
into the command line

Description:
    Script loadest newest .odb file, extracts displacement information and stores it in a csv file with the same name
    as the original .odb file

Remarks:
Variable names in the odb files are always capitalized!

written by Ryan Alberdi
modified by Max Schmitz on 04/10/2021

change log:
v3:
    - output file now contains time and data of when it was created

v4:
    - function added which automatically searches for the newest .odb file in working directory
    - output .csv file now has the same name as the .odb file
v5:
    - add progress bar to extraction - NO, does not work with Abaqus
    
'''


def get_newest_file_name(data_path, job_name='max_analysis_job', extension='.odb'):
    """
        searches the current working directory
        and returns the name of the newest csv file

        Make sure the file name is in the format:
        -> mm-dd_hh-mm-ss_max_analysis_job.odb

        args:
            - data_path: (string or Path object) specify where to look for the file
            - job_name: (string) name of the job to search for after creation date
            - extension: (optional) specifies file extension to look up
    """
    cur_path = data_path
    files_in_folder = [f for f in listdir(cur_path) if isfile(join(cur_path, f))]
    newest_creation_date = np.zeros([10, 1], dtype=np.int32).ravel()
    for file in files_in_folder:
        _, file_extension = path.splitext(file)
        if file_extension == extension:
            if file[15:15 + len(job_name)] == job_name and file.find('2dfft') == -1:
                time_idxs = []
                for idx, c in enumerate(file):
                    if c.isdigit() and idx <= 15:
                        time_idxs.append(int(idx))
                creation_date = np.array([int(file[i]) for i in time_idxs])
                for elem in zip(creation_date, newest_creation_date):
                    if elem[0] > elem[1]:
                        newest_creation_date = creation_date
                        break
                newest_file_name = file
    return newest_file_name


# ______________________ Script starts here: ______________________


def extract_displacement(d_path=None):
    """
    Extract the displacement information from Abaqus .odb file. Make sure that
     this file is run from the Abaqus Python interpreter, since extraction is
     not possible otherwise.
    """
    if d_path is None:
        d_path = getcwd()

    print('___________ extraction started ___________')

    # Specify files
    try:
        odbName = get_newest_file_name(data_path=d_path, job_name='max_analysis_job_upgraded', extension='.odb')
    except Exception as ex:
        traceback.print_exc()
        print('Cannot load odb Names. Not enough storage accessible, remove unused files!')
        odbName = None

    # odbName = '09-20_19-23-59_max_analysis_job.odb'
    print('Name of .odb file to analyze:')
    print(odbName)

    dispFileName = odbName[0:-4] + '_disp.csv'
    # import pdb; pdb.set_trace()
    dispComponent = 0  # 0-> write x-component | 1-> write y-component

    # Specify model data
    stepName = 'excitation_explicit_analysis'
    instanceName = 'PLATE_COMPLETE_MERGE-1'  # 'Plate_complete_MERGE-1'  # 'COATING_INSTANCE'
    nodeSetName = 'SAMPLING_NODES'  # 'Sampling_Nodes'  # 'SAMPLING_NODES'

    # Open output file
    odb = openOdb(odbName, readOnly=False)
    outputStep = odb.steps[stepName]
    samplingNodes = odb.rootAssembly.instances[instanceName].nodeSets[nodeSetName]

    # Write displacement history to file
    numFrames = len(outputStep.frames)
    # dispFile = open(dispFileName,'w')
    with open(dispFileName, 'w') as dispFile:
        writer = csv.writer(dispFile)
        rows = []
        for frame in range(numFrames):
            frameTime = outputStep.frames[frame].frameValue

            frameDisp = outputStep.frames[frame].fieldOutputs['U']
            samplingNodesDisp = frameDisp.getSubset(region=samplingNodes)

            row = ['%10.10E' % (
                frameTime)]  # % is inserting values from dict into string: format % values (printf-style String Formatting)
            for node in samplingNodesDisp.values:
                row.append('%10.10E' % (node.data[dispComponent]))
            rows.append(row)
        writer.writerows(rows)

    print('Output file created:')
    print(dispFileName)
    print('___________ extraction finished ___________')


if __name__ == '__main__':
    extract_displacement()
