import os

'''
Script calls a command line task and executes it.

Goal: using the command line automated through Python
'''

ans = os.system('python calculate_group_velocity.py')

print(f'ans = {ans}')

'''
ans = 0 if task was successful
ans = 1 if error occured in calculate_group_velocity.py
ans = 2 if error in task (no such file or directory)
'''