import os
import re

def read_file(file_name):
    current_path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(current_path, 'customer', file_name)
    cnt = 0
    observation_list = {}
    flag = True
    with open(path, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            if not flag and line.startswith('#'):
                break
            if flag and line.startswith('#'):
                continue
            if line.startswith(' ') or line.startswith('\n'):
                observation_list[cnt] = []
                cnt += 1
                flag = False
            else:
                tmp = re.split(r'[\,]', line.rstrip())
                observation_list[cnt] = tmp
                cnt += 1
                flag = False
    return observation_list
