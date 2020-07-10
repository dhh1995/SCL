#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : list_fails.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
# Date   : 02/02/2020
#
# Distributed under terms of the MIT license.

'''
To visualize the examples as images.

# Usage
python3 list_fails.py $fail_cases.json
'''

import argparse
import json
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('fail_file', type=str,
    help='the file contains the list of fail cases')
parser.add_argument('--keys', '-k', type=str, nargs='+',
    default=['pred', 'label'], help='the keys to be print')
args = parser.parse_args()


def main():
    with open(args.fail_file, 'r') as f:
        counter = 0
        while True:
            line = f.readline()
            if len(line) == 0:
                break
            if counter > 0:
                print('continue?')
                answer = input()
                if answer.startswith('y'):
                    continue
                else:
                    break
            fail_cases = json.loads(line)
            for i in fail_cases['fail_cases']:
                print('case_prefix: {}'.format(i['file_prefix']))
                message = ''
                for k in args.keys:
                    if len(message) > 0:
                        message += ', '
                    message += '{}:{}'.format(k, i[k])
                print(message)
            counter += 1


if __name__ == '__main__':
    main()
