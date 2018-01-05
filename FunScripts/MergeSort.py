#!/usr/bin/python
'''Code to perform a merge sorting algorithm'''

from __future__ import division
import numpy as np
import random
import argparse

def GetArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', '--N', type=int, \
        required=False, default=10, action='store', help='Length of random list')    
    args = parser.parse_args()
    return args

def merge(left, right):

    result = []

    # as long as the two lists are not empty:
    while len(left) > 0 and len(right)>0:
        if left[0] <= right[0]:
            result.append(left[0])
            left = left[1:]
        else:
            result.append(right[0])
            right = right[1:]

    # if one of them is empty
    while len(left) > 0:
        result.append(left[0])
        left = left[1:]
    while len(right) > 0:
        result.append(right[0])
        right = right[1:]
    print result
    return result

def split_list(List):
    left, right = [], []
    for i in range(len(List)):
        if i < len(List)/2:
            left.append(List[i])
        else:
            right.append(List[i])

    return left, right

def topdown_merge(Rlist):
    # ensure original random list is not modified
    numbers = list(Rlist)
    print 'Top down merge sort...'
    print numbers
    LEN = len(numbers)

    # if the list is less than one element: sorted already
    if LEN <= 1:
        return numbers

    left, right = split_list(numbers)

    left = topdown_merge(left)
    right = topdown_merge(right)


    return merge(left,right)

def bottomup_merge(Rlist):
    # ensure original random list is not modified
    numbers = list(Rlist)
    print 'Bottum up merge sort...'
    print numbers
    LEN = len(numbers)

    return


def main():
    args        = GetArgs()
    N           = args.N
    # create list of random integers
    RandInt = random.sample(range(N), N)
    print 'Random integer array:', RandInt
    topdown_merge(RandInt)



    return

if __name__ == "__main__":
    main()