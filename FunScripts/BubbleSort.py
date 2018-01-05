#!/usr/bin/python
'''Code to perform a bubble sorting algorithm'''

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

def simple_bubblesort(Rlist):
    # ensure original random list is not modified
    numbers = list(Rlist)
    print 'Bubble sort...'
    LEN = len(numbers)

    count  = 0
    for i in range(LEN-1):
        for j in range(1, LEN):
            if numbers[j-1] > numbers[j]:
                numbers[j-1], numbers[j] = numbers[j], numbers[j-1]
            #print numbers
            count +=1
    print count, 'iterations required.'

    return numbers

def fast_bubblesort(Rlist):
    '''Opmitise algorithm. If the last elements are swapped already we do not need to check again'''
    
    # ensure original random list is not modified
    numbers = list(Rlist)
    print 'Optimised bubble sort...'
    LEN = len(numbers)

    count  = 0
    swapped = False 
    for i in range(LEN-1):
        swapped = True
        for j in range(1, LEN):
            if numbers[j-1] > numbers[j]:
                numbers[j-1], numbers[j] = numbers[j], numbers[j-1]
                swapped = False
            #print numbers
            count +=1
        # break for loop if no more swappes 
        # have been performed in previous inner loop
        if swapped == True:
            break
    print count, 'iterations required.'

    return numbers


def main():
    args        = GetArgs()
    N           = args.N
    # create list of random integers
    RandInt = random.sample(range(N), N)
    print 'Random integer array:', RandInt
    simple_bubblesort(RandInt)
    fast_bubblesort(RandInt)


    return

if __name__ == "__main__":
    main()