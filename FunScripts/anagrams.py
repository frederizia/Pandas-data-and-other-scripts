#!/usr/bin/python
'''Code to find anagrams of random letters'''

from __future__ import division
import numpy as np
import argparse
import sys
import itertools

def GetArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--string', type=str,\
        required=False, default='None', action='store',\
        help = 'String of random letters for anagram')
    args = parser.parse_args()
    return args

def main():

    args        = GetArgs()
    letters     = args.string

    if letters == 'None':
        print 'Please enter some letters.'
        sys.exit(1)


    # check that letters only contrains letters

    # import dictionary
    words_file = 'google-10000-english.txt'#'words.txt'
    words = open(words_file).read().split('\n')


    # create all combinations of letters
    for i in range(2,len(letters)):
        combs = [''.join(p) for p in itertools.permutations(letters, i)]
        #print set(combs).intersection(set(words))
        for c in combs:
            if c in words:
                print 'Anagram found:', c
    

    # attach to list
    # find unique words
    # print all elements



    return


if __name__ == "__main__":
    main()