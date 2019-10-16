#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 15:48:35 2019

@author: eduardo
"""
import pandas as pd
import os
import sys

citing_titles_file = "inf_users.txt"
input_abs_folder = "abs"
output_abs_folder = "abstracts/"

def read_folder_content(folder):
    return os.listdir(folder)


def read_input_file(input_file):
    pass



################
if __name__ == "__main__":
    print("[INFO] Creating the abstract texts")
    
    print("[INFO] Reading the names of the abstract files...")
    file_names_list = read_folder_content(input_abs_folder)
    print(file_names_list)
    
