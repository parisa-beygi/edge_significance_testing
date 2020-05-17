# This module reads a .csv file and outputs the corresponding bipartite graph in the form of biadjacency matrix

import csv
import numpy as np


def read_movies_lens(file_dir):
    bin_mat = np.zeros((943, 1682))

    f = open(file_dir, 'r')
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        if int(row[2]) >= 3:
            bin_mat[int(row[0])-1, int(row[1])-1] = 1

    # bin_mat[0] = np.zeros(1682)
    for i in range(bin_mat.shape[0]):
        if all(v == 0 for v in bin_mat[i]):
            print "Zero row", i


    # print bin_mat
    # print bin_mat.shape
    return bin_mat



