import numpy as np
from src.bicm import BiCM
from src.gcm import GCM
from pdb import set_trace
import time
from preprocess import make_bi


if __name__ == "__main__":
    # set_trace()
    start_time = time.time()

    # mat = np.array([[1, 1, 0, 0], [0, 1, 1, 1], [0, 0, 0, 1]])
    # mat = np.random.randint(2, size=(150,1200))

    # mat = make_bi.read_movies_lens("../dataset/ml-100k/u.data")

###########################################
    mat = np.array([[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0]])

    print mat
    print mat.shape

    gcm = GCM(mat)

    # cm = BiCM(bin_mat=mat)

    time_bef_make_bicm = time.time()
    print "make_bicm started!"
    # cm.make_bicm()
    gcm.make_cm()
    print "make_bicm took {} seconds.".format(time.time() - time_bef_make_bicm)

    print "input network:"
    print mat

    print "link probabilitie accroding to BiCM:"
    print gcm.adj_matrix

    # cm.print_max_degree_differences()
    #
    # cm.save_biadjacency(filename="../outputs/adj_mat.csv", delim='\t')
    #
    time_bef_calc_pvals = time.time()
    # cm.lambda_motifs( False, filename = "../outputs/pvalues_movies.csv", delim = '\t', binary = False)
    gcm.lambda_motifs(filename = "../outputs/pvalues_gcm.csv", delim = '\t', binary = False)
    print "lambda_motifs took {} seconds.".format(time.time() - time_bef_calc_pvals)
    #
    # # cm.lambda_motifs( False, filename = "outputs/pvalues_col.csv", delim = '\t', binary = False)
    #
    #
