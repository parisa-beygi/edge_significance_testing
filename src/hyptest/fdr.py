from multipy.fdr import lsu
# from multipy.data import neuhaus

from numpy import genfromtxt
import time

start_read = time.time()
print 'reading pvals from file ...'

pvals = genfromtxt('../../outputs/pvalues_gcm.csv', delimiter=',')

print '>>>>>>   Data read in {} seconds'.format(time.time() - start_read)

# pvals = neuhaus()
print pvals
print "type(pvals): ", type(pvals)
print len(pvals)

start_testing = time.time()
significant_pvals = lsu(pvals, q=0.4)
print '>>>>>>   Hypotheses tested in {} seconds'.format(time.time() - start_testing)

print "type(significant_pvals): ", type(significant_pvals)
print significant_pvals


# print(zip(['{:.4f}'.format(p) for p in pvals], significant_pvals))
