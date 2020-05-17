import ctypes
import multiprocessing
import scipy.optimize as opt
import numpy as np
from poibin.poibin import PoiBin
from pdb import set_trace



class GCM(object):
    """General Configuration Model for undirected networks.

    This class implements the General Configuration Model (GCM), which can
    be used as a null model for the analysis of undirected
    networks. The class provides methods for calculating the adjacency matrix
    of the null model and for quantifying node similarities in terms of
    p-values.
    """

    def __init__(self, bin_mat):
        """Initialize the parameters of the GCM.

        :param input_adj_mat: binary input matrix describing the adjacency matrix
                of an undirected graph.
        :type bin_mat: numpy.array
        """
        self.bin_mat = np.array(bin_mat, dtype=np.int64)
        self.check_input_matrix_is_binary()
        self.check_input_matrix_is_symmetric()
        self.num_nodes = self.bin_mat.shape[0]
        self.dseq = self.set_degree_seq()
        self.dim = self.dseq.size
        self.sol = None             # solution of the equation system
        self.adj_matrix = None      # biadjacency matrix of the null model
        self.input_queue = None     # queue for parallel processing
        self.output_queue = None    # queue for parallel processing


    def check_input_matrix_is_binary(self):
        """Check that the input matrix is binary, i.e. entries are 0 or 1.

        :raise AssertionError: raise an error if the input matrix is not
            binary
        """
        assert np.all(np.logical_or(self.bin_mat == 0, self.bin_mat == 1)), \
            "Input matrix is not binary."


    def check_input_matrix_is_symmetric(self):
        assert np.all(self.bin_mat == self.bin_mat.T), "Input matrix is not symmetric"


    def set_degree_seq(self):
        """Return the node degree sequence of the input matrix.

        :returns: node degree sequence [node degrees]
        :rtype: numpy.array

        :raise AssertionError: raise an error if the length of the returned
            degree sequence does not correspond to the total number of nodes
        """
        set_trace()
        # dseq = np.empty(self.num_nodes)
        dseq = np.squeeze(np.sum(self.bin_mat, axis=0))
        assert dseq.size == (self.num_nodes)
        return dseq


    def make_cm(self, x0=None, method='hybr', jac=None, tol=None,
                  callback=None, options=None):
        self.sol = self.solve_equations(x0=x0, method=method, jac=jac, tol=tol,
                                        callback=callback, options=options)
        set_trace()
        # create CM adjacency matrix:
        self.adj_matrix = self.get_adjacency_matrix(self.sol.x)
        # self.print_max_degree_differences()
        # assert self.test_average_degrees(eps=1e-2)



    def solve_equations(self, x0=None, method='hybr', jac=None, tol=None,
                        callback=None, options=None):

        # use Jacobian if the hybr solver is chosen
        if method is 'hybr':
            jac = self.jacobian

        # set initial conditions
        if x0 is None:
            x0 = self.dseq / np.sqrt(np.sum(self.dseq))
        else:
            if not len(x0) == self.dim:
                msg = "One initial condition for each parameter is required."
                raise ValueError(msg)

        # solve equation system
        sol = opt.root(fun=self.equations, x0=x0, method=method, jac=jac,
                       tol=tol, options=options, callback=callback)

        # check whether system has been solved successfully
        print "Solver successful:", sol.success
        print sol.message
        if not sol.success:
            errmsg = "Try different initial conditions and/or a" + \
                     "different solver, see documentation at " + \
                     "https://docs.scipy.org/doc/scipy-0.19.0/reference/" + \
                     "generated/scipy.optimize.root.html"
            print errmsg
        return sol

    def equations(self, xx):
        """Return the equations of the log-likelihood maximization problem.

        Note that the equations for the row-nodes depend only on the
        column-nodes and vice versa, see [Saracco2015]_.

        :param xx: Lagrange multipliers which have to be solved
        :type xx: numpy.array
        :returns: equations to be solved (:math:`f(x) = 0`)
        :rtype: numpy.array
        """
        eq = -self.dseq
        for i in xrange(0, self.dim):
            for j in xrange(0, self.dim):
                if j != i:
                    dum = xx[i] * xx[j] / (1. + xx[i] * xx[j])
                    eq[i] += dum
        return eq

    def jacobian(self, xx):
        """Return a NumPy array with the Jacobian of the equation system.

        :param xx: Lagrange multipliers which have to be solved
        :type xx: numpy.array
        :returns: Jacobian
        :rtype: numpy.array
        """
        jac = np.zeros((self.dim, self.dim))
        for i in xrange(0, self.dim):
            for j in xrange(0, self.dim):
                if j != i:
                    xxi = xx[i] / (1.0 + xx[i] * xx[j]) ** 2
                    xxj = xx[j] / (1.0 + xx[i] * xx[j]) ** 2
                    jac[i, i] += xxj
                    jac[i, j] = xxi
        return jac

    def get_adjacency_matrix(self, xx):
        set_trace()

        """ Calculate the adjacency matrix of the null model.

        The adjacency matrix describes the CM null model, i.e. the optimal
        average graph :math:`<G>^*` with the average link probabilities
        :math:`<G>^*_{rc} = p_{rc}` ,
        :math:`p_{rc} = \\frac{x_r \\cdot x_c}{1 + x_r\\cdot x_c}.`
        :math:`x` are the solutions of the equation system which has to be
        solved for the null model.
        Note that :math:`r` and :math:`c` are taken from opposite bipartite
        node sets, thus :math:`r \\neq c`.

        :param xx: solutions of the equation system (Lagrange multipliers)
        :type xx: numpy.array
        :returns: adjacency matrix of the null model
        :rtype: numpy.array

        :raises ValueError: raise an error if :math:`p_{rc} < 0` or
            :math:`p_{rc} > 1` for any :math:`r, c`
        """

        mat = np.zeros((self.dim, self.dim))
        for i in range(self.dim):
            for j in range(i+1, self.dim):
                pij = xx[i] * xx[j] / (1 + xx[i] * xx[j])
                mat[i,j] = pij
                mat[j,i] = pij

        # account for machine precision:
        mat += np.finfo(np.float).eps
        if np.any(mat < 0):
            errmsg = 'Error in get_adjacency_matrix: probabilities < 0 in ' \
                     + str(np.where(mat < 0))
            raise ValueError(errmsg)
        elif np.any(mat > (1. + np.finfo(np.float).eps)):
            errmsg = 'Error in get_adjacency_matrix: probabilities > 1 in' \
                     + str(np.where(mat > 1))
            raise ValueError(errmsg)
        assert mat.shape == self.bin_mat.shape, \
            "Adjacency matrix has wrong dimensions."
        return mat


    def lambda_motifs(self, parallel=True, filename=None,
            delim='\t', binary=True, num_chunks=4):
        """Calculate and save the p-values of the :math:`\\Lambda`-motifs.

        For each node couple in the bipartite layer specified by ``bip_set``,
        calculate the p-values of the corresponding :math:`\\Lambda`-motifs
        according to the link probabilities in the biadjacency matrix of the
        BiCM null model.

        The results can be saved either as a binary ``.npy`` or a
        human-readable ``.csv`` file, depending on ``binary``.

        .. note::

            * The total number of p-values that are calculated is split into
              ``num_chunks`` chunks, which are processed sequentially in order
              to avoid memory allocation errors. Note that a larger value of
              ``num_chunks`` will lead to less memory occupation, but comes at
              the cost of slower processing speed.

            * The output consists of a one-dimensional array of p-values. If
              the bipartite layer ``bip_set`` contains ``n`` nodes, this means
              that the array will contain :math:`\\binom{n}{2}` entries. The
              indices ``(i, j)`` of the nodes corresponding to entry ``k`` in
              the array can be reconstructed using the method
              :func:`BiCM.flat2_triumat_idx`. The number of nodes ``n``
              can be recovered from the length of the array with
              :func:`BiCM.flat2_triumat_dim`

            * If ``binary == False``, the ``filename`` should end with
              ``.csv``. If ``binary == True``, it will be saved in binary NumPy
              ``.npy`` format and the suffix ``.npy`` will be appended
              automatically. By default, the file is saved in binary format.

        :param bip_set: select row-nodes (``True``) or column-nodes (``False``)
        :type bip_set: bool
        :param parallel: select whether the calculation of the p-values should
            be run in parallel (``True``) or not (``False``)
        :type parallel: bool
        :param filename: name of the output file
        :type filename: str
        :param delim: delimiter between entries in the ``.csv``file, default is
            ``\\t``
        :type delim: str
        :param binary: if ``True``, the file will be saved in the binary
            NumPy format ``.npy``, otherwise as ``.csv``
        :type binary: bool
        :param num_chunks: number of chunks of p-value calculations that are
            performed sequentially
        :type num_chunks: int
        :raise ValueError: raise an error if the parameter ``bip_set`` is
            neither ``True`` nor ``False``
        """
        set_trace()

        adj_mat = self.adj_matrix
        bin_mat = self.bin_mat

        n = self.triumat2flat_dim(self.num_nodes)
        pval = np.ones(shape=(n, ), dtype='float') * (-0.1)

        # handle layers of dimension 2 separately
        if n == 1:
            nlam = np.dot(bin_mat[0, :], bin_mat[1, :].T)
            plam = adj_mat[0, :] * adj_mat[1, :]
            pb = PoiBin(plam)
            pval[0] = pb.pval(nlam)
        else:
            # if the dimension of the network is too large, split the
            # calculations # of the p-values in ``m`` intervals to avoid memory
            # allocation errors
            if n > 100:
                kk = self.split_range(n, m=num_chunks)
            else:
                kk = [0]
            # calculate p-values for index intervals
            for i in range(len(kk) - 1):
                k1 = kk[i]
                k2 = kk[i + 1]
                nlam = self.get_lambda_motif_block(bin_mat, k1, k2)
                plam = self.get_plambda_block(adj_mat, k1, k2)
                pv = self.get_pvalues_q(plam, nlam, k1, k2)
                pval[k1:k2] = pv
            # last interval
            k1 = kk[len(kk) - 1]
            k2 = n - 1
            nlam = self.get_lambda_motif_block(bin_mat, k1, k2)
            plam = self.get_plambda_block(adj_mat, k1, k2)
            # for the last entry we have to INCLUDE k2, thus k2 + 1
            pv = self.get_pvalues_q(plam, nlam, k1, k2 + 1)
            pval[k1:] = pv
        # check that all p-values have been calculated
#        assert np.all(pval >= 0) and np.all(pval <= 1)
        if filename is None:
            fname = 'p_values_gcm'
            if not binary:
                fname +=  '.csv'
        else:
            fname = filename
        # account for machine precision:
        pval += np.finfo(np.float).eps
        self.save_array(pval, filename=fname, delim=delim,
                         binary=binary)


    def get_lambda_motif_block(self, mm, k1, k2):
        set_trace()
        """Return a subset of :math:`\\Lambda`-motifs as observed in ``mm``.

        Given the binary input matrix ``mm``, count the number of
        :math:`\\Lambda`-motifs for all the node couples specified by the
        interval :math:`\\left[k_1, k_2\\right[`.


        .. note::

            * The :math:`\\Lambda`-motifs are counted between the **row-nodes**
              of the input matrix ``mm``.

            * If :math:`k_2 \equiv \\binom{mm.shape[0]}{2}`, the interval
              becomes :math:`\\left[k_1, k_2\\right]`.

        :param mm: binary matrix
        :type mm: numpy.array
        :param k1: lower interval limit
        :type k1: int
        :param k2: upper interval limit
        :type k2: int
        :returns: array of observed :math:`\\Lambda`-motifs
        :rtype: numpy.array
        """
        ndim = mm.shape[0]
        # if the upper limit is the largest possible index, i.e. corresponds to
        # the node couple (ndim - 2, ndim - 1), where node indices start from 0,
        # include the result
        if k2 == (ndim * (ndim - 1) / 2 - 1):
            flag = 1
        else:
            flag = 0
        aux = np.ones(shape=(k2 - k1 + flag, )) * (-1) # -1 as a test
        [i1, j1] = self.flat2triumat_idx(k1, ndim)
        [i2, j2] = self.flat2triumat_idx(k2, ndim)

        # if limits have the same row index
        if i1 == i2:
            aux[:k2 - k1] = np.dot(mm[i1, :], mm[j1:j2, :].T)
        # if limits have different row indices
        else:
            k = 0
            # get values for lower limit row
            fi = np.dot(mm[i1, :], mm[j1:, :].T)
            aux[:len(fi)] = fi
            k += len(fi)
            # get values for intermediate rows
            for i in range(i1 + 1, i2):
                mid = np.dot(mm[i, :], mm[i + 1:, :].T)
                aux[k : k + len(mid)] = mid
                k += len(mid)
            # get values for upper limit row
            if flag == 1:
                aux[-1] = np.dot(mm[ndim - 2, :], mm[ndim - 1, :].T)
            else:
                la =  np.dot(mm[i2, :], mm[i2 + 1 : j2, :].T)
                aux[k:] = la
        return aux


    def get_plambda_block(self, adj_mat, k1, k2):
        """Return a subset of the :math:`\\Lambda` probability matrix.

        Given the biadjacency matrix ``biad_mat`` with
        :math:`\\mathbf{M}_{rc} = p_{rc}`, which describes the probabilities of
        row-node ``r`` and column-node ``c`` being linked, the method returns
        the matrix

        :math:`P(\\Lambda)_{ij} = \\left(M_{i\\alpha_1} \\cdot M_{j\\alpha_1},
        M_{i\\alpha_2} \\cdot M_{j\\alpha_2}, \\ldots\\right),`

        for all the node couples in the interval
        :math:`\\left[k_1, k_2\\right[`.  :math:`(i, j)` are two **row-nodes**
        of ``biad_mat`` and :math:`\\alpha_k` runs over the nodes in the
        opposite layer.

        .. note::

            * The probabilities are calculated between the **row-nodes** of the
              input matrix ``biad_mat``.

            * If :math:`k_2 \equiv \\binom{biad\\_mat.shape[0]}{2}`, the
              interval becomes :math:`\\left[k1, k2\\right]`.

        :param biad_mat: biadjacency matrix
        :type biad_mat: numpy.array
        :param k1: lower interval limit
        :type k1: int
        :param k2: upper interval limit
        :type k2: int
        :returns: :math:`\\Lambda`-motif probability matrix
        :rtype: numpy.array
        """
        set_trace()
        [ndim1, ndim2] = adj_mat.shape
        # if the upper limit is the largest possible index, i.e. corresponds to
        # the node couple (ndim - 2, ndim - 1), where node indices start from 0,
        # include the result
        if k2 == (ndim1 * (ndim1 - 1) / 2 - 1):
            flag = 1
        else:
            flag = 0
        paux = np.ones(shape=(k2 - k1 + flag, ndim2), dtype='float') * (-0.1)
        [i1, j1] = self.flat2triumat_idx(k1, ndim1)
        [i2, j2] = self.flat2triumat_idx(k2, ndim1)

        # if limits have the same row index
        if i1 == i2:
            paux[:k2 - k1, :] = adj_mat[i1, ] * adj_mat[j1:j2, :]
        # if limits have different indices
        else:
            k = 0
            # get values for lower limit row
            fi = adj_mat[i1, :] * adj_mat[j1:, :]
            paux[:len(fi), :] = fi
            k += len(fi)
            # get values for intermediate rows
            for i in range(i1 + 1, i2):
                mid = adj_mat[i, :] * adj_mat[i + 1:, :]
                paux[k : k + len(mid), :] = mid
                k += len(mid)
            # get values for upper limit row
            if flag == 1:
                paux[-1, :] = adj_mat[ndim1 - 2, :] * adj_mat[ndim1 - 1, :]
            else:
                la = adj_mat[i2, :] * adj_mat[i2 + 1:j2, :]
                paux[k:, :] = la
        return paux


    def get_pvalues_q(self, plam_mat, nlam_mat, k1, k2, parallel=True):
        """Calculate the p-values of the observed :math:`\\Lambda`-motifs.

        For each number of :math:`\\Lambda`-motifs in ``nlam_mat`` for the node
        interval :math:`\\left[k1, k2\\right[`, construct the Poisson Binomial
        distribution using the corresponding
        probabilities in ``plam_mat`` and calculate the p-value.

        :param plam_mat: array containing the list of probabilities for the
            single observations of :math:`\\Lambda`-motifs
        :type plam_mat: numpy.array (square matrix)
        :param nlam_mat: array containing the observations of
            :math:`\\Lambda`-motifs
        :type nlam_mat: numpy.array (square matrix)
        :param k1: lower interval limit
        :type k1: int
        :param k2: upper interval limit
        :type k2: int
        :param parallel: if ``True``, the calculation is executed in parallel;
            if ``False``, only one process is started
        :type parallel: bool
        """
        set_trace()
        n = len(nlam_mat)
        # the array must be sharable to be accessible by all processes
        shared_array_base = multiprocessing.Array(ctypes.c_double, n)
        pval_mat = np.frombuffer(shared_array_base.get_obj())

        # number of processes running in parallel has to be tested.
        # good guess is multiprocessing.cpu_count() +- 1
        if parallel:
            num_procs = multiprocessing.cpu_count() - 1
        elif not parallel:
            num_procs = 1
        else:
            num_procs = 1
        self.input_queue = multiprocessing.Queue()
        self.output_queue = multiprocessing.Queue()

        p_inqueue = multiprocessing.Process(target=self.add2inqueue,
                                            args=(num_procs, plam_mat, nlam_mat,
                                                k1, k2))
        p_outqueue = multiprocessing.Process(target=self.outqueue2pval_mat,
                                             args=(num_procs, pval_mat))
        ps = [multiprocessing.Process(target=self.pval_process_worker,
                                      args=()) for i in range(num_procs)]
        # start queues
        p_inqueue.start()
        p_outqueue.start()
        # start processes
        for p in ps:
            p.start()       # each process has an id, p.pid
        p_inqueue.join()
        for p in ps:
            p.join()
        p_outqueue.join()
        return pval_mat

    def add2inqueue(self, nprocs, plam_mat, nlam_mat, k1, k2):
        """Add elements to the in-queue to calculate the p-values.

        :param nprocs: number of processes running in parallel
        :type nprocs: int
        :param plam_mat: array containing the list of probabilities for the
            single observations of :math:`\\Lambda`-motifs
        :type plam_mat: numpy.array (square matrix)
        :param nlam_mat: array containing the observations of
            :math:`\\Lambda`-motifs
        :type nlam_mat: numpy.array (square matrix)
        :param k1: lower interval limit
        :type k1: int
        :param k2: upper interval limit
        :type k2: int
        """
        n = len(plam_mat)
        # add tuples of matrix elements and indices to the input queue
        for k in xrange(k1, k2):
            self.input_queue.put((k - k1, plam_mat[k - k1, :],
                                  nlam_mat[k - k1]))

        # add as many poison pills "STOP" to the queue as there are workers
        for i in xrange(nprocs):
            self.input_queue.put("STOP")

    def pval_process_worker(self):
        """Calculate p-values and add them to the out-queue."""
        # take elements from the queue as long as the element is not "STOP"
        for tupl in iter(self.input_queue.get, "STOP"):
            pb = PoiBin(tupl[1])
            pv = pb.pval(int(tupl[2]))
            # add the result to the output queue
            self.output_queue.put((tupl[0], pv))
        # once all the elements in the input queue have been dealt with, add a
        # "STOP" to the output queue
        self.output_queue.put("STOP")

    def outqueue2pval_mat(self, nprocs, pvalmat):
        """Put the results from the out-queue into the p-value array."""
        # stop the work after having met nprocs times "STOP"
        for work in xrange(nprocs):
            for val in iter(self.output_queue.get, "STOP"):
                k = val[0]
                pvalmat[k] = val[1]


    def split_range(self, n, m=4):
        """Split the interval :math:`\\left[0,\ldots, n\\right]` in ``m`` parts.

        :param n: upper limit of the range
        :type n: int
        :param m: number of part in which range should be split
        :type n: int
        :returns: delimiter indices for the ``m`` parts
        :rtype: list
        """
        return [i * n / m for i in range(m)]


    @staticmethod
    def triumat2flat_dim(n):
        """Return the size of the triangular part of a ``n x n`` matrix.

        :param n: the dimension of the square matrix
        :type n: int
        :returns: number of elements in the upper triangular part of the matrix
            (excluding the diagonal)
        :rtype: int
        """
        return n * (n - 1) / 2

    @staticmethod
    def flat2triumat_idx(k, n):
        """Convert an array index into the index couple of a triangular matrix.

        ``k`` is the index of an array of length :math:`\\binom{n}{2}{2}`,
        which contains the elements of an upper triangular matrix of dimension
        ``n`` excluding the diagonal. The function returns the index couple
        :math:`(i, j)` that corresponds to the entry ``k`` of the flat array.

        .. note::
            * :math:`k \\in \left[0,\\ldots, \\binom{n}{2} - 1\\right]`
            * returned indices:
                * :math:`i \\in [0,\\ldots, n - 1]`
                * :math:`j \\in [i + 1,\\ldots, n - 1]`

        :param k: flattened array index
        :type k: int
        :param n: dimension of the square matrix
        :type n: int
        :returns: matrix index tuple (row, column)
        :rtype: tuple
        """
        # row index of array index k in the the upper triangular part of the
        # square matrix
        r = n - 2 - int(0.5 * np.sqrt(-8 * k + 4 * n * (n - 1) - 7) - 0.5)
        # column index of array index k in the the upper triangular part of the
        # square matrix
        c = k + 1 + r * (3 - 2 * n + r) / 2
        return r, c

    @staticmethod
    def save_array(mat, filename, delim='\t', binary=False):
        """Save the array ``mat`` in the file ``filename``.

        The array can either be saved as a binary NumPy ``.npy`` file or as a
        human-readable ``.npy`` file.

        .. note::

            * The relative path has to be provided in the filename, e.g.
              *../data/pvalue_matrix.csv*.

            * If ``binary==True``, NumPy
              automatically appends the format ending ``.npy`` to the file.

        :param mat: array
        :type mat: numpy.array
        :param filename: name of the output file
        :type filename: str
        :param delim: delimiter between values in file
        :type delim: str
        :param binary: if ``True``, save as binary ``.npy``, otherwise as a
            ``.csv`` file
        :type binary: bool
        """
        if binary:
            np.save(filename, mat)
        else:
            np.savetxt(filename, mat, delimiter=delim)
