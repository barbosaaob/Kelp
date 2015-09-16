"""
Kelp

Kelp multidimensional projection technique.
http://www.lcad.icmc.usp.br/~barbosa
"""
from __future__ import print_function
from projection import projection
from force import force

try:
    import scipy.spatial.distance as pdist
    import numpy as np
except ImportError as msg:
    error = ", please install the following packages:\n"
    error += "    NumPy      (http://www.numpy.org)\n"
    error += "    SciPy      (http://www.scipy.org)"
    raise ImportError(str(msg) + error)


class Kelp(projection.Projection):
    """
    Kelp projection.
    """
    def __init__(self, data, data_class, sample=None, sample_projection=None,
                 kernel_param=None):
        """
        Class initialization.
        """
        assert type(data) is np.ndarray, "*** ERROR (Kelp): Data is of wrong \
                type!"

        projection.Projection.__init__(self, data, data_class)
        self.sample = sample
        self.sample_projection = sample_projection
        self.kernel_param = kernel_param

    def project(self, tol=1e-6):
        """
        Projection method.

        Projection itself.
        """
        ninst, dim = self.data.shape    # number os instances, data dimension
        k = len(self.sample)            # number os sample instances
        p = self.projection_dim         # visual space dimension
        x = self.data
        xs = self.data[self.sample, :]
        param = self.kernel_param

        if param is None:
            param = 2 * np.sum(np.var(x, ddof=1, keepdims=True,
                                      axis=0)) / x.shape[1]
            self.kernel_param = param

        K_uncentered = self.kernel_matrix(xs)
        K = self.center_kernel_matrix(K_uncentered)

        # kernel matrix eigendecomposition
        evals, evecs = np.linalg.eig(K)
        idx = evals.argsort()[::-1]
        evals, evecs = evals[idx], evecs[:, idx]
        nz_idx = 0
        for i in range(k):
            if evals[i] >= tol:
                nz_idx = i
                evecs[:, i] = evecs[:, i] / np.sqrt(evals[i])

        # make evals to be evals of covariance matrix
        evals[range(nz_idx + 1)] = float(k) / evals[range(nz_idx + 1)]

        # projection it self
        self.projection = np.zeros((ninst, p))
        ones = np.ones(k) / float(k)
        ONES = np.ones((k, k)) / float(k)
        evecs_nz = evecs[:, range(nz_idx + 1)]
        evals_nz = np.diag(evals[range(nz_idx + 1)])
        T = (1.0 / k) * np.dot(np.dot(np.dot(np.dot(evecs_nz, evals_nz),
                                             evecs_nz.T), K),
                               self.sample_projection)
        for pt in range(ninst):
            aux = xs - x[pt, :]
            sqdist = np.diag(np.dot(aux, aux.T))
            Kx = np.exp(-sqdist / param)
            Kxc = Kx - np.dot(K_uncentered, ones) - np.dot(ONES, Kx) + \
                np.dot(np.dot(ONES, K_uncentered), ones)
            self.projection[pt, :] = np.dot(Kxc.T, T)

    def kernel_matrix(self, x):
        """
        Computes gaussian kernel matrix.
        """
        sqdist = pdist.squareform(pdist.pdist(x)) ** 2
        return np.exp(-sqdist / self.kernel_param)

    def center_kernel_matrix(self, kmat):
        """
        Centralizes kernel matrix.
        """
        k = kmat.shape[0]
        Ik = np.ones((k, k)) / float(k)
        return kmat - np.dot(Ik, kmat) - np.dot(kmat, Ik) + \
            np.dot(np.dot(Ik, kmat), Ik)


def run():
    import time
    import sys
    print("Loading data set... ", end="")
    sys.stdout.flush()
    data_file = np.loadtxt("iris.data")
    print("Done.")
    ninst, dim = data_file.shape
    sample_size = int(np.ceil(np.sqrt(ninst)))
    data = data_file[:, range(dim - 1)]
    data_class = data_file[:, dim - 1]
    sample = np.random.permutation(ninst)
    sample = sample[range(sample_size)]

    # force
    start_time = time.time()
    print("Projecting samples... ", end="")
    sys.stdout.flush()
    f = force.Force(data[sample, :], [])
    f.project()
    sample_projection = f.get_projection()
    print("Done. (" + str(time.time() - start_time) + "s.)")

    # kelp
    start_time = time.time()
    print("Projecting... ", end="")
    sys.stdout.flush()
    kelp = Kelp(data, data_class, sample, sample_projection)
    kelp.project()
    print("Done. (" + str(time.time() - start_time) + "s.)")
    kelp.plot()


if __name__ == "__main__":
    run()
