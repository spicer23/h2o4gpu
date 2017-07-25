from ctypes import c_int, c_float, c_double, pointer
from numpy import ndarray
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from h2ogpuml.types import ORD, cptr, make_settings, make_solution, make_info, change_settings, change_solution, \
    Solution, FunctionVector
from h2ogpuml.libs.cpu import pogsCPU
from h2ogpuml.libs.gpu import pogsGPU
from h2ogpuml.solvers.utils import devicecount
import sys


# TODO: catch Ctrl-C

class Pogs(object):
    def __init__(self, A, **kwargs):

        for key, value in kwargs.items():
            if key == "n_gpus":
                n_gpus = value
                # print("The value of {} is {}".format(key, value))
                # sys.stdout.flush()

        n_gpus, deviceCount = devicecount(n_gpus=n_gpus)

        if not pogsCPU:
            print(
                '\nWarning: Cannot create a pogs CPU Solver instance without linking Python module to a compiled H2OGPUML CPU library')

        if not pogsGPU:
            print(
                '\nWarning: Cannot create a pogs GPU Solver instance without linking Python module to a compiled H2OGPUML GPU library')
            print('> Use CPU or add CUDA libraries to $PATH and re-run setup.py\n\n')

        if ((n_gpus == 0) or (pogsGPU is None) or (deviceCount == 0)):
            print("\nUsing CPU GLM solver\n")
            self.solver = BaseSolver(A, pogsCPU)
        else:
            print("\nUsing GPU GLM solver with %d GPUs\n" % n_gpus)
            self.solver = BaseSolver(A, pogsGPU)

        assert self.solver != None, "Couldn't instantiate Pogs Solver"

        self.info = self.solver.info
        self.solution = self.solver.pysolution

    def init(self, A, **kwargs):
        self.solver.init(A, **kwargs)

    def fit(self, f, g, **kwargs):
        self.solver.fit(f, g, **kwargs)

    def finish(self):
        self.solver.finish()

    def __delete__(self):
        self.solver.finish()


class BaseSolver(object):
    def __init__(self, A, lib, **kwargs):
        try:
            self.dense = isinstance(A, ndarray) and len(A.shape) == 2
            self.CSC = isinstance(A, csc_matrix)
            self.CSR = isinstance(A, csr_matrix)

            assert self.dense or self.CSC or self.CSR
            assert A.dtype == c_float or A.dtype == c_double
            assert lib and (lib == pogsCPU or lib == pogsGPU)

            self.m = A.shape[0]
            self.n = A.shape[1]
            self.A = A
            self.lib = lib
            self.wDev = 0

            self.double_precision = A.dtype == c_double
            self.settings = make_settings(self.double_precision, **kwargs)
            self.pysolution = Solution(self.double_precision, self.m, self.n)
            self.solution = make_solution(self.pysolution)
            self.info = make_info(self.double_precision)
            self.order = ORD["ROW_MAJ"] if (self.CSR or self.dense) else ORD["COL_MAJ"]

            if self.dense and not self.double_precision:
                self.work = self.lib.h2ogpuml_init_dense_single(self.wDev, self.order, self.m, self.n, cptr(A, c_float))
            elif self.dense:
                self.work = self.lib.h2ogpuml_init_dense_double(self.wDev, self.order, self.m, self.n,
                                                                cptr(A, c_double))
            elif not self.double_precision:
                self.work = self.lib.h2ogpuml_init_sparse_single(self.wDev, self.order, self.m, self.n, A.nnz,
                                                                 cptr(A.data, c_float),
                                                                 cptr(A.indices, c_int), cptr(A.indptr, c_int))
            else:
                self.work = self.lib.h2ogpuml_init_sparse_double(self.wDev, self.order, self.m, self.n, A.nnz,
                                                                 cptr(A.data, c_double),
                                                                 cptr(A.indices, c_int), cptr(A.indptr, c_int))


        except AssertionError:
            print("data must be a (m x n) numpy ndarray or scipy csc_matrix containing float32 or float64 values")

    def init(self, A, lib, **kwargs):
        if not self.work:
            self.__init__(A, lib, **kwargs)
        else:
            print("H2OGPUML_work data structure already intialized, cannot re-initialize without calling finish()")

    def fit(self, f, g, **kwargs):
        try:
            # assert f,g types
            assert isinstance(f, FunctionVector)
            assert isinstance(g, FunctionVector)

            # assert f,g lengths
            assert f.length() == self.m
            assert g.length() == self.n

            # pass previous rho through, if not first run (rho=0)
            if self.info.rho > 0:
                self.settings.rho = self.info.rho

            # apply user inputs
            change_settings(self.settings, **kwargs)
            change_solution(self.pysolution, **kwargs)

            if not self.work:
                print("no viable H2OGPUML_work pointer to call solve(). call Solver.init( args... ) first")
                return
            elif not self.double_precision:
                self.lib.h2ogpuml_solve_single(self.work, pointer(self.settings), pointer(self.solution),
                                               pointer(self.info),
                                               cptr(f.a, c_float), cptr(f.b, c_float), cptr(f.c, c_float),
                                               cptr(f.d, c_float), cptr(f.e, c_float), cptr(f.h, c_int),
                                               cptr(g.a, c_float), cptr(g.b, c_float), cptr(g.c, c_float),
                                               cptr(g.d, c_float), cptr(g.e, c_float), cptr(g.h, c_int))
            else:
                self.lib.h2ogpuml_solve_double(self.work, pointer(self.settings), pointer(self.solution),
                                               pointer(self.info),
                                               cptr(f.a, c_double), cptr(f.b, c_double), cptr(f.c, c_double),
                                               cptr(f.d, c_double), cptr(f.e, c_double), cptr(f.h, c_int),
                                               cptr(g.a, c_double), cptr(g.b, c_double), cptr(g.c, c_double),
                                               cptr(g.d, c_double), cptr(g.e, c_double), cptr(g.h, c_int))

        except AssertionError:
            print("\nf and g must be objects of type FunctionVector with:")
            print(">length of f = m, # of rows in solver's data matrix A")
            print(">length of g = n, # of columns in solver's data matrix A")

    def finish(self):
        if not self.work:
            print("no viable H2OGPUML_work pointer to call finish(). call Solver.init( args... ) first")
            pass
        elif not self.double_precision:
            self.lib.h2ogpuml_finish_single(self.work)
            self.work = None
        else:
            self.lib.h2ogpuml_finish_double(self.work)
            self.work = None
        print("shutting down... H2OGPUML_work freed in C++")

    def __delete__(self):
        self.finish()
