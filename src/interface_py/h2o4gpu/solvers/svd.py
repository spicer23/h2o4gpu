#- * - encoding : utf - 8 - * -
"""
:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
import sys
import warnings
import numpy as np

from ctypes import c_int, c_float, c_double, c_void_p, c_size_t, POINTER, \
    pointer, cast, addressof
from ..libs.lib_svd import GPUlib, CPUlib
from ..solvers.utils import device_count, _get_data, _data_info, \
    _convert_to_ptr, _check_equal
from ..typecheck.typechecks import (assert_is_type, numpy_ndarray,
                                    pandas_dataframe)

class SVDH2O(object):
    """H2O SVD Solver for GPUs

    :param int n_threads : Number of threads to use in the gpu.
    Each thread is an independent model builder. Default is None.

    :param gpu_id : int, optional, default: 0
        ID of the GPU on which the algorithm should run.

    :param int n_gpus : Number of gpu's to use in GLM solver. Default is -1.

    :param order : Order of data.  Default is None, and internally
        determined (unless using _ptr methods) whether
        row 'r' or column 'c' major order.

    :param int verbose : Print verbose information to the console if set to > 0.
        Default is 0.
    """

    class info:
        pass

    class solution:
        pass

    def __init__(self,
                 n_threads=None,
                 gpu_id=0,
                 n_gpus=-1,
                 order=None,
                 verbose=0):

        assert_is_type(n_threads, int, None)
        assert_is_type(gpu_id, int)
        assert_is_type(n_gpus, int)
        assert_is_type(verbose, int)

        if order is not None:
            assert_is_type(order, str)
            assert order in ['r',
                             'c'], \
                "Order should be set to 'r' or 'c' but got " + order
            self.ord = ord(order)
        else:
            self.ord = None
        self.dtype = None

        self.n = 0
        self.m_train = 0
        self.source_dev = 0  # assume Dev=0 is source of data for upload_data
        self.source_me = 0  # assume thread=0 is source of data for upload_data

        self.uploaded_data = 0
        self.did_fit_ptr = 0
        self.verbose = verbose
        self._shared_a = 0
        self._standardize = 0

        (self.n_gpus, devices) = device_count(n_gpus)
        gpu_id = gpu_id % devices
        self._gpu_id = gpu_id
        self._total_n_gpus = devices

        if n_threads is None:
            n_threads = (1 if self.n_gpus == 0 else self.n_gpus)

        self.n_threads = n_threads

        gpu_lib = GPUlib().get()
        cpu_lib = CPUlib().get()

        if self.n_gpus == 0 or gpu_lib is None or devices == 0:
            if verbose > 0:
                print('Using CPU GLM solver %d %d' % (self.n_gpus, devices))
            self.lib = cpu_lib
        elif self.n_gpus > 0 or gpu_lib is None or devices == 0:
            if verbose > 0:
                print('Using GPU GLM solver with %d GPUs' % self.n_gpus)
            self.lib = gpu_lib
        else:
            raise RuntimeError("Couldn't instantiate GLM Solver")

    def fit(self,
            matrix=None,
            free_input_data=1):

        assert_is_type(matrix, numpy_ndarray, pandas_dataframe, None)
        assert_is_type(free_input_data, int)

        source_dev = 0
        if not matrix is None:

            self.prepare_and_upload_data(
                matrix=matrix,
                source_dev=source_dev)

        else:
            pass

        self.fit_ptr(
            self.m_train,
            self.n,
            self.double_precision,
            self.ord,
            self.a,
            free_input_data=free_input_data,
            source_dev=source_dev)
        return self

    def prepare_and_upload_data(self,
                                matrix=None,
                                source_dev=0):

        train_x_np, m_train, n1, fortran1, self.ord, self.dtype = _get_data(
            matrix,
            ismatrix=True,
            order=self.ord,
            dtype=self.dtype)

        fortran_list = [fortran1]
        _check_equal(fortran_list)

        a = self.upload_data(train_x_np, source_dev)

        self.a = a
        return a

    def fit_ptr(
            self,
            m_train,
            n,
            double_precision,
            order,
            a,  # trainX_ptr or train_xptr
            free_input_data=0,
            source_dev=0):

        self._fitorpredict_ptr(
            source_dev,
            m_train,
            n,
            double_precision,
            order,
            a,
            free_input_data=free_input_data)

    def _fitorpredict_ptr(
            self,
            source_dev,
            m_train,
            n,
            double_precision,
            order,
            a,
            free_input_data=0):

        assert_is_type(source_dev, int, None)
        assert_is_type(m_train, int, None)
        assert_is_type(n, int, None)
        assert_is_type(double_precision, float, int, None)
        assert_is_type(order, int, str, None)
        assert_is_type(a, c_void_p, None)
        assert_is_type(free_input_data, int)

        self.source_dev = source_dev
        self.m_train = m_train
        self.n = n
        self.a = a

        self.did_fit_ptr = 1

        if order is not None:  # set order if not already set
            if order in ['r', 'c']:
                self.ord = ord(order)
            else:
                self.ord = order

        if hasattr(self,
                   'double_precision') and self.double_precision is not None:
            which_precision = self.double_precision
        else:
            which_precision = double_precision
            self.double_precision = double_precision

        c_size_t_p = POINTER(c_size_t)
        if which_precision == 1:
            c_svd = self.lib.svd_ptr_double
            self.dtype = np.float64
            self.myctype = c_double
            if self.verbose > 0:
                print('double precision fit')
                sys.stdout.flush()
        else:
            c_svd = self.lib.svd_ptr_float
            self.dtype = np.float32
            self.myctype = c_float
            if self.verbose > 0:
                print('single precision fit')
                sys.stdout.flush()

        #call svd in C backend
        c_svd(
            c_int(source_dev),
            c_int(1),
            c_int(self._shared_a),
            c_int(self.n_threads),
            c_int(self._gpu_id),
            c_int(self.n_gpus),
            c_int(self._total_n_gpus),
            c_int(self.ord),
            c_size_t(m_train),
            c_size_t(n),
            c_int(self._standardize),
            c_int(self.verbose),
            a)

        if free_input_data == 1:
            self.free_data()

#PROCESS OUTPUT
#save pointers


#Properties and setters of properties

    @property
    def total_n_gpus(self):
        return self._total_n_gpus

    @property
    def gpu_id(self):
        return self._gpu_id

    @gpu_id.setter
    def gpu_id(self, value):
        assert_is_type(value, int)
        assert value >= 0, "GPU ID must be non-negative."
        self._gpu_id = value

    @property
    def shared_a(self):
        return self._shared_a

    @shared_a.setter
    def shared_a(self, value):
        #add check
        self.__shared_a = value

#Free up memory functions

    def free_data(self):

        #NOTE : For now, these are automatically freed
        #when done with fit-- ok, since not used again

        if self.uploaded_data == 1:
            self.uploaded_data = 0
            if self.double_precision == 1:
                self.lib.modelfree1_double(self.a)
                self.lib.modelfree1_double(self.b)
                self.lib.modelfree1_double(self.c)
                self.lib.modelfree1_double(self.d)
                self.lib.modelfree1_double(self.e)
            else:
                self.lib.modelfree1_float(self.a)
                self.lib.modelfree1_float(self.b)
                self.lib.modelfree1_float(self.c)
                self.lib.modelfree1_float(self.d)
                self.lib.modelfree1_float(self.e)

    def finish(self):
        self.free_data()

    def upload_data(self,
                    train_x,
                    source_dev=0):
        """Upload the data through the backend library"""
        if self.uploaded_data == 1:
            self.free_data()
        self.uploaded_data = 1

        self.double_precision1, m_train, n1 = _data_info(train_x, self.verbose)
        self.m_train = m_train
        self.double_precision = self.double_precision1  # either one

        a = c_void_p(0)
        if self.double_precision == 1:
            self.dtype = np.float64

            if self.verbose > 0:
                print('Detected np.float64')
                sys.stdout.flush()
        else:
            self.dtype = np.float32

            if self.verbose > 0:
                print('Detected np.float32')
                sys.stdout.flush()

#make these types consistent
        A = _convert_to_ptr(train_x)

        if self.double_precision == 1:
            c_upload_data = self.lib.make_ptr_double
        elif self.double_precision == 0:
            c_upload_data = self.lib.make_ptr_float
        else:
            print('Unknown numpy type detected')
            print(train_x.dtype)
            sys.stdout.flush()
            return a
        '''
        int sourceDev, int datatype, int sharedA,
			int nThreads, int gpu_id, int nGPUs, int totalnGPUs, const char ord, size_t mTrain, size_t n,
			int standardize, int verbose, void *matrix
        '''
        status = c_upload_data(
            c_int(self.source_dev),
            c_int(1),
            c_int(self._shared_a),
            c_int(self.n_threads),
            c_int(self.gpu_id),
            c_int(self.n_gpus),
            c_int(self.n_gpus),
            c_int(self.ord),
            c_size_t(self.m_train),
            c_size_t(self.n),
            c_int(self._standardize),
            c_int(self.verbose),
            A)

        assert status == 0, 'Failure uploading the data'

        self.a = a
        return a

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        #fetch the constructor or the original constructor before
        #deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            #No explicit constructor to introspect
            return []

        #introspect the constructor arguments to find the model parameters
        #to represent
        from ..utils.fixes import signature
        init_signature = signature(init)
        #Consider the constructor parameters excluding 'self'
        parameters = [
            p for p in init_signature.parameters.values()
            if p.name != 'self' and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("h2o4gpu GLM estimator should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention." %
                                   (cls, init_signature))
        #Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        :param bool deep : If True, will return the parameters for this
            estimator and contained subobjects that are estimators.

        :returns dict params : Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            #We need deprecation warnings to always be on in order to
            #catch deprecated param values.
            #This is set in utils / __init__.py but it gets overwritten
            #when running under python3 somehow.
            warnings.simplefilter("always", DeprecationWarning)
            try:
                with warnings.catch_warnings(record=True) as w:
                    value = getattr(self, key, None)
                if w and w[0].category == DeprecationWarning:
                    #if the parameter is deprecated, don't show it
                    continue
            finally:
                warnings.filters.pop(0)

            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator."""
        if not params:
            #Simple optimization to gain speed(inspect is slow)
            return self
        valid_params = self.get_params(deep=True)
        from ..externals import six
        for key, value in six.iteritems(params):
            split = key.split('__', 1)
            if len(split) > 1:
                #nested objects case
                name, sub_name = split
                if name not in valid_params:
                    raise ValueError('Invalid parameter %s for estimator %s. '
                                     'Check the list of available parameters '
                                     'with `estimator.get_params().keys()`.' %
                                     (name, self))
                sub_object = valid_params[name]
                sub_object.set_params(**{sub_name: value})
            else:
                #simple objects case
                if key not in valid_params:
                    raise ValueError('Invalid parameter %s for estimator %s. '
                                     'Check the list of available parameters '
                                     'with `estimator.get_params().keys()`.' %
                                     (key, self.__class__.__name__))
                setattr(self, key, value)
        return self