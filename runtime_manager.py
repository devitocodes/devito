from _ctypes import ArgumentError
import cgen_wrapper as cgen
import numpy as np


class RuntimeManager(object):
    def __init__(self, function_descriptor, jit):
        self.jit_manager = jit
        self.function_descriptor = function_descriptor

    def wrap_function(self, function, function_descriptor):
        """ Wrap the function by converting the python style arguments(simply passing object references)
            to C style (pointer + int dimensions)
        """
        def wrapped_function(*args):
            num_params = len(function_descriptor.params)
            assert len(args) == num_params, "Expected %d parameters, got %d" % (num_params, len(args))
            arg_list = []
            num_matrix_params = len(function_descriptor.matrix_params)
            for i in range(num_params):
                if i < num_matrix_params:
                    param_ref = np.asarray(args[i], dtype=self.dtype)  # Force the parameter provided as argument to be ndarray (not tuple or list)
                else:
                    param_ref = args[i]  # No conversion necessary for a scalar value
                arg_list += [param_ref]
            try:
                function(*arg_list)
            except ArgumentError as inst:
                assert False, "Argument Error (%s), provided arguments: " % str(inst)+" ,".join([str(type(arg)) for arg in arg_list])

        return wrapped_function

    def _prepare_wrapped_function(self):
        # Compile code if this hasn't been done yet
        self.compile()
        # Load compiled binary
        self.__load_library(src_lib=self.src_lib)

        # Type: double* in C
        array_nd = np.ctypeslib.ndpointer(dtype=self.dtype, flags='C')
        wrapped_functions = []
        for function_descriptor in self._function_descriptors:
            # Pointer to the function in the compiled library
            library_function = getattr(self._library, function_descriptor.name)
            if self._mic_flag:
                wrapped_functions.append(self.create_a_function(self._stream, library_function))
            else:
                # Ctypes needs an array describing the function parameters, prepare that array
                argtypes = [array_nd for i in function_descriptor.matrix_params]
                argtypes += [cgen.convert_dtype_to_ctype(param[0]) for param in function_descriptor.value_params]
                library_function.argtypes = argtypes
                wrapped_functions.append(self.wrap_function(library_function, function_descriptor))
        return wrapped_functions

    def create_a_function(self, mic_stream, mic_library):
        # Create function dynamically to execute mic offload using pymic
        def function_template(*args, **kwargs):
            mic_stream.invoke(mic_library, args)
            mic_stream.sync()
        return function_template

    def get_wrapped_functions(self):
        if self._wrapped_functions is None:
            self._wrapped_functions = self._prepare_wrapped_functions()
        return self._wrapped_functions