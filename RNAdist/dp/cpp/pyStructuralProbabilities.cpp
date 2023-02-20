#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PYBIND11_DETAILED_ERROR_MESSAGES
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "structuralProbabilities.h"

namespace py = pybind11;


typedef struct {
    PyObject_HEAD
    void *ptr;
    void *ty;
    int own;
    PyObject *next;
    PyObject *dict;
} SwigPyObject;

static vrna_fold_compound_t *swigFcToFc(PyObject *swig_fold_compound) {
    SwigPyObject *swf = (SwigPyObject*) swig_fold_compound;
    vrna_fold_compound_t *fc = (vrna_fold_compound_t*) swf->ptr;
    return fc;
}

static py::array structProbabilities(py::args args){
    vrna_fold_compound_t *fc = swigFcToFc(args[0].ptr());
    vector <vector<double>> probabilities =  structuralProbabilities(fc);
    py::array probabilities_array =  py::cast(probabilities);
    return probabilities_array;
}

PYBIND11_MODULE(RNAsProbs, m){
    m.def("cpp_struct_probs", structProbabilities, "Calculates Structural Probabilities of RNA");
}