#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PYBIND11_DETAILED_ERROR_MESSAGES
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "clotePontyExpectedDistance.h"
#include "../../cpp/RNAHelpers.h"

namespace py = pybind11;


static py::array py_clote_ponty_expected_distance(py::args args){
    vrna_fold_compound_t *fc = swigFcToFc(args[0].ptr());
    vector <vector<double>> expectedDistances =  clote_ponty_expected_distance(fc);
    py::array edArray =  py::cast(expectedDistances);
    return edArray;
}


PYBIND11_MODULE(CPExpectedDistance, m){
m.def("clote_ponty_expected_distance", py_clote_ponty_expected_distance, "Calculates Clote Ponty Expected Distance matrix of an ViennaRNA fold compound");
}

