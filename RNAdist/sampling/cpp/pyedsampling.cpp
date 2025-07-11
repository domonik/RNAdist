#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PYBIND11_DETAILED_ERROR_MESSAGES
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "../../cpp/RNAHelpers.h"

#include "edsampling.h"

namespace py = pybind11;


extern "C"
{
  #include "ViennaRNA/fold_compound.h"
}


static py::array trackSampledDistances(py::args args){
    vrna_fold_compound_t *fc = swigFcToFc(args[0].ptr());
    int nr_samples = args[1].cast<int>();
    size_t n = fc->length;
    vector<uint16_t> counts;
    py::array_t<int> count_array;

    if (args.size() == 2) {
        counts = trackDistances(fc, nr_samples);
        count_array = py::array_t<int>({n, n, n});
        auto r = count_array.mutable_unchecked<3>();
        for (size_t k = 0; k < n; ++k)
            for (size_t l = 0; l < n; ++l)
                for (size_t m = 0; m < n; ++m)
                    r(k, l, m) = counts[k * n * n + l * n + m];// call original version
    } else {
        int i = args[2].cast<int>();
        int j = args[3].cast<int>();
        counts = trackDistances(fc, nr_samples, i, j);
        count_array = py::array_t<int>({static_cast<py::ssize_t>(n)});
        auto r = count_array.mutable_unchecked<1>();
        for (size_t k = 0; k < n; ++k){
            r(k) = counts[k];
        }

    }



    return count_array;
}

static py::array edSampling(py::args args){
    vrna_fold_compound_t *fc = swigFcToFc(args[0].ptr());

    int nr_samples = args[1].cast<int>();
    bool undirected = args[2].cast<bool>();
    vector <vector<double>> ed_vector =  edSampleRedundant(fc, nr_samples, undirected);
    py::array ed_array =  py::cast(ed_vector);
    return ed_array;
}

static py::array edNRSampling(py::args args){
    vrna_fold_compound_t *fc = swigFcToFc(args[0].ptr());

    int nr_samples = args[1].cast<int>();
    bool undirected = args[2].cast<bool>();

    vector <vector<double>> ed_vector =  edSampleNonRedundant(fc, nr_samples, undirected);
    py::array ed_array =  py::cast(ed_vector);
    return ed_array;
}

static py::array edPThresholdSampling(py::args args){
    vrna_fold_compound_t *fc = swigFcToFc(args[0].ptr());

    double threshold = args[1].cast<double>();
    bool undirected = args[2].cast<bool>();

    vector <vector<double>> ed_vector =  edPThresholdSample(fc, threshold, undirected);
    py::array ed_array =  py::cast(ed_vector);
    return ed_array;
}

static double edIJ(py::args args){
    vrna_fold_compound_t *fc = swigFcToFc(args[0].ptr());
    int i = args[1].cast<int>();
    int j = args[2].cast<int>();
    int nr_samples = args[3].cast<int>();
    return expectedDistanceIJ(fc, nr_samples, i, j);

}

PYBIND11_MODULE(sampling, m) {
    m.def("cpp_sampling", edSampling, "Samples redundant from possible RNA structures");
    m.def("cpp_distance_tracking", trackSampledDistances, "Tracks histogram of sampled distances");
    m.def("cpp_nr_sampling", edNRSampling, "Samples non-redundant from possible RNA structures");
    m.def("cpp_sampling_ij", edIJ, "Return expected distance between i and j");
    m.def("cpp_pthreshold_sampling", edPThresholdSampling, "Samples non-redundant from possible RNA structures until "
                                                   "probability threshold is reached");
}

