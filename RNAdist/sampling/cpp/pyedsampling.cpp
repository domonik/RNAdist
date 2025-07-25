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

py::dict convertStructureCacheToPython(const StructureCache& cache) {
    py::dict result;
    for (const auto& [key, value] : cache) {
        py::bytes py_key(reinterpret_cast<const char*>(key.data()), key.size());
        result[py_key] = value;
    }
    return result;
}

StructureCache convertPythonToStructureCache(const py::dict& dict) {
    StructureCache cache;
    for (const auto& item : dict) {
        // Extract the key as a py::bytes object
        py::bytes py_key = py::reinterpret_borrow<py::bytes>(item.first);

        // Convert py::bytes to std::string, then to std::vector<uint8_t>
        std::string key_str = static_cast<std::string>(py_key);
        std::vector<uint8_t> key_vec(key_str.begin(), key_str.end());

        // Extract the value and cast to int
        int value = py::cast<int>(item.second);

        // Insert into the StructureCache
        cache.emplace(std::move(key_vec), value);
    }
    return cache;
}

static std::string convertBitRepresentationToStructure(py::args args){
    py::bytes py_key = args[0];
    size_t structure_len = args[1].cast<size_t>();
    std::string temp = py_key;
    std::vector<uint8_t> key(temp.begin(), temp.end());
    std::string structure = decodeStructure(key, structure_len);
    return structure;


};


static py::array distancesFromStructureString(py::str& pystructure) {
    string structure = pystructure.cast<string>();
    size_t n = structure.size();
    vector <uint16_t> distances(n * n, 0);
    fillDistancesFromStructureString(structure, distances);

    py::array_t<uint16_t> dist_array = py::array_t<uint16_t>({n, n});

    auto r = dist_array.mutable_unchecked<2>();
    for (size_t k = 0; k < n; ++k)
        for (size_t l = k; l < n; ++l) {
            uint16_t d = distances[k * n + l];
            r(k, l) = d;
            r(l, k) = d;
        }

    return dist_array;
}


static py::array histogramFromStructureCache(py::dict py_cache, int n) {
    StructureCache cache = convertPythonToStructureCache(py_cache);

    vector<uint16_t> counts = distancesFromStructureCache(cache, n);

    py::array_t<uint16_t> count_array = py::array_t<uint16_t>({n, n, n});

    auto r = count_array.mutable_unchecked<3>();
    for (size_t k = 0; k < n; ++k)
        for (size_t l = k; l < n; ++l)
            for (size_t m = 0; m < n; ++m) {
                uint16_t d = counts[k * n * n + m * n + l];
                r(k, l, m) = d;// call original version
                r(l, k, m) = d;// call original version

            }

    return count_array;
}



static std::tuple<py::array, py::dict> trackSampledDistances(py::args args){
    vrna_fold_compound_t *fc = swigFcToFc(args[0].ptr());
    int nr_samples = args[1].cast<int>();
    size_t n = fc->length;
    vector<uint16_t> counts;
    StructureCache cache;
    py::array_t<uint16_t> count_array;

    if (args.size() == 2) {
        std::tie(counts, cache) = trackDistances(fc, nr_samples);
        count_array = py::array_t<uint16_t>({n, n, n});
        auto r = count_array.mutable_unchecked<3>();
        for (size_t k = 0; k < n; ++k)
            for (size_t l = k; l < n; ++l)
                for (size_t m = 0; m < n; ++m) {
                    uint16_t d = counts[k * n * n + m * n + l];
                    r(k, l, m) = d;// call original version
                    r(l, k, m) = d;// call original version

                }

    } else {
        int i = args[2].cast<int>();
        int j = args[3].cast<int>();
        std::tie(counts, cache) = trackDistances(fc, nr_samples, i, j);
        count_array = py::array_t<uint16_t>({static_cast<py::ssize_t>(n)});
        auto r = count_array.mutable_unchecked<1>();
        for (size_t k = 0; k < n; ++k){
            r(k) = counts[k];
        }

    }

    py::dict py_structure_cache = convertStructureCacheToPython(cache);



    return {count_array, py_structure_cache};
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
    m.def("cpp_bit_to_structure", convertBitRepresentationToStructure, "Converts bit representation to dot brakcet structure");
    m.def("cpp_histogram_from_structure_cache", &histogramFromStructureCache,py::arg("cache"), py::arg("n"), "Return histogram array from structure cache");
    m.def("cpp_nr_sampling", edNRSampling, "Samples non-redundant from possible RNA structures");
    m.def("cpp_sampling_ij", edIJ, "Return expected distance between i and j");
    m.def("distances_from_structure", distancesFromStructureString, "Returns distance matrix for a given structure");
    m.def("cpp_pthreshold_sampling", edPThresholdSampling, "Samples non-redundant from possible RNA structures until "
                                                   "probability threshold is reached");
}

