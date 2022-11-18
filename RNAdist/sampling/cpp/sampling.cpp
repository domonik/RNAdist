#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PYBIND11_DETAILED_ERROR_MESSAGES
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "sampling.h"
#include <ViennaRNA/utils/structures.h>
namespace py = pybind11;

extern "C"
{
  #include "ViennaRNA/fold_compound.h"
  #include "ViennaRNA/eval.h"
  #include "ViennaRNA/part_func.h"
  #include "ViennaRNA/boltzmann_sampling.h"
  #include "ViennaRNA/mfe.h"

}


static vrna_fold_compound_t *swig_fc_to_fc(PyObject *swig_fold_compound) {
    SwigPyObject *swf = (SwigPyObject*) swig_fold_compound;
    vrna_fold_compound_t *fc = (vrna_fold_compound_t*) swf->ptr;
    return fc;
}

struct nr_en_data {
  vrna_fold_compound_t  *fc;
  double                kT;
  double                ens_en;
  double                *prob_sum;
  py::array_t<double>*   expected_distance;
};


struct r_data {
  vrna_fold_compound_t  *fc;
  py::array_t<double>*   expected_distance;
  double nr_samples;
};


void add_distances_callback(const char *structure, void *data)
{
  if (structure) {
    struct nr_en_data     *d      = (struct nr_en_data *)data;
    vrna_fold_compound_t  *fc     = d->fc;
    double                kT      = d->kT;
    double                ens_en  = d->ens_en;
    double                *prob_sum  = d->prob_sum;
    py::array_t<double>*   exp_d = d->expected_distance;


    double                e         = vrna_eval_structure(fc, structure);
    double                prob      = exp((ens_en - e) / kT);
    auto pt = vrna_ptable(structure);
    prob_sum[0] += prob;
    Graph g(fc->length);
    g.pairTableToGraph(pt);
    g.AddDistances(exp_d, prob);
    delete pt;
  }
}

py::array_t<double> sample_non_redundant(vrna_fold_compound_t *fc, double cutoff){
    double ens_en = vrna_pf(fc, NULL);
    double kT = fc->exp_params->kT / 1000.;
    double prob_sum = 0.f;
    py::array_t<double> foo = py::array_t<double>({ fc->length, fc->length });
    foo[py::make_tuple(py::ellipsis())] = 0.f;
    struct nr_en_data dat;
        dat.fc      = fc;
        dat.kT      = kT;
        dat.ens_en  = ens_en;
        dat.prob_sum = &prob_sum;
        dat.expected_distance = &foo;
    vrna_pbacktrack_mem_t nonredundant_memory = NULL;
    int nr_structures = 1;
    while (prob_sum < cutoff && nr_structures != 0) {
        nr_structures = vrna_pbacktrack_resume_cb(fc,
                       1,
                       &add_distances_callback,
                       (void *)&dat,
                       &nonredundant_memory,
                       VRNA_PBACKTRACK_NON_REDUNDANT
                       );
    }
    vrna_pbacktrack_mem_free(nonredundant_memory);
    return foo;
}

void addDistancesRedundantCallback(const char *structure, void *data)
{
    if (structure) {
        struct r_data     *d      = (struct r_data *)data;
        vrna_fold_compound_t  *fc     = d->fc;
        py::array_t<double>*   exp_d = d->expected_distance;
        auto pt = vrna_ptable(structure);
        Graph g(fc->length);
        g.pairTableToGraph(pt);
        g.AddDistances(exp_d, 1/d->nr_samples);
        delete pt;
    }
}

py::array_t<double> sample_redundant(vrna_fold_compound_t *fc, int nr_samples){
    py::array_t<double> expected_distances = py::array_t<double>({ fc->length, fc->length });
    vrna_pf(fc, NULL);
    expected_distances[py::make_tuple(py::ellipsis())] = 0.f;
    struct r_data dat;
        dat.fc = fc;
        dat.expected_distance = &expected_distances;
        dat.nr_samples = nr_samples;
    vrna_pbacktrack_cb(
            fc,
            nr_samples,
            &add_distances_redundant_callback,
            (void *)&dat,
            VRNA_PBACKTRACK_DEFAULT
            );
    return expected_distances;
}



static py::array_t<double> nr_sampling(py::args args)
{
    // vrna_fold_compound_t *fc = swig_fc_to_fc(args[0]);
    vrna_fold_compound_t *fc = swig_fc_to_fc(args[0].ptr());
    double cutoff = args[1].cast<double>();
    double mfe = (double)vrna_mfe(fc, NULL);
    vrna_exp_params_rescale(fc, &mfe);
    py::array_t<double> expected_distance = sample_non_redundant(fc, cutoff);


    return expected_distance;
}

static py::array_t<double> sampling(py::args args)
{
    // vrna_fold_compound_t *fc = swig_fc_to_fc(args[0]);
    vrna_fold_compound_t *fc = swig_fc_to_fc(args[0].ptr());
    int nr_samples = args[1].cast<int>();
    double mfe = (double)vrna_mfe(fc, NULL);
    vrna_exp_params_rescale(fc, &mfe);
    py::array_t<double> expected_distance = sample_redundant(fc, nr_samples);


    return expected_distance;
}


PYBIND11_MODULE(sampling2, m) {
  m.def("cpp_nr_sampling", nr_sampling, "Samples non redundant from possible RNA structures");
  m.def("cpp_sampling", sampling, "Samples redundant from possible RNA structures");
}