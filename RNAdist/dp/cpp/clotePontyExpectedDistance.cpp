//
// Created by domonik on 18.11.24.
//
#include "../../cpp/RNAHelpers.h"
#include "clotePontyExpectedDistance.h"

using namespace std;





static double get_Z(vrna_fold_compound_t *fc, int i, int j){
    int idx = (((fc->exp_matrices->length + 1 - i) * (fc->exp_matrices->length - i)) / 2) + fc->exp_matrices->length + 1 - j;
    return fc->exp_matrices->q[idx];
}

static double get_ZB(vrna_fold_compound_t *fc, int i, int j){
    int idx = (((fc->exp_matrices->length + 1 - i) * (fc->exp_matrices->length - i)) / 2) + fc->exp_matrices->length + 1 - j;
    return fc->exp_matrices->qb[idx];
}

static double new_exp_E_ext_stem(vrna_fold_compound_t *fc, unsigned int i, unsigned int j){
    unsigned int type;
    int enc5, enc3;
    enc5 = enc3 = -1;

    type = vrna_get_ptype_md(fc->sequence_encoding2[i],
                             fc->sequence_encoding2[j],
                             &(fc->params->model_details));

    if (i > 1)
        enc5 = fc->sequence_encoding[i - 1];
    if (j < fc->length)
        enc3 = fc->sequence_encoding[j + 1];

    return (double)vrna_exp_E_ext_stem(type,enc5,enc3,fc->exp_params);
}


vector <vector<double>> clote_ponty_expected_distance(vrna_fold_compound_t *fc) {
    double mfe = (double)vrna_mfe(fc, NULL);
    vrna_exp_params_rescale(fc, &mfe);
    vrna_pf(fc, NULL);


    vector <vector<double>> e_distance(fc->length, vector<double>(fc->length));
    double z;
    unsigned int j;
    double prev_res;

    for (unsigned int l = 1; l < fc->exp_matrices->length + 1; l = l+1){
        for (unsigned int i = 1;  i < fc->exp_matrices->length + 1 - l; i = i+1){
            j = i + l;
            prev_res =  e_distance[i-1][j-2];
            z = prev_res * fc->exp_matrices->scale[1] + get_Z(fc, i, j);
            for (unsigned int  k = i+1;  k <= j; k = k+1) {
                prev_res = e_distance[i-1][k-2];
                z +=  (prev_res + get_Z(fc, i, k-1)) * get_ZB(fc, k, j) * new_exp_E_ext_stem(fc, k, j);
            }

            e_distance[i-1][j-1] = z ;
        }
    }
    for (unsigned int i = 1; i < fc->exp_matrices->length + 1; i = i+1){
        for (unsigned int j = i+1;  j < fc->exp_matrices->length + 1; j = j+1){
            e_distance[i-1][j-1] /= get_Z(fc, i, j);
        }
    }
    return e_distance;
}
