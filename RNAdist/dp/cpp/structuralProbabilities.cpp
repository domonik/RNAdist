//
// Created by rabsch on 17.02.23.
//
#include<bits/stdc++.h>
#include "structuralProbabilities.h"



using namespace std;

extern "C"
{
#include "ViennaRNA/fold_compound.h"
#include <ViennaRNA/part_func_window.h>

}

void structProbCallback(double *pr, int pr_size, int i, int max, unsigned int what, void *data) {
    struct struct_prob_callback_data     *d      = (struct struct_prob_callback_data *)data;
    vector <vector<double>> *probs = d->probabilities;
    vector <vector<double>>& probsref = *probs;

    what = what & ~VRNA_PROBS_WINDOW_UP;
    int x;
    if (what == VRNA_EXT_LOOP) {
        x = 0;
    } else if (what == VRNA_HP_LOOP) {
        x = 1;
    }
    else if (what == VRNA_INT_LOOP) {
        x = 2;
    }
    else if (what == VRNA_MB_LOOP) {
        x = 3;
    } else {
        throw std::invalid_argument("Structural Probabilities Callback received invalid Argument");
    }
    probsref[x][i-1] += pr[1];
}

vector <vector<double>> structuralProbabilities(vrna_fold_compound_t *fc) {
    vector <vector<double>> structprobs(4, vector<double>(fc->length));
    struct struct_prob_callback_data data;
    data.probabilities = &structprobs;
    vrna_probs_window(
            fc,
            fc->length,
            VRNA_PROBS_WINDOW_UP | VRNA_PROBS_WINDOW_UP_SPLIT,
            &structProbCallback,
            (void *) &data
    );
    return structprobs;
}

