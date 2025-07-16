//
// Created by rabsch on 17.02.23.
//

#ifndef RNADIST_STRUCTURALPROBABILITIES_H
#define RNADIST_STRUCTURALPROBABILITIES_H

#endif //RNADIST_STRUCTURALPROBABILITIES_H

#include<bits/stdc++.h>

using namespace std;

extern "C"
{
#include "ViennaRNA/fold_compound.h"
#include "ViennaRNA/eval.h"
#include "ViennaRNA/part_func.h"
#include "ViennaRNA/sampling/basic.h"
#include "ViennaRNA/mfe.h"

}

void structProbCallback(double *pr, int pr_size, int i, int max, unsigned int what, void *data);
vector <vector<double>> structuralProbabilities(vrna_fold_compound_t *fc);



struct struct_prob_callback_data {
    vector<vector<double>>  *probabilities;
};