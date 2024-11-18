//
// Created by rabsch on 15.11.22.
//

#ifndef RNADIST_EDSAMPLING_H
#define RNADIST_EDSAMPLING_H

#endif //RNADIST_EDSAMPLING_H

#include<bits/stdc++.h>

using namespace std;


extern "C"
{
#include "ViennaRNA/fold_compound.h"
#include "ViennaRNA/eval.h"
#include "ViennaRNA/part_func.h"
#include "ViennaRNA/mfe.h"

#include "ViennaRNA/sampling/basic.h"

}

void addDistancesRedundantCallback(const char *structure, void *data);
void addDistancesNonRedundantCallback(const char *structure, void *data);
void addShortestPathDirected(short * pairtable, vector <vector<double>> &e_distances, double weight);


vector <vector<double>> edSampleRedundant(vrna_fold_compound_t *fc, int nr_samples, bool undirected);
vector <vector<double>> edSampleNonRedundant(vrna_fold_compound_t *fc, int nr_samples, bool undirected);
vector <vector<double>> edPThresholdSample(vrna_fold_compound_t *fc, double threshold, bool undirected);
double expectedDistanceIJ(vrna_fold_compound_t *fc, int nr_samples, int i, int j);

struct sampling_data {
    vrna_fold_compound_t  *fc;
    vector<vector<double>>  *expected_distance;
    double nr_samples;
    bool undirected;
};


struct nr_sampling_data {
    vrna_fold_compound_t        *fc;
    double                      kT;
    double                      ens_en;
    double                      *prob_sum;
    vector<vector<double>>      *expected_distance;
    bool                        undirected;
};

struct ij_sampling_data {
    double nr_samples;
    double *distance;
    int i;
    int j;

};