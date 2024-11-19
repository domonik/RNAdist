#include "edsampling.h"
#include "RNAGraph.h"

#include<bits/stdc++.h>


using namespace std;


extern "C"
{
  #include "ViennaRNA/fold_compound.h"
  #include "ViennaRNA/sampling/basic.h"
  #include "ViennaRNA/mfe.h"
  #include <ViennaRNA/utils/structures.h>
}



void addShortestPathDirected(short * pairtable, vector <vector<double>> &e_distances, double weight) {
    vector <vector<double>> tmp_distances(pairtable[0], vector<double>(pairtable[0]));
    for (int i = 1; i<=pairtable[0]; i++) {
        for (int j = i + 1; j<=pairtable[0]; j++) {
            double distance = 0.;
            int k = j;
            while (k > i){
                if (pairtable[k] == 0){
                    distance += tmp_distances[i-1][k-2] + 1;
                    break;
                } else if ((j > pairtable[k]) && (pairtable[k] >= i)){
                    k = pairtable[k];
                    distance += 1;
                    if (k > i){
                        k -= 1;
                        distance += 1;
                    }
                } else {
                    k -= 1;
                    distance += 1;
                }
            }
            tmp_distances[i-1][j-1] += distance;

        }
    }
    for (int i = 1; i<=pairtable[0]; i++) {
        for (int j = i + 1; j<=pairtable[0]; j++) {
            e_distances[i-1][j-1] += tmp_distances[i-1][j-1] * weight;
        }
    }

}


void addDistancesRedundantCallback(const char *structure, void *data)
{

    if (structure) {
        struct sampling_data     *d      = (struct sampling_data *)data;
        vector <vector<double>>*  exp_d = d->expected_distance;
        bool undirected = d->undirected;
        short *pt = vrna_ptable(structure);
        if (undirected){
            Graph g(pt);
            g.addDistances(*exp_d, 1/d->nr_samples);
            free(pt);
        } else {
            addShortestPathDirected(pt, *exp_d, 1/d->nr_samples);
        }

    }
}

void addDistancesNonRedundantCallback(const char *structure, void *data)
{
    if (structure) {
        struct nr_sampling_data     *d      = (struct nr_sampling_data *)data;
        vrna_fold_compound_t  *fc     = d->fc;
        double                kT      = d->kT;
        double                ens_en  = d->ens_en;
        double                *prob_sum  = d->prob_sum;
        vector <vector<double>>*   exp_d = d->expected_distance;


        double                e         = vrna_eval_structure(fc, structure);
        double                prob      = exp((ens_en - e) / kT);
        bool undirected = d->undirected;
        prob_sum[0] += prob;
        short *pt = vrna_ptable(structure);
        if (undirected) {
            Graph g(pt);
            g.addDistances(*exp_d, prob);
            free(pt);
        } else {
            addShortestPathDirected(pt, *exp_d, prob);

        }


    }
}


vector <vector<double>> edSampleRedundant(vrna_fold_compound_t *fc, int nr_samples, bool undirected) {
    vector <vector<double>> e_distance(fc->length, vector<double>(fc->length));
    double mfe = (double)vrna_mfe(fc, NULL);
    vrna_exp_params_rescale(fc, &mfe);
    vrna_pf(fc, NULL);
    struct sampling_data data;
    data.fc = fc;
    data.expected_distance = &e_distance;
    data.nr_samples = nr_samples;
    data.undirected = undirected;

    vrna_pbacktrack_cb(
            fc,
            nr_samples,
            &addDistancesRedundantCallback,
            (void *) &data,
            VRNA_PBACKTRACK_DEFAULT
    );
    return e_distance;
}

vector <vector<double>> edSampleNonRedundant(vrna_fold_compound_t *fc, int nr_samples, bool undirected) {
    vector <vector<double>> e_distance(fc->length, vector<double>(fc->length));
    double mfe = (double)vrna_mfe(fc, NULL);
    vrna_exp_params_rescale(fc, &mfe);
    double ens_en = vrna_pf(fc, NULL);
    double kT = fc->exp_params->kT / 1000.;
    double prob_sum = 0.f;


    struct nr_sampling_data data;
    data.fc = fc;
    data.expected_distance = &e_distance;
    data.ens_en = ens_en;
    data.kT = kT;
    data.prob_sum = &prob_sum;
    data.undirected = undirected;

    vrna_pbacktrack_cb(
            fc,
            nr_samples,
            &addDistancesNonRedundantCallback,
            (void *) &data,
            VRNA_PBACKTRACK_NON_REDUNDANT
    );
    return e_distance;
}

vector <vector<double>> edPThresholdSample(vrna_fold_compound_t *fc, double threshold, bool undirected) {
    vector <vector<double>> e_distance(fc->length, vector<double>(fc->length));
    double mfe = (double)vrna_mfe(fc, NULL);
    vrna_exp_params_rescale(fc, &mfe);
    double ens_en = vrna_pf(fc, NULL);
    double kT = fc->exp_params->kT / 1000.;
    double prob_sum = 0.f;


    struct nr_sampling_data data;
    data.fc = fc;
    data.expected_distance = &e_distance;
    data.ens_en = ens_en;
    data.kT = kT;
    data.prob_sum = &prob_sum;
    data.undirected = undirected;

    vrna_pbacktrack_mem_t nonredundant_memory = NULL;

    int nr_structures = 1;

    while ((prob_sum < threshold) && (nr_structures != 0)) {
        nr_structures = vrna_pbacktrack_resume_cb(
                fc,
                1,
                &addDistancesNonRedundantCallback,
                (void *) &data,
                &nonredundant_memory,
                VRNA_PBACKTRACK_NON_REDUNDANT
        );
    }
    vrna_pbacktrack_mem_free(nonredundant_memory);
    return e_distance;
}


void distanceIJCallback(const char *structure, void *data)
{

    if (structure) {
        struct ij_sampling_data     *d      = (struct ij_sampling_data *)data;
        int i = d->i;
        int j = d->j;
        double *distance = d->distance;
        short *pt = vrna_ptable(structure);
        Graph g(pt);
        double sp = g.shortestPath(i , j);

        distance[0] +=  sp / d->nr_samples;

        free(pt);

    }
}

double expectedDistanceIJ(vrna_fold_compound_t *fc, int nr_samples, int i, int j) {
    double mfe = (double)vrna_mfe(fc, NULL);
    vrna_exp_params_rescale(fc, &mfe);
    vrna_pf(fc, NULL);
    double distance = 0.f;
    struct ij_sampling_data data;
    data.nr_samples = nr_samples;
    data.i = i;
    data.j = j;
    data.distance = &distance;
    vrna_pbacktrack_cb(
            fc,
            nr_samples,
            &distanceIJCallback,
            (void *) &data,
            VRNA_PBACKTRACK_DEFAULT
    );
    return distance;
}

