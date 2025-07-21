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

inline uint8_t encodeDotBracket(char c) {
    switch (c) {
        case '.': return 0;
        case '(': return 1;
        case ')': return 2;
        default: throw std::invalid_argument("Invalid character");
    }
}

char decodeDotBracket(uint8_t val) {
    switch (val) {
        case 0: return '.';
        case 1: return '(';
        case 2: return ')';
        default: return '?';
    }
}


std::vector<uint8_t> encodeStructure(const std::string& s) {
    std::vector<uint8_t> packed;
    packed.reserve((s.size() + 3) / 4);
    uint8_t current = 0;
    int bitsFilled = 0;

    for (char c : s) {
        uint8_t val = encodeDotBracket(c);
        current = current * 3 + val;
        bitsFilled++;

        if (bitsFilled == 4) {  // 3^4 = 81 < 256, pack 4 digits in 1 byte
            packed.push_back(current);
            current = 0;
            bitsFilled = 0;
        }
    }

    if (bitsFilled > 0) {
        // Pad remaining digits (acts like base-3 left padding)
        while (bitsFilled < 4) {
            current *= 3;
            bitsFilled++;
        }
        packed.push_back(current);
    }

    return packed;
}

std::string decodeStructure(const std::vector<uint8_t>& packed, size_t originalLength) {
    std::string result;
    for (uint8_t byte : packed) {
        uint8_t temp = byte;
        uint8_t vals[4] = {};
        for (int i = 3; i >= 0; --i) {
            vals[i] = temp % 3;
            temp /= 3;
        }
        for (int i = 0; i < 4 && result.size() < originalLength; ++i) {
            result += decodeDotBracket(vals[i]);
        }
    }
    return result;
}



void trackDistancesCallback(const char *structure, void *data) {
    if (structure) {
        struct tracking_data *d = (struct tracking_data *) data;

        std::unordered_map<std::vector<uint8_t>, int, PackedKeyHash, PackedKeyEqual>*cache = d->cache;
        // short *pt_raw = vrna_ptable(structure);
        auto encoded = encodeStructure(structure);
        (*cache)[encoded] += 1;
    }
}

void trackDistancesCallbackIJ(const char *structure, void *data) {
    if (structure) {
        struct tracking_data *d = (struct tracking_data *) data;

        vector <uint16_t>*counts = d->counts;

        short *pt = vrna_ptable(structure);
        Graph g(pt);
        int sp = g.shortestPath(d->i , d->j);
        (*counts)[sp] += 1;
        free(pt);

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

void ensurePartitionFunctionReady(vrna_fold_compound_t *fc) {
    if (fc->exp_params == NULL || fc->exp_matrices == NULL) {
        double mfe = static_cast<double>(vrna_mfe(fc, NULL));
        vrna_exp_params_rescale(fc, &mfe);
        vrna_pf(fc, NULL);
    }
}




StructureCache sampleStructuresAndDistances(vrna_fold_compound_t *fc, int nr_samples) {

    StructureCache cache;


    struct tracking_data data;
    data.fc = fc;
    data.cache = &cache;

    vrna_pbacktrack_cb(
            fc,
            nr_samples,
            &trackDistancesCallback,
            (void *) &data,
            VRNA_PBACKTRACK_DEFAULT
    );


    return cache;


}



vector <uint16_t> distancesFromStructureCache(const StructureCache& cache, int n) {
    vector<uint16_t> counts(n * n * n, 0);
    vector <uint16_t> distances(n * n);
    for (const auto &[key, value]: cache) {
        // key is std::vector<uint8_t>
        // value is int
        std::string structure = decodeStructure(key, n);
        short *pt = vrna_ptable(structure.c_str());
        Graph g(pt);
        g.fillShortestPaths(distances);
        for (int i = 0; i < n; ++i) {
            int iidx = i * n * n;
            for (int j = i; j < n; ++j) {
                uint16_t d = distances[i * n + j];
                counts[iidx + d * n + j] += value;
            }
        }

        free(pt);
    }
    return counts;

}

vector <uint16_t> distancesFromStructureCache(const StructureCache& cache, int n, int i, int j) {
    vector<uint16_t> counts(n, 0);
    for (const auto &[key, value]: cache) {
        std::string structure = decodeStructure(key, n);
        short *pt = vrna_ptable(structure.c_str());
        Graph g(pt);
        int sp = g.shortestPath(i , j);
        counts[sp] += value;
        free(pt);
    }
    return counts;

}


tuple<vector <uint16_t>, StructureCache> trackDistances(vrna_fold_compound_t *fc, int nr_samples, int i, int j) {
    int n = fc->length;

    ensurePartitionFunctionReady(fc);

    StructureCache cache = sampleStructuresAndDistances(fc, nr_samples);

    vector <uint16_t> counts = distancesFromStructureCache(cache, n, i, j);

    return {counts, cache};
}

tuple<vector <uint16_t>, StructureCache>  trackDistances(vrna_fold_compound_t *fc, int nr_samples) {
    int n = fc->length;
    ensurePartitionFunctionReady(fc);

    cout << "Done with pf calc";
    StructureCache cache = sampleStructuresAndDistances(fc, nr_samples);
    cout << "Done with sampling";


    vector <uint16_t> counts = distancesFromStructureCache(cache, n);
    cout << "Done with distance Calc";



    return {counts, cache};

}


vector <vector<double>> edSampleRedundant(vrna_fold_compound_t *fc, int nr_samples, bool undirected) {
    vector <vector<double>> e_distance(fc->length, vector<double>(fc->length));
    ensurePartitionFunctionReady(fc);
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
        int sp = g.shortestPath(i , j);

        distance[0] +=  static_cast<double>(sp) / d->nr_samples;

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

