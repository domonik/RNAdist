//
// Created by domonik on 18.11.24.
//
#include<bits/stdc++.h>


using namespace std;


extern "C"
{
#include "ViennaRNA/fold_compound.h"
#include <ViennaRNA/fold_compound.h>
#include <ViennaRNA/utils/basic.h>
#include <ViennaRNA/utils/strings.h>
#include <ViennaRNA/mfe.h>
#include <ViennaRNA/part_func.h>
#include <ViennaRNA/model.h>
#include <numpy/arrayobject.h>
#include <ViennaRNA/loops/external.h>
#include <ViennaRNA/alphabet.h>
}


vector <vector<double>> clote_ponty_expected_distance(vrna_fold_compound_t *fc);