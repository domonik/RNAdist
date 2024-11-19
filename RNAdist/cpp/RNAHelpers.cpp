//
// Created by rabsch on 18.11.24.
//

#include "RNAHelpers.h"



vrna_fold_compound_t *swigFcToFc(PyObject *swig_fold_compound) {
    SwigPyObject *swf = (SwigPyObject*) swig_fold_compound;
    vrna_fold_compound_t *fc = (vrna_fold_compound_t*) swf->ptr;
    return fc;
}

