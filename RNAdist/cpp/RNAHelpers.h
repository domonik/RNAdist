//
// Created by rabsch on 18.11.24.
//

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PYBIND11_DETAILED_ERROR_MESSAGES

#ifndef RNADIST_RNAHELPERS_H
#define RNADIST_RNAHELPERS_H
#endif //RNADIST_RNAHELPERS_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

extern "C"
{
#include "ViennaRNA/fold_compound.h"
}



typedef struct {
    PyObject_HEAD
    void *ptr;
    void *ty;
    int own;
    PyObject *next;
    PyObject *dict;
} SwigPyObject;

vrna_fold_compound_t *swigFcToFc(PyObject *swig_fold_compound);

