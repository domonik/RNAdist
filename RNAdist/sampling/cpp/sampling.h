#include<bits/stdc++.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;


extern "C"
{
  #include "ViennaRNA/fold_compound.h"
}

using namespace std;

// This class represents a directed graph using
// adjacency list representation
class Graph
{
    int V;    // No. of vertices

    // Pointer to an array containing adjacency
    // lists
    vector<list<int>> adj;
public:
    Graph(int V);  // Constructor

    // function to add an edge to graph
    void addEdge(int v, int w);


    void pairTableToGraph(short *pairtable);
    void AddDistances(py::array_t<double>* distances, double weight);
};

Graph::Graph(int V)
{
    this->V = V;
    adj.resize(V);
}

void Graph::addEdge(int v, int w)
{
    adj[v].push_back(w); // Add w to v’s list.
}

void Graph::pairTableToGraph(short *pairtable)
{
    for (int i = 0; i< pairtable[0]; i++){
        if (pairtable[i+1] > 0){
                addEdge(i, pairtable[i+1] - 1);
        }
        if(i < pairtable[0] - 1){
            addEdge(i, i+1);
            addEdge(i+1, i);
        }

    }
}

void Graph::AddDistances(py::array_t<double>* distances, double weight){
    auto idx = distances->mutable_unchecked<2>();
    for (int s=0; s<V; s++){
        queue<int> q;
        vector<bool> used(V);
        vector<int> d(V);
        q.push(s);
        used[s] = true;
        while (!q.empty()) {
            int i = q.front();
            q.pop();
            for (int j: adj[i]) {
                if (!used[j]) {
                    used[j] = true;
                    q.push(j);
                    d[j] = d[i] + 1;
                    idx(s, j) += d[j] * weight;
                }
            }
        }

    }
}


typedef struct {
    PyObject_HEAD
    void *ptr;
    void *ty;
    int own;
    PyObject *next;
    PyObject *dict;
} SwigPyObject;


