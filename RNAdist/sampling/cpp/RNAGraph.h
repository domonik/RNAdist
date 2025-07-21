

#ifndef RNADIST_RNAGRAPH_H
#define RNADIST_RNAGRAPH_H

#endif //RNADIST_RNAGRAPH_H

#include<bits/stdc++.h>

using namespace std;


class Graph {
    int V;    // No. of vertices

    // Pointer to an array containing adjacency
    // lists
    vector <list<int>> adj;


    bool filled;

    void resizePaths();


public:
    Graph(int V);  // Constructor
    Graph(short *pairtable); // Constructor for ViennaRNA pairtable


    // function to add an edge to graph
    void addEdge(int v, int w);

    // function to fill the shortest paths
    vector <vector<uint16_t>> getShortestPaths();
    void fillShortestPaths(vector <uint16_t>& shortestPaths);

    void addDistances(vector <vector<double>> &e_distances, double weight);
    void distanceHistogram(vector<uint16_t> &distances);
    // function to get shortest path between two nodes
    int shortestPath(int i, int j);

};

