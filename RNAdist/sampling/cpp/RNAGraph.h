

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
    vector <vector<int>> shortestPaths;

    void fillShortestPaths();

    bool filled;

    void resizePaths();


public:
    Graph(int V);  // Constructor
    Graph(short *pairtable); // Constructor for ViennaRNA pairtable


    // function to add an edge to graph
    void addEdge(int v, int w);

    // function to fill the shortest paths
    vector <vector<int>> getShortestPaths();

    void addDistances(vector <vector<double>> &e_distances, double weight);

    // function to get shortest path between two nodes
    double shortestPath(int i, int j);

};

