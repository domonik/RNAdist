#include "RNAGraph.h"


Graph::Graph(int V)
{
    this->V = V;
    adj.resize(V);
}

Graph::Graph(short *pairtable)
{
    this->V = pairtable[0];
    adj.resize(V);
    this->filled = false;

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

void Graph::addEdge(int v, int w)
{
    adj[v].push_back(w); // Add w to v’s list.
}



void Graph::fillShortestPaths(vector<uint16_t>& shortestPaths){
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
                    shortestPaths[s*V+j] = d[j];
                }
            }
        }

    }
}

int Graph::shortestPath(int i, int j) {
    queue<int> q;
    vector<bool> used(V, false);
    vector<int> d(V, -1);  // Distance from i to each node

    q.push(i);
    used[i] = true;
    d[i] = 0;

    while (!q.empty()) {
        int current = q.front();
        q.pop();

        for (int k : adj[current]) {
            if (!used[k]) {
                used[k] = true;
                d[k] = d[current] + 1;
                q.push(k);
            }
            if (k == j) {
                return d[k];
            }
        }
    }

    throw std::range_error("No path found from i to j");
}


inline uint16_t& get_count(std::vector<uint16_t>& counts, int i, int j, int d, int V ) {
    return counts[i * V * V + j * V + d];
}

void Graph::distanceHistogram(vector<uint16_t> &distances) {
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
                        get_count(distances, s, j, d[j], V) += 1;
                    }
            }
            }
        }
    }


void Graph::addDistances(vector <vector<double>> &e_distances, double weight) {
    for (int s = 0; s < V; s++) {
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
                    e_distances[s][j] += d[j] * weight;
                }
            }
        }
    }
}







