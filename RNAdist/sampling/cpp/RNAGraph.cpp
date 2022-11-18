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

void Graph::resizePaths(){
    shortestPaths.resize(V);
    for (int i = 0; i < V; ++i){
        shortestPaths[i].resize(V);
    }
}

void Graph::fillShortestPaths(){
    resizePaths();
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
                    shortestPaths[s][j] += d[j];
                }
            }
        }

    }
    filled = true;
}

void Graph::addDistances(vector <vector<double>> &e_distances, double weight){
    if (filled){
        for(int i = 0; i != V; i++) {
            for(int j = 0; j != V; j++) {
                e_distances[i][j] += shortestPaths[i][j] * weight;
            }
        }
    }
    else {
        resizePaths();
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
                        shortestPaths[s][j] += d[j];
                        e_distances[s][j] += shortestPaths[s][j] * weight;
                    }
                }
            }
        }
    }
    filled = true;
}

vector<vector<int>> Graph::getShortestPaths(){
    if (!filled){
        fillShortestPaths();
    }
    return shortestPaths;
}






