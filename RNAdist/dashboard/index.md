# Welcome

This is the RNAdist Webserver used to calculate nucleotide distances on the ensemble of RNA secondary structures.

## What does it do

The thermodynamic ensemble of RNA secondary structures refers to the collection of all possible conformations an RNA 
molecule can adopt, each associated with a specific probability determined by its free energy.

Each RNA secondary structure can be represented as a graph, where nucleotides are nodes and connections are edges. 
For simplicity, both backbone links and hydrogen bonds between base pairs are assigned an edge distance of 1, allowing 
the structure to be analyzed using standard graph-theoretical methods.

By sampling structures from the RNA thermodynamic ensemble, one can compute the distances between any two nucleotides
$i$ and $j$ across the sampled graphs. This generates a distribution of distances, reflecting how often different spatial 
separations occur in the ensemble and providing insights into the flexibility and connectivity of the RNA molecule.

also [Watch the Video](#demo)

 
## How to use
Use the Navbar at the top to enter your session token. All computations will be associated with this token, so make 
sure to save it if you want to access your data later.

Next, go to the [**Submission**](/submission) page to enter your RNA sequence and folding/sampling parameters. 
Once your job is complete, it will appear as "finished" in the submissions table.

Finally, visit the [**Visualization**](/visualization) page to explore nucleotide distance distributions and view the 
sampled RNA structures.
<div class="warning">
<strong>Warning:</strong> Note that data will be stored no longer than 7 days and there is the possibility that it gets deleted even earlier 
depending on website traffic. Further all our jobs will be public. Meaning everyone that knows your token can access it.
</div>
