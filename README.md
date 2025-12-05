# Setting up environment
```
mamba env create -f conda/environment.yml
conda activate abide-hgnn
```

# Data installation
To install data (you can copy and paste these):
```
git clone git@github.com:preprocessed-connectomes-project/abide.git
rm -fr abide/.git
rm abide/.gitignore
python abide/download_abide_preproc.py -d rois_ho -p cpac -s filt_noglobal -o .
```

# Generate graphs
```
python rois_to_graph.py && python rois_to_hypergraph.py
```
