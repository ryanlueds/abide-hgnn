# Setting up environment
  1. make sure you are in your `conda/` folder, then enter those commands in the terminal.
  2. ```mamba env create -f environment.yml```
  3. ```conda activate abide-hgnn```

# Data installation
To install data (you can copy and paste these):
```
git clone git@github.com:preprocessed-connectomes-project/abide.git
cd abide
rm -fr .git
rm .gitignore
python download_abide_preproc.py -d rois_cc200 -p cpac -s filt_noglobal -o ..
```

# Generate graphs
```
python rois_to_graph.py && python rois_to_hypergraph.py
```
