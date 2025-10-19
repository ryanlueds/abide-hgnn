To install data (you can either copy these commands as is if you're in the root directory of this repo):
```
git clone git@github.com:preprocessed-connectomes-project/abide.git
cd abide
python abide/download_abide_preproc.py -d rois_cc200 -p cpac -s filt_noglobal -o ..
```

If you want to clone the preprocessed abide data elsewhere, change the `-o ..` to `-o path/to/abide-hgnn`.
