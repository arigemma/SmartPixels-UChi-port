# Welcome to our SmartPixels ML project
## Usage
Setup environment (w/ optional GPU support)
```
source setup.sh
```
Look at the notebook `playground.ipynb` for data exploration.

Optional: Calculate feature scaling
```
python scaling.py --data-config config/v3.yml --prefix scalers/myscaler
```
Train a model
```
python train.py -c config/v3.yml -a conv2d --task regression
```
Evaluate a model
```
python evaluate.py --input /path/to/model.pt -c config/v3.yml
```

## The data
There are two datasets on zenodo: https://zenodo.org/records/10783560 (3 versions) and https://zenodo.org/records/17180303 (1 version).
The latest versions of both contain charge deposits shape `(t, x, y) = (20, 13, 21)` (or permutations thereof). Meaning short videos with 20 frames of 13x21 (x,y) each. There are 13 labels (w/ example values):
```
x-entry              129.128448
y-entry              -13.157045
z-entry              100.000000
n_x                   -1.716407
n_y                    0.092777
n_z                   -0.630184
number_eh_pairs    15612.000000
y-local                2.733230
pt                    -0.637653
cotAlpha               2.723660
cotBeta               -0.147222
y-midplane            -5.795941
x-midplane            -7.054556
```
### Remarks
For the dataset `17180303`, the first few files of each dataset seem to have a different pT distribution than the rest. E.g. mean absolute values:
![alt text](image.png)
## Existing Papers
"Smartpixels: Towards on-sensor ..." (https://arxiv.org/pdf/2312.11676) uses mixture density network to regress 4 variables (x, y, alpha, beta). 

"Sensor Co-design for smartpixels" (https://arxiv.org/pdf/2510.06588v1) analyzes the y-profile of the data. reduces the data to 14-dim arrays (y0 and 13 y-pixels) 

