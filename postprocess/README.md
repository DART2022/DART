# DART GUI post processing

Please refer to the [step3 of DART GUI](https://dart2022.github.io/)
for the exporting process.


## Requirements

1. Download `mano_v1_2.zip` from the [MANO website](https://mano.is.tue.mpg.de/), 
unzip the file and copy `MANO_RIGHT.pkl` to `extra_data/hand_wrist/` folder:

2. Install `pytorch3d` according to official [INSTALL.md](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)

## Usage

1. Copy the `xxx\Build_Hand\ExportData\2022-08-02_17-07-21` to `convert.py`.

2. `Python convert.py` and all the ground truths are exported to `output.pkl` (include mano pose, 2d/3d joint).