# DARTset

This is the toolkit for the DARTset project.

To visualize the data, please follow the instructions:

## Environment

Please make sure you have the following dependencies installed:
* numpy
* cv2
* imageio
* pytorch
* pytorch3d (>= 0.6)
* [manotorch](https://github.com/lixiny/manotorch.git)



## Data

Please download the data from [DropBox](https://tinyurl.com/2p875pa3) and put them in the `data/DARTset` folder.

Then download [MANO](https://mano.is.tue.mpg.de) from the official website and put it in the `assets` folder.

Your directory should look like this:

```
.
├── DARTset.py
├── DARTset_utils.py
├── assets
│   └── mano_v1_2
├── data
│   └── DARTset
│       ├── train
│       │   ├── 0
│       │   ├── 0_wbg
│       │   ├── part_0.pkl
│       │   |-- ...
│       └── test
```

## Visualization

```python
python DARTset.py
```

You can modify this [line](https://github.com/DART2022/DARTset/blob/f619f609b1902b344fc5bbba57d080763a5496eb/DARTset.py#L175) in DARTset.py to change the `train/test` data split.