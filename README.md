# Kinetic classification in rodents
Keras implementation to automatically label rodent behaviour of on video data, based on paper [Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://arxiv.org/abs/1705.07750). This is a modified version of the pioneer [sebastiantiesmeyer](https://github.com/sebastiantiesmeyer/autoscore_3d) implementation.

The model uses transfer learning of the model by the original paper. That is, their network is re-used and the last layer is dropped and replaced by layers suited for our purposes. The model currently assumes a dataset with all video frames labeled for _exploration_ stacked in one `.h5` file. This specific repository is restructured from the master to accommodate different metrics and flexible use.

This repository is by no means a package. It is currently merely under construction.


###### Environment
It is suggested to create an anaconda environment for this repository using the environment.yml file:
```
cd ~/autoscore_3d
conda env create -f environment.yml
```
To use the environment `source activate autoscore1` (improved name pending).

Original Repository info
======
## keras-kinetics-i3d
Keras implementation (including pretrained weights) of Inflated 3d Inception architecture reported in the paper [Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://arxiv.org/abs/1705.07750).

Original implementation by the authors can be found in this [repository](https://github.com/deepmind/kinetics-i3d).

## Sample Data (for Evaluation)
Similar to the original implementation, we evaluate the keras models using the RGB sample and Optical Flow sample (processed from video data) provided in the repository of the authors (see the **data/** directory). Details about the [preprocessing techniques](https://github.com/deepmind/kinetics-i3d#sample-data-and-preprocessing) applied to the data are specified in the authors' repository.

## Usage
```
python evaluate_sample.py

or

[For help]
python evaluate_sample.py -h
```

With default flags settings, the `evaluate_sample.py` script builds two I3d Inception architecture (2 stream: RGB and Optical Flow), loads their respective pretrained weights and evaluates RGB sample and Optical Flow sample obtained from video data.  

You can set flags to evaluate model using only one I3d Inception architecture (RGB or Optical Flow) as shown below:

```
# For RGB
python evaluate_sample.py --eval-type rgb

# For Optical Flow
python evaluate_sample.py --eval-type flow
```

Addtionally, as described in the paper (and the authors repository), there are two types of pretrained weights for RGB and Optical Flow models respectively. These are;
- RGB I3d Inception:
    - Weights Pretrained on Kinetics dataset only
    - Weights pretrained on Imagenet and Kinetics datasets
- Optical Flow I3d Inception:
    - Weights Pretrained on Kinetics dataset only
    - Weights pretrained on Imagenet and Kinetics datasets

The above usage examples loads weights pretrained on Imagenet and Kinetics datasets. To load weight pretrained on Kinetics dataset only add the flag **--no-imagenet-pretrained** to the above commands. See an example below:

```

# RGB I3d Inception model pretrained on kinetics dataset only
python evaluate_sample.py --eval-type rgb --no-imagenet-pretrained
```

## Requirements
- Keras
- Keras Backend: Tensorflow (tested) or Theano (not tested) or CNTK (not tested)
- h5py

## License
- All code in this repository are licensed under the MIT license as specified by the LICENSE file.
- The i3d (rgb and flow) pretrained weights were ported from the ones released [Deepmind](https://deepmind.com) in this [repository](https://github.com/deepmind/kinetics-i3d) under [Apache-2.0 License](https://github.com/deepmind/kinetics-i3d/blob/master/LICENSE)
