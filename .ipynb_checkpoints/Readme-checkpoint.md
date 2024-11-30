# Image Super-Resolution via Iterative Refinement for soil tomography

## Prerequisites

Install dependencies:

pip install -r requirements.txt

Ensure you have a compatible GPU environment

Download a pretrained model from (link) and put it into pretrained_models folder

Put your data into example data lr_32 folder

## Running Inference

Prepare your configuration file (e.g., config/inference/32_256/deep.yaml).

Use the following command:

python sr3.py --phase val -c config/inference/32_256/deep.yaml

--phase: Specify the operation mode (train or val). For inference, use val.

--config: Path to the configuration file (YAML).

--num_iterations: Number of inference iterations to improve the structural quality of the super-resolved image (default: 15).


## Acknowledgments

This repository is based on [Image-Super-Resolution-via-Iterative-Refinement](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement) by [Janspiry](https://github.com/Janspiry). The original implementation has been modified to [briefly describe your major changes, e.g., "support grayscale CT data and implement additional features for soil CT analysis"].
