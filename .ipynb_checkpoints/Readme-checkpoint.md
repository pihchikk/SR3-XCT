# Image Super-Resolution via Iterative Refinement for soil tomography

## Prerequisites

Install dependencies:

pip install -r requirements.txt

Ensure you have a compatible GPU environment (e.g., NVIDIA GeForce RTX 4090).

## Running Inference

To run inference and generate super-resolved images:

Prepare your configuration file (e.g., config/inference/32_256/deep.yaml).

Use the following command:

python sr3.py --phase val -c config/inference/32_256/deep.yaml

--phase: Specify the operation mode (train or val). For inference, use val.

--config: Path to the configuration file (YAML or JSON).

--num_iterations: Number of iterations for generating and evaluating super-resolved images (default: 15).

Output

## Acknowledgments

This repository is based on [Image-Super-Resolution-via-Iterative-Refinement](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement) by [Janspiry](https://github.com/Janspiry). The original implementation has been modified to [briefly describe your major changes, e.g., "support grayscale CT data and implement additional features for soil CT analysis"].
