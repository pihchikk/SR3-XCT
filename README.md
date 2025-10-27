# SR3-XCT

![License](https://img.shields.io/github/license/pihchikk/SR3-XCT?style=flat&logo=opensourceinitiative&logoColor=white&color=blue)
[![OSA-improved](https://img.shields.io/badge/improved%20by-OSA-yellow)](https://github.com/aimclub/OSA)

Built with:

![numpy](https://img.shields.io/badge/NumPy-013243.svg?style=flat&logo=NumPy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458.svg?style=flat&logo=pandas&logoColor=white)
![tqdm](https://img.shields.io/badge/tqdm-FFC107.svg?style=flat&logo=tqdm&logoColor=black)

## Overview

SR3-XCT enhances the quality of soil tomography images through advanced super-resolution techniques, enabling clearer and more detailed visualizations. This project provides significant value by improving the interpretability of low-resolution images, which is crucial for accurate analysis and decision-making in soil studies.

## Table of Contents

- [Core features](#core-features)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## Core features

1. **Image Super-Resolution**: The core functionality of the SR3-XCT project is to perform image super-resolution using iterative refinement techniques, enhancing the quality of low-resolution images.
2. **Custom Dataset Handling**: The project includes a custom dataset class (LRHRDataset) that supports loading low-resolution and high-resolution images from various data sources, including LMDB and image files.
3. **Model Training and Optimization**: The project provides a structured approach to model training, including parameter optimization and loss calculation, allowing for efficient training of the super-resolution model.
4. **Image Quality Metrics**: The project implements various image quality metrics, such as PSNR and SSIM, to evaluate the performance of the super-resolution outputs against ground truth images.
5. **Checkpointing and Resuming Training**: The project includes functionality to save and load model checkpoints, allowing users to resume training from a specific state without losing progress.

## Installation

Install SR3-XCT using one of the following methods:

**Build from source:**

1. Clone the SR3-XCT repository:
   ```sh
   git clone https://github.com/pihchikk/SR3-XCT
   ```

2. Navigate to the project directory:
   ```sh
   cd SR3-XCT
   ```

3. Install the project dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Getting Started

(Instructions for getting started will be added here.)

## Contributing

- **[Report Issues](https://github.com/pihchikk/SR3-XCT/issues)**: Submit bugs found or log feature requests for the project.

## License

This project is protected under the Apache License 2.0. For more details, refer to the [LICENSE](https://github.com/pihchikk/SR3-XCT/tree/main/LICENSE) file.

## Citation

If you use this software, please cite it as below.

### APA format:

    pihchikk (2024). SR3-XCT repository [Computer software]. https://github.com/pihchikk/SR3-XCT

### BibTeX format:

    @misc{SR3-XCT,
        author = {pihchikk},
        title = {SR3-XCT repository},
        year = {2024},
        publisher = {github.com},
        journal = {github.com repository},
        howpublished = {\url{https://github.com/pihchikk/SR3-XCT.git}},
        url = {https://github.com/pihchikk/SR3-XCT.git}
    }