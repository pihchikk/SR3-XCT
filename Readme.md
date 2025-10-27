![License](https://img.shields.io/github/license/pihchikk/SR3-XCT?style=flat&logo=opensourceinitiative&logoColor=white&color=blue)

# Image Super-Resolution via Iterative Refinement for Soil Tomography

This repository implements diffusion-based super-resolution for soil tomography, adapted from [Image-Super-Resolution-via-Iterative-Refinement](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement).

---
<p align="center">
<table>
  <tr>
    <td><img src="example data/lr_32/input_image.png" width="400"/></td>
    <td><img src="example data/hr_256/input_image.png" width="400"/></td>
  </tr>
  <tr>
    <td align="center">Low-resolution input (32x32)</td>
    <td align="center">High-resolution output (256x256)</td>
  </tr>
</table>
</p>

## Installation

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

## Running inference using pretrained models

1. Ensure the model weights (```.pth``` or ```.safetensors```)  are in pretrained models folder
   
2. **Prepare Input Data**

   Place your low-resolution input images in the ```example_data/lr_32``` or ```example_data/lr_64``` folder

4. **Running inference**

   Run for __x4__ upscaling:
   ```bash
   python sr3.py --phase val -c config/inference/64_256/x4_deep.yaml
   ```
   
   Run for__x8__ upscaling:
   ```bash
   python sr3.py --phase val -c config/inference/32_256/x8.yaml
   ```

    Arguments 

    ```--phase```: Specify the mode (val for inference).  
    ```-c```: Path to the YAML configuration file.  

## Contributing

- **[Report Issues](https://github.com/pihchikk/SR3-XCT/issues)**: Submit bugs found or log feature requests for the project.

## License

This project is protected under the Apache License 2.0. For more details, refer to the [LICENSE](https://github.com/pihchikk/SR3-XCT/tree/main/LICENSE) file.

## Citation

If you use this software, please cite it as below...
