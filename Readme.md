# Image Super-Resolution via Iterative Refinement for Soil Tomography

This repository implements super-resolution techniques for soil tomography, adapted from [Image-Super-Resolution-via-Iterative-Refinement](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement).

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

## Setup and running inference using pretrained models

1. **Install Dependencies**  
   Run:
   ```bash
   pip install -r requirements.txt
   ```

2. [Download the pretrained model](https://drive.google.com/file/d/12eU2cIx4NetzOgkx3rppj-4vpPVxuU0M/view?usp=sharing) and put it into pretrained models folder
   
3. **Prepare Input Data**
   Place your low-resolution input images in the example_data/lr_32/ folder

5. **Running inference**
   Run:
   ```bash
   python sr3.py --phase val -c config/inference/32_256/deep.yaml
   ```

    Arguments 

    ```--phase```: Specify the mode (val for inference).  
    ```-c```: Path to the YAML configuration file.  
    ```--num_iterations```: (Optional)  Number of refinement steps for better structure (default is 1).

