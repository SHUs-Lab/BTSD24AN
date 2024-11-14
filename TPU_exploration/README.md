# Exploration of optimal TPU architecture for DETR DNN model

This project converts DETR (Detection Transformer) architecture into a format suitable for Scale-Sim simulation and finds optimal TPU architectures through Design Space Exploration using Genetic Algorithm.

## Installation
1. Scale-Sim, a cycle-accurate simulation tool that estimates latency
     ```
     $ pip3 install scalesim
     ```
  
3. Required Python libraries to load pre-trained DETR and create the Genetic Algorithm
   ```
   $ pip3 install torch transformers pygad
   ```

## Usage

1. Generate DETR architecture CSV file
```bash
python get_detr_csv.py
```
This script:
- Loads pretrained DETR model from Transformers library
- Converts Transformer layers to custom convolutional representations
- Generates a CSV file compatible with Scale-Sim

2. Run Design Space Exploration
```bash
python dse.py
```
This script performs optimization using Genetic Algorithm to find optimal TPU configurations.

## Files
- `get_detr_csv.py`: Generates Scale-Sim compatible CSV from DETR architecture
- `dse.py`: Design Space Exploration using Genetic Algorithm
- `detr.csv`: CSV file produced by get_detr_csv.py
TPU architecture exploration was performed using Genetic Algorithm from [PyGAD](https://pygad.readthedocs.io/en/latest/) library, while the latency evaluation was employed using the [Scale-Sim](https://github.com/scalesim-project/scale-sim-v2) tool.

