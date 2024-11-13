import pygad
import numpy as np
import scalesim
from configparser import ConfigParser
import os
import pandas as pd

parameter_evolution = []
csv_file = 'parameter_evolution.csv'

def fitness_function(ga_instance, solution, solution_idx):
    array_height = int(solution[0])
    array_width = int(solution[1])
    ifmap_sram_sz_kb = int(solution[2])
    filter_sram_sz_kb = int(solution[3])
    ofmap_sram_sz_kb = int(solution[4])
    dataflow = solution[5]
    bandwidth = int(solution[6])
    memory_banks = int(solution[7])


    # TPU configuration for current epoch
    config = {
        "ArrayHeight": array_height,
        "ArrayWidth": array_width,
        "IfmapSramSzkB": ifmap_sram_sz_kb,
        "FilterSramSzkB": filter_sram_sz_kb,
        "OfmapSramSzkB": ofmap_sram_sz_kb,
        "Dataflow": dataflows[int(dataflow)],
        "Bandwidth": bandwidth,
        "MemoryBanks": memory_banks
    }
    
    experiment_name = f"{array_height}_{array_width}_{ifmap_sram_sz_kb}_{filter_sram_sz_kb}_{ofmap_sram_sz_kb}_{dataflow}_{bandwidth}_{memory_banks}"
    fileparser.read('dse_results/google.cfg')

    fileparser['general']['run_name'] = experiment_name
    fileparser['architecture_presets']={
        'ArrayHeight': array_height,
        'ArrayWidth': array_width,
        'IfmapSramSzkB': ifmap_sram_sz_kb,
        'FilterSramSzkB': filter_sram_sz_kb,
        'OfmapSramSzkB': ofmap_sram_sz_kb,
        'IfmapOffset':    0,
        'FilterOffset':   10000000,
        'OfmapOffset':    20000000,
        'Dataflow': dataflows[int(dataflow)],
        'Bandwidth': bandwidth,
        'MemoryBanks': memory_banks
    }
    
    # Write the TPU configuration to a file, for Scale-Sim
    with open('dse_results/google.cfg', 'w') as configfile:
        fileparser.write(configfile)

    # build the Scale-Sim command and run the simulation
    command = f"python scalesim/scale.py -c dse_results/google.cfg -t dse_results/detr.csv -p dse_results"
    os.system(command)

    # get only the total cycles number from simulation results
    results = pd.read_csv(f'dse_results/{experiment_name}/COMPUTE_REPORT.csv')
    cycles = results.iloc[:, 1].sum()
    
    parameter_evolution = [array_height,
    array_width,
    ifmap_sram_sz_kb,
    filter_sram_sz_kb,
    ofmap_sram_sz_kb,
    dataflow,
    bandwidth,
    memory_banks,
    cycles]
    
    df = pd.DataFrame([parameter_evolution], columns = ["ArrayHeight", "ArrayWidth", "IfmapSramSzkB", "FilterSramSzkB", "OfmapSramSzkB", "Dataflow", "Bandwidth", "MemoryBanks", "Cycles"])
    if not os.path.isfile(csv_file):
       df.to_csv(csv_file, index=False)
    else:
       df.to_csv(csv_file, mode='a', header=False, index=False)

    print(f"Experiment {experiment_name} finished with {cycles} cycles")
   
    # The fitness function should minimize the number of cycles, hence we return negative cycles
    return -cycles

fileparser = ConfigParser()

# Dataflow mapping strategies: weight/output/input stationary
dataflows = ["ws", "os", "is"]

# Genetic algorithm parameters
ga_instance = pygad.GA(
    num_generations=100,    # Number of generations
    num_parents_mating=20,  # Number of solutions to be selected as parents in the mating pool
    fitness_func=fitness_function,
    sol_per_pop=100,        # Number of solutions in the population
    num_genes=8,            # Number of parameters we are optimizing

    gene_space=[
        np.arange(128, 513, 128),              # ArrayHeight
        np.arange(128, 513, 128),              # ArrayWidth
        np.arange(1024.0, 10*1024.0, 1024.0),  # IfmapSramSzkB
        np.arange(1024.0, 10*1024.0, 1024.0),  # FilterSramSzkB
        np.arange(1024.0, 10*1024.0, 1024.0),  # OfmapSramSzkB
        range(len(dataflows)),                 # Dataflow type
        np.arange(500, 1201, 100),             # Bandwidth
        range(1, 4)                            # MemoryBanks
    ],
    parent_selection_type="rank",
    mutation_percent_genes=10,  # Mutation percentage
    mutation_type="adaptive",   # Type of mutation
    mutation_num_genes=[2, 1]
)

# Run the Genetic Algorithm
ga_instance.run()

# Print best solution of genetic algorithm
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}")
print("Fitness value of the best solution = {solution_fitness}")

# Decode and print the optimal TPU configuration
decoded_solution = {
    "ArrayHeight": int(solution[0]),
    "ArrayWidth": int(solution[1]),
    "IfmapSramSzkB": int(solution[2]),
    "FilterSramSzkB": int(solution[3]),
    "OfmapSramSzkB": int(solution[4]),
    "Dataflow": dataflows[int(solution[5])],
    "Bandwidth": int(solution[6]),
    "MemoryBanks": int(solution[7])
}
print("Decoded solution:", decoded_solution)
