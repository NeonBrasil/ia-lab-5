import numpy as np
import matplotlib.pyplot as plt
from nn_approximators import run_specific_test, run_all_tests

# Define different architectures for each test
architectures = {
    'teste2.npy': [
        (10,),           # Simple architecture with one hidden layer
        (15, 5),         # Medium complexity with two hidden layers
        (20, 10, 5)      # More complex with three hidden layers
    ],
    'teste3.npy': [
        (20,),           # Larger single hidden layer
        (30, 15),        # Larger two hidden layers
        (40, 20, 10)     # Larger three hidden layers
    ],
    'teste4.npy': [
        (25,),           # Larger single hidden layer
        (35, 20),        # Larger two hidden layers
        (50, 30, 15)     # Larger three hidden layers
    ],
    'teste5.npy': [
        (30,),           # Larger single hidden layer
        (40, 25),        # Larger two hidden layers
        (60, 40, 20)     # Larger three hidden layers
    ]
}

# Run tests for each file with its specific architectures
def run_custom_tests():
    test_files = ['teste2.npy', 'teste3.npy', 'teste4.npy', 'teste5.npy']
    
    for test_file in test_files:
        print(f"\n\n===== Processing {test_file} with custom architectures =====")
        run_specific_test(test_file, architectures[test_file])

if __name__ == "__main__":
    print("Starting neural network approximation tests...")
    print("Running custom tests for each test file with specific architectures")
    run_custom_tests()