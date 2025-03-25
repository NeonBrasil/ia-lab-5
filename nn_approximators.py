import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MaxAbsScaler
import time
from datetime import datetime

# Function to run a single simulation with a specific architecture
def run_simulation(x, y, hidden_layer_sizes, activation='tanh', solver='adam', 
                  learning_rate='adaptive', max_iter=400, verbose=False):
    # Create and train the neural network
    regr = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        activation=activation,
        solver=solver,
        learning_rate=learning_rate,
        n_iter_no_change=max_iter,
        verbose=verbose
    )
    
    regr.fit(x, y)
    y_est = regr.predict(x)
    
    # Return the trained model and its predictions
    return regr, y_est

# Function to run multiple simulations and calculate statistics
def run_multiple_simulations(x, y, hidden_layer_sizes, num_simulations=10, **kwargs):
    errors = []
    best_model = None
    best_error = float('inf')
    best_y_est = None
    
    for i in range(num_simulations):
        regr, y_est = run_simulation(x, y, hidden_layer_sizes, **kwargs)
        error = regr.best_loss_
        errors.append(error)
        
        # Keep track of the best model
        if error < best_error:
            best_error = error
            best_model = regr
            best_y_est = y_est
    
    # Calculate statistics
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    
    return {
        'mean_error': mean_error,
        'std_error': std_error,
        'best_model': best_model,
        'best_y_est': best_y_est,
        'all_errors': errors
    }

# Function to test different architectures on a dataset
def test_architectures(test_file, architectures, num_simulations=10):
    # Load data
    print(f'Loading {test_file}')
    arquivo = np.load(test_file)
    x = arquivo[0]
    
    # Scale the target values
    scale = MaxAbsScaler().fit(arquivo[1])
    y = np.ravel(scale.transform(arquivo[1]))
    
    results = []
    
    # Test each architecture
    for i, arch in enumerate(architectures):
        print(f'Testing architecture {i+1}: {arch}')
        result = run_multiple_simulations(x, y, arch, num_simulations=num_simulations)
        results.append(result)
        
        print(f'  Mean error: {result["mean_error"]:.6f}')
        print(f'  Std error: {result["std_error"]:.6f}')
    
    return x, y, results

# Function to plot results
def plot_results(test_name, x, y, results, architectures):
    num_archs = len(architectures)
    
    # Create a figure with subplots
    fig = plt.figure(figsize=[15, 5 * (num_archs + 1)])
    
    # Plot original function
    plt.subplot(num_archs + 1, 3, 1)
    plt.title('Original Function')
    plt.plot(x, y, color='green')
    
    # Plot bar chart of mean errors with error bars for std
    plt.subplot(num_archs + 1, 3, 2)
    arch_labels = [f'Arch {i+1}' for i in range(num_archs)]
    mean_errors = [result['mean_error'] for result in results]
    std_errors = [result['std_error'] for result in results]
    
    bars = plt.bar(arch_labels, mean_errors, yerr=std_errors, capsize=10)
    plt.title('Mean Error with Std Dev')
    plt.ylabel('Error')
    
    # Add architecture descriptions
    plt.subplot(num_archs + 1, 3, 3)
    plt.axis('off')
    arch_text = 'Architecture Descriptions:\n\n'
    for i, arch in enumerate(architectures):
        arch_text += f'Arch {i+1}: {arch}\n'
        arch_text += f'  Mean Error: {results[i]["mean_error"]:.6f}\n'
        arch_text += f'  Std Error: {results[i]["std_error"]:.6f}\n\n'
    plt.text(0, 0.5, arch_text, fontsize=12)
    
    # Plot each architecture's best result
    for i in range(num_archs):
        # Plot learning curve
        plt.subplot(num_archs + 1, 3, 3*i + 4)
        plt.title(f'Learning Curve - Arch {i+1}')
        plt.plot(results[i]['best_model'].loss_curve_, color='red')
        
        # Plot approximation
        plt.subplot(num_archs + 1, 3, 3*i + 5)
        plt.title(f'Function Approximation - Arch {i+1}')
        plt.plot(x, y, linewidth=1, color='green', label='Original')
        plt.plot(x, results[i]['best_y_est'], linewidth=2, color='blue', label='Approximated')
        plt.legend()
        
        # Plot error distribution
        plt.subplot(num_archs + 1, 3, 3*i + 6)
        plt.title(f'Error Distribution - Arch {i+1}')
        plt.hist(results[i]['all_errors'], bins=10, alpha=0.7)
        plt.axvline(results[i]['mean_error'], color='red', linestyle='dashed', linewidth=2)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'{test_name}_{timestamp}.png')
    plt.show()

# Main function to run all tests
def run_all_tests():
    # Define architectures to test for each problem
    # Format: [(hidden_layer_sizes_1), (hidden_layer_sizes_2), (hidden_layer_sizes_3)]
    architectures = [
        # Simple architectures
        (10,),
        # Medium complexity
        (15, 5),
        # More complex
        (20, 10, 5)
    ]
    
    # Test files
    test_files = ['teste2.npy', 'teste3.npy', 'teste4.npy', 'teste5.npy']
    
    for test_file in test_files:
        print(f'\n\n===== Processing {test_file} =====')
        test_name = test_file.split('.')[0]
        
        # For teste3, teste4, and teste5, we might need more complex architectures
        if test_file in ['teste3.npy', 'teste4.npy', 'teste5.npy']:
            custom_architectures = [
                (20,),
                (30, 15),
                (40, 20, 10)
            ]
        else:
            custom_architectures = architectures
        
        # Run the test
        x, y, results = test_architectures(test_file, custom_architectures, num_simulations=10)
        
        # Plot the results
        plot_results(test_name, x, y, results, custom_architectures)

# Run a specific test
def run_specific_test(test_file, custom_architectures=None):
    if custom_architectures is None:
        custom_architectures = [
            (10,),
            (15, 5),
            (20, 10, 5)
        ]
    
    print(f'\n\n===== Processing {test_file} =====')
    test_name = test_file.split('.')[0]
    
    # Run the test
    x, y, results = test_architectures(test_file, custom_architectures, num_simulations=10)
    
    # Plot the results
    plot_results(test_name, x, y, results, custom_architectures)

# If this script is run directly
if __name__ == "__main__":
    run_all_tests()