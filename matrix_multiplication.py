import numpy as np
import time
import multiprocessing
import matplotlib.pyplot as plt

# Define matrix size for testing
N = 200

# Generate random matrices with random values
A = np.random.rand(N, N)
B = np.random.rand(N, N)

# ---------------- basic Implementation ----------------
def basic_matrix_multiplication(A, B):
    """Performs matrix multiplication using a triple nested loop (basic approach)."""
    N = A.shape[0]
    C = np.zeros((N, N))  # Initialize result matrix with zeros
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i, j] += A[i, k] * B[k, j]  # Standard matrix multiplication formula
    return C

# ---------------- Optimized NumPy Implementation ----------------
def numpy_matrix_multiplication(A, B):
    """Uses NumPy's optimized dot product for efficient matrix multiplication."""
    return np.dot(A, B)

# ---------------- Parallelized Implementation using Multiprocessing ----------------
def parallel_multiply_worker(A, B, start, end, result_queue):
    """Worker function that computes part of the matrix multiplication for parallel processing."""
    N = A.shape[0]
    C_partial = np.zeros((end - start, N))  # Initialize partial result matrix
    for i in range(start, end):
        for j in range(N):
            C_partial[i - start, j] = np.dot(A[i, :], B[:, j])  # Compute dot product for row
    result_queue.put((start, C_partial))  # Store partial result in queue

def parallel_matrix_multiplication(A, B, num_processes=4):
    """Divides matrix multiplication among multiple processes for faster execution."""
    N = A.shape[0]  
    chunk_size = N // num_processes  # Determine how many rows each process will handle
    result_queue = multiprocessing.Queue()  
    processes = []  
    # Create and start multiple processes
    for i in range(num_processes):
        start = i * chunk_size  
        end = N if i == num_processes - 1 else (i + 1) * chunk_size  
        # Assign a process to compute a portion of the matrix multiplication
        process = multiprocessing.Process(target=parallel_multiply_worker, args=(A, B, start, end, result_queue))
        processes.append(process)  
        process.start()  

    C = np.zeros((N, N))  # Initialize result matrix

    # Retrieve and assemble computed chunks
    for _ in range(num_processes):
        start, C_partial = result_queue.get()  
        C[start:start + C_partial.shape[0], :] = C_partial  

    # Ensure all processes complete execution
    for process in processes:
        process.join()  

    return C  # Return final result


# ---------------- Measure Execution Times ----------------
times = {}

# Measure execution time for basic matrix multiplication
start_time = time.time()
basic_matrix_multiplication(A, B)
times['Basic'] = time.time() - start_time

# Measure execution time for NumPy optimized multiplication
start_time = time.time()
numpy_matrix_multiplication(A, B)
times['NumPy Optimized'] = time.time() - start_time

# Measure execution time for parallelized implementation
start_time = time.time()
parallel_matrix_multiplication(A, B, num_processes=4)
times['Parallelized'] = time.time() - start_time

# ---------------- Visualization ----------------
plt.figure()
plt.bar(times.keys(), times.values(), color=['red', 'blue', 'green'])
plt.xlabel("Implementation Method")
plt.ylabel("Execution Time (seconds)")
plt.title(f"Matrix Multiplication Performance (Size: {N}x{N})")
plt.show()  # Display the graph

# Print execution times for each method
for method, exec_time in times.items():
    print(f"{method}: {exec_time:.6f} seconds")
