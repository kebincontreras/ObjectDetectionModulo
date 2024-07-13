import torch
import time

def measure_time(tensor_size, num_iterations):
    a = torch.randn(tensor_size, tensor_size, device='cuda')
    b = torch.randn(tensor_size, tensor_size, device='cuda')
    c = torch.randn(tensor_size, tensor_size, device='cuda')

    start_time = time.time()

    for _ in range(num_iterations):
        d = torch.matmul(a, b)
        e = d + c
        f = torch.relu(e)

    end_time = time.time()
    elapsed_time = end_time - start_time

    return elapsed_time

if __name__ == "__main__":
    tensor_size = 20000
    num_iterations = 1000

    time_taken = measure_time(tensor_size, num_iterations)
    print(f"Tiempo tomado para {num_iterations} operaciones complejas en matrices de tamaño {tensor_size}x{tensor_size}: {time_taken:.6f} segundos")
