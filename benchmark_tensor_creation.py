
import torch
import numpy as np
import time
import timeit

def benchmark():
    batch_size = 64
    # Simulate data
    batch_action = tuple(np.random.randint(0, 10, size=batch_size).tolist())
    batch_reward = tuple(np.random.randn(batch_size).tolist())
    batch_done = tuple(np.random.randint(0, 2, size=batch_size).astype(float).tolist())
    batch_gamma = tuple(np.random.uniform(0.9, 0.99, size=batch_size).tolist())

    # is_weights is a numpy array
    is_weights = np.random.uniform(0, 1, size=batch_size).astype(np.float32)

    device = torch.device("cpu") # Testing on CPU as sandbox has no GPU

    print(f"Benchmarking with batch_size={batch_size} on {device}")

    def method_original():
        action_batch = torch.tensor(batch_action, dtype=torch.long).unsqueeze(1).to(device)
        reward_batch = torch.tensor(batch_reward, dtype=torch.float32).to(device)
        done_batch = torch.tensor(batch_done, dtype=torch.float32).to(device)
        weights_batch = torch.tensor(is_weights, dtype=torch.float32).to(device)
        gamma_batch = torch.tensor(batch_gamma, dtype=torch.float32).to(device)
        return action_batch, reward_batch, done_batch, weights_batch, gamma_batch

    def method_optimized_device_arg():
        action_batch = torch.tensor(batch_action, dtype=torch.long, device=device).unsqueeze(1)
        reward_batch = torch.tensor(batch_reward, dtype=torch.float32, device=device)
        done_batch = torch.tensor(batch_done, dtype=torch.float32, device=device)
        weights_batch = torch.tensor(is_weights, dtype=torch.float32, device=device)
        gamma_batch = torch.tensor(batch_gamma, dtype=torch.float32, device=device)
        return action_batch, reward_batch, done_batch, weights_batch, gamma_batch

    def method_as_tensor_numpy():
        # Convert tuple to numpy first?
        # Note: converting tuple to numpy adds overhead.
        # But for is_weights which is already numpy:
        weights_batch = torch.as_tensor(is_weights, dtype=torch.float32, device=device)

        # For tuples, maybe sticking to torch.tensor is fine, or converting to numpy first?
        # Let's try converting tuples to numpy arrays first
        action_batch = torch.as_tensor(batch_action, dtype=torch.long, device=device).unsqueeze(1)
        reward_batch = torch.as_tensor(batch_reward, dtype=torch.float32, device=device)
        done_batch = torch.as_tensor(batch_done, dtype=torch.float32, device=device)
        gamma_batch = torch.as_tensor(batch_gamma, dtype=torch.float32, device=device)
        return action_batch, reward_batch, done_batch, weights_batch, gamma_batch

    # Warmup
    for _ in range(100):
        method_original()
        method_optimized_device_arg()
        method_as_tensor_numpy()

    # Measure
    n_iter = 10000

    t0 = timeit.timeit(method_original, number=n_iter)
    t1 = timeit.timeit(method_optimized_device_arg, number=n_iter)
    t2 = timeit.timeit(method_as_tensor_numpy, number=n_iter)

    print(f"Original: {t0:.4f} s")
    print(f"Optimized (device arg): {t1:.4f} s")
    print(f"Optimized (as_tensor): {t2:.4f} s")

    print(f"Speedup (device arg): {t0/t1:.2f}x")
    print(f"Speedup (as_tensor): {t0/t2:.2f}x")

if __name__ == "__main__":
    benchmark()
