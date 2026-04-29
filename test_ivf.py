import torch
import time
from TQ_engine_lib import TurboQuant

def test_ivf():
    # Setup
    dim = 768
    num_vectors = 50000  # 50k vectors
    bits = 4
    n_list = 128
    n_probe = 8
    
    print(f"Generating {num_vectors} random vectors of dim {dim}...")
    vectors = torch.nn.functional.normalize(torch.randn(num_vectors, dim), dim=-1)
    query = torch.nn.functional.normalize(torch.randn(dim), dim=-1)
    

    # 1. Standard (Exhaustive) Search
    print("\n--- Standard Search (Exhaustive) ---")
    tq_standard = TurboQuant(dim=dim, bits=bits, use_ivf=False)
    
    t0 = time.time()
    tq_standard.index(vectors)
    print(f"Index time: {time.time() - t0:.4f} s")
    
    # Warmup
    _ = tq_standard.search(query, top_k=10)
    
    t0 = time.time()
    std_indices, std_scores = tq_standard.search(query, top_k=10)
    std_time = time.time() - t0
    print(f"Search time: {std_time:.6f} s")
    
    # 2. IVF Search
    print(f"\n--- IVF Search (n_list={n_list}, n_probe={n_probe}) ---")
    tq_ivf = TurboQuant(dim=dim, bits=bits, use_ivf=True, ivf_nlist=n_list, ivf_nprobe=n_probe)
    
    t0 = time.time()
    tq_ivf.index(vectors)
    print(f"Index time (including K-Means): {time.time() - t0:.4f} s")
    
    # Warmup
    _ = tq_ivf.search(query, top_k=10)
    
    t0 = time.time()
    ivf_indices, ivf_scores = tq_ivf.search(query, top_k=10)
    ivf_time = time.time() - t0
    print(f"Search time: {ivf_time:.6f} s")
    
    # Calculate overlap
    std_idx_set = set(std_indices.cpu().numpy().tolist())
    ivf_idx_set = set(ivf_indices.cpu().numpy().tolist())
    overlap = len(std_idx_set.intersection(ivf_idx_set))
    
    print("\n--- Results ---")
    print(f"Speedup: {std_time / ivf_time:.2f}x")
    print(f"Recall@10 compared to standard search: {overlap/10 * 100:.1f}%")

if __name__ == '__main__':
    test_ivf()
