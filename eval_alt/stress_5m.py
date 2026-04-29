import os
import sys
import time
import torch
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from TQ_engine_lib import tq_native_lib
from TQ_engine_lib.quantizer import TQEngine
import gc
import psutil

def get_current_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def rotate_forward(x, rot_op):
    return torch.matmul(x, rot_op)

def stress_test():
    DIM = 768
    TOTAL_TOKENS = 5_000_000
    N_QUERIES = 20
    DATA_DIR = "data/stress_5m"
    TQ_DIR = f"{DATA_DIR}/tq_data"
    
    results = []

    # --- 1. RAW Baseline ---
    raw_path = f"{DATA_DIR}/raw_768.npy"
    if os.path.exists(raw_path):
        RAW_BATCH = 250_000 
        print(f"\nBenchmarking RAW (Float32) - Batch: {RAW_BATCH}...")
        buffer = np.zeros((RAW_BATCH, DIM), dtype=np.float32)
        total_time = 0
        with open(raw_path, "rb") as f:
            for _ in range(N_QUERIES):
                f.seek(128)
                t_start = time.perf_counter()
                for i in range(0, TOTAL_TOKENS, RAW_BATCH):
                    actual = min(RAW_BATCH, TOTAL_TOKENS - i)
                    f.readinto(buffer[:actual].data)
                    _ = np.dot(buffer[:actual], np.random.rand(DIM).astype(np.float32))
                total_time += (time.perf_counter() - t_start)
        results.append(("RAW (F32)", RAW_BATCH, get_current_memory_mb(), total_time / N_QUERIES))
        # Clear RAW data
        del buffer
        gc.collect()

    # --- 2. SQ Variants ---
    sq_variants = {
        "SQ 2-bit": {"file": "sq2_768.npy", "batch": 4_300_000, "bits": 2},
        "SQ 4-bit": {"file": "sq4_768.npy", "batch": 2_100_000, "bits": 4},
    }
    centroids_dummy = np.random.rand(256).astype(np.float32)
    norms_dummy = np.ones(TOTAL_TOKENS, dtype=np.float32)

    for name, cfg in sq_variants.items():
        path = f"{DATA_DIR}/{cfg['file']}"
        if not os.path.exists(path): continue
        BATCH = cfg['batch']
        print(f"Benchmarking {name} - Batch: {BATCH}...")
        
        packed_dim = (DIM * cfg['bits']) // 8
        buffer = np.zeros((BATCH, packed_dim), dtype=np.uint8)
        
        total_time = 0
        with open(path, "rb") as f:
            for _ in range(N_QUERIES):
                f.seek(128)
                q = np.random.rand(DIM).astype(np.float32)
                t_start = time.perf_counter()
                for i in range(0, TOTAL_TOKENS, BATCH):
                    actual = min(BATCH, TOTAL_TOKENS - i)
                    f.readinto(buffer[:actual].data)
                    _ = tq_native_lib.sq_scan(
                        q, buffer[:actual], centroids_dummy, norms_dummy[i:i+actual], DIM, cfg['bits']
                    )
                total_time += (time.perf_counter() - t_start)
        results.append((name, BATCH, get_current_memory_mb(), total_time / N_QUERIES))
        # Clear SQ data
        del buffer
        gc.collect()

    # --- 3. PQ Variants ---
    pq_variants = {
        "PQ 2-bit": {"file": "pq2_768.npy", "batch": 4_000_000, "m": 192},
        "PQ 4-bit": {"file": "pq4_768.npy", "batch": 1_750_000, "m": 384},
    }
    for name, cfg in pq_variants.items():
        path = f"{DATA_DIR}/{cfg['file']}"
        if not os.path.exists(path): continue
        BATCH = cfg['batch']
        print(f"Benchmarking {name} - Batch: {BATCH}...")
        
        m = cfg['m']
        buffer = np.zeros((BATCH, m), dtype=np.uint8)
        dist_table = np.random.rand(m, 256).astype(np.float32)
        
        total_time = 0
        with open(path, "rb") as f:
            for _ in range(N_QUERIES):
                f.seek(128)
                t_start = time.perf_counter()
                for i in range(0, TOTAL_TOKENS, BATCH):
                    actual = min(BATCH, TOTAL_TOKENS - i)
                    f.readinto(buffer[:actual].data)
                    _ = tq_native_lib.pq_scan(buffer[:actual], dist_table)
                total_time += (time.perf_counter() - t_start)
        results.append((name, BATCH, get_current_memory_mb(), total_time / N_QUERIES))
        # Clear PQ data
        del buffer
        gc.collect()

    # --- 4. TurboQuant ---
    # Batch sizes targeting ~1GB RAM
    batch_map = {"2bit": 4_000_000, "4bit": 1_500_000}
    for bit_folder in ["2bit", "4bit"]:
        tq_dir = f"{TQ_DIR}/{bit_folder}"
        if not os.path.exists(tq_dir): continue
        
        BATCH = batch_map[bit_folder]
        config = np.load(f"{tq_dir}/config.npz")
        centroids = config['centroids']
        sq_bits = int(config['sq_bits'])
        qjl_scale = float(config['qjl_scale'])
        rot_op_t = torch.from_numpy(config['rot_op'])

        print(f"Benchmarking TQ {bit_folder} - Batch: {BATCH}...")
        
        # Paths
        sq_path = f"{tq_dir}/tq_sq_codes_{DIM}.npy"
        signs_path = f"{tq_dir}/tq_qjl_signs_{DIM}.npy"
        norms_path = f"{tq_dir}/tq_norms_{DIM}.npy"
        res_path = f"{tq_dir}/tq_res_norms_{DIM}.npy"
        
        # Pre-allocate buffers for the batch
        sq_buffer = np.zeros((BATCH, DIM//2 if bit_folder=="4bit" else DIM//8), dtype=np.uint8)
        signs_buffer = np.zeros((BATCH, DIM//8), dtype=np.uint8)
        norms_buffer = np.zeros(BATCH, dtype=np.float32)
        res_buffer = np.zeros(BATCH, dtype=np.float32)

        total_time = 0
        peak_mem = 0
        
        # Open all files
        f_sq = open(sq_path, "rb")
        f_signs = open(signs_path, "rb")
        f_norms = open(norms_path, "rb")
        f_res = open(res_path, "rb")

        for _ in range(N_QUERIES):
            query = torch.nn.functional.normalize(torch.randn(1, DIM), dim=-1)
            q_rot = rotate_forward(query, rot_op_t).squeeze(0).numpy().astype(np.float32)
            
            # Reset file pointers (skip .npy header)
            f_sq.seek(128); f_signs.seek(128); f_norms.seek(128); f_res.seek(128)
            
            t_start = time.perf_counter()
            for i in range(0, TOTAL_TOKENS, BATCH):
                actual = min(BATCH, TOTAL_TOKENS - i)
                
                # Batch read from files
                f_sq.readinto(sq_buffer[:actual].data)
                f_signs.readinto(signs_buffer[:actual].data)
                f_norms.readinto(norms_buffer[:actual].data)
                f_res.readinto(res_buffer[:actual].data)
                
                _ = tq_native_lib.tq_scan(
                    q_rot, sq_buffer[:actual], centroids, norms_buffer[:actual],
                    signs_buffer[:actual], res_buffer[:actual], q_rot,
                    qjl_scale, DIM, sq_bits
                )
            total_time += (time.perf_counter() - t_start)
            peak_mem = max(peak_mem, get_current_memory_mb())
            
        # Cleanup TQ session
        f_sq.close(); f_signs.close(); f_norms.close(); f_res.close()
        del sq_buffer, signs_buffer, norms_buffer, res_buffer
        gc.collect()
        
    # --- 5. TurboQuant IVF ---
    for bit_folder in ["2bit", "4bit"]:
        ivf_prefix = f"tq_ivf_{bit_folder}"
        ivf_dir = f"{TQ_DIR}/ivf_{bit_folder}"
        
        if os.path.exists(ivf_dir):
            print(f"Benchmarking TQ IVF {bit_folder}...")
            # Load meta
            ivf_meta = np.load(f"{ivf_dir}/{ivf_prefix}_ivf_meta.npz")
            pq_meta = np.load(f"{ivf_dir}/{ivf_prefix}_pq_meta.npz")
            
            coarse_centroids = torch.from_numpy(ivf_meta['coarse_centroids'])
            list_offsets = ivf_meta['list_offsets']
            
            centroids = pq_meta['centroids']
            sq_bits = int(pq_meta['sq_bits'])
            qjl_scale = float(pq_meta['qjl_scale'])
            rot_op_t = torch.from_numpy(pq_meta['rot_op'])
            
            n_probe = int(ivf_meta.get('ivf_nprobe', 32))
            
            sq_path = f"{ivf_dir}/{ivf_prefix}_sq_codes.npy"
            signs_path = f"{ivf_dir}/{ivf_prefix}_qjl_signs.npy"
            norms_path = f"{ivf_dir}/{ivf_prefix}_norms.npy"
            res_path = f"{ivf_dir}/{ivf_prefix}_res_norms.npy"
            
            # Calculate max cluster size
            max_cluster_size = int(np.max(np.diff(list_offsets)))
            
            sq_dim = DIM//2 if bit_folder == "4bit" else DIM//8
            sq_buffer = np.zeros((max_cluster_size, sq_dim), dtype=np.uint8)
            signs_buffer = np.zeros((max_cluster_size, DIM//8), dtype=np.uint8)
            norms_buffer = np.zeros(max_cluster_size, dtype=np.float32)
            res_buffer = np.zeros(max_cluster_size, dtype=np.float32)
            
            total_time = 0
            peak_mem = 0
            
            f_sq = open(sq_path, "rb")
            f_signs = open(signs_path, "rb")
            f_norms = open(norms_path, "rb")
            f_res = open(res_path, "rb")
            
            for _ in range(N_QUERIES):
                query = torch.nn.functional.normalize(torch.randn(1, DIM), dim=-1)
                q_rot = rotate_forward(query, rot_op_t).squeeze(0).numpy().astype(np.float32)
                
                t_start = time.perf_counter()
                
                q_sq = (query ** 2).sum(dim=1, keepdim=True)
                c_sq = (coarse_centroids ** 2).sum(dim=1).unsqueeze(0)
                dist = q_sq + c_sq - 2 * torch.mm(query, coarse_centroids.t())
                top_c_indices = dist.topk(n_probe, dim=1, largest=False).indices.squeeze(0).cpu().tolist()
                
                for c_idx in top_c_indices:
                    start = list_offsets[c_idx]
                    end = list_offsets[c_idx+1]
                    count = end - start
                    if count == 0: continue
                    
                    f_sq.seek(128 + start * sq_dim)
                    f_signs.seek(128 + start * (DIM//8))
                    f_norms.seek(128 + start * 4)
                    f_res.seek(128 + start * 4)
                    
                    f_sq.readinto(sq_buffer[:count].data)
                    f_signs.readinto(signs_buffer[:count].data)
                    f_norms.readinto(norms_buffer[:count].data)
                    f_res.readinto(res_buffer[:count].data)
                    
                    _ = tq_native_lib.tq_scan(
                        q_rot, sq_buffer[:count], centroids, norms_buffer[:count],
                        signs_buffer[:count], res_buffer[:count], q_rot,
                        qjl_scale, DIM, sq_bits
                    )
                
                total_time += (time.perf_counter() - t_start)
                peak_mem = max(peak_mem, get_current_memory_mb())
                
            f_sq.close(); f_signs.close(); f_norms.close(); f_res.close()
            del sq_buffer, signs_buffer, norms_buffer, res_buffer
            gc.collect()
            
            results.append((f"TQ-IVF {bit_folder}", f"probe:{n_probe}", peak_mem, total_time / N_QUERIES))

    # Final Report
    raw_lat = next((lat for name, b, r, lat in results if "RAW" in name), 1.0)
    
    print("\n" + "="*108)
    print(f"{'Method':<15} | {'Batch':<11} | {'Peak RAM':<12} | {'Latency':<10} | {'QPS':<10} | {'Speedup'}")
    print("-" * 108)
    for name, batch, ram, lat in results:
        qps = 1.0 / lat if lat > 0 else 0
        speedup = raw_lat / lat if lat > 0 else 0
        if isinstance(batch, int):
            print(f"{name:<15} | {batch:>11,} | {ram:>9.1f} MB | {lat:>9.4f}s | {qps:>9.2f} | {speedup:>8.1f}x")
        else:
            print(f"{name:<15} | {str(batch):>11} | {ram:>9.1f} MB | {lat:>9.4f}s | {qps:>9.2f} | {speedup:>8.1f}x")
    print("="*108)

if __name__ == "__main__":
    stress_test()
