import os
import sys
import torch
import numpy as np
import math
import gc

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from TQ_engine_lib.quantizer import TQEngine, ProdQuantized

def get_orthogonal_matrix(dim, device='cpu', seed=42):
    # Dùng lại logic rotation của TurboQuant
    torch.manual_seed(seed)
    random_matrix = torch.randn(dim, dim, device=device)
    q, r = torch.linalg.qr(random_matrix)
    d = torch.diag(r)
    q *= d.sign()
    return q

def prepare_tq_ivf_out_of_core():
    DIM = 768
    TOTAL_TOKENS = 5_000_000
    CHUNK_SIZE = 100_000
    N_LIST = 1024
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for BITS in [2, 4]:
        OUT_DIR = f"data/stress_5m/tq_data/ivf_{BITS}bit"
        os.makedirs(OUT_DIR, exist_ok=True)
        
        print(f"\n=======================================================")
        print(f"Preparing {TOTAL_TOKENS} vectors for TQ-IVF {BITS}-bit out-of-core...")
        
        engine = TQEngine(dim=DIM, bits=BITS, device=device, use_ivf=True, ivf_nlist=N_LIST)
        
        # 1. Train K-Means centroids using a sample
        print("1. Training Coarse Centroids...")
        sample_vectors = torch.nn.functional.normalize(torch.randn(100_000, DIM, device=device), dim=-1)
        coarse_centroids = engine._train_kmeans(sample_vectors, N_LIST)
        del sample_vectors
        gc.collect()
        
        # Temporary storage for each cluster
        tmp_dir = os.path.join(OUT_DIR, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        
        files_sq = [open(os.path.join(tmp_dir, f"sq_{i}.bin"), "wb") for i in range(N_LIST)]
        files_qjl = [open(os.path.join(tmp_dir, f"qjl_{i}.bin"), "wb") for i in range(N_LIST)]
        files_norms = [open(os.path.join(tmp_dir, f"norms_{i}.bin"), "wb") for i in range(N_LIST)]
        files_res = [open(os.path.join(tmp_dir, f"res_{i}.bin"), "wb") for i in range(N_LIST)]
        files_ids = [open(os.path.join(tmp_dir, f"ids_{i}.bin"), "wb") for i in range(N_LIST)]
        
        counts = np.zeros(N_LIST, dtype=np.int64)
        
        global_id_offset = 0
        
        # 2. Process in chunks
        print("2. Quantizing and Partitioning Chunks...")
        for chunk_idx in range(0, TOTAL_TOKENS, CHUNK_SIZE):
            actual = min(CHUNK_SIZE, TOTAL_TOKENS - chunk_idx)
            print(f"   -> Processing chunk {chunk_idx} to {chunk_idx + actual}...")
            
            batch = torch.nn.functional.normalize(torch.randn(actual, DIM, device=device), dim=-1)
            
            # Assign to clusters
            b_sq = (batch ** 2).sum(dim=1, keepdim=True)
            c_sq = (coarse_centroids ** 2).sum(dim=1).unsqueeze(0)
            dist = b_sq + c_sq - 2 * torch.mm(batch, coarse_centroids.t())
            assignments = dist.argmin(dim=1)
            
            # We process each cluster
            for c_idx in range(N_LIST):
                mask = (assignments == c_idx)
                count = mask.sum().item()
                if count > 0:
                    cluster_x = batch[mask]
                    pq = engine._quantize_flat(cluster_x, online_clustering=False)
                    
                    files_sq[c_idx].write(pq.sq_codes.tobytes())
                    files_qjl[c_idx].write(pq.qjl_signs.tobytes())
                    files_norms[c_idx].write(pq.norms.tobytes())
                    files_res[c_idx].write(pq.res_norms.tobytes())
                    
                    global_ids = mask.nonzero(as_tuple=False).view(-1).cpu().numpy() + global_id_offset
                    files_ids[c_idx].write(global_ids.astype(np.int64).tobytes())
                    
                    counts[c_idx] += count
                    
            global_id_offset += actual
            del batch, dist, assignments
            gc.collect()
            
        # Close all temp files
        for f in files_sq + files_qjl + files_norms + files_res + files_ids:
            f.close()
            
        # 3. Merge into flat arrays
        print("3. Merging partitions into IVF Flat Arrays...")
        prefix = f"tq_ivf_{BITS}bit"
        
        sq_dim = DIM // 2 if BITS == 4 else DIM // 8
        qjl_dim = DIM // 8
        
        merged_sq = np.zeros((TOTAL_TOKENS, sq_dim), dtype=np.uint8)
        merged_qjl = np.zeros((TOTAL_TOKENS, qjl_dim), dtype=np.uint8)
        merged_norms = np.zeros(TOTAL_TOKENS, dtype=np.float32)
        merged_res = np.zeros(TOTAL_TOKENS, dtype=np.float32)
        merged_ids = np.zeros(TOTAL_TOKENS, dtype=np.int64)
        
        list_offsets = [0]
        current_offset = 0
        
        for c_idx in range(N_LIST):
            count = counts[c_idx]
            if count == 0:
                list_offsets.append(current_offset)
                continue
                
            start = current_offset
            end = current_offset + count
            
            merged_sq[start:end] = np.fromfile(os.path.join(tmp_dir, f"sq_{c_idx}.bin"), dtype=np.uint8).reshape(count, sq_dim)
            merged_qjl[start:end] = np.fromfile(os.path.join(tmp_dir, f"qjl_{c_idx}.bin"), dtype=np.uint8).reshape(count, qjl_dim)
            merged_norms[start:end] = np.fromfile(os.path.join(tmp_dir, f"norms_{c_idx}.bin"), dtype=np.float32)
            merged_res[start:end] = np.fromfile(os.path.join(tmp_dir, f"res_{c_idx}.bin"), dtype=np.float32)
            merged_ids[start:end] = np.fromfile(os.path.join(tmp_dir, f"ids_{c_idx}.bin"), dtype=np.int64)
            
            current_offset += count
            list_offsets.append(current_offset)
            
        np.save(os.path.join(OUT_DIR, f"{prefix}_sq_codes.npy"), merged_sq)
        np.save(os.path.join(OUT_DIR, f"{prefix}_qjl_signs.npy"), merged_qjl)
        np.save(os.path.join(OUT_DIR, f"{prefix}_norms.npy"), merged_norms)
        np.save(os.path.join(OUT_DIR, f"{prefix}_res_norms.npy"), merged_res)
        
        # Save Metadata
        config = {
            "dim": DIM,
            "bits": BITS,
            "use_ivf": True,
            "ivf_nlist": N_LIST,
            "ivf_nprobe": 32
        }
        
        np.savez(os.path.join(OUT_DIR, f"{prefix}_ivf_meta.npz"), 
                 coarse_centroids=coarse_centroids.cpu().numpy(),
                 list_offsets=np.array(list_offsets, dtype=np.int64),
                 vector_ids=merged_ids,
                 **config)
                 
        np.savez(os.path.join(OUT_DIR, f"{prefix}_pq_meta.npz"),
                 centroids=engine.sq_quantizer.centroids.cpu().numpy(),
                 dim=DIM,
                 sq_bits=BITS - 1,
                 total_bits=BITS,
                 qjl_scale=engine.qjl_scale,
                 rot_op=engine.rot_op_np)
                 
        # Cleanup tmp
        import shutil
        shutil.rmtree(tmp_dir)
                 
        print(f"Done! IVF Index {BITS}-bit built and saved to {OUT_DIR}")

if __name__ == "__main__":
    prepare_tq_ivf_out_of_core()
