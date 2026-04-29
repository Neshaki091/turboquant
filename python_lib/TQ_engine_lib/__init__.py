import os
import sys
import subprocess
import shutil
import torch

def build_native_core():
    """Tự động biên dịch lõi Rust SIMD nếu thiếu hoặc trên máy mới."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    core_dir = os.path.join(base_dir, "core")
    
    if not os.path.exists(core_dir):
        print("TurboQuant Error: 'core' directory not found. Cannot build Rust core.")
        return False

    print("TurboQuant: Checking and compiling Rust SIMD core (please wait)...")
    
    # 1. Check Cargo (Rust)
    try:
        subprocess.run(["cargo", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("TurboQuant Error: 'cargo' not found. Please install Rust (https://rustup.rs/).")
        return False

    # 2. Run cargo build
    try:
        subprocess.run(["cargo", "build", "--release"], cwd=core_dir, check=True)
    except subprocess.CalledProcessError as e:
        print(f"TurboQuant Error: Failed to build Rust core: {e}")
        return False

    # 3. Find and copy binary (.dll/.so)
    target_dir = os.path.join(core_dir, "target", "release")
    # Cargo produces .dll on Windows, but Python expects .pyd
    search_ext = ".dll" if os.name == "nt" else ".so"
    
    found_lib = None
    for f in os.listdir(target_dir):
        if f.startswith("tq_native_lib") and f.endswith(search_ext):
            found_lib = f
            break
    
    if found_lib:
        src = os.path.join(target_dir, found_lib)
        dst = os.path.join(base_dir, "tq_native_lib.pyd")
        if os.name != "nt":
            dst = os.path.join(base_dir, "tq_native_lib.so")
            
        shutil.copy2(src, dst)
        print(f"TurboQuant: Build successful! Saved to {dst}")
        
        # --- NEW: Cleanup after build ---
        try:
            print("TurboQuant: Cleaning up build artifacts...")
            shutil.rmtree(os.path.join(core_dir, "target"), ignore_errors=True)
            lock_file = os.path.join(core_dir, "Cargo.lock")
            if os.path.exists(lock_file):
                os.remove(lock_file)
        except Exception as e:
            print(f"TurboQuant Warning: Cleanup failed: {e}")
            
        return True
    else:
        print("TurboQuant Error: Binary file not found after build.")
        return False

# Try importing Rust core
try:
    from . import tq_native_lib
except ImportError:
    if build_native_core():
        try:
            from . import tq_native_lib
        except ImportError:
            print("TurboQuant Warning: Built successfully but import failed. Please restart Python.")
    else:
        print("TurboQuant Warning: Native SIMD mode disabled. Performance will be low.")

from .quantizer import TQEngine, ProdQuantized
from .codebook import ScalarQuantizer

__version__ = "0.4.0"

class TurboQuant:
    """
    High-level API for TurboQuant Vector Search.
    
    Hỗ trợ tự động biên dịch và nén SQ+QJL với IVF.
    """
    def __init__(self, dim: int, bits: int = 4, device: str = None, 
                 use_ivf: bool = False, ivf_nlist: int = 1024, ivf_nprobe: int = 32):
        self.engine = TQEngine(dim=dim, bits=bits, device=device, 
                               use_ivf=use_ivf, ivf_nlist=ivf_nlist, ivf_nprobe=ivf_nprobe)
        self.pq_data = None

    def index(self, vectors: torch.Tensor, online_clustering: bool = False):
        """Lập chỉ mục dữ liệu."""
        self.pq_data = self.engine.quantize(vectors, online_clustering=online_clustering)
        print(f"TurboQuant: Đã lập chỉ mục {vectors.shape[0]} vectors.")

    def search(self, query: torch.Tensor, top_k: int = 10):
        """Tìm kiếm Top-K."""
        if self.pq_data is None:
            raise ValueError("Index trống. Vui lòng gọi .index() trước.")
        return self.engine.native_cosine_search(query, self.pq_data, top_k=top_k)

    def save_index(self, directory: str, prefix: str):
        """Lưu chỉ mục xuống đĩa."""
        import os
        import numpy as np
        from .quantizer import IVFData, ProdQuantized
        
        os.makedirs(directory, exist_ok=True)
        if self.pq_data is None: raise ValueError("Index trống.")
            
        is_ivf = isinstance(self.pq_data, IVFData)
        config = {"dim": self.engine.dim, "bits": self.engine.bits, "use_ivf": is_ivf}
        
        if is_ivf:
            config["ivf_nlist"] = self.pq_data.n_list
            config["ivf_nprobe"] = self.pq_data.n_probe
            np.savez(os.path.join(directory, f"{prefix}_ivf_meta.npz"), 
                     coarse_centroids=self.pq_data.coarse_centroids.cpu().numpy(),
                     list_offsets=self.pq_data.list_offsets,
                     vector_ids=self.pq_data.vector_ids, **config)
            pq = self.pq_data.pq_data
        else:
            np.savez(os.path.join(directory, f"{prefix}_meta.npz"), **config)
            pq = self.pq_data
            
        np.save(os.path.join(directory, f"{prefix}_sq_codes.npy"), pq.sq_codes)
        np.save(os.path.join(directory, f"{prefix}_qjl_signs.npy"), pq.qjl_signs)
        np.save(os.path.join(directory, f"{prefix}_norms.npy"), pq.norms)
        np.save(os.path.join(directory, f"{prefix}_res_norms.npy"), pq.res_norms)
        
        np.savez(os.path.join(directory, f"{prefix}_pq_meta.npz"),
                 centroids=pq.centroids, dim=pq.dim, sq_bits=pq.sq_bits,
                 total_bits=pq.total_bits, qjl_scale=pq.qjl_scale, rot_op=pq.rot_op)
        print(f"TurboQuant: Đã lưu tại {directory} với prefix '{prefix}'.")

    def load_index(self, directory: str, prefix: str):
        """Tải chỉ mục từ đĩa."""
        import os
        import numpy as np
        from .quantizer import IVFData, ProdQuantized
        
        meta_ivf_path = os.path.join(directory, f"{prefix}_ivf_meta.npz")
        meta_flat_path = os.path.join(directory, f"{prefix}_meta.npz")
        is_ivf = os.path.exists(meta_ivf_path)
        
        if is_ivf:
            meta = np.load(meta_ivf_path)
            self.engine.use_ivf = True
            self.engine.ivf_nlist = int(meta["ivf_nlist"])
            self.engine.ivf_nprobe = int(meta["ivf_nprobe"])
        else:
            meta = np.load(meta_flat_path)
            self.engine.use_ivf = False
            
        self.engine.dim = int(meta["dim"])
        self.engine.bits = int(meta["bits"])
        pq_meta = np.load(os.path.join(directory, f"{prefix}_pq_meta.npz"))
        
        pq = ProdQuantized(
            sq_codes=np.load(os.path.join(directory, f"{prefix}_sq_codes.npy")),
            qjl_signs=np.load(os.path.join(directory, f"{prefix}_qjl_signs.npy")),
            norms=np.load(os.path.join(directory, f"{prefix}_norms.npy")),
            res_norms=np.load(os.path.join(directory, f"{prefix}_res_norms.npy")),
            centroids=pq_meta["centroids"], dim=int(pq_meta["dim"]),
            sq_bits=int(pq_meta["sq_bits"]), total_bits=int(pq_meta["total_bits"]),
            qjl_scale=float(pq_meta["qjl_scale"]), rot_op=pq_meta["rot_op"]
        )
        
        if is_ivf:
            self.pq_data = IVFData(
                coarse_centroids=torch.from_numpy(meta["coarse_centroids"]).to(self.engine.device),
                pq_data=pq, vector_ids=meta["vector_ids"], list_offsets=meta["list_offsets"],
                n_list=self.engine.ivf_nlist, n_probe=self.engine.ivf_nprobe)
        else:
            self.pq_data = pq
        print(f"TurboQuant: Đã tải chỉ mục từ {directory}.")
