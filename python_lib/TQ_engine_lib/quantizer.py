import torch
import numpy as np
import math
from dataclasses import dataclass
from .codebook import ScalarQuantizer
from .rotation import get_orthogonal_matrix, rotate_forward, rotate_backward
from .tq_bridge import tq_native

@dataclass
class ProdQuantized:
    sq_codes: np.ndarray
    qjl_signs: np.ndarray
    norms: np.ndarray
    centroids: np.ndarray
    dim: int
    sq_bits: int
    total_bits: int
    qjl_scale: float
    rot_op: np.ndarray
    res_norms: np.ndarray

@dataclass
class IVFData:
    coarse_centroids: torch.Tensor
    pq_data: ProdQuantized
    vector_ids: np.ndarray
    list_offsets: np.ndarray
    n_list: int
    n_probe: int

class TQEngine:
    def __init__(self, dim: int = 768, bits: int = 4, device: str = None, use_ivf: bool = False, ivf_nlist: int = 1024, ivf_nprobe: int = 32):
        if bits not in [2, 4]:
            raise ValueError(f"TurboQuant currently only supports 2-bit (1+1) and 4-bit (3+1) configurations. Received: {bits}")
            
        self.dim = dim
        self.bits = bits
        self.sq_bits = bits - 1
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_ivf = use_ivf
        self.ivf_nlist = ivf_nlist
        self.ivf_nprobe = ivf_nprobe
        
        # 1. Initialize Scalar Quantizer (Stage 1)
        self.sq_quantizer = ScalarQuantizer(dim=dim, bits=self.sq_bits, device=self.device)
        
        # 2. Pure TurboQuant: Exact Dimension Orthogonal Rotation
        rot_op_t = get_orthogonal_matrix(dim, device=self.device)
        self.rot_op_np = rot_op_t.cpu().numpy().astype(np.float32)
        
        # 3. Calculate QJL Scale
        if self.sq_bits == 1:
            self.qjl_scale = 0.585 / math.sqrt(self.dim)
        else:
            self.qjl_scale = math.sqrt(2.0 / math.pi) / math.sqrt(self.dim)

    def _train_kmeans(self, x: torch.Tensor, n_clusters: int, n_iter: int = 30, sample_size: int = 65536) -> torch.Tensor:
        N, D = x.shape
        if N > sample_size:
            perm = torch.randperm(N, device=self.device)
            x_sample = x[perm[:sample_size]]
        else:
            x_sample = x

        perm = torch.randperm(len(x_sample), device=self.device)
        centroids = x_sample[perm[:n_clusters]].clone()

        for _ in range(n_iter):
            x_sq = (x_sample ** 2).sum(dim=1, keepdim=True)
            c_sq = (centroids ** 2).sum(dim=1).unsqueeze(0)
            dist = x_sq + c_sq - 2 * torch.mm(x_sample, centroids.t())
            
            assignments = dist.argmin(dim=1)
            
            new_centroids = torch.zeros_like(centroids)
            counts = torch.bincount(assignments, minlength=n_clusters).unsqueeze(1).float()
            
            new_centroids.scatter_add_(0, assignments.unsqueeze(1).expand(-1, D), x_sample)
            new_centroids = torch.where(counts > 0, new_centroids / counts, centroids)
            centroids = new_centroids
            
        return centroids

    def quantize(self, x: torch.Tensor, online_clustering: bool = False):
        if x.device.type != self.device:
            x = x.to(self.device)
            
        if self.use_ivf:
            coarse_centroids = self._train_kmeans(x, self.ivf_nlist)
            
            # Assign
            assignments = []
            batch_size = 100000
            for i in range(0, len(x), batch_size):
                batch = x[i:i+batch_size]
                b_sq = (batch ** 2).sum(dim=1, keepdim=True)
                c_sq = (coarse_centroids ** 2).sum(dim=1).unsqueeze(0)
                dist = b_sq + c_sq - 2 * torch.mm(batch, coarse_centroids.t())
                assignments.append(dist.argmin(dim=1))
            assignments = torch.cat(assignments)
            
            sq_codes_list = []
            qjl_signs_list = []
            norms_list = []
            res_norms_list = []
            vector_ids_list = []
            list_offsets = [0]
            
            current_offset = 0
            
            # Khởi tạo một ProdQuantized dummy để lưu metadata chung
            dummy_pq = None
            
            for c_idx in range(self.ivf_nlist):
                mask = (assignments == c_idx)
                count = mask.sum().item()
                if count > 0:
                    cluster_x = x[mask]
                    cluster_pq = self._quantize_flat(cluster_x, online_clustering)
                    
                    sq_codes_list.append(cluster_pq.sq_codes)
                    qjl_signs_list.append(cluster_pq.qjl_signs)
                    norms_list.append(cluster_pq.norms)
                    res_norms_list.append(cluster_pq.res_norms)
                    vector_ids_list.append(mask.nonzero(as_tuple=False).view(-1).cpu().numpy())
                    
                    if dummy_pq is None:
                        dummy_pq = cluster_pq
                
                current_offset += count
                list_offsets.append(current_offset)
                
            if dummy_pq is None:
                raise ValueError("All IVF clusters are empty!")
                
            flat_pq = ProdQuantized(
                sq_codes=np.concatenate(sq_codes_list, axis=0) if sq_codes_list else np.array([]),
                qjl_signs=np.concatenate(qjl_signs_list, axis=0) if qjl_signs_list else np.array([]),
                norms=np.concatenate(norms_list, axis=0) if norms_list else np.array([]),
                centroids=dummy_pq.centroids,
                dim=dummy_pq.dim,
                sq_bits=dummy_pq.sq_bits,
                total_bits=dummy_pq.total_bits,
                qjl_scale=dummy_pq.qjl_scale,
                rot_op=dummy_pq.rot_op,
                res_norms=np.concatenate(res_norms_list, axis=0) if res_norms_list else np.array([])
            )
            
            return IVFData(
                coarse_centroids=coarse_centroids, 
                pq_data=flat_pq, 
                vector_ids=np.concatenate(vector_ids_list, axis=0) if vector_ids_list else np.array([]),
                list_offsets=np.array(list_offsets, dtype=np.int64),
                n_list=self.ivf_nlist,
                n_probe=self.ivf_nprobe
            )
        else:
            return self._quantize_flat(x, online_clustering)

    def _quantize_flat(self, x: torch.Tensor, online_clustering: bool = False) -> ProdQuantized:
        """
        Nén bộ dữ liệu vector x (N, D) sang định dạng TurboQuant (SQ+QJL).
        """
        dist_device = x.device
        rot_op_t = torch.from_numpy(self.rot_op_np).to(dist_device)

        # 1. ROTATE
        x_rot = rotate_forward(x, rot_op_t)
        
        # 2. Extract Norms
        norms = torch.norm(x, dim=-1)
        
        # 3. Stage 1: SQ
        if online_clustering:
            self.sq_quantizer.fit(x_rot)
        
        sq_q = self.sq_quantizer.quantize(x_rot)
        x_hat_1 = self.sq_quantizer.reconstruct(sq_q.indices)
        
        # 4. Stage 2: QJL Residual (Pure TQ: Signs in Rotated Space)
        residual = x_rot - x_hat_1
        res_norms = torch.norm(residual, dim=-1)
        
        # Signs of residual vector directly in rotated space
        signs = (residual > 0).to(torch.uint8).cpu().numpy()
        qjl_signs = np.packbits(signs, axis=-1, bitorder='little')
        
        return ProdQuantized(
            sq_codes=sq_q.indices.cpu().numpy().astype(np.uint8),
            qjl_signs=qjl_signs.astype(np.uint8),
            norms=norms.cpu().numpy().astype(np.float32),
            centroids=self.sq_quantizer.centroids.cpu().numpy().astype(np.float32),
            dim=self.dim,
            sq_bits=self.sq_bits,
            total_bits=self.bits,
            qjl_scale=self.qjl_scale,
            rot_op=self.rot_op_np,
            res_norms=res_norms.cpu().numpy().astype(np.float32)
        )

    def native_cosine_search(self, query: torch.Tensor, pq, top_k: int = 100) -> tuple[torch.Tensor, torch.Tensor]:
        if query.device.type != self.device:
            query = query.to(self.device)
            
        if isinstance(pq, IVFData):
            if query.dim() == 1:
                query_2d = query.unsqueeze(0)
            else:
                query_2d = query
                
            q_sq = (query_2d ** 2).sum(dim=1, keepdim=True)
            c_sq = (pq.coarse_centroids ** 2).sum(dim=1).unsqueeze(0)
            dist = q_sq + c_sq - 2 * torch.mm(query_2d, pq.coarse_centroids.t())
            
            top_c_indices = dist.topk(pq.n_probe, dim=1, largest=False).indices.squeeze(0).cpu().tolist()
            
            all_scores = []
            all_ids = []
            
            for c_idx in top_c_indices:
                start = pq.list_offsets[c_idx]
                end = pq.list_offsets[c_idx+1]
                if start == end: continue
                
                sub_pq = ProdQuantized(
                    sq_codes=pq.pq_data.sq_codes[start:end],
                    qjl_signs=pq.pq_data.qjl_signs[start:end],
                    norms=pq.pq_data.norms[start:end],
                    centroids=pq.pq_data.centroids,
                    dim=pq.pq_data.dim,
                    sq_bits=pq.pq_data.sq_bits,
                    total_bits=pq.pq_data.total_bits,
                    qjl_scale=pq.pq_data.qjl_scale,
                    rot_op=pq.pq_data.rot_op,
                    res_norms=pq.pq_data.res_norms[start:end]
                )
                
                list_indices, list_scores = self._native_cosine_search_flat(query, sub_pq, top_k)
                global_ids = pq.vector_ids[start:end][list_indices.cpu().numpy()]
                
                all_scores.append(list_scores)
                all_ids.append(torch.from_numpy(global_ids))
                
            if not all_scores:
                return torch.tensor([]), torch.tensor([])
                
            merged_scores = torch.cat(all_scores)
            merged_ids = torch.cat(all_ids)
            
            final_top_k = min(top_k, len(merged_scores))
            final_scores, final_idx = merged_scores.topk(final_top_k)
            
            return merged_ids[final_idx], final_scores
        else:
            return self._native_cosine_search_flat(query, pq, top_k)

    def _native_cosine_search_flat(self, query: torch.Tensor, pq: ProdQuantized, top_k: int = 100) -> tuple[torch.Tensor, torch.Tensor]:
        q_t = query.to(self.device).float()
        rot_op_t = torch.from_numpy(pq.rot_op).to(self.device)
        
        if q_t.dim() == 1:
            q_t = q_t.unsqueeze(0)
            
        # 1. Query Rotation (Stage 2 query is the SAME as Stage 1 query)
        q_rot = rotate_forward(q_t, rot_op_t).squeeze(0)
        q_np = q_rot.cpu().numpy().astype(np.float32)
        
        # 2. Preparation for Rust (Strict C-Contiguous)
        query_1d = np.array(q_np, dtype=np.float32, order='C')
        sq_codes_2d = np.array(pq.sq_codes, dtype=np.uint8, order='C')
        centroids_1d = np.array(pq.centroids, dtype=np.float32, order='C')
        norms_1d = np.array(pq.norms, dtype=np.float32, order='C')
        qjl_signs_2d = np.array(pq.qjl_signs, dtype=np.uint8, order='C')
        res_norms_1d = np.array(pq.res_norms, dtype=np.float32, order='C')
        
        # Pure TQ: qjl_query IS query_1d
        scores = tq_native.tq_scan(
            query_1d, sq_codes_2d, centroids_1d, norms_1d,
            qjl_signs_2d, res_norms_1d, query_1d,
            float(pq.qjl_scale), int(self.dim), int(self.sq_bits)
        )
        
        scores_t = torch.from_numpy(scores).view(-1)
        top_scores, top_indices = torch.topk(scores_t, min(top_k, len(scores_t)))
        return top_indices, top_scores
