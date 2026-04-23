import math
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, NamedTuple, List

from tq_engine.codebook import get_codebook_tensors, get_polar_codebooks
from tq_engine.rotation import (
    generate_rotation_matrix,
    generate_qjl_matrix,
    rotate_forward,
    rotate_backward,
)
from tq_engine.polar import recursive_polar_transform, recursive_polar_reconstruct

# =============================================================================
# Cấu trúc dữ liệu kết quả nén
# =============================================================================

class PolarQuantized(NamedTuple):
    """Kết quả nén Polar (Recursive Polar Transform)"""
    angle_indices: List[torch.Tensor] # Các chỉ số góc (List Level-wise)
    final_radius: torch.Tensor       # Bán kính cuối cùng (Level L)
    bits_list: List[int]             # Số bit dùng cho từng level


class ProdQuantized(NamedTuple):
    """Kết quả nén cho Inner Product (Algorithm 2 - Polar + QJL)"""
    polar_q: PolarQuantized     # Kết quả nén Polar stage 1
    qjl_signs: torch.Tensor     # Các bit dấu (+1/-1) nén từ stage 2 QJL
    residual_norms: torch.Tensor # Độ dài vector phần dư (residual)
    norms: torch.Tensor         # Độ dài vector gốc (để benchmark hoặc fallback)


# =============================================================================
# Các hàm bổ trợ đóng gói Bit (Bit-packing) để tiết kiệm bộ nhớ
# =============================================================================

def _pack_indices(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Đóng gói các chỉ số integer vào uint8 để tiết kiệm RAM."""
    d = indices.shape[-1]
    batch_shape = indices.shape[:-1]

    if bits == 1:
        vals_per_byte = 8
    elif bits == 2:
        vals_per_byte = 4
    elif bits <= 4:
        vals_per_byte = 2
        bits = 4
    else:
        return indices.to(torch.uint8)

    padded_d = ((d + vals_per_byte - 1) // vals_per_byte) * vals_per_byte
    if padded_d > d:
        indices = F.pad(indices.to(torch.uint8), (0, padded_d - d), value=0)

    reshaped = indices.to(torch.uint8).reshape(*batch_shape, -1, vals_per_byte)
    shifts = torch.arange(vals_per_byte, device=indices.device, dtype=torch.uint8) * bits
    packed = (reshaped << shifts).sum(dim=-1, dtype=torch.uint8)
    return packed


def _unpack_indices(packed: torch.Tensor, bits: int, d: int) -> torch.Tensor:
    """Giải nén các chỉ số từ uint8 về lại tensor integer."""
    batch_shape = packed.shape[:-1]

    if bits == 1:
        vals_per_byte = 8
    elif bits == 2:
        vals_per_byte = 4
    elif bits <= 4:
        vals_per_byte = 2
        bits = 4
    else:
        return packed.long()

    mask = (1 << bits) - 1
    shifts = torch.arange(vals_per_byte, device=packed.device, dtype=torch.uint8) * bits
    unpacked = ((packed.unsqueeze(-1) >> shifts) & mask)
    unpacked = unpacked.reshape(*batch_shape, -1)
    return unpacked[..., :d].long()


# =============================================================================
# ENGINE 1: Nén Polar (Tối ưu theo kiến trúc mới)
# =============================================================================

class TQEnginePolar(torch.nn.Module):
    def __init__(self, dim: int, levels: int = 4, bits_list: Optional[List[int]] = None, device: torch.device = None, dtype: torch.dtype = torch.float32, seed: int = 42):
        super().__init__()
        self.dim = dim
        self.levels = levels
        # Mặc định: 4 bit cho L1, 2 bit cho các level sau theo paper
        self.bits_list = bits_list or ([4] + [2] * (levels - 1))
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Khởi tạo ma trận quay Pi
        self.register_buffer("Pi", generate_rotation_matrix(dim, self.device, dtype, seed=seed))

        # Khởi tạo bảng mã cho từng level
        self.codebooks = get_polar_codebooks(levels, self.bits_list, self.device, dtype)
        
    @torch.no_grad()
    def quantize(self, x: torch.Tensor, micro_batch: int = 100000) -> PolarQuantized:
        """Nén vector sang dạng Polar angles."""
        y = rotate_forward(x.float(), self.Pi)
        final_radius, angles = recursive_polar_transform(y, self.levels)
        
        packed_angles = []
        for l in range(1, self.levels + 1):
            angle = angles[l-1]
            centroids, boundaries = self.codebooks[l-1]
            n_clusters = len(centroids)
            # searchsorted trả về vị trí insert trong boundaries (len n_clusters+1)
            # Index hợp lệ: [0, n_clusters-1]
            indices = torch.searchsorted(boundaries.contiguous(), angle.contiguous())
            # Clamp để tránh out-of-bounds
            indices = indices.clamp(0, n_clusters - 1)
            packed_angles.append(_pack_indices(indices, self.bits_list[l-1]))
            
        return PolarQuantized(angle_indices=packed_angles, final_radius=final_radius, bits_list=self.bits_list)

    def dequantize(self, q: PolarQuantized) -> torch.Tensor:
        """Giải nén và quay ngược về không gian gốc."""
        unpacked_angles = []
        for l in range(1, self.levels + 1):
            bits = q.bits_list[l-1]
            # Tính toán dim của level này: dim / 2^l
            level_dim = self.dim // (2**l)
            indices = _unpack_indices(q.angle_indices[l-1], bits, level_dim)
            centroids, _ = self.codebooks[l-1]
            unpacked_angles.append(centroids[indices])
            
        y_hat = recursive_polar_reconstruct(q.final_radius, unpacked_angles)
        x_hat = rotate_backward(y_hat, self.Pi)
        return x_hat


# =============================================================================
# ENGINE 2: Nén Inner Product (Tối ưu cho tính toán Attention)
# =============================================================================

class TQEngine(torch.nn.Module):
    def __init__(self, dim: int, bits: int = 3, use_qjl: bool = True, device: torch.device = None, dtype: torch.dtype = torch.float32, seed: int = 42):
        super().__init__()
        self.dim = dim
        self.bits = bits
        self.use_qjl = use_qjl
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Cấu hình PolarQuant: levels=4 cho d=128
        levels = 4 if dim >= 64 else 2
        
        # Bit allocation: Phân bổ lại bits dựa trên tổng ngân sách `bits`
        # Level 1 quan trọng nhất nên cần nhiều bits hơn (thường là bits+1)
        # Các level sau ít quan trọng hơn, lấy phần còn lại (tối thiểu là 2)
        bits_l1 = bits + 1
        bits_rest = max(2, bits - 1)
        bits_list = [bits_l1] + [bits_rest] * (levels - 1)
        
        self.polar_quantizer = TQEnginePolar(dim=dim, levels=levels, bits_list=bits_list, device=self.device, dtype=dtype, seed=seed)


        if self.use_qjl:
            self.register_buffer("S", generate_qjl_matrix(dim, self.device, dtype, seed=seed + 1000))
            self.qjl_scale = math.sqrt(math.pi / 2.0) / dim
        else:
            self.S, self.qjl_scale = None, 0.0

    def _pack_qjl_signs(self, projected: torch.Tensor) -> torch.Tensor:
        signs = (projected > 0).to(torch.uint8)
        d = signs.shape[-1]
        if d % 8 != 0: signs = F.pad(signs, (0, 8 - d % 8), value=0)
        signs_reshaped = signs.reshape(*signs.shape[:-1], -1, 8)
        powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=signs.device, dtype=torch.uint8)
        return (signs_reshaped * powers).sum(dim=-1, dtype=torch.uint8)

    def _unpack_qjl_signs(self, packed: torch.Tensor) -> torch.Tensor:
        powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=packed.device, dtype=torch.uint8)
        unpacked = ((packed.unsqueeze(-1) & powers) > 0).float()
        return 2.0 * unpacked.reshape(*packed.shape[:-1], -1)[..., :self.dim] - 1.0

    @torch.no_grad()
    def quantize(self, x: torch.Tensor, micro_batch: int = 100000) -> ProdQuantized:
        """Thực hiện nén 2 giai đoạn: Polar + QJL."""
        norms = x.norm(dim=-1)
        polar_q = self.polar_quantizer.quantize(x, micro_batch=micro_batch)
        x_hat = self.polar_quantizer.dequantize(polar_q)
        
        residual = x - x_hat
        res_norms = residual.norm(dim=-1)
        
        if self.use_qjl and self.S is not None:
            projected = torch.matmul(residual.float(), self.S.T)
            packed_signs = self._pack_qjl_signs(projected)
        else:
            packed_signs = torch.zeros((*residual.shape[:-1], (self.dim + 7) // 8), device=self.device, dtype=torch.uint8)
        
        return ProdQuantized(
            polar_q=polar_q,
            qjl_signs=packed_signs,
            residual_norms=res_norms,
            norms=norms
        )

    def dequantize(self, q: ProdQuantized) -> torch.Tensor:
        """Giải nén vector từ cả 2 stage."""
        x_polar = self.polar_quantizer.dequantize(q.polar_q)
        if self.use_qjl and self.S is not None:
            signs = self._unpack_qjl_signs(q.qjl_signs)
            x_qjl = torch.matmul(signs, self.S) * (self.qjl_scale * q.residual_norms.unsqueeze(-1))
            return x_polar + x_qjl
        return x_polar

    def attention_score(self, query: torch.Tensor, quantized_key: ProdQuantized) -> torch.Tensor:
        """Tính toán tích vô hướng query-key trực tiếp trên dạng nén (Asymmetric)."""
        k_polar = self.polar_quantizer.dequantize(quantized_key.polar_q)
        scores_polar = torch.matmul(query.float(), k_polar.float().transpose(-2, -1))

        if self.use_qjl and self.S is not None:
            q_sketched = torch.matmul(query.float(), self.S.T)
            signs = self._unpack_qjl_signs(quantized_key.qjl_signs)
            scores_qjl = torch.matmul(q_sketched, signs.transpose(-2, -1)) * (self.qjl_scale * quantized_key.residual_norms.unsqueeze(-2))
            return scores_polar + scores_qjl.to(scores_polar.dtype)
        return scores_polar
