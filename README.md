# 🚀 TurboQuant: High-Performance Vector Search Engine (SQ+QJL & IVF)

![TurboQuant Benchmark](https://img.shields.io/badge/Algorithm-SQ+QJL%20%2B%20IVF-blue.svg) ![Scale](https://img.shields.io/badge/Scale-5_Million_Vectors-orange.svg) ![Recall](https://img.shields.io/badge/Recall@1-90%25-brightgreen.svg) ![Speedup](https://img.shields.io/badge/Speedup-21x-red.svg)

**Author:** HUYNH CONG LUYEN (Senior Student at the University of Transport Ho Chi Minh City)  
**Tác giả:** HUỲNH CÔNG LUYỆN (Sinh viên năm cuối, Trường Đại học Giao thông Vận tải Thành phố Hồ Chí Minh)

---

## 📖 Overview / Tổng quan
Dự án này triển khai **TurboQuant** - một công cụ tìm kiếm vector mật độ cao (Dense Vector Search) tối ưu cho các hệ thống RAG quy mô lớn. Bằng cách kết hợp thuật toán nén **SQ+QJL** và cấu trúc chỉ mục **IVF (Inverted File)**, TurboQuant cho phép tìm kiếm trên hàng triệu vector với tốc độ cực nhanh và tài nguyên RAM cực thấp.

### Điểm nhấn thuật toán:
1. **SQ+QJL Optimization:** Nén vector về 2/4-bit bằng phép xoay trực giao và bù sai số bằng định lý Johnson-Lindenstrauss.
2. **IVF Integration:** Phân cụm dữ liệu để giảm không gian tìm kiếm, tăng tốc độ truy vấn gấp nhiều lần.
3. **Rust SIMD Core:** Lõi tính toán được viết bằng Rust với tập lệnh AVX2/FMA, tự động biên dịch ngay khi sử dụng.
4. **Out-of-Core Processing:** Khả năng truy vấn dữ liệu trực tiếp từ ổ cứng, giúp xử lý 5 triệu vector chỉ với ~230MB RAM.

---

## 🏆 Kết quả thực nghiệm (5 Triệu Vectors / 768-dim)

### 1. Hiệu năng tìm kiếm (Stress Test)
| Phương pháp | QPS | RAM Tiêu thụ | Speedup |
| :--- | :--- | :--- | :--- |
| RAW (Float32) | 0.06 | 932 MB | 1.0x |
| SQ 4-bit | 0.88 | 991 MB | 15.9x |
| PQ 4-bit | 0.59 | 865 MB | 10.6x |
| **TQ-IVF 4-bit (Ours)** | **1.17** | **236 MB** | **21.1x** |

### 2. Độ chính xác (Recall@K)
| Phương pháp | Recall@1 | Recall@10 | Trạng thái |
| :--- | :--- | :--- | :--- |
| SQ 4-bit | 83% | 100% | Kém ổn định |
| PQ 4-bit | 81% | 99% | Cần training lâu |
| **TQ-IVF 4-bit** | **90%** | **99%** | **Tốt nhất (No Training)** |

---

## 🛠 Hướng dẫn cài đặt & Sử dụng

### Yêu cầu hệ thống:
- **Python 3.10+**
- **Rust Compiler (Cargo)**: Để tự động build lõi SIMD.
- **CPU hỗ trợ AVX2/FMA** (Hầu hết các CPU Intel Core i thế hệ 4+).

### Sử dụng nhanh:
TurboQuant được thiết kế để sử dụng cục bộ (**Local Priority**) trong thư mục dự án của bạn.

```python
from TQ_engine_lib import TurboQuant

# 1. Khởi tạo với IVF
engine = TurboQuant(dim=768, bits=4, use_ivf=True, ivf_nlist=256)

# 2. Lập chỉ mục
engine.index(my_vectors)

# 3. Tìm kiếm
top_indices, scores = engine.search(query_vector, top_k=10)
```

## 📂 Cấu trúc dự án
- `TQ_engine_lib/`: Thư viện lõi (Python + Rust).
- `eval_alt/`: Bộ công cụ benchmark toàn diện.
- `benchmark_report_alt.md`: Báo cáo chi tiết kết quả đo lường.

---
*Dự án thuộc khuôn khổ đề tài nghiên cứu tốt nghiệp về tối ưu hóa hệ thống truy vấn vector quy mô lớn.*
