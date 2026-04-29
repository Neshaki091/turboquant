# TurboQuant Performance & Accuracy Report (TQ_engine_lib)
**Generated at:** 2026-04-29 22:02:30

## 1. System Configuration
| Component | Specification |
| :--- | :--- |
| **CPU** | Intel(R) Core(TM) i5-10300H CPU @ 2.50GHz |
| **RAM** | 15.84 GB |
| **OS** | Windows 11 |
| **SIMD** | AVX2, FMA (TurboQuant Native SIMD Active - via TQ_engine_lib) |
| **Core Library** | **TQ_engine_lib** |
| **Python** | 3.13.13 |
| **PyTorch** | 2.11.0+cpu |

## 2. Benchmark Parameters
- **Total Vectors:** 5,000,000 (Stress Test)
- **Dimension:** 768
- **Batch Sizes (TQ):** 4M (2bit), 1.5M (4bit)
- **Batch Sizes (SQ):** 4.3M (2bit), 2.1M (4bit)
- **Queries:** 5 queries per iteration (Stress), 50 queries (Recall)
- **PQ Configuration:** Custom training (10@start, 50@1k, 100@10k, 50@13k, 46@end | 20 iterations). M=96 (2b), M=192 (4b).

## 3. Performance Results (Stress Test 5M)
```text
TurboQuant Native (Rust SIMD) Active - SQ+QJL Mode

Benchmarking RAW (Float32) - Batch: 250000...
Benchmarking SQ 2-bit - Batch: 4300000...
Benchmarking SQ 4-bit - Batch: 2100000...
Benchmarking PQ 2-bit - Batch: 4000000...
Benchmarking PQ 4-bit - Batch: 1750000...
Benchmarking TQ 2bit - Batch: 4000000...
Benchmarking TQ 4bit - Batch: 1500000...
Benchmarking TQ IVF 2bit...
Benchmarking TQ IVF 4bit...

============================================================================================================
Method          | Batch       | Peak RAM     | Latency    | QPS        | Speedup
------------------------------------------------------------------------------------------------------------
RAW (F32)       |     250,000 |     932.1 MB |   18.1062s |      0.06 |      1.0x
SQ 2-bit        |   4,300,000 |    1009.1 MB |    0.9548s |      1.05 |     19.0x
SQ 4-bit        |   2,100,000 |     991.2 MB |    1.1420s |      0.88 |     15.9x
PQ 2-bit        |   4,000,000 |     955.4 MB |    0.7484s |      1.34 |     24.2x
PQ 4-bit        |   1,750,000 |     865.8 MB |    1.7055s |      0.59 |     10.6x
TQ-IVF 2bit     |    probe:32 |     233.2 MB |    0.8768s |      1.14 |     20.6x
TQ-IVF 4bit     |    probe:32 |     236.7 MB |    0.8572s |      1.17 |     21.1x
============================================================================================================
```

## 4. Accuracy Results (Recall@K)
```text
TurboQuant Native (Rust SIMD) Active - SQ+QJL Mode

===============================================================================================
TURBOQUANT COMPREHENSIVE RECALL: SQ vs PQ vs TQ vs TQ-IVF
===============================================================================================
Config: PQ trained on Highly Fragmented 256 samples (10@start, 50@1k, 100@10k, 50@13k, 46@end).
===============================================================================================
Dataset: 28378 vectors x 768d | 100 queries

Computing Ground Truth...
  ---- SQ 2-bit...
  PQ 2-bit/dim (M=96, sub_dim=8 | Highly Fragmented Training: 256 samples, 20 iter)...
  ---- TQ 2-bit (SQ+QJL Native)...
  ---- TQ-IVF 2-bit (SQ+QJL Native with IVF)...
  ---- SQ 4-bit...
  PQ 4-bit/dim (M=192, sub_dim=4 | Highly Fragmented Training: 256 samples, 20 iter)...
  ---- TQ 4-bit (SQ+QJL Native)...
  ---- TQ-IVF 4-bit (SQ+QJL Native with IVF)...

==============================================================================================================
TABLE 1: TOP-1 IN K PROBABILITY (Is the true best result within predicted Top-K?)
==============================================================================================================
Method       | P@K=1  | P@K=2  | P@K=4  | P@K=8  | P@K=16 | P@K=32 | P@K=64 |      QPS
--------------------------------------------------------------------------------------------------------------
SQ 2-bit     |   5.0% |  11.0% |  16.0% |  25.0% |  31.0% |  35.0% |  50.0% |    308.3
PQ 2-bit     |  53.0% |  78.0% |  86.0% |  95.0% |  99.0% | 100.0% | 100.0% |    303.6
TQ 2-bit     |  68.0% |  86.0% |  94.0% |  99.0% |  99.0% |  99.0% | 100.0% |     21.7
TQ-IVF 2-bit |  67.0% |  86.0% |  93.0% |  97.0% |  97.0% |  97.0% |  98.0% |     18.4
SQ 4-bit     |  83.0% |  93.0% |  99.0% | 100.0% | 100.0% | 100.0% | 100.0% |    329.6
PQ 4-bit     |  81.0% |  92.0% |  97.0% |  99.0% | 100.0% | 100.0% | 100.0% |    324.6
TQ 4-bit     |  90.0% |  94.0% |  99.0% | 100.0% | 100.0% | 100.0% | 100.0% |     20.5
TQ-IVF 4-bit |  90.0% |  93.0% |  98.0% |  98.0% |  98.0% |  98.0% |  98.0% |     17.1

==============================================================================================================
TABLE 2: SET RECALL@K (Percentage of actual Top-K items found in predicted Top-K)
==============================================================================================================
Method       | R@K=1  | R@K=2  | R@K=4  | R@K=8  | R@K=16 | R@K=32 | R@K=64 |      QPS
--------------------------------------------------------------------------------------------------------------
SQ 2-bit     |   5.0% |   7.5% |   8.8% |  13.5% |  13.4% |  14.6% |  15.9% |    308.3
PQ 2-bit     |  53.0% |  61.0% |  59.8% |  58.9% |  61.7% |  64.2% |  65.8% |    303.6
TQ 2-bit     |  68.0% |  70.0% |  70.2% |  77.2% |  76.4% |  77.0% |  77.9% |     21.7
TQ-IVF 2-bit |  67.0% |  70.0% |  71.0% |  76.4% |  75.5% |  75.4% |  74.9% |     18.4
SQ 4-bit     |  83.0% |  78.0% |  78.0% |  78.6% |  78.9% |  81.1% |  82.0% |    329.6
PQ 4-bit     |  81.0% |  79.0% |  79.2% |  76.1% |  78.2% |  79.3% |  80.9% |    324.6
TQ 4-bit     |  90.0% |  84.5% |  88.5% |  88.1% |  88.2% |  89.4% |  89.3% |     20.5
TQ-IVF 4-bit |  90.0% |  82.5% |  87.8% |  85.8% |  85.8% |  85.9% |  84.9% |     17.1

==============================================================================================================
(*) PQ trained on custom fragmented 256 samples (10@start, 50@1k, 100@10k, 50@13k, 46@end).
(*) PQ configuration: M=96 for 2-bit, M=192 for 4-bit.
```

## 5. Execution Summary
- **Stress Test Duration:** 668.76s
- **Recall Test Duration:** 54.45s
- **Total Time:** 723.21s
- **Status:** All tests completed successfully.
