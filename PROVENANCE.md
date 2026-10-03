# 公开实现对应关系

本文件说明公开目标与论文正式实现的对应关系。完整文件身份见[source-manifest.tsv](source-manifest.tsv)；本地原始位置、构建记录和差异检查保存在不提交的内部审计文件中。

## 目标与论文结果

| 库目标 | 方案目录 | 后端目标 | 论文结果 |
| --- | --- | --- | --- |
| `xl_baseline` | XL-WBSM4 | `WBMatrix` | 表4原始、表5原始 |
| `xl_backend` | XL-WBSM4 | `WBMatrixGFNI` | 表4后端、表5后端 |
| `xl_construction_optimized` | XL-WBSM4 | `WBMatrixGFNI` | 表4本文；不构成新的执行优化 |
| `masked_baseline` | Masked-WBSM4 | `WBMatrix_xl` | 表4/5简化原始 |
| `masked_backend` | Masked-WBSM4 | `WBMatrixGFNI_xl` | 表4/5简化后端 |
| `masked_construction_optimized` | Masked-WBSM4 | `WBMatrixGFNI_xl` | 表4简化本文 |
| `masked_optimized` | Masked-WBSM4 | `WBMatrixGFNI_xl` | 表5简化本文 |
| `masked_enhanced_baseline` | Masked-WBSM4 | `WBMatrix_xl` | 表4/5增强原始 |
| `masked_enhanced_backend` | Masked-WBSM4 | `WBMatrixGFNI` | 表4/5增强后端 |
| `masked_enhanced_optimized` | Masked-WBSM4 | `WBMatrixGFNI` | 表4/5增强本文 |
| `baiwu_baseline` | Bai-Wu-WBSM4 | `WBMatrix_bai` | 表4/5原始 |
| `baiwu_backend` | Bai-Wu-WBSM4 | `WBMatrixGFNI_bai` | 表4/5后端 |
| `baiwu_optimized` | Bai-Wu-WBSM4 | `WBMatrixGFNI_bai` | 表4/5本文 |
| `wsise_baseline` | WSISE | `WBMatrix_wsise` | 表4/5原始 |
| `wsise_backend` | WSISE | `WBMatrixGFNI_wsise` | 表4/5后端 |
| `wsise_optimized` | WSISE | `WBMatrixGFNI_wsise` | 表5本文；沿用构表，不增补表4结果 |
| `wbdl_baseline` | WBDL | `WBMatrix_wbdl` | 表4/5原始 |
| `wbdl_backend` | WBDL | `WBMatrixGFNI_legacy` | 表4/5后端 |
| `wbdl_optimized` | WBDL | `WBMatrixGFNI_legacy` | 表4/5本文 |
| `chen_baseline` | Chen-SE-WBSM4 | `WBMatrix_self_equivalence_native` | 表4/5原始 |
| `chen_construction_optimized` | Chen-SE-WBSM4 | `WBMatrix_self_equivalence；构表内局部GFNI` | 表4本文；附带表执行仅用于验证 |
| `chen_simd` | Chen-SE-WBSM4 | `WBMatrix_self_equivalence_native` | 表5本文；独立SIMD执行实现 |
| `gong_baseline` | Gong-SE-WBSM4 | `WBMatrix_self_equivalence_native` | 表4原始 |
| `gong_backend` | Gong-SE-WBSM4 | `WBMatrixGFNI_self_equivalence` | 表4后端；既有仿射组合与互逆对生成优化 |
| `gong_gfni` | Gong-SE-WBSM4 | `WBMatrix_self_equivalence_native；执行专用GFNI` | 表6本文 |
| `gong_avx2` | Gong-SE-WBSM4 | `WBMatrix_self_equivalence` | 表6公开32分组AVX2对照 |

每个库目标对应`test_<目标>_encrypt/decrypt`及`bench_<目标>_encrypt/decrypt/construction`。附带测试或benchmark不代表论文新增该列结果；构表专用目标的执行入口仅供功能核对。

## 生成和执行入口

各方案的`test/release_adapter.c`直接连接实际实例生成与执行接口；`test/adapter_common.c`连接功能参考与吞吐计时程序。解密方向只在生成阶段选择逆序轮密钥，原名称的生成接口仍默认加密；Gong保留原有encmode，WBDL保留原生逆向执行。

简化Masked的表4构表目标与表5吞吐目标分别保留正式源码；Chen GFNI构表与SIMD执行也分别保留。XL专用构表不增加方案专用执行优化，WSISE未另列本文构表数据。Gong的`gong_backend`保留已有构表优化，`gong_gfni`与`gong_avx2`分别对应表6的8分组GFNI与公开32分组AVX2。

## 上游身份

- WBMatrix：`0043eb53e4ee1d7641c06c0155b5f2357bce18e9`。
- WBMatrix-GFNI：`9fee4d178158f73764ad8e1cadaf6e0477ba19bb`，标签`v2.0.0`。
- XL：`84de39f8d82f206197cddb38177d758efc11a3b8`。
- Bai–Wu：`d5fde8cb9f2fce8c19281c5624e0f1a3c178c5d3`。
- WSISE：`96b78b0bb04385cf186890e518bb825113c1d06a`。
- SCNUcrypto SIMD：`5f36390beb98bcbd914a35998e231ad9141a2773`。

源码来源和许可证见[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。本文自行实现的Masked与WBDL不冒称上游官方源码。

## 矩阵参考身份

不同方案正式代码携带的矩阵参考不一定等于WBMatrix公开主库的同一文件。集中保存各实际版本，并明确后端选择，未用更新后的矩阵函数静默替换它们。

| 参考接口 | WBMatrix.c SHA-256 |
| --- | --- |
| bai | `c709afea1064664c3b8cd313b79774dab3373a6d29eeb9630fd6a8dba1f524e8` |
| legacy | `2332aa314c048c1efd5f8a42d04f4e6669d83094aa16147e0f3747b14b4387c8` |
| self_equivalence | `b704656fb5ac4c3b09adbf9211881cdfdaeab79cf650c5db899de670347b963e` |
| wbdl | `a3624207aab51ab717aa9744093d3a50f9f3775b62283538338e8e5785b5b633` |
| wsise | `af8f56c826bc044c5a9261031eabae68d243b69a3750f3d14b2b4ca6398b0a32` |
| xl | `a3624207aab51ab717aa9744093d3a50f9f3775b62283538338e8e5785b5b633` |

WBMatrix-GFNI公开核心11个文件与固定提交的文件字节一致。兼容接口所选57个函数，以及Chen已选配置中的40个构表/表执行函数，分别核对原函数：除标识符规范和已验证的密钥方向适配外，计算体保持不变。Gong只保留最终8分组GFNI及其所需尾批函数。

## 非算法性整理

目录、包含路径、目标名称与测试入口规范；提取必要计算函数；移除未使用候选声明及开发说明；加入解密生成方向接口，默认加密语义不变；为Chen必需矩阵数据提供基于当前工程目录的位置。没有改变intrinsic顺序、批宽、展开方式或数据布局，没有用发布启动检查数值替换论文结果。

构表与吞吐各按其原计时范围组织；Chen完整构表使用选定实现内部的total计数，不把固定矩阵缓存加载或验证哈希计入。吞吐保留CLOCK_MONOTONIC_RAW及完整调用计时。
