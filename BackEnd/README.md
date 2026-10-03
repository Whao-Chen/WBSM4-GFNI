# 矩阵后端

## 来源与固定版本

| 项目 | 来源 | 固定提交/标签 | 用途与许可证 |
| --- | --- | --- | --- |
| WBMatrix | https://github.com/Nexus-TYF/WBMatrix | `0043eb53e4ee1d7641c06c0155b5f2357bce18e9` | 固定尺寸矩阵与仿射运算基线，Apache-2.0 |
| WBMatrix-GFNI | https://github.com/Whao-Chen/WBMatrixGFNI/ | `9fee4d178158f73764ad8e1cadaf6e0477ba19bb`，标签`v2.0.0` | GFNI优化后端，Apache-2.0 |

`WBMatrix/include`和`WBMatrix/src`保存基准库；`WBMatrix-GFNI/include`和`WBMatrix-GFNI/src`的11个核心文件保持上述公开提交的字节内容。原有LICENSE、NOTICE与来源说明均保留，见[WBMatrix-GFNI/provenance](WBMatrix-GFNI/provenance/README.md)。

## 实验接口与参考库

不同方案的正式实验曾使用各自基础项目携带的WBMatrix版本。为了保持实际生成器和执行函数，`WBMatrix/reference/`集中保存对应参考实现；它们不是额外提出的矩阵算法，也不在各方案目录重复复制。`WBMatrix-GFNI/compat/`保存正式实验所需的已选矩阵与仿射计算接口。兼容接口按依赖关系提取原函数，删除未使用的候选声明，规范部分标识符；不改变运算、布局或指令顺序。

| 库目标 | 使用关系 |
| --- | --- |
| `WBMatrix` / `WBMatrixGFNI` | XL原始/后端及专用构表；增强Masked后端与优化实现使用后者 |
| `WBMatrix_xl` / `WBMatrixGFNI_xl` | 简化Masked原始/后端及优化；增强Masked原始使用前者 |
| `WBMatrix_bai` / `WBMatrixGFNI_bai` | Bai–Wu原始/后端及优化 |
| `WBMatrix_wsise` / `WBMatrixGFNI_wsise` | WSISE原始/后端及执行优化 |
| `WBMatrix_wbdl` / `WBMatrixGFNI_legacy` | WBDL原始/后端及优化 |
| `WBMatrix_self_equivalence` | Gong公开AVX2、Chen GFNI构表的基础矩阵函数 |
| `WBMatrix_self_equivalence_native` | Chen原始与SIMD执行、Gong原始及专用GFNI执行的构表基础 |
| `WBMatrixGFNI_self_equivalence` | Gong已有仿射组合与互逆仿射对生成优化 |

`WBMatrixGFNI_<profile>`将所选优化接口与对应参考接口链接，保留未被替换的基础函数。兼容头文件优先使用正式实验所需的类型定义。原始参考文件的精确身份以[PROVENANCE.md](../PROVENANCE.md)的源码指纹为准，不把经过适配的文件冒称为远程最新版本。

## 构建与版权

各方案通过`add_subdirectory`和目标链接选择后端。完整编译选项见[options.cmake](options.cmake)，ISA要求由实际计算函数决定。

已有SCNUCrypto版权头和Apache-2.0声明保留，WBMatrix与WBMatrix-GFNI各自的LICENSE不变。完整第三方说明见[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。
