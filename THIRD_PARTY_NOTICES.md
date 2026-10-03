# 第三方来源与许可证

根目录Apache-2.0覆盖本仓库新增代码及明确授权的代码，不覆盖或删除第三方既有版权声明。

| 项目 | 仓库 | 使用提交/身份 | 作者归属 | 原始许可证 | 使用位置与修改 |
| --- | --- | --- | --- | --- | --- |
| WBMatrix | https://github.com/Nexus-TYF/WBMatrix | `0043eb53e4ee1d7641c06c0155b5f2357bce18e9`；参考接口指纹见PROVENANCE | SCNUCrypto，原文件版权声明；Tang Y等人的WBMatrix工作 | Apache-2.0 | BackEnd/WBMatrix；保留各基础项目所用参考代码，规范部分包含路径 |
| WBMatrix-GFNI | https://github.com/Whao-Chen/WBMatrixGFNI/ | `9fee4d178158f73764ad8e1cadaf6e0477ba19bb`，`v2.0.0`；兼容接口另记指纹 | WBMatrix原作者及WBMatrix-GFNI贡献者 | Apache-2.0 | BackEnd/WBMatrix-GFNI；公开核心原字节保留，兼容代码提取已选函数并规范标识符 |
| Xiao–Lai White-box SM4 | https://github.com/Nexus-TYF/Xiao-Lai-White-box-SM4 | `84de39f8d82f206197cddb38177d758efc11a3b8` | Nexus-TYF/SCNUCrypto；算法Xiao Y、Lai X | Apache-2.0 | XL-WBSM4及相应矩阵参考；构表专用代码来自本文实验，加入方向接口和测试适配 |
| Bai–Wu White-box SM4 | https://github.com/Nexus-TYF/Bai-Wu-White-box-SM4 | `d5fde8cb9f2fce8c19281c5624e0f1a3c178c5d3` | Nexus-TYF/SCNUCrypto；算法Bai K、Wu C | Apache-2.0 | Bai-Wu-WBSM4及矩阵参考；保留本文已选构表与执行实现，加入解密方向接口 |
| WSISE White-box SM4 | https://github.com/Nexus-TYF/WSISE-White-box-SM4 | `96b78b0bb04385cf186890e518bb825113c1d06a` | Nexus-TYF/SCNUCrypto；算法Yao S、Chen J | Apache-2.0 | WSISE及矩阵参考；保留后端与专用执行实现，加入方向接口 |
| SIMD White-Box Block Cipher | https://github.com/scnucrypto/SIMD-White-Box-Block-Cipher | `5f36390beb98bcbd914a35998e231ad9141a2773` | SCNUcrypto；公开SIMD工作Chen L、Tang Y、Zhao L等；Chen方案Chen J、Luo Y、Liu J等；Gong方案Gong Z、Tang Y、Yi N | 见下述授权说明 | Chen-SE-WBSM4、Gong-SE-WBSM4及其矩阵参考；保留本文构表/GFNI执行和公开AVX2对照，提取必要接口并加入测试 |

## SCNUcrypto授权

Upstream repository does not contain an explicit license file covering its root and the selected Chen/Gong directories. 该上游的其他子目录许可证不能自动解释为覆盖本次使用源码。

权利人已确认有权授权本次使用的Chen/Gong源码在本仓库按Apache-2.0发布，并要求保留作者归属。本仓库据此发布所选源码；不将此授权误述为上游整个仓库原本带有Apache-2.0许可证。文件中已有版权头、作者说明继续保留。

公开SIMD工作：Chen L, Tang Y, Zhao L, et al. *SIMD optimizations of white-box block cipher implementations with the self-equivalence framework*. Information Security and Cryptology, 2025: 129–149。算法文献见各方案README。

## 本文作者实现与已有技术

Masked-WBSM4与WBDL的baseline由本文作者根据原论文自行实现，不是算法原作者的官方源码。论文优化、方向接口、测试和构建整理采用Apache-2.0。算法来源分别为Zhao D等人的Masked-WBSM4工作和Yang Y T等人的WBDL工作。

标准SM4 S盒的GFNI仿射—求逆—仿射计算采用Guo的已有方法：*Efficient Constant-Time Implementation of SM4 with Intel GFNI instruction set extension and Arm NEON coprocessor*, Cryptology ePrint 2022/1154，https://eprint.iacr.org/2022/1154 。本仓库保留论文正式实验中所用实现，不将该方法列为本文新提出的S盒算法。

正确性测试链接系统OpenSSL的SM4实现，仅作独立参考；不在仓库中重新分发OpenSSL代码。本仓库不包含未使用的普通SM4查表移植文件。

## 修改范围

目录与文件名规范、包含路径调整、CMake目标链接、必要函数选择、方向接口及测试适配。保留所选核心函数的运算、intrinsic顺序、批宽与数据布局；没有加入新的性能优化。源码指纹与目标映射见[PROVENANCE.md](PROVENANCE.md)。
