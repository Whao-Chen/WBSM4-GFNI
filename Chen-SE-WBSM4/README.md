# Chen-SE-WBSM4

## 方案与来源

baseline来自 https://github.com/scnucrypto/SIMD-White-Box-Block-Cipher，固定上游提交 `5f36390beb98bcbd914a35998e231ad9141a2773`；未追随远程最新HEAD。

算法文献：Chen J, Luo Y, Liu J, et al. A white-box implementation of SM4 with self-equivalence encoding. The Computer Journal, 2024, 67(3): 1087-1098.

相关SIMD实现：Chen L, Tang Y, Zhao L, et al. SIMD optimizations of white-box block cipher implementations with the self-equivalence framework. Information Security and Cryptology, 2025: 129–149。原代码来自SCNUcrypto，保留作者归属；上游无覆盖这两目录的明确许可证，本次经权利人授权按Apache-2.0发布。

## 版本与后端关系

| 库目标 | 矩阵后端/实现关系 | 论文对应结果 |
| --- | --- | --- |
| chen_baseline | WBMatrix_self_equivalence_native | 表4/5原始 |
| chen_construction_optimized | WBMatrix_self_equivalence；构表内局部GFNI | 表4本文；附带表执行仅用于验证 |
| chen_simd | WBMatrix_self_equivalence_native | 表5本文；独立SIMD执行实现 |

矩阵代码不在本目录重复放置，来源和兼容接口见[BackEnd](../BackEnd/README.md)。原始、后端与优化目标按上述真实差异组织。

## 优化方法

构表结合局部GFNI、128比特机器字分片和奇偶计算；执行通过128比特向量加载和异或处理宽表项。GFNI构表优化与SIMD执行优化是两个独立实验实现，不是已集成实现的两项结果。

## 目录

`include/`为接口与参数类型，`src/`为生成及执行实现，`test/`为功能验证、构表与吞吐入口。 `data/p.txt`和`data/p8.txt`为原实现所需的预存矩阵；通过CMake提供位置，不依赖原工作目录。GFNI构表先用Guo方法生成标准S盒与逆表，SIMD执行实现则以普通S盒数组初始化表。

## 构建

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
```

依赖Linux x86-64、GCC、CMake ≥ 3.16、OpenSSL开发包。ISA选项见[options.cmake](../BackEnd/options.cmake)，宽向量目标需GFNI、AVX2和所用AVX-512扩展，功能测试先检查CPU支持。

## 加密与解密测试

```bash
./build/test_chen_simd_encrypt
./build/test_chen_simd_decrypt
cd build
ctest --output-on-failure
```

每个库目标均有两个方向的测试。方向在实例生成时选择；普通SM4参考仅用于正确性核对，不进入白盒执行内核。 解密实例采用逆序轮密钥，加解密共用原执行内核，各实例参数按其方向生成。

实际调用关系见`test/release_adapter.c`，生成方向接口见`include/generation.h`（如有）。

## Benchmark

```bash
taskset -c 2 ./build/bench_chen_simd_encrypt --messages 1048576 --samples 20
taskset -c 2 ./build/bench_chen_simd_decrypt --messages 1048576 --samples 20
taskset -c 2 ./build/bench_chen_baseline_construction --samples 21
```

构表程序输出TSC及百万TSC计数，可加`--decrypt`。吞吐程序输出每样本Mbps、时间和校验和，默认20次，默认长度16 B至1 MiB；`--messages`指定逗号分隔的16 B整数倍长度，`--min-total-bytes`和`--warmup-bytes`控制累计处理量与预热。完整调用纳入计时，构表、分配和正确性检查位于吞吐计时外。 GFNI构表使用内部`total`计时字段，固定矩阵缓存预先准备，不将验证执行吞吐当作表5的SIMD吞吐。

论文结果以根README为准，发布检查产生的新样本不替代论文数值。

## License与第三方说明

新增代码采用[Apache-2.0](../LICENSE)，已有版权声明不变。完整来源、作者和修改范围见[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)与[PROVENANCE.md](../PROVENANCE.md)。
