# XL-WBSM4

## 方案与来源

baseline来自 https://github.com/Nexus-TYF/Xiao-Lai-White-box-SM4，固定上游提交 `84de39f8d82f206197cddb38177d758efc11a3b8`；未追随远程最新HEAD。

算法文献：Xiao Y, Lai X. White-box cryptography and a SMS4 implementation. 2009.

## 版本与后端关系

| 库目标 | 矩阵后端/实现关系 | 论文对应结果 |
| --- | --- | --- |
| xl_baseline | WBMatrix | 表4原始、表5原始 |
| xl_backend | WBMatrixGFNI | 表4后端、表5后端 |
| xl_construction_optimized | WBMatrixGFNI | 表4本文；不构成新的执行优化 |

矩阵代码不在本目录重复放置，来源和兼容接口见[BackEnd](../BackEnd/README.md)。原始、后端与优化目标按上述真实差异组织。

## 优化方法

复用构表过程中共同的仿射操作数和固定SM4线性因子，准备适合GFNI的输入仿射及Q-box输出映射表示，减少中间矩阵生成和重复表示转换。专用优化仅作用于构表，加密与解密继续采用相同的标量执行函数和所链接的矩阵后端。

## 目录

`include/`为接口与参数类型，`src/`为生成及执行实现，`test/`为功能验证、构表与吞吐入口。

## 构建

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
```

依赖Linux x86-64、GCC、CMake ≥ 3.16、OpenSSL开发包。ISA选项见[options.cmake](../BackEnd/options.cmake)，宽向量目标需GFNI、AVX2和所用AVX-512扩展，功能测试先检查CPU支持。

## 加密与解密测试

```bash
./build/test_xl_construction_optimized_encrypt
./build/test_xl_construction_optimized_decrypt
cd build
ctest --output-on-failure
```

每个库目标均有两个方向的测试。方向在实例生成时选择；普通SM4参考仅用于正确性核对，不进入白盒执行内核。 解密实例采用逆序轮密钥，加解密共用原执行内核，各实例参数按其方向生成。

实际调用关系见`test/release_adapter.c`，生成方向接口见`include/generation.h`（如有）。

## Benchmark

```bash
taskset -c 2 ./build/bench_xl_construction_optimized_encrypt --messages 1048576 --samples 20
taskset -c 2 ./build/bench_xl_construction_optimized_decrypt --messages 1048576 --samples 20
taskset -c 2 ./build/bench_xl_baseline_construction --samples 21
```

构表程序输出TSC及百万TSC计数，可加`--decrypt`。吞吐程序输出每样本Mbps、时间和校验和，默认20次，默认长度16 B至1 MiB；`--messages`指定逗号分隔的16 B整数倍长度，`--min-total-bytes`和`--warmup-bytes`控制累计处理量与预热。完整调用纳入计时，构表、分配和正确性检查位于吞吐计时外。

论文结果以根README为准，发布检查产生的新样本不替代论文数值。

## License与第三方说明

新增代码采用[Apache-2.0](../LICENSE)，已有版权声明不变。完整来源、作者和修改范围见[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)与[PROVENANCE.md](../PROVENANCE.md)。
