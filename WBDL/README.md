# WBDL

## 方案与来源

本目录中的baseline为本文作者根据原论文算法描述自行实现的版本，并非原论文作者提供的官方源码。

算法文献：Yang Y T, Dong H, Zhang Y, et al. WBDL: An Improved Dynamic White-Box Cryptographic Algorithm for SM4. Journal of Cryptologic Research, 2023, 10(4): 796-812.

## 版本与后端关系

| 库目标 | 矩阵后端/实现关系 | 论文对应结果 |
| --- | --- | --- |
| wbdl_baseline | WBMatrix_wbdl | 表4/5原始 |
| wbdl_backend | WBMatrixGFNI_legacy | 表4/5后端 |
| wbdl_optimized | WBMatrixGFNI_legacy | 表4/5本文 |

矩阵代码不在本目录重复放置，来源和兼容接口见[BackEnd](../BackEnd/README.md)。原始、后端与优化目标按上述真实差异组织。

## 优化方法

利用独立表项与固定参数批量生成候选表，保留原方案数据相关的动态选表与逆向执行逻辑。

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
./build/test_wbdl_optimized_encrypt
./build/test_wbdl_optimized_decrypt
cd build
ctest --output-on-failure
```

每个库目标均有两个方向的测试。方向在实例生成时选择；普通SM4参考仅用于正确性核对，不进入白盒执行内核。 本方案保留数据相关动态选表和原有逆向执行，以动态参考验证，不能等同普通单主密钥SM4。

实际调用关系见`test/release_adapter.c`，生成方向接口见`include/generation.h`（如有）。

## Benchmark

```bash
taskset -c 2 ./build/bench_wbdl_optimized_encrypt --messages 1048576 --samples 20
taskset -c 2 ./build/bench_wbdl_optimized_decrypt --messages 1048576 --samples 20
taskset -c 2 ./build/bench_wbdl_baseline_construction --samples 21
```

构表程序输出TSC及百万TSC计数，可加`--decrypt`。吞吐程序输出每样本Mbps、时间和校验和，默认20次，默认长度16 B至1 MiB；`--messages`指定逗号分隔的16 B整数倍长度，`--min-total-bytes`和`--warmup-bytes`控制累计处理量与预热。完整调用纳入计时，构表、分配和正确性检查位于吞吐计时外。

论文结果以根README为准，发布检查产生的新样本不替代论文数值。

## License与第三方说明

新增代码采用[Apache-2.0](../LICENSE)，已有版权声明不变。完整来源、作者和修改范围见[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)与[PROVENANCE.md](../PROVENANCE.md)。
