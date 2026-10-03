# Bai-Wu-WBSM4

## 方案与来源

baseline来自 https://github.com/Nexus-TYF/Bai-Wu-White-box-SM4，固定上游提交 `d5fde8cb9f2fce8c19281c5624e0f1a3c178c5d3`；未追随远程最新HEAD。

算法文献：Bai K, Wu C. A Secure White-Box SM4 Implementation. Security and Communication Networks, 2016.

## 版本与后端关系

| 库目标 | 矩阵后端/实现关系 | 论文对应结果 |
| --- | --- | --- |
| baiwu_baseline | WBMatrix_bai | 表4/5原始 |
| baiwu_backend | WBMatrixGFNI_bai | 表4/5后端 |
| baiwu_optimized | WBMatrixGFNI_bai | 表4/5本文 |

矩阵代码不在本目录重复放置，来源和兼容接口见[BackEnd](../BackEnd/README.md)。原始、后端与优化目标按上述真实差异组织。

## 优化方法

TR构表固定第一输入，对第二输入每批64项计算标准SM4 S盒并应用输出编码；执行阶段采用16分组交错方式组织TD/TR查询。

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
./build/test_baiwu_optimized_encrypt
./build/test_baiwu_optimized_decrypt
cd build
ctest --output-on-failure
```

每个库目标均有两个方向的测试。方向在实例生成时选择；普通SM4参考仅用于正确性核对，不进入白盒执行内核。 解密实例采用逆序轮密钥，加解密共用原执行内核，各实例参数按其方向生成。

实际调用关系见`test/release_adapter.c`，生成方向接口见`include/generation.h`（如有）。

## Benchmark

```bash
taskset -c 2 ./build/bench_baiwu_optimized_encrypt --messages 1048576 --samples 20
taskset -c 2 ./build/bench_baiwu_optimized_decrypt --messages 1048576 --samples 20
taskset -c 2 ./build/bench_baiwu_baseline_construction --samples 21
```

构表程序输出TSC及百万TSC计数，可加`--decrypt`。吞吐程序输出每样本Mbps、时间和校验和，默认20次，默认长度16 B至1 MiB；`--messages`指定逗号分隔的16 B整数倍长度，`--min-total-bytes`和`--warmup-bytes`控制累计处理量与预热。完整调用纳入计时，构表、分配和正确性检查位于吞吐计时外。

论文结果以根README为准，发布检查产生的新样本不替代论文数值。

## License与第三方说明

新增代码采用[Apache-2.0](../LICENSE)，已有版权声明不变。完整来源、作者和修改范围见[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)与[PROVENANCE.md](../PROVENANCE.md)。
