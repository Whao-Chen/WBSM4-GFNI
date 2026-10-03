# WBSM4-GFNI

本仓库提供论文《基于 GFNI 的白盒 SM4 构表与并行执行优化》的实现代码和实验程序。

## 摘要

白盒SM4（White-Box SM4，WBSM4）用于在执行过程可被观察和干预的环境中保护SM4密钥，但编码计算与查找表处理增加了实例生成和执行开销。针对不同白盒构造中仿射运算的阶段分布与表示差异，本文结合Galois 域新指令（Galois Field New Instructions，GFNI）与单指令多数据（Single Instruction, Multiple Data，SIMD）技术开展优化。构表阶段复用固定矩阵因子和编码参数，减少矩阵表示转换，并批量生成独立表项；执行阶段结合多分组仿射求值、分支融合与并行查表，对已表化的大位宽仿射层采用向量加载和表项异或。在Intel Core i7-11800H平台上，对采用查找表与随机仿射编码的Xiao-Lai WBSM4（XL-WBSM4），以及在其基础上引入布尔掩码的简化版和代数增强版Masked-WBSM4进行测试，三种实现的完整构表相对各自原始实现加速比分别为100.213、227.341和746.991。对1 MiB消息，两种掩码实现的实测中位吞吐率分别达到719.734和299.441 Mbps，采用内部状态扩充的白盒SM4实现（WSISE）为757.440 Mbps，Gong等人基于自等价编码的白盒SM4实现为380.679 Mbps。实验结果表明，结合具体编码结构组织参数复用与并行计算，能够降低白盒实例生成开销并提高批量执行吞吐率。

## 代码结构

```text
BackEnd/          WBMatrix、WBMatrix-GFNI及实验所用兼容接口
XL-WBSM4/         Xiao–Lai白盒SM4
Masked-WBSM4/     简化版与全32轮代数增强版
Bai-Wu-WBSM4/     Bai–Wu白盒SM4
WSISE/            内部状态扩充方案
WBDL/             动态选表方案
Chen-SE-WBSM4/    Chen自等价方案
Gong-SE-WBSM4/    Gong自等价方案及公开AVX2对照
```

各方案包含自己的`include/`、`src/`、`test/`、CMake与中文README。矩阵后端集中于`BackEnd/`。

## 实现与基线

| 方案 | 原始实现来源 | 矩阵后端 | 本文优化目标 | 加密 | 解密 |
| --- | --- | --- | --- | --- | --- |
| XL-WBSM4 | Nexus-TYF/Xiao-Lai-White-box-SM4 | WBMatrix / WBMatrix-GFNI | xl_construction_optimized（仅构表） | 支持 | 支持 |
| 简化Masked | 本文作者根据原论文实现 | WBMatrix / WBMatrix-GFNI兼容接口 | masked_construction_optimized / masked_optimized | 支持 | 支持 |
| 增强Masked | 本文作者根据原论文实现 | WBMatrix / WBMatrix-GFNI | masked_enhanced_optimized | 支持 | 支持 |
| Bai–Wu | Nexus-TYF/Bai-Wu-White-box-SM4 | WBMatrix / WBMatrix-GFNI兼容接口 | baiwu_optimized | 支持 | 支持 |
| WSISE | Nexus-TYF/WSISE-White-box-SM4 | WBMatrix / WBMatrix-GFNI兼容接口 | wsise_optimized（执行） | 支持 | 支持 |
| WBDL | 本文作者根据原论文实现 | WBMatrix / WBMatrix-GFNI兼容接口 | wbdl_optimized | 动态算法 | 原动态算法的逆向执行 |
| Chen SE | SCNUcrypto公开代码及本文适配 | WBMatrix，无论文后端单列结果 | chen_construction_optimized / chen_simd（独立实现） | 支持 | 支持 |
| Gong SE | SCNUcrypto公开代码 | WBMatrix / WBMatrix-GFNI兼容接口 | gong_gfni；公开对照gong_avx2 | 支持 | 支持 |

目标后缀`baseline`表示原始实现，`backend`表示后端实现，其他目标的具体对应关系见各方案README与[PROVENANCE.md](PROVENANCE.md)。并非每个方案均存在三列结果，未测结果不作补填。

## 后端

[WBMatrix](https://github.com/Nexus-TYF/WBMatrix)是固定尺寸矩阵与仿射运算基线；[WBMatrix-GFNI](https://github.com/Whao-Chen/WBMatrixGFNI/)是优化后端仓库。本仓库固定使用已核验源码，同时保留正式实验中实际使用的参考库与兼容接口，详见[BackEnd/README.md](BackEnd/README.md)。

## 优化方法

**XL-WBSM4**：复用构表过程中共同的仿射操作数和固定SM4线性因子，准备适合GFNI的输入仿射及Q-box输出映射表示，减少中间矩阵生成和重复表示转换。专用优化仅作用于构表，加密与解密继续采用相同的标量执行函数和所链接的矩阵后端。

**Masked-WBSM4**：构表时固定其余输入，将64个候选输入组织到向量通道中生成双输入表项；执行时对16个独立分组组织GFNI仿射求值、并行查表和掩码状态更新。简化版与全32轮代数增强版分别保留原有掩码机制。

**Bai-Wu-WBSM4**：TR构表固定第一输入，对第二输入每批64项计算标准SM4 S盒并应用输出编码；执行阶段采用16分组交错方式组织TD/TR查询。

**WSISE**：对固定32比特显式仿射映射准备字节子矩阵，在16个分组间并行GFNI求值，并结合并行查表和状态更新。构表采用已有矩阵后端，没有另列本文专用构表结果。

**WBDL**：利用独立表项与固定参数批量生成候选表，保留原方案数据相关的动态选表与逆向执行逻辑。

**Chen-SE-WBSM4**：构表结合局部GFNI、128比特机器字分片和奇偶计算；执行通过128比特向量加载和异或处理宽表项。GFNI构表优化与SIMD执行优化是两个独立实验实现，不是已集成实现的两项结果。

**Gong-SE-WBSM4**：将实例中固定32比特仿射映射预处理为GFNI表示，执行时每批8个分组并行求值，尾部逐分组处理；对照保留Chen等人的公开32分组AVX2 SIMD查表实现。

简化/增强Masked、Bai–Wu和WBDL构表采用Guo的GFNI标准SM4 S盒方法（Cryptology ePrint 2022/1154）；Chen构表先生成标准S盒与相关逆表。执行阶段均使用各自白盒表或编码参数，不以普通SM4替代。XL、Gong及WSISE的标准S盒仍查表。

## 构建

依赖：Linux x86-64、GCC、CMake ≥ 3.16和OpenSSL开发包（独立正确性参考）。

```bash
sudo apt-get install build-essential cmake libssl-dev
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_MASKED=ON
cmake --build build -j2
```

根目录选项`BUILD_XL`、`BUILD_MASKED`、`BUILD_BAIWU`、`BUILD_WSISE`、`BUILD_WBDL`、`BUILD_CHEN`、`BUILD_GONG`默认均为OFF，可分别开启。也可进入任一方案目录独立构建：

```bash
cd Gong-SE-WBSM4
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
```

当前配置面向论文x86平台。宽向量目标需AVX2、GFNI及实际使用的AVX-512扩展（包括F/DQ/BW/VL、VPOPCNTDQ、VBMI和BITALG）。完整选项见`BackEnd/options.cmake`；`-march=native`生成面向构建机器的程序，不应将该二进制直接分发给不同CPU。

## 测试

在相应构建目录执行`ctest --output-on-failure`，或分别执行`test_<目标>_encrypt`与`test_<目标>_decrypt`。测试覆盖标准已知答案、三组密钥和实例、批宽边界及尾批，并使用OpenSSL SM4交叉核对。WBDL按其动态参考与原有逆向逻辑验证，不视为普通单密钥SM4。CPU不支持必要扩展时测试返回77，不能解释为通过。

## 性能测试

每个方案单独提供`bench_<目标>_construction`、`bench_<目标>_encrypt`与`bench_<目标>_decrypt`，不提供一键运行全部实验的脚本。例如：

```bash
taskset -c 2 ./build/Masked-WBSM4/bench_masked_optimized_encrypt --messages 1048576 --samples 20
taskset -c 2 ./build/Masked-WBSM4/bench_masked_optimized_decrypt --messages 1048576 --samples 20
taskset -c 2 ./build/Masked-WBSM4/bench_masked_construction_optimized_construction --samples 21
```

独立构建时可执行文件直接位于该方案的`build/`。吞吐入口保留完整调用的计时边界，使用`CLOCK_MONOTONIC_RAW`；默认测试16、64、128、256、272、512、528、8912、1048576 B，可用`--messages`指定逗号分隔的16 B整数倍长度。`--samples`、`--min-total-bytes`和`--warmup-bytes`控制重复次数、每样本累计字节数与预热。输出包括原始样本Mbps、时间、TSC差值与校验和。短消息按其原长度重复完整调用。

构表入口默认21次，可加`--decrypt`生成解密实例，输出TSC计数与百万TSC计数；TSC差值不等于动态频率下的核心执行周期。实例对象分配位于计时外；Chen构表沿用所选实现的内部完整构表区间，固定矩阵缓存准备位于该区间外。

## 实验平台

论文平台：Intel Core i7-11800H，8核16线程，40 GB内存，WSL2 Ubuntu 20.04，GCC 9.4.0，`-O2`、`-march=native`，单线程，固定逻辑CPU 2。各目标实际ISA选项按原实现保留；Chen GFNI构表采用AVX2/GFNI配置。

## 实验结果

以下摘要与表格来自论文最终稿，引用编号沿用论文。不同处理器、编译器、系统负载和频率策略会导致实测性能变化，下表为论文实验平台上的结果，不以发布整理时的启动检查结果替换。

### 表4 白盒SM4实现的完整构表开销比较

原始实现、WBMatrix-GFNI及本文列单位为百万TSC计数。加速比为原始实现计数除以本文实现计数。

| 方案 | 原始实现 | WBMatrix-GFNI[20] | 本文 | 相对原始实现加速比 |
| --- | --- | --- | --- | --- |
| XL-WBSM4[3] | 32.726 | 17.588 | 0.327 | 100.213 |
| 简化版Masked-WBSM4[10] | 235.540 | 11.293 | 1.036 | 227.341 |
| 代数增强版Masked-WBSM4[10] | 1975.669 | 55.929 | 2.645 | 746.991 |
| Bai-Wu[12] | 4849.418 | 210.502 | 14.286 | 339.446 |
| WSISE[13] | 95.072 | 3.461 | -- | -- |
| WBDL[14] | 70.636 | 1.067 | 0.768 | 91.965 |
| Chen SE-WBSM4[5] | 15021.430 | -- | 123.091 | 122.035 |
| Gong SE-WBSM4[28] | 5.701 | 0.053 | -- | -- |

### 表5 白盒SM4实现处理1 MiB消息的吞吐率比较

吞吐率单位为Mbps。加速比为本文吞吐率除以原始实现吞吐率。

| 方案 | 原始实现 Mbps | WBMatrix-GFNI[20] Mbps | 本文 Mbps | 相对原始实现加速比 |
| --- | --- | --- | --- | --- |
| XL-WBSM4[3] | 3.543 | 47.014 | -- | -- |
| 简化版Masked[10] | 3.718 | 118.050 | 719.734 | 193.605 |
| 代数增强版Masked[10] | 2.855 | 20.857 | 299.441 | 104.868 |
| Bai-Wu[12] | 19.805 | 25.979 | 49.934 | 2.521 |
| WSISE[13] | 4.244 | 150.849 | 757.440 | 178.489 |
| WBDL[14] | 4.135 | 89.442 | 96.297 | 23.286 |
| Chen SE[5] | 0.496 | -- | 15.922 | 32.072 |

### 表6 Gong SE-WBSM4在不同消息长度下的吞吐率比较

| 单次消息长度 | Chen等人的SIMD方法[21]/Mbps | 本文GFNI实现/Mbps |
| --- | --- | --- |
| 16 B | 21.377 | 107.414 |
| 64 B | 84.332 | 103.694 |
| 128 B | 162.367 | 316.835 |
| 256 B | 326.802 | 349.665 |
| 272 B | 334.985 | 298.672 |
| 512 B | 664.866 | 360.995 |
| 528 B | 359.032 | 346.134 |
| 8912 B | 690.420 | 356.850 |
| 1 MiB | 683.047 | 380.679 |

`--`表示无对应结果或不适用。XL本文专用优化仅用于构表，其47.014 Mbps来自矩阵后端替换。Chen表4与表5分别对应GFNI构表实现和独立SIMD执行实现。

## 加密与解密

SM4加密和解密采用相同的轮计算结构，相应白盒实例按照不同轮密钥顺序生成，表结构相同不表示表内容相同。本文配对测试未观察到稳定的方向性执行性能差异，因此论文主表以统一吞吐率指标报告执行性能。WBDL保留其动态算法语义及逆向执行方式。

## 引用

If you use this code, please cite the corresponding paper/preprint.

本文尚未提供正式出版标识，不编造DOI。基础算法、软件及GFNI S盒方法来源见各方案README和[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## License

新增代码与经授权的SCNUcrypto代码采用[Apache License 2.0](LICENSE)。第三方已有版权和许可证继续保留，见[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
