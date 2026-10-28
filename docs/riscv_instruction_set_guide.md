# RISC-V 指令集与机器码：从 32 位编码到程序执行

本文面向第一次系统接触指令集的读者，回答四个核心问题：

1. RISC-V 指令集到底是什么；
2. 一条 32 位机器码为什么能表达一项操作；
3. `opcode`、`funct3`、`funct7` 和寄存器字段分别做什么；
4. 多条普通 RISC-V 指令和 Vortex 扩展指令如何组合成 Kernel 和 AI 模型。

---

## 1. 先抓住本质：指令集是软件和硬件之间的合同

指令集架构（Instruction Set Architecture，ISA）定义软件可以向处理器发出哪些命令，以及每条命令的二进制编码和可见行为。

它规定：

- 有哪些寄存器；
- 哪些二进制编码代表加法、访存、跳转等操作；
- 指令从哪里取得输入、把结果写到哪里；
- 内存访问、异常、中断和特权操作遵守什么规则。

它不规定：

- 处理器有几级流水线；
- 加法器内部如何实现；
- 有多少个执行单元；
- Cache 有多大；
- 一条指令具体需要几个时钟周期。

因此同一套 RISC-V 指令集可以由简单的顺序执行 CPU、乱序 CPU、Vortex GPU 或模拟器分别实现。

可以把三层关系写成：

```text
高级程序：C/C++、MLIR、LLVM IR
            ↓ 编译
指令集：ADD、LOAD、BRANCH、VX_SPLIT……
            ↓ 硬件实现
微架构：译码器、流水线、ALU、寄存器、Cache、Warp 控制单元
```

---

## 2. RISC-V 为什么采用“基础指令集 + 扩展”

RISC-V 不是要求所有芯片实现同一大包指令，而是由一个基础整数指令集和若干可选扩展组成。

例如：

```text
RV64IMAC
```

可以拆成：

| 名称 | 含义 |
|---|---|
| `RV64` | 整数寄存器宽度 XLEN 为 64 位 |
| `I` | 基础整数指令 |
| `M` | 整数乘法和除法 |
| `A` | 原子操作 |
| `C` | 16 位压缩指令 |

`RV64` 表示整数寄存器宽度（XLEN）为 64 位，不表示每条指令都是 64 位。基础指令通常采用 32 位编码，启用 `C` 扩展后还可以出现 16 位压缩指令。

常见扩展包括：

| 扩展 | 主要能力 | 指令示例 |
|---|---|---|
| `I` | 整数、访存、分支、跳转 | `add`、`lw`、`beq`、`jal` |
| `M` | 整数乘除 | `mul`、`div`、`rem` |
| `A` | 原子操作 | `lr`、`sc`、`amoadd` |
| `F` | 单精度浮点 | `fadd.s`、`fmul.s` |
| `D` | 双精度浮点 | `fadd.d`、`fmul.d` |
| `C` | 16 位压缩编码 | `c.addi`、`c.lw` |
| `V` | 向量计算 | `vadd.vv`、`vle64.v` |
| `B`/`Zb*` | 位操作 | 旋转、计数、位提取 |
| `Zicsr` | CSR 访问 | `csrrw`、`csrrs`、`csrrc` |
| `Zifencei` | 指令流同步 | `fence.i` |

`G` 是常用通用组合的简称，通常展开为 `I+M+A+F+D+Zicsr+Zifencei`；`RV64GC` 在此基础上再加入压缩指令 `C`。

多字母扩展使用下划线分隔：

```text
rv64imafdcv_zba_zbb_xvortex
```

- `Z...` 通常表示标准的细分扩展；
- `X...` 表示非标准或厂商自定义扩展；
- Vortex 3 可以视为在 RISC-V 上增加 `Xvortex` 类自定义能力。

扩展的本质不是“加载一个软件库”，而是告诉编译器和处理器：这一批机器指令可以合法使用。

---

## 3. 程序可以看到哪些寄存器

基础 RISC-V 整数架构通常有 32 个整数寄存器：

```text
x0, x1, x2, ... , x31
```

其中 `x0` 永远等于 0，写入它的结果会被丢弃。

ABI 又给部分寄存器规定了便于编程的名字：

| 物理编号 | ABI 名称 | 常见用途 |
|---|---|---|
| `x0` | `zero` | 常数 0 |
| `x1` | `ra` | 返回地址 |
| `x2` | `sp` | 栈指针 |
| `x5～x7` | `t0～t2` | 临时值 |
| `x8～x9` | `s0～s1` | 被调用者保存值 |
| `x10～x17` | `a0～a7` | 函数参数和返回值 |

这里需要区分 ISA 和 ABI：

- ISA 规定 `x0` 的硬件行为，以及指令如何引用 `x0～x31`；
- ABI 约定 `x1` 用作返回地址、`x2` 用作栈指针、`x10` 用作第一个参数；
- 除了 `x0` 等特殊规则，硬件通常只看到寄存器编号，不理解变量名。

可选扩展还会引入：

```text
f0～f31       浮点寄存器
v0～v31       向量寄存器
CSR           控制和状态寄存器
PC            程序计数器
```

Vortex 还定义线程 ID、Warp ID、Core ID、线程掩码等专用 CSR。

---

## 4. 一条 32 位机器码不是“小指令的拼接”

本节集中讨论最常见的 32 位指令。未启用 `C` 扩展时，基础指令按 32 位边界取指；启用 `C` 后，低两位不等于 `11` 的编码可以表示 16 位压缩指令，而普通 32 位指令的低两位为 `11`。处理器先判断指令长度，再按照对应格式译码。

一条 32 位机器码是一个整体，只是这个整体被划分成多个字段。字段分别告诉硬件：

```text
做哪一类操作
具体做哪一种操作
读取哪些寄存器
结果写入哪个寄存器
是否携带立即数或地址偏移
```

以最常见的 R 类型为例：

```text
 31      25 24    20 19    15 14  12 11     7 6       0
+----------+--------+--------+------+---------+---------+
|  funct7  |  rs2   |  rs1   |funct3|   rd    | opcode  |
+----------+--------+--------+------+---------+---------+
    7位       5位      5位      3位     5位       7位
```

总长度为：

```text
7 + 5 + 5 + 3 + 5 + 7 = 32 位
```

字段含义如下：

| 字段 | 作用 |
|---|---|
| `opcode` | 第一层大类，例如整数、Load、Store、分支、浮点、向量、自定义 |
| `funct3` | 在大类中继续区分操作 |
| `funct7` | 再进一步区分具体指令或格式 |
| `rs1` | 第一个源寄存器编号 |
| `rs2` | 第二个源寄存器编号 |
| `rd` | 目标寄存器编号 |

所以“组合”有两个不同层次：

```text
多个字段组合成一条 32 位指令
多条 32 位指令组合成程序
```

---

## 5. opcode 只确定第一层类别

常见 7 位 opcode 如下：

| 类别 | 二进制 opcode | 十六进制 |
|---|---:|---:|
| Load | `0000011` | `0x03` |
| 整数立即数运算 OP-IMM | `0010011` | `0x13` |
| AUIPC | `0010111` | `0x17` |
| Store | `0100011` | `0x23` |
| 整数寄存器运算 OP | `0110011` | `0x33` |
| LUI | `0110111` | `0x37` |
| Branch | `1100011` | `0x63` |
| JALR | `1100111` | `0x67` |
| JAL | `1101111` | `0x6F` |
| System/CSR | `1110011` | `0x73` |
| 浮点运算 OP-FP | `1010011` | `0x53` |
| 向量运算 OP-V | `1010111` | `0x57` |
| Custom-0 | `0001011` | `0x0B` |

`opcode` 类似“部门编号”，通常还需要 `funct3`、`funct7` 等字段才能确定具体指令。

例如 `ADD` 和 `SUB` 共享：

```text
opcode = 0110011
funct3 = 000
```

只通过 `funct7` 区分：

```text
ADD：funct7 = 0000000
SUB：funct7 = 0100000
```

硬件译码逻辑可以抽象为：

```cpp
if (opcode == 0x33 && funct3 == 0) {
    if (funct7 == 0x00)
        execute_add();
    else if (funct7 == 0x20)
        execute_sub();
}
```

---

## 6. 六种主要的 32 位编码格式

### 6.1 R 类型：寄存器与寄存器运算

```text
funct7 | rs2 | rs1 | funct3 | rd | opcode
```

典型指令：

```asm
add x5, x6, x7
sub x5, x6, x7
mul x5, x6, x7
```

### 6.2 I 类型：立即数、Load、JALR、部分 System 指令

```text
imm[11:0] | rs1 | funct3 | rd | opcode
```

典型指令：

```asm
addi x5, x6, 10
ld   x5, 16(x6)
jalr x0, 0(x1)
```

### 6.3 S 类型：Store

```text
imm[11:5] | rs2 | rs1 | funct3 | imm[4:0] | opcode
```

典型指令：

```asm
sd x5, 16(x6)
```

Store 没有 `rd`，因为它把寄存器数据写入内存。原来 `rd` 所在的位被用于保存立即数的一部分。

### 6.4 B 类型：条件分支

```text
imm[12] | imm[10:5] | rs2 | rs1 | funct3 | imm[4:1] | imm[11] | opcode
```

典型指令：

```asm
beq x5, x6, label
bne x5, x6, label
blt x5, x6, label
```

分支目标至少按 2 字节对齐，因此最低地址位不需要存入指令。立即数位经过排列，可以保持 `rs1`、`rs2`、`funct3` 和 `opcode` 的位置与其他格式一致，简化硬件。

### 6.5 U 类型：高位立即数

```text
imm[31:12] | rd | opcode
```

典型指令：

```asm
lui   x5, 0x12345
auipc x5, 0x10
```

### 6.6 J 类型：长距离 PC 相对跳转

```text
imm[20] | imm[10:1] | imm[11] | imm[19:12] | rd | opcode
```

典型指令：

```asm
jal x1, function
```

RISC-V 将不同格式中的寄存器字段尽量固定在相同位置，并尽量把有符号立即数的符号位放在第 31 位。这是其译码简单、硬件规则统一的重要原因。

---

## 7. 完整示例一：ADD 如何编码成 32 位机器码

汇编指令：

```asm
add x5, x6, x7
```

语义：

```text
x5 = x6 + x7
```

字段取值：

```text
funct7 = 0000000       ADD
rs2    = 00111         x7
rs1    = 00110         x6
funct3 = 000           加减运算组
rd     = 00101         x5
opcode = 0110011       整数寄存器运算
```

按照 R 类型字段位置组合：

```text
0000000 00111 00110 000 00101 0110011
```

对应的 32 位十六进制数：

```text
0x007302B3
```

RISC-V 常用小端存储，因此在目标文件或内存中的四个字节通常是：

```text
B3 02 73 00
```

处理器执行时：

```text
1. 低 7 位 opcode=0x33，进入整数寄存器运算译码；
2. funct3=0、funct7=0，确定具体操作为 ADD；
3. 从寄存器 x6、x7 读取数据；
4. ALU 完成加法；
5. 把结果写入 x5。
```

同样的寄存器换成 `SUB`：

```asm
sub x5, x6, x7
```

只把 `funct7` 改为 `0100000`：

```text
0x407302B3
```

由此可以看到，机器码不是一个不可解释的随机数字，而是一张压缩成 32 位的操作表单。

---

## 8. 完整示例二：ADDI 如何携带常数

汇编指令：

```asm
addi x5, x6, 10
```

语义：

```text
x5 = x6 + 10
```

它使用 I 类型：

```text
imm    = 000000001010   十进制 10
rs1    = 00110          x6
funct3 = 000            ADDI
rd     = 00101          x5
opcode = 0010011        OP-IMM
```

最终机器码：

```text
0x00A30293
```

小端字节序：

```text
93 02 A3 00
```

这里没有 `rs2`，因为第二个输入直接存放在 `imm` 字段中。

`ADDI` 的 12 位立即数按照有符号数解释，并在运算前符号扩展到 XLEN。因而相同字段既能表示正数，也能表示负数。

---

## 9. 完整示例三：多条指令组成一个函数

C 代码：

```c
long calculate(long a, long b) {
    long c = a + b;
    if (c == 0)
        return 1;
    return c * 2;
}
```

使用基础 RV64I 可以形成：

```asm
calculate:
    add   a0, a0, a1       # c = a + b
    bne   a0, zero, .L1    # c != 0 时跳转
    addi  a0, zero, 1      # 返回值 = 1
    jalr  zero, 0(ra)      # 返回调用者

.L1:
    slli  a0, a0, 1        # 左移1位，相当于乘2
    jalr  zero, 0(ra)      # 返回调用者
```

这里没有一条“执行 calculate 函数”的超级指令。函数由算术、比较、跳转和返回指令组成。

处理器通常在完成一条 32 位顺序指令后令 `PC=PC+4`；`bne`、`jal`、`jalr` 等控制流指令会根据条件或目标地址改写 PC。硬件内部可以采用流水线并行执行，但对软件呈现的结果必须符合这套 ISA 顺序语义。

层次关系是：

```text
字段组合成一条机器指令
多条机器指令组合成基本块
多个基本块组成函数
多个函数和 Kernel 组成程序或 AI 模型
```

---

## 10. 扩展会怎样改变编译结果

### 10.1 没有 M 扩展

C 代码：

```c
long multiply(long a, long b) {
    return a * b;
}
```

如果目标只有 `RV64I`，编译器不能使用 `mul`，可能需要调用软件辅助函数：

```asm
call __muldi3
```

### 10.2 启用 M 扩展

目标是 `RV64IM` 时，可以直接生成：

```asm
mul a0, a0, a1
ret
```

这就是 `RISCVSubtarget` 中硬件能力信息影响代码生成的实际方式：编译器查询当前目标是否支持 `M`，再决定使用硬件 `mul` 还是软件实现。

### 10.3 启用 V 扩展

数组加法在标量模式下一次处理一个元素：

```asm
ld    t0, 0(a0)
ld    t1, 0(a1)
add   t2, t0, t1
sd    t2, 0(a2)
```

启用标准 RISC-V `V` 扩展后，可以一次处理一组元素：

```asm
vsetvli  t0, a3, e64, m1, ta, ma
vle64.v  v1, (a0)
vle64.v  v2, (a1)
vadd.vv  v3, v1, v2
vse64.v  v3, (a2)
```

标准 `V` 扩展解决向量寄存器上的 SIMD 计算；Vortex 扩展主要解决线程、Warp、发散和同步，两者不是同一个概念。

---

## 11. Vortex 如何使用 RISC-V 的 Custom Opcode

RISC-V 预留了 Custom Opcode 空间，允许硬件厂商增加自定义指令。

Vortex 使用：

```text
CUSTOM0 opcode = 0001011 = 0x0B
CUSTOM1 opcode = 0101011 = 0x2B
```

`0x0B` 本身只说明“这是一条 Custom-0 指令”，还不能确定具体是 `VX_TMC` 还是 `VX_SPLIT`。Vortex 继续使用 `funct3`、`funct7`，有时还使用固定的 `rs2` 或 `rd` 位区分具体操作。

| opcode | funct7 | funct3 | 额外固定字段 | Vortex 指令 |
|---:|---:|---:|---|---|
| `0x0B` | 0 | 0 | `rd=0, rs2=0` | `VX_TMC` |
| `0x0B` | 0 | 1 | `rd=0` | `VX_WSPAWN` |
| `0x0B` | 0 | 2 | `rs2=0` | `VX_SPLIT` |
| `0x0B` | 0 | 2 | `rs2=1` | `VX_SPLIT_N` |
| `0x0B` | 0 | 3 | `rd=0, rs2=0` | `VX_JOIN` |
| `0x0B` | 0 | 4 | `rd=0` | `VX_BAR` |
| `0x0B` | 0 | 5 | `rd=0` | `VX_PRED` |
| `0x0B` | 0 | 5 | `rd=1` | `VX_PRED_N` |

普通 RISC-V 处理器只知道 `0x0B` 属于预留的自定义空间，并不知道 Vortex 对它的解释。没有 Vortex 解码器的处理器通常会产生非法指令异常。

---

## 12. 完整示例四：VX_SPLIT 的机器码

汇编指令：

```asm
vx_split x6, x5
```

语义：

```text
使用 x5 中的条件划分活动线程，
把之后重新汇合需要的状态令牌写入 x6。
```

字段为：

```text
funct7 = 0000000
rs2    = 00000          正向条件
rs1    = 00101          x5，分支条件
funct3 = 010            SPLIT
rd     = 00110          x6，重汇合令牌
opcode = 0001011        CUSTOM0
```

组合后的二进制：

```text
0000000 00000 00101 010 00110 0001011
```

十六进制机器码：

```text
0x0002A30B
```

小端字节序：

```text
0B A3 02 00
```

反向版本：

```asm
vx_split_n x6, x5
```

只把固定的 `rs2` 字段从 0 改为 1：

```text
0x0012A30B
```

Vortex 硬件译码可以抽象为：

```cpp
if (opcode == 0x0B && funct7 == 0) {
    switch (funct3) {
    case 0: execute_vx_tmc();    break;
    case 1: execute_vx_wspawn(); break;
    case 2: execute_vx_split();  break;
    case 3: execute_vx_join();   break;
    case 4: execute_vx_bar();    break;
    case 5: execute_vx_pred();   break;
    }
}
```

其中 `rs2[0]` 或 `rd[0]` 可以继续表示正向/反向形式。

---

## 13. 普通 RISC-V 指令与 Vortex 指令如何组合

源程序：

```c
int tid = vx_thread_id();

if (tid & 1)
    path_a();
else
    path_b();
```

可能形成：

```asm
csrr      t0, tid          # 读取 Vortex 线程 ID CSR
andi      t1, t0, 1        # 普通 RISC-V：计算条件
vx_split  t2, t1           # Vortex：拆分活动线程
bne       t1, zero, .LA    # 普通 RISC-V：条件跳转

.LB:
    ...                    # B 路径
    j .LJOIN

.LA:
    ...                    # A 路径

.LJOIN:
    vx_join t2             # Vortex：恢复汇合线程
```

这里展示了 Vortex 的基本设计方式：

```text
普通计算、访存、函数调用、跳转
    使用标准 RISC-V 指令

线程分流、线程汇合、Barrier、线程掩码
    使用 Vortex 自定义指令
```

Vortex 没有替换 RISC-V，而是在其基础上增加 GPU/SIMT 控制能力。

---

## 14. 伪指令为什么没有独立机器码

汇编器为了便于程序员阅读，提供了伪指令。例如：

```asm
li  a0, 1
mv  a0, a1
j   label
ret
```

它们通常会展开成真实指令：

```asm
addi a0, zero, 1      # li a0, 1
addi a0, a1, 0        # mv a0, a1
jal  zero, label      # j label
jalr zero, 0(ra)      # ret
```

Vortex 的 `VX_MOV` 也是编译器内部伪指令。它用于暂时保护发散分支条件，随后在 Machine IR Pass 中被消除，因此最终 ELF 中没有独立的 `VX_MOV` 机器码。

---

## 15. 从源代码到最终机器码

完整编译过程可以表示为：

```text
C/C++、ONNX 或模型前端
          ↓
高层计算图 / MLIR
          ↓
LLVM IR
          ↓ 指令选择
Machine IR：ADD、LD、VX_SPLIT 等抽象机器指令
          ↓ 寄存器分配、分支修正、伪指令展开
MCInst
          ↓ MCCodeEmitter
32位机器码字
          ↓ 汇编、链接
ELF 可执行文件或 Kernel 二进制
          ↓ 取指与译码
RISC-V/Vortex 硬件执行
```

在 VOLT 中：

- `RISCVInstrInfoVX.td` 定义 Vortex 指令字段和编码；
- LLVM TableGen 根据 `.td` 文件生成编码器、汇编器和反汇编器表；
- `RISCVGenMCCodeEmitter.inc` 把寄存器编号填入字段并输出最终 32 位数值；
- Vortex RTL 的 `VX_decode.sv` 使用相同字段规则完成硬件译码。

编译器和硬件必须对每个字段的含义完全一致，否则相同的 32 位数会在两端得到不同解释。

---

## 16. 指令集、ABI 和微架构不要混淆

| 层次 | 解决的问题 | 示例 |
|---|---|---|
| ISA | 二进制指令是什么意思 | `0x007302B3` 表示 `add x5,x6,x7` |
| ABI | 函数如何传参、返回、保存寄存器 | `a0` 传第一个参数，`ra` 保存返回地址 |
| 微架构 | 硬件怎样执行指令 | 五级流水线、乱序执行、Warp 调度 |
| Runtime | 怎样装载程序并启动 Kernel | 内存分配、参数传递、设备启动 |

例如 `hasVendorXVortex=true` 表示目标硬件具备 Vortex ISA 能力；编译器据此选择 Vortex Pass 和指令。它不会启动硬件，真正的装载与 Kernel 调度由 Runtime 完成。

---

## 17. 最容易混淆的几个问题

### 17.1 opcode 是否等于具体指令

通常不等于。`opcode` 只确定第一层大类，完整指令一般由：

```text
opcode + funct3 + funct7 + 其他固定字段
```

共同确定。

### 17.2 32 位机器码是否由多条简单指令拼成

不是。它由多个字段组成，但仍然只是一条指令。复杂功能由多条完整指令顺序执行实现。

### 17.3 十六进制机器码和内存字节为什么顺序不同

机器码字：

```text
0x007302B3
```

小端内存字节：

```text
B3 02 73 00
```

数值没有改变，只是多字节数在内存中的排列顺序不同。

### 17.4 标准 V 扩展和 Vortex 是否相同

不同：

- 标准 `V` 扩展使用向量寄存器完成 SIMD 计算；
- Vortex 扩展提供线程、Warp、分支发散和同步控制；
- 两者可以同时存在，也可以分别存在。

### 17.5 一个 AI 模型是否对应一条机器指令

不是。一个 AI 模型通常被拆成多个算子和 Kernel；每个 Kernel 又由大量普通 RISC-V 指令、可选向量/浮点指令以及 Vortex 控制指令组成。

---

## 18. 一张图记住整个关系

```text
RV64IMAFDCV_Xvortex
│
├── RV64：64位整数寄存器
├── I：整数、Load/Store、Branch、Jump
├── M：乘除法
├── A：原子操作
├── F/D：浮点
├── C：压缩编码
├── V：向量计算
└── Xvortex：线程、Warp、分流、汇合、同步

每条指令：
opcode + funct字段 + 寄存器字段 + 立即数字段
                ↓
             32位机器码
                ↓
多条机器码组成基本块、函数、Kernel 和完整程序
```

理解 RISC-V 时，可以一直使用下面这个判断顺序：

```text
先看目标启用了哪些 ISA 扩展
再看 opcode 属于哪个指令大类
再看 funct 字段确定具体操作
再看 rs/rd/imm 确定操作数
最后把多条指令连起来理解程序行为
```
