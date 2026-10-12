# Vesper

**带大模型规划环节的望远镜巡天调度智能体（Rust）。** Vesper 是 **starwing** 队在
[GOSIM 2026 Agentic Observer 黑客松](https://create.gosim.org/survey26/)（survey26，`participant-agent-protocol-v4`）提交的最终版本。

[English README](README.md)

*Vesper* 是昏星，黄昏时西方天空亮起的第一颗星。这个名字对应我们最关键的发现：A 型卡的胜负在季初黄昏的西天带（赤经 300–360°）。
那里的暗类星体只在季初几周可见，必须趁早拍，错过就再也拍不到了。

本仓库只包含最终提交版本的代码（平台版本 `222b7453`），编译后与提交的二进制逐字节相同（见[复现提交版本](#复现提交版本)）。

## 智能体做什么

每局是一台多光纤光谱望远镜的一个模拟观测季。引擎每一步发来决策请求，智能体回答以下四种动作之一：

- `observe`：指向、光纤分配、曝光时长、程序；
- `wait`；
- `report`：举报仪器故障；
- `finish`。

计分包括以下几项：

- 每个目标最好一次曝光的科学分，再乘以程序档倍数；
- 漏拍必做目标（required）的罚分；
- 截止前完成有奖观测请求的奖励；
- 故障举报：报对得分，免费额度用完后报错扣分；
- 均匀度项。

"真实"卡上，零奖励请求里还带一份值班日志。日志是多语言聊天，夹杂暗号、凯撒位移和摩斯编码、多个时区、更正和传言。
它会预告要举报的导星相机故障、平场测试、停机、关闭和地形。

Vesper 是规则搜索规划器加 6 个大模型环节。每个环节默认开启，输出经规则信任域校验并夹紧后生效，任何失败都退回规则。

## 成绩

**决赛最终排名：第 2 名（共 102 支有排名的队伍）。** 最终排名由隐藏任务卡 E–H 决定。主办方用各队的送测版本和各队自己的模型密钥评测，
结果公布在[决赛成绩页](https://create.gosim.org/survey26/platform/leaderboard/final-hidden)：

| 名次 | 队伍 | **总分（E–H 均分）** | E | F | G | H |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Pleiades | 33,575.79 | 37,542.8 | 31,005.2 | 31,412.0 | 34,343.3 |
| **2** | **starwing（Vesper）** | **33,051.03** | 37,193.9 | 30,722.2 | 30,307.0 | 33,981.0 |
| 3 | 迷星叫 | 32,598.75 | 37,312.4 | 30,206.4 | 29,441.2 | 33,435.0 |

正式赛（线上阶段）最终版本，无模型评测：

| 卡 | A | B | C | D | **A–D 均分** | A1 | B1 | C1 | D1 | **超级榜** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 分数 | 27,636.8 | 44,686.4 | 29,013.9 | 33,255.5 | **33,648.2** | 37,312.7 | 57,196.7 | 28,747.4 | 61,648.5 | **174,842.9** |

超级榜按 0.2 × A–D 合计 + 0.8 × A1–D1 合计计算，A1–D1 是带值班日志的真实卡。线上阶段截止时，我们在 A–D 榜和超级榜都排第 2。

线上阶段的进展（A–D 均分）：

| 版本 | A–D 均分 |
|---|---:|
| 官方 Python 示例（带 Kimi） | 29,298 |
| 官方 Rust 示例（带 Kimi） | 29,945 |
| 第一轮合并改进 | 31,847 |
| 密集搜索、季节 LP、完成或跳过、铺块 | 33,103 |
| 双线程宽搜索、值班日志解码器 | 33,456 – 33,561 |
| 线上截止时的版本 | 33,597 |
| **Vesper（最终）** | **33,648** |

## 工作原理

**规划器（`src/planner/`）。** 每个决策做一次搜索，选出指向、光纤、曝光时长和程序，目标是最大化 `gain - lambda * T`。
时间价格 lambda 由全季可达速率决定。搜索的支撑模块有：

- 季节 LP 分配视场时间（`src/seasonlp.rs`）；
- 天区紧迫度（`src/urgency.rs`）；
- 每个目标的程序档概率（`src/band.rs`）；
- 必做目标和观测请求的截止期价值（`src/planner/values.rs`、`src/requests.rs`）。对季初就关窗的必做目标，还会计入它剩余的好夜数和同一天区带里的竞争目标数。

光纤在格内选目标时，会计入"这次用掉、以后本可以拍满"的期权成本（`src/planner/search.rs` 的 `option_cost`，对应 `PRO_FOV`）。

**故障（`src/fault.rs`）。** 用变点检测器监视每次曝光的通量读数。

**值班日志（`src/planner/dutylog/`、`src/planner/duty.rs`）。** 规则解码器把日志转成结构化事实，然后：

- 预告的故障按时举报；
- 平场测试和停机等过去；
- 避开关闭的方向，包括最长一次曝光期间就会到来的关闭；
- 按山脊高度设置地形限制。

日志漏掉的故障，在读数持续 35 分钟低于正常 1% 时仍会举报。

**保护与配速。** 每个动作都经过输出护栏（`src/guard.rs`）。配速（`src/main.rs` 的 `pace`）保证不超出 900 秒 CPU 预算。

### 大模型环节

| 环节 | 位置 | 触发（默认开启） | 大模型输出决定什么 | 校验/夹紧 | 回退 |
|---|---|---|---|---|---|
| 自然语言理解 | `src/llm/request_triage.rs`、`week_plan.rs`、`duty_log.rs` | 新的有奖请求、新的 7 天预报、新的值班日志 | 请求优先级、预报周计划、解码器没把握的日志行的含义 | 优先级限制在 [1, 1.5]（规则判定可行时）或 [0.5, 1.5]；日志事实逐字段核对 | 优先级 1、解码器的事实 |
| 数据解析 | `duty_log.rs` → `DutyLog::resolve` → `duty.rs`；`fault_judge.rs` | 新日志、规则判定要举报、新预报 | 日志文本转为结构化事实，驱动举报、停机等待、关闭方向和地形；解读故障检测状态 | 模型给出的故障时刻只在免费额度内举报；关闭 ≤ 16 小时、≤ 3 个方向 | 规则决定 |
| 任务规划 | `week_plan.rs` → `main.rs : night_advice` | 新的 7 天预报 | 每夜的坏夜标记、要避开的方向、时间价格系数 `lambda_scale` | 坏夜只能落在规则判定的坏夜上；`lambda_scale` 在 [0.98, 1.02] | 规则计划 |
| 动作决策 | `fault_judge.rs` → `main.rs : maybe_report` | 规则判定要举报时 | 现在举报，还是确认一次曝光后再报 | 只能推迟一次曝光，不能取消也不能新增 | 规则举报 |
| 工具调用 | `src/llm/client.rs` | 每次提问 | 通过函数调用（`tools`）作答 | 按 schema 和信任域校验 | 文本 JSON，再不行退回规则 |
| 计划调整 | `week_plan.rs` + `main.rs : week_context` | 每次预报 | 根据上一段回顾重新规划下周 | 同任务规划 | 同任务规划 |

**额度与稳健性（`src/llm/client.rs`）。**

- **调用上限：** 每局最多 16 次 HTTP 请求（含重试）；带值班日志的卡另加 24 次，这些问题在后台提问，决策不等待。
- **重试与退避：** 最多重试 1 次；遇到 429 按 Retry-After 退避，非有限值忽略。
- **熔断：** 遇到 401/403、额度用尽或连续失败时熔断。
- **超时：** 所有等待都有上限，连接超时也有上限。

没有密钥或设置了 `OBSERVER_MODEL_DISABLED=1` 时，只用规则运行。

## 编译与运行

```bash
CC_x86_64_unknown_linux_musl=gcc cargo build --release --locked --target x86_64-unknown-linux-musl
# -> target/x86_64-unknown-linux-musl/release/rust-pro（静态二进制，与提交版本相同）
```

二进制通过 stdin/stdout 使用 v4 JSON-lines 协议，可以用官方仓库的本地运行器运行，也可以在平台上运行。

本仓库的 `observer.project.json` 是在 `rust:1-bookworm` 里从源码编译的写法。实际提交的 ZIP 用的是预编译的 `bin/rust-pro`，
对应的配置是 `image: debian:bookworm-slim`、`build: []`、`run: ["./bin/rust-pro"]`，环境变量与本仓库相同。

### 配置

| 变量 | 含义 |
|---|---|
| `OPENAI_API_KEY`（或 `KIMI_API_KEY`） | 模型密钥，不设则只用规则 |
| `OPENAI_BASE_URL` | 默认 `https://api.kimi.com/coding/v1`，任何 OpenAI 兼容端点都可以 |
| `OPENAI_MODEL` | 模型名 |
| `LLM_MAX_CALLS`、`LLM_TIMEOUT` | 调用上限（16）、每问超时（40 秒） |
| `PRO_PACE_SAFETY=2.0` | **最终设置**。配速系数：智能体估算 CPU 时把两个搜索线程相加，默认值 0.75 下 CPU 只用了不到 30% 就开始收窄搜索 |
| `PRO_FOV=0.3` | **最终设置**。光纤期权成本强度（默认 0.5） |
| `PRO_<NAME>` | 覆盖任意规划常数（见 `src/planner/mod.rs` 的 `Params::load`）；`PRO_FIXED_LEVEL=0` 固定搜索档位 |

## 复现提交版本

```bash
CC_x86_64_unknown_linux_musl=gcc cargo build --release --locked --target x86_64-unknown-linux-musl
sha256sum target/x86_64-unknown-linux-musl/release/rust-pro
# e4afbcfa3d9315003b0e948a6ff0ef87c7afe259dfeb7e1b33f490be9da51349
# = 提交 ZIP 中的 bin/rust-pro（rustc 1.97.0，目标 x86_64-unknown-linux-musl）
```

运行时设置 `PRO_PACE_SAFETY=2.0 PRO_FOV=0.3`，与提交时一致。

## 仓库范围

本仓库只包含提交版本的代码。开发工具都不在这里，包括：

- 单元测试、本地大模型桩；
- 逐字节一致的加速裁判、集群批测框架；
- 合成卡生成器、值班日志金标集；
- 实验笔记。

源码里仍有指向 `tests/unit/` 的 `#[cfg(test)]` 声明，这和提交版本完全一样，所以本仓库不能运行 `cargo test`。
注释里的 `notes/23` 之类指的是内部实验笔记。

## 许可

自写代码采用 [Apache License 2.0](LICENSE)。智能体以官方 Rust 示例为起点，派生自该示例的部分仍按
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) 授权（署名、非商业），详见 [NOTICE](NOTICE)。

任务卡、模拟数据和评测代码属于主办方，使用时请引用 GOSIM 2026 Agentic Observer Hackathon（https://create.gosim.org/survey26/）。
