# 🏭 DK-SAR: 双知识增强自适应推理 HAZOP 分析系统

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![HAZOP](https://img.shields.io/badge/Domain-Chemical%20Safety-red.svg)]()
[![LangGraph](https://img.shields.io/badge/Framework-LangGraph-orange.svg)]()

> **基于多智能体 LLM 的自动化 HAZOP 分析系统**  
> 用户输入化工异常场景，系统自动完成「案例检索 → 报告生成 → 物理校验 → 修正输出」全流程

---

## ✨ 功能演示

输入：`汽油储罐液位计显示正常，但进料流量持续高于出料流量约25%，已持续20分钟`

系统自动输出：
- 📌 **分析摘要**：核心偏差 / 首要怀疑 / 最高风险 / 首要处置动作
- 🔗 **推理证据链**：输入事实 → 规则触发 → 案例支持 → 物理校验 → 结论收敛
- 🔍 **分层原因分析**：首要怀疑 / 次要怀疑 / 待验证（每条注明来源）
- ⚠️ **后果链**：直接后果 → 升级后果 → 极端后果
- 🛡️ **现有保护措施**及有效性评估
- 💡 **三层建议**：立即处置 / 短期整改 / 长期改进
- 🧠 **双知识库反思过程**：负样本库命中详情 + 每轮修正记录

---

## 🏗️ 系统架构

```
用户输入
   ↓
[智能体1] 上下文解析 → 结构化工艺上下文
   ↓
[智能体2] RAG增强分析
   ├── 正样本库检索（Top-3 相似CSB事故案例）
   └── 生成结构化 HAZOP 报告初稿
   ↓
[智能体3] 物理反思校验（双层架构）
   ├── Layer1: 向量+模糊匹配（负样本库，55条物理谬误）
   └── Layer2: LLM 五维物理检查（守恒定律/时序因果/数值合理性）
   ↓
通过? ──否──→ 修正指导 → 回到[智能体2]（最多3轮）
   ↓ 是
输出最终 HAZOP 报告
```

对应论文：**DK-SAR: Dual Knowledge-enhanced Self-Adaptive Reasoning for Automated HAZOP Analysis**

---

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/yilubanhan123c-pixel/dk-sar.git
cd dk-sar
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 API Key

```bash
# 复制配置模板
cp .env.example .env

# 编辑 .env，填入你的 API Key
# 推荐：通义千问（免费）https://bailian.aliyun.com
# DASHSCOPE_API_KEY=sk-你的key
```

### 4. 启动系统

```bash
python app.py
```

浏览器会自动打开 `http://localhost:7860`，首次启动会下载 Embedding 模型（约300MB）。

---

## 📁 项目结构

```
dk-sar/
├── agents/
│   ├── context_agent.py      # 智能体1：上下文解析
│   ├── rag_agent.py          # 智能体2：RAG增强分析（含证据链生成）
│   └── reflection_agent.py   # 智能体3：双层物理反思校验
├── utils/
│   ├── llm.py                # LLM API 封装
│   ├── vector_store.py       # ChromaDB 向量库封装
│   └── logger.py             # 结构化日志
├── prompts/
│   └── prompt_manager.py     # Prompt 版本管理
├── data/
│   ├── positive/cases.json   # 正样本库（36个CSB事故案例）
│   ├── negative/fallacies.json # 负样本库（55条物理谬误）
│   └── schema/hazop_schema.json # HAZOP 报告 Schema
├── main.py                   # LangGraph 多智能体编排
├── app.py                    # Gradio 网页界面
├── eval.py                   # 论文四指标评估（PCC/CCC/RDI/LCC）
├── feedback.py               # 用户反馈统计
├── config.py                 # 配置文件
└── requirements.txt
```

---

## 🔧 技术栈

| 组件 | 技术选型 |
|------|---------|
| 大语言模型 | 通义千问 qwen-plus |
| 多智能体编排 | LangGraph 状态机 |
| 向量数据库 | ChromaDB（本地持久化） |
| Embedding | all-MiniLM-L6-v2 |
| 网页界面 | Gradio 4.x |
| 日志 | loguru |

---

## 📊 知识库说明

- **正样本库**：36 个 CSB 真实事故案例，涵盖反应、换热、储运、蒸馏等工艺类型，用于 RAG 检索相似情景
- **负样本库**：55 条物理谬误记录，每条包含错误陈述、物理检查点和正确理解，用于快速筛查错误描述
- **模式仓库**：基于 IEC 61882 标准的 HAZOP 报告 JSON Schema

---

## 📊 评估指标

系统实现论文定义的四项自动评估指标：

| 指标 | 全称 | 权重 | 说明 |
|------|------|------|------|
| PCC | 物理概念覆盖率 | 35% | 涵盖的物理机理类别数 |
| CCC | 因果链完整性 | 30% | 原因→后果链的完整程度 |
| RDI | 建议详细度 | 20% | 措施的可操作性 |
| LCC | 案例关联度 | 15% | 历史案例的相关性 |

---

## 📝 引用

如果本项目对你的研究有帮助，请引用：

```bibtex
@article{chen2025dksar,
  title={DK-SAR: Dual Knowledge-enhanced Self-Adaptive Reasoning for Automated HAZOP Analysis},
  author={Chen, Shitou},
  year={2025}
}
```

---

## 📄 License

MIT License - 详见 [LICENSE](LICENSE) 文件
