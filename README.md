# 🏭 DK-SAR: 双知识增强自适应推理 HAZOP 分析系统

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![HAZOP](https://img.shields.io/badge/Domain-Chemical%20Safety-red.svg)]()

> **基于多智能体 LLM 的自动化 HAZOP 分析系统**  
> 用户输入化工异常场景，系统自动完成「案例检索 → 报告生成 → 物理校验 → 修正输出」全流程

---

## ✨ 功能演示

输入：`反应釜温度从80°C升至120°C，搅拌器仍在运行`

系统自动输出：
- ✅ 3-5 个可能原因（含类型分类）
- ✅ 完整后果链分析  
- ✅ 现有保护措施评估
- ✅ 优先级排序的建议措施
- ✅ 参考了哪些历史事故案例
- ✅ 经过了几轮物理一致性校验

---

## 🏗️ 系统架构

```
用户输入
   ↓
[智能体1] 上下文解析 → 结构化工艺上下文
   ↓
[智能体2] RAG增强分析
   ├── 正样本库检索（Top-3 相似案例）
   └── 生成 HAZOP 初稿
   ↓
[智能体3] 物理反思校验
   ├── Layer1: 向量+模糊匹配（负样本库）
   └── Layer2: LLM 五维物理检查
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
git clone https://github.com/your-username/dk-sar.git
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
│   ├── rag_agent.py          # 智能体2：RAG增强分析
│   └── reflection_agent.py   # 智能体3：物理反思校验
├── utils/
│   ├── llm.py                # LLM API 封装
│   └── vector_store.py       # ChromaDB 向量库封装
├── data/
│   ├── positive/cases.json   # 正样本库（CSB事故案例）
│   ├── negative/fallacies.json # 负样本库（物理谬误）
│   └── schema/hazop_schema.json # HAZOP 报告 Schema
├── main.py                   # 多智能体编排逻辑
├── app.py                    # Gradio 网页界面
├── config.py                 # 配置文件
└── requirements.txt
```

---

## 🔧 技术栈

| 组件 | 技术选型 |
|------|---------|
| 大语言模型 | 通义千问 qwen-plus / GPT-4o-mini |
| 向量数据库 | ChromaDB（本地） |
| Embedding | all-MiniLM-L6-v2 |
| 网页界面 | Gradio 4.x |
| 知识库格式 | JSON |

---

## 📊 知识库说明

- **正样本库**：8 个 CSB 真实事故案例，涵盖反应、换热、储运、蒸馏等工艺类型
- **负样本库**：12 条物理谬误记录，每条包含错误陈述、物理检查点和正确理解
- **模式仓库**：基于 IEC 61882 标准的 HAZOP 报告 JSON Schema

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
