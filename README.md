# DK-SAR: 双知识增强自适应推理 HAZOP 分析系统

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![HAZOP](https://img.shields.io/badge/Domain-Chemical%20Safety-red.svg)]()
[![Vision](https://img.shields.io/badge/Mode-Text%20%2B%20Vision-0f766e.svg)]()

> 面向化工安全场景的多智能体分析系统，支持 HAZOP 文本分析与现场图片风险识别双入口。

## 功能特色

- HAZOP 文本分析：输入化工异常场景后，系统自动完成上下文解析、RAG 检索、报告生成、物理反思校验与结构化输出。
- 现场拍照识别：一线巡检人员可上传现场照片，AI 自动识别安全风险，并关联可能的 HAZOP 偏差。
- 双知识库增强：正样本库用于相似事故检索，负样本库用于物理谬误拦截与反思修正。
- 结构化结果页：输出 HAZOP 报告、知识溯源、反思日志、论文指标评估和原始 JSON。
- 一键回填：图片识别结果可直接转写为文本 HAZOP 输入草稿。

## 目标用户

- 安全工程师：进行正式 HAZOP 节点分析和风险梳理。
- 科研人员：验证 AI 在化工过程安全领域的推理表现。
- 一线巡检人员：通过拍照快速识别现场隐患并联动文本分析。

## 系统架构

```text
双入口界面
├─ 文本分析入口
│  ├─ Context Agent：提取设备、参数、偏差方向
│  ├─ RAG Agent：检索正样本事故案例并生成 HAZOP 报告
│  └─ Reflection Agent：负样本库匹配 + 物理一致性校验
└─ 图片分析入口
   ├─ 上传现场照片与补充说明
   ├─ DashScope qwen-vl-max 多模态识别
   └─ 转写为 HAZOP 文本分析输入
```

## 项目结构

```text
dk-sar/
├─ agents/
├─ data/
│  ├─ positive/cases.json
│  ├─ negative/fallacies.json
│  └─ schema/hazop_schema.json
├─ prompts/
├─ utils/
├─ app.py
├─ main.py
├─ eval.py
├─ feedback.py
├─ config.py
└─ requirements.txt
```

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/yilubanhan123c-pixel/dk-sar.git
cd dk-sar
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
cp .env.example .env
```

`.env` 至少需要配置：

```bash
DASHSCOPE_API_KEY=sk-你的key
# OPENAI_API_KEY=sk-你的key
```

说明：

- `DASHSCOPE_API_KEY` 现在同时用于文本推理能力和通义千问 VL 多模态图片识别。
- 首次运行时会自动构建向量索引，并可能下载 embedding 模型。

### 4. 启动系统

```bash
python app.py
```

浏览器会自动打开 `http://localhost:7860`。

## 页面说明

### Tab 1: HAZOP 文本分析

- 顶部 Banner 展示项目名称、英文副标题和技术徽章。
- 左侧输入异常场景，并提供 5 个预置示例。
- 中间显示实时进度：上下文解析、RAG 检索、物理反思。
- 右侧结果页包含：
  - HAZOP 报告
  - 知识溯源
  - 反思日志
  - 论文指标评估
  - 原始 JSON

### Tab 2: 现场拍照识别

- 上传化工现场照片，可附带补充说明。
- 使用 `qwen-vl-max` 识别现场隐患、风险等级、建议措施和 HAZOP 关联。
- 可一键将识别结果转写为文本分析输入。

## 数据集扩展

- 正样本库新增多类典型事故案例，包括聚合反应飞温、氯气泄漏、氢气爆炸、蒸馏塔液泛、LNG rollover、粉尘爆炸等。
- 负样本库新增物理谬误样本，包括压力与沸点、氮气窒息、快速泄压、水锤、爆炸极限、防爆误解等主题。

## 技术栈

- 大语言模型：通义千问 / DashScope
- 多模态视觉：`qwen-vl-max`
- 多智能体编排：LangGraph
- 向量数据库：ChromaDB
- Web UI：Gradio

## 效果截图

- 建议补充新的双入口界面截图：
  - HAZOP 文本分析页
  - 现场拍照识别页

## 引用

```bibtex
@article{chen2025dksar,
  title={DK-SAR: Dual Knowledge-enhanced Self-Adaptive Reasoning for Automated HAZOP Analysis},
  author={Chen, Shitou},
  year={2025}
}
```

## License

MIT License
