import base64
import html
import json
import mimetypes
import os
import re
from pathlib import Path

import gradio as gr

import config
from agents.context_agent import ContextAgent
from agents.rag_agent import RAGAgent
from agents.reflection_agent import ReflectionAgent
from eval import evaluate_report, format_scores_markdown
from feedback import format_stats_markdown, get_stats, save_feedback
from main import initialize_system

os.environ["no_proxy"] = "localhost,127.0.0.1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

EXAMPLES = [
    ["反应釜温度从80°C升至120°C，搅拌器仍在运行，冷却水流量正常。"],
    ["换热器出口温度持续升高，超过设计温度15°C，操作人员停止了搅拌。"],
    ["储罐液位计显示正常，但进料流量持续大于出料流量，差值约20%。"],
    ["管道发生少量可燃气体泄漏，现场人员认为气体会自然扩散，未采取措施。"],
    ["反应器压力上升，泄压阀自动开启，操作人员判断系统已经安全继续操作。"],
]

TEXT_TAB_EXAMPLES = [
    "反应釜热失控",
    "液氨中毒窒息",
    "储罐泄漏扩散",
    "换热器结垢堵塞",
    "安全阀超压排放",
]

IMAGE_SYSTEM_PROMPT = """你是一名资深化工安全工程师和 HAZOP 分析专家。
请基于用户上传的现场照片，输出结构化中文分析，要求专业但易懂，适合一线巡检人员阅读。

请按以下结构输出：
1. 风险等级：仅可使用 高风险 / 中风险 / 低风险
2. 风险识别：逐条列出照片中观察到的异常、隐患或不安全行为
3. 判断依据：说明你为什么这样判断，可引用设备状态、环境特征、PPE、泄漏迹象、腐蚀、标识缺失等
4. 建议措施：给出现场可执行的处置建议，优先给出立即措施
5. HAZOP 关联：如果能关联到 HAZOP 偏差，请指出可能的参数与偏差方向，例如 温度偏高、压力偏高、流量偏低、液位偏高、泄漏、成分异常

在完整风险评估报告末尾，必须额外输出以下固定标签块，且不得省略：
[SCENE_SUMMARY]
用一句话概括图片中最核心的异常工况。必须是你根据图片内容生成的真实总结，不超过 50 个汉字，不要写成模板说明。
[/SCENE_SUMMARY]

如果照片信息不足，请明确说明“不足以确认”的部分，不要编造细节。"""

_current_result = {
    "report": None,
    "input": "",
    "scores": None,
    "reflection_history": [],
    "image_seed": "",
}
EXPORT_PATH = "last_report.json"

THEME = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="emerald",
    neutral_hue="slate",
)

CUSTOM_CSS = """
html, body, .gradio-container {
  background: #FFFFFF !important;
  color: #1F2937 !important;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif !important;
}
.gradio-container * {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif !important;
}
.gr-block, .gr-panel, .gr-box, .gr-form, .gr-group, .gr-accordion, .gradio-container .block {
  border: 1px solid #E5E7EB !important;
  box-shadow: none !important;
  background: #FFFFFF !important;
}
.gradio-container {
  max-width: 1440px !important;
  margin: 0 auto !important;
}
.gradio-container .main,
.gradio-container .contain,
.gradio-container .wrap {
  gap: 10px !important;
}
.gradio-container .gr-row,
.gradio-container .gr-column {
  gap: 10px !important;
}
.gradio-container .tabs,
.gradio-container .tabitem,
.gradio-container .tab-nav,
.gradio-container .tab-nav button {
  box-shadow: none !important;
}
.gradio-container .tab-nav {
  gap: 6px !important;
  padding-bottom: 4px !important;
}
.gradio-container .tab-nav button {
  border: 1px solid #E5E7EB !important;
  background: #FFFFFF !important;
  border-radius: 6px !important;
  padding: 8px 12px !important;
  font-size: 13px !important;
  min-height: 36px !important;
}
.gradio-container .tab-nav button.selected {
  border-color: #0F766E !important;
  color: #0F766E !important;
  background: #F8FAFC !important;
}
.gradio-container .form,
.gradio-container .wrap,
.gradio-container .panel {
  box-shadow: none !important;
}
.gradio-container .block-label,
.gradio-container label {
  font-size: 12px !important;
  font-weight: 600 !important;
  color: #4B5563 !important;
  letter-spacing: 0.01em;
}
.gradio-container textarea,
.gradio-container input[type="text"] {
  min-height: 42px !important;
  border: 1px solid #D1D5DB !important;
  border-radius: 6px !important;
  box-shadow: none !important;
  font-size: 14px !important;
}
.gradio-container textarea {
  line-height: 1.5 !important;
}
.gradio-container .gr-textbox,
.gradio-container .gr-code,
.gradio-container .gr-markdown,
.gradio-container .gr-file,
.gradio-container .gr-image,
.gradio-container .gr-slider {
  margin: 0 !important;
}
.gradio-container .gr-button,
.gradio-container button {
  min-height: 38px !important;
  border-radius: 6px !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  padding: 8px 14px !important;
}
.gradio-container button.primary {
  background: #0F766E !important;
  border: 1px solid #0F766E !important;
}
.gradio-container button.secondary {
  background: #FFFFFF !important;
  border: 1px solid #D1D5DB !important;
  color: #1F2937 !important;
}
.gradio-container .gr-accordion {
  border-radius: 6px !important;
}
.gradio-container .gr-accordion .label-wrap {
  min-height: 38px !important;
}
.gradio-container .gr-markdown.prose {
  font-size: 14px !important;
}
.gradio-container .gr-code textarea,
.gradio-container .gr-code pre {
  font-size: 12px !important;
}
.gradio-container .prose,
.gradio-container label,
.gradio-container p,
.gradio-container span,
.gradio-container div {
  color: #1F2937;
}
.gradio-container .secondary-text,
.dk-top-meta,
.dk-subtitle,
.dk-helper {
  color: #6B7280 !important;
}
.dk-banner {
  border: 1px solid #E5E7EB;
  background: #FFFFFF;
  padding: 14px 16px;
  margin-bottom: 10px;
}
.dk-banner-row {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 16px;
}
.dk-title {
  margin: 0;
  font-size: 1.35rem;
  font-weight: 700;
  color: #1F2937;
}
.dk-subtitle {
  margin-top: 4px;
  font-size: 0.9rem;
}
.dk-top-meta {
  white-space: nowrap;
  font-size: 0.82rem;
  padding-top: 4px;
}
.dk-footer {
  margin-top: 12px;
  padding: 12px 0 0 0;
  border-top: 1px solid #E5E7EB;
  color: #6B7280;
  font-size: 0.9rem;
}
.gradio-container .examples,
.gradio-container .examples-table,
.gradio-container .examples-table table,
.gradio-container .examples-table td,
.gradio-container .examples-table tr {
  text-align: left !important;
}
button.primary, .gradio-container button.primary {
  box-shadow: none !important;
}
.dk-section-title {
  margin: 0 0 4px 0;
  font-size: 14px;
  font-weight: 700;
  color: #111827;
}
.dk-progress {
  border: 1px solid #E5E7EB;
  background: #FFFFFF;
  padding: 10px 12px;
  border-radius: 6px;
}
.dk-progress {
  font-size: 13px !important;
  line-height: 1.55 !important;
}
.dk-progress-stack {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.dk-progress-item {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  border: 1px solid #E5E7EB;
  border-radius: 6px;
  padding: 9px 10px;
  background: #FFFFFF;
}
.dk-progress-icon {
  width: 20px;
  flex: 0 0 20px;
  font-size: 14px;
  line-height: 1.4;
}
.dk-progress-body {
  min-width: 0;
  flex: 1 1 auto;
}
.dk-progress-title {
  font-size: 13px;
  font-weight: 700;
  color: #111827;
}
.dk-progress-desc {
  margin-top: 2px;
  font-size: 12px;
  color: #6B7280;
}
.dk-progress-item.is-pending {
  background: #FFFFFF;
}
.dk-progress-item.is-running {
  background: #FEFCE8;
  border-color: #FCD34D;
}
.dk-progress-item.is-running .dk-progress-title,
.dk-progress-item.is-running .dk-progress-icon {
  color: #B45309;
}
.dk-progress-item.is-done {
  background: #ECFDF5;
  border-color: #86EFAC;
}
.dk-progress-item.is-done .dk-progress-title,
.dk-progress-item.is-done .dk-progress-icon {
  color: #047857;
}
.dk-progress-item.is-error {
  background: #FEF2F2;
  border-color: #FCA5A5;
}
.dk-progress-item.is-error .dk-progress-title,
.dk-progress-item.is-error .dk-progress-icon {
  color: #B91C1C;
}
.dk-helper {
  margin-bottom: 4px;
  font-size: 12px;
}
.dk-empty-state {
  min-height: 360px;
  display: flex;
  align-items: center;
  justify-content: center;
  border: 1px dashed #D1D5DB;
  border-radius: 8px;
  background: #FAFAFA;
  text-align: center;
  padding: 24px;
}
.dk-empty-state-icon {
  font-size: 26px;
  line-height: 1;
  margin-bottom: 10px;
}
.dk-empty-state-title {
  font-size: 15px;
  font-weight: 700;
  color: #111827;
}
.dk-empty-state-desc {
  margin-top: 6px;
  font-size: 13px;
  color: #6B7280;
}
.dk-risk-badge-row {
  margin-bottom: 10px;
}
.dk-risk-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 700;
}
.dk-risk-badge.high {
  background: #DC2626;
  color: #FFFFFF;
}
.dk-risk-badge.medium {
  background: #EA580C;
  color: #FFFFFF;
}
.dk-risk-badge.low {
  background: #0F766E;
  color: #FFFFFF;
}
.dk-image-report {
  border: 1px solid #E5E7EB;
  border-radius: 6px;
  background: #FFFFFF;
  overflow: hidden;
}
.dk-image-report-body {
  padding: 12px;
  font-size: 13px;
  color: #1F2937;
  line-height: 1.7;
}
.dk-image-report-body p {
  margin: 0 0 8px 0;
}
.dk-image-report-body strong {
  color: #111827;
}
.dk-risk-highlight {
  color: #B91C1C;
  font-weight: 700;
}
.dk-seed-box textarea,
.dk-seed-box input {
  background: #F0F9FF !important;
}
.dk-report {
  border: 1px solid #E5E7EB;
  border-radius: 6px;
  overflow: hidden;
  background: #FFFFFF;
}
.dk-report-section {
  border-top: 1px solid #E5E7EB;
}
.dk-report-section:first-child {
  border-top: none;
}
.dk-report-title {
  padding: 10px 12px;
  font-size: 13px;
  font-weight: 700;
  color: #111827;
  background: #F9FAFB;
  border-bottom: 1px solid #E5E7EB;
}
.dk-report-table {
  width: 100%;
  border-collapse: collapse;
  table-layout: fixed;
}
.dk-report-table th,
.dk-report-table td {
  padding: 9px 12px;
  border-bottom: 1px solid #E5E7EB;
  vertical-align: top;
  text-align: left;
  font-size: 13px;
  color: #1F2937;
  word-break: break-word;
}
.dk-report-table th {
  width: 180px;
  font-weight: 700;
  color: #111827;
  background: #F9FAFB;
}
.dk-report-table tr:nth-child(even) td,
.dk-report-table tr:nth-child(even) th {
  background: #FCFCFD;
}
.dk-report-table tr:last-child th,
.dk-report-table tr:last-child td {
  border-bottom: none;
}
.dk-report-list {
  margin: 0;
  padding-left: 18px;
}
.dk-report-list li {
  margin: 0 0 4px 0;
}
.dk-source-wrap {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.dk-source-section {
  border: 1px solid #E5E7EB;
  border-radius: 6px;
  background: #FFFFFF;
  overflow: hidden;
}
.dk-source-title {
  padding: 10px 12px;
  font-size: 13px;
  font-weight: 700;
  color: #111827;
  background: #F9FAFB;
  border-bottom: 1px solid #E5E7EB;
}
.dk-chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  padding: 12px;
}
.dk-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
  border: 1px solid #D1D5DB;
  border-radius: 999px;
  background: #F8FAFC;
  font-size: 12px;
  color: #334155;
}
.dk-timeline {
  padding: 12px 12px 4px 12px;
}
.dk-timeline-item {
  display: grid;
  grid-template-columns: 26px 1fr;
  gap: 10px;
}
.dk-timeline-rail {
  display: flex;
  flex-direction: column;
  align-items: center;
}
.dk-timeline-dot {
  width: 12px;
  height: 12px;
  border-radius: 999px;
  background: #0F766E;
  margin-top: 3px;
  flex: 0 0 auto;
}
.dk-timeline-line {
  width: 2px;
  flex: 1 1 auto;
  background: #D1D5DB;
  margin-top: 4px;
  min-height: 44px;
}
.dk-timeline-item:last-child .dk-timeline-line {
  background: transparent;
}
.dk-timeline-card {
  border: 1px solid #E5E7EB;
  border-radius: 6px;
  background: #FFFFFF;
  padding: 10px 12px;
  margin-bottom: 10px;
}
.dk-timeline-step {
  font-size: 12px;
  font-weight: 700;
  color: #6B7280;
  margin-bottom: 6px;
}
.dk-timeline-tag {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  background: #ECFEFF;
  color: #0F766E;
  font-size: 12px;
  font-weight: 700;
  margin-bottom: 6px;
}
.dk-timeline-desc {
  font-size: 13px;
  color: #1F2937;
  line-height: 1.6;
}
.dk-metric-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
  padding: 12px;
}
.dk-metric-card {
  border: 1px solid #E5E7EB;
  border-radius: 6px;
  background: #FFFFFF;
  padding: 12px;
}
.dk-metric-value {
  font-size: 24px;
  font-weight: 700;
  color: #111827;
  line-height: 1.1;
}
.dk-metric-label {
  margin-top: 6px;
  font-size: 12px;
  color: #6B7280;
}
.dk-metric-note {
  margin-top: 4px;
  font-size: 12px;
  color: #4B5563;
  line-height: 1.45;
}
.dk-reflection-wrap {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.dk-reflection-empty {
  border: 1px dashed #D1D5DB;
  border-radius: 6px;
  background: #FAFAFA;
  padding: 18px;
  color: #6B7280;
  font-size: 13px;
}
.dk-reflection-accordion {
  border: 1px solid #E5E7EB;
  border-radius: 6px;
  overflow: hidden;
  background: #FFFFFF;
}
.dk-reflection-summary {
  list-style: none;
  cursor: pointer;
  padding: 12px 14px;
  font-size: 14px;
  font-weight: 700;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}
.dk-reflection-summary::-webkit-details-marker {
  display: none;
}
.dk-reflection-summary-text {
  display: flex;
  align-items: center;
  gap: 8px;
}
.dk-reflection-status {
  font-size: 12px;
  font-weight: 700;
}
.dk-reflection-body {
  padding: 12px 14px 14px 14px;
  border-top: 1px solid rgba(229, 231, 235, 0.9);
}
.dk-reflection-issue {
  border: 1px solid #FED7AA;
  background: #FFF7ED;
}
.dk-reflection-issue .dk-reflection-summary {
  background: #FFF7ED;
  color: #9A3412;
}
.dk-reflection-issue .dk-reflection-status {
  color: #C2410C;
}
.dk-reflection-pass {
  border: 1px solid #A7F3D0;
  background: #ECFDF5;
}
.dk-reflection-pass .dk-reflection-summary {
  background: #ECFDF5;
  color: #065F46;
}
.dk-reflection-pass .dk-reflection-status {
  color: #047857;
}
.dk-reflection-subtitle {
  font-size: 12px;
  font-weight: 700;
  color: #4B5563;
  margin: 0 0 8px 0;
}
.dk-reflection-list {
  margin: 0;
  padding-left: 18px;
}
.dk-reflection-list li {
  margin: 0 0 6px 0;
  color: #1F2937;
  font-size: 13px;
  line-height: 1.55;
}
.dk-reflection-badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 700;
  margin-right: 6px;
  margin-bottom: 4px;
}
.dk-reflection-badge.issue {
  background: #FFEDD5;
  color: #C2410C;
}
.dk-reflection-badge.hit {
  background: #DBEAFE;
  color: #1D4ED8;
}
.dk-summary-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
  padding: 12px;
}
.dk-summary-card {
  border: 1px solid #E5E7EB;
  border-radius: 6px;
  padding: 10px 12px;
  background: #FFFFFF;
}
.dk-summary-label {
  font-size: 12px;
  font-weight: 700;
  color: #6B7280;
  margin-bottom: 4px;
}
.dk-summary-value {
  font-size: 14px;
  color: #1F2937;
  line-height: 1.5;
}
.dk-summary-core .dk-summary-value {
  font-weight: 700;
}
.dk-summary-risk {
  background: #FEF2F2;
  border-color: #FCA5A5;
}
.dk-summary-risk .dk-summary-label,
.dk-summary-risk .dk-summary-value {
  color: #B91C1C;
}
.dk-summary-action {
  background: #EFF6FF;
  border-color: #93C5FD;
}
.dk-summary-action .dk-summary-label,
.dk-summary-action .dk-summary-value {
  color: #1D4ED8;
}
"""


def render_report(report: dict) -> str:
    if not report:
        return "<div class='dk-report'><div class='dk-report-title'>HAZOP 报告</div><table class='dk-report-table'><tr><th>状态</th><td>请先在左侧输入场景并开始分析。</td></tr></table></div>"

    summary = report.get("summary", {})
    node = report.get("node_info", {})
    sections = []

    def esc(value) -> str:
        if value is None:
            return ""
        return html.escape(str(value))

    def list_html(items) -> str:
        if not items:
            return "-"
        return "<ul class='dk-report-list'>" + "".join(
            f"<li>{esc(item)}</li>" for item in items
        ) + "</ul>"

    def rows_html(rows) -> str:
        return "".join(
            f"<tr><th>{esc(label)}</th><td>{value if isinstance(value, str) else esc(value)}</td></tr>"
            for label, value in rows
        )

    if summary:
        sections.append(
            "<div class='dk-report-section'>"
            "<div class='dk-report-title'>分析摘要</div>"
            "<div class='dk-summary-grid'>"
            f"<div class='dk-summary-card dk-summary-core'>"
            f"<div class='dk-summary-label'>核心偏差</div>"
            f"<div class='dk-summary-value'>{esc(summary.get('core_deviation', 'N/A'))}</div>"
            f"</div>"
            f"<div class='dk-summary-card'>"
            f"<div class='dk-summary-label'>首要怀疑</div>"
            f"<div class='dk-summary-value'>{esc(summary.get('top_suspect', 'N/A'))}</div>"
            f"</div>"
            f"<div class='dk-summary-card dk-summary-risk'>"
            f"<div class='dk-summary-label'>最高风险</div>"
            f"<div class='dk-summary-value'>{esc(summary.get('highest_risk', 'N/A'))}</div>"
            f"</div>"
            f"<div class='dk-summary-card dk-summary-action'>"
            f"<div class='dk-summary-label'>首要动作 / 立即处置</div>"
            f"<div class='dk-summary-value'>{esc(summary.get('immediate_action', 'N/A'))}</div>"
            f"</div>"
            "</div></div>"
        )

    sections.append(
        "<div class='dk-report-section'>"
        "<div class='dk-report-title'>节点信息</div>"
        "<table class='dk-report-table'>"
        + rows_html(
            [
                ("设备", esc(node.get("equipment", "N/A"))),
                ("参数", esc(node.get("parameter", "N/A"))),
                ("偏差方向", esc(node.get("deviation_direction", "N/A"))),
                ("正常值", esc(node.get("normal_value", "N/A"))),
                ("当前值", esc(node.get("current_value", "N/A"))),
            ]
        )
        + "</table></div>"
    )

    for index, deviation in enumerate(report.get("deviations", []), start=1):
        causes = deviation.get("causes", {})
        cause_items = []
        if isinstance(causes, dict):
            for key, title in [
                ("primary", "首要怀疑"),
                ("secondary", "次要怀疑"),
                ("pending", "待验证项"),
            ]:
                items = causes.get(key, [])
                if items:
                    rendered = [
                        f"[{item.get('type', '未分类')}] {item.get('description', '')}"
                        for item in items
                    ]
                    cause_items.append((title, list_html(rendered)))

        consequences = deviation.get("consequences", [])
        consequence_items = []
        if consequences:
            for item in consequences:
                if isinstance(item, dict):
                    consequence_items.append(
                        f"{item.get('stage', '后果')}：{item.get('description', '')}"
                    )
                else:
                    consequence_items.append(str(item))

        safeguard_items = []
        safeguards = deviation.get("safeguards", [])
        if safeguards:
            for safeguard in safeguards:
                safeguard_items.append(
                    f"{safeguard.get('measure', '')}："
                    f"{safeguard.get('effectiveness', '未说明')}，"
                    f"{safeguard.get('effectiveness_reason', '暂无说明')}"
                )

        recommendation_rows = []
        recommendations = deviation.get("recommendations", {})
        if isinstance(recommendations, dict):
            for key, title in [
                ("immediate", "立即处置"),
                ("short_term", "短期整改"),
                ("long_term", "长期改进"),
            ]:
                items = recommendations.get(key, [])
                if items:
                    recommendation_rows.append(
                        (title, list_html([item.get("action", "") for item in items]))
                    )

        section_rows = []
        section_rows.extend(cause_items)
        if consequence_items:
            section_rows.append(("后果链", list_html(consequence_items)))
        if safeguard_items:
            section_rows.append(("现有保护措施", list_html(safeguard_items)))
        section_rows.extend(recommendation_rows)
        if not section_rows:
            section_rows.append(("状态", "暂无结构化信息"))

        sections.append(
            "<div class='dk-report-section'>"
            f"<div class='dk-report-title'>偏差条目 {index}</div>"
            "<table class='dk-report-table'>"
            + rows_html(section_rows)
            + "</table></div>"
        )

    return "<div class='dk-report'>" + "".join(sections) + "</div>"


def render_sources_panel(report: dict) -> str:
    if not report:
        return (
            "<div class='dk-source-wrap'>"
            "<div class='dk-source-section'>"
            "<div class='dk-source-title'>知识溯源</div>"
            "<div class='dk-chip-row'><span class='dk-chip'>🔖 分析完成后，这里会展示参考案例</span></div>"
            "</div>"
            "</div>"
        )

    meta = report.get("analysis_metadata", {})
    evidence_chain = report.get("evidence_chain", [])
    ref_ids = meta.get("referenced_cases", [])
    ref_names = meta.get("referenced_names", {})
    issues = meta.get("physical_issues_found", [])
    confidence_level = meta.get("confidence_level", "N/A")
    confidence_reason = meta.get("confidence_reason", "暂无说明")
    reflection_rounds = meta.get("reflection_rounds", 0)
    issue_count = len(issues)

    def esc(value) -> str:
        return html.escape(str(value if value is not None else ""))

    chips = []
    if ref_ids:
        for case_id in ref_ids:
            case_name = ref_names.get(case_id, "")
            label = f"{case_id} - {case_name}" if case_name else case_id
            chips.append(f"<span class='dk-chip'>🔖 {esc(label)}</span>")
    else:
        chips.append("<span class='dk-chip'>🔖 未检索到明确参考案例</span>")

    timeline_items = []
    if evidence_chain:
        for item in evidence_chain:
            timeline_items.append(
                "<div class='dk-timeline-item'>"
                "<div class='dk-timeline-rail'>"
                "<div class='dk-timeline-dot'></div>"
                "<div class='dk-timeline-line'></div>"
                "</div>"
                "<div class='dk-timeline-card'>"
                f"<div class='dk-timeline-step'>Step {esc(item.get('step', '?'))}</div>"
                f"<div class='dk-timeline-tag'>{esc(item.get('type', '未分类'))}</div>"
                f"<div class='dk-timeline-desc'>{esc(item.get('content', ''))}</div>"
                "</div></div>"
            )
    else:
        timeline_items.append(
            "<div class='dk-timeline-item'>"
            "<div class='dk-timeline-rail'><div class='dk-timeline-dot'></div><div class='dk-timeline-line'></div></div>"
            "<div class='dk-timeline-card'>"
            "<div class='dk-timeline-step'>Step -</div>"
            "<div class='dk-timeline-tag'>证据链</div>"
            "<div class='dk-timeline-desc'>暂无证据链数据。</div>"
            "</div></div>"
        )

    metrics = (
        "<div class='dk-metric-grid'>"
        f"<div class='dk-metric-card'><div class='dk-metric-value'>{esc(confidence_level)}</div><div class='dk-metric-label'>置信水平</div><div class='dk-metric-note'>{esc(confidence_reason)}</div></div>"
        f"<div class='dk-metric-card'><div class='dk-metric-value'>{esc(reflection_rounds)}</div><div class='dk-metric-label'>反思轮次</div><div class='dk-metric-note'>多智能体校验与修正次数</div></div>"
        f"<div class='dk-metric-card'><div class='dk-metric-value'>{esc(issue_count)}</div><div class='dk-metric-label'>物理问题数</div><div class='dk-metric-note'>识别到的物理一致性问题数量</div></div>"
        "</div>"
    )

    return (
        "<div class='dk-source-wrap'>"
        "<div class='dk-source-section'>"
        "<div class='dk-source-title'>参考案例</div>"
        f"<div class='dk-chip-row'>{''.join(chips)}</div>"
        "</div>"
        "<div class='dk-source-section'>"
        "<div class='dk-source-title'>证据链</div>"
        f"<div class='dk-timeline'>{''.join(timeline_items)}</div>"
        "</div>"
        "<div class='dk-source-section'>"
        "<div class='dk-source-title'>置信与校验</div>"
        f"{metrics}"
        "</div>"
        "</div>"
    )


def render_reflection_panel(reflection_history: list) -> str:
    if not reflection_history:
        return (
            "<div class='dk-reflection-wrap'>"
            "<div class='dk-reflection-empty'>分析完成后，这里会展示系统如何拦截物理幻觉与修正推理过程。</div>"
            "</div>"
        )

    def esc(value) -> str:
        return html.escape(str(value if value is not None else ""))

    panels = []
    has_issue_round = any(not record.get("passed") for record in reflection_history)

    for record in reflection_history:
        passed = record.get("passed", False)
        round_num = record.get("round", "?")
        issues = record.get("issues", [])
        fallacy_hits = record.get("fallacy_hits", [])
        open_attr = " open" if (not passed or not has_issue_round) else ""
        box_class = "dk-reflection-pass" if passed else "dk-reflection-issue"
        icon = "✅" if passed else "⚠️"
        status_text = "已通过" if passed else f"发现 {len(issues)} 个问题"

        issue_items = []
        for issue in issues:
            issue_type = esc(issue.get("issue_type", "未分类"))
            description = esc(issue.get("description", ""))
            issue_items.append(
                f"<li><span class='dk-reflection-badge issue'>{issue_type}</span>{description}</li>"
            )

        hit_items = []
        for hit in fallacy_hits:
            label = esc(hit.get("fallacy_id", "命中"))
            proposition = esc(hit.get("matched_proposition", ""))
            hit_items.append(
                f"<li><span class='dk-reflection-badge hit'>{label}</span>{proposition}</li>"
            )

        body_parts = []
        if hit_items:
            body_parts.append(
                "<div class='dk-reflection-subtitle'>负样本库命中</div>"
                f"<ul class='dk-reflection-list'>{''.join(hit_items)}</ul>"
            )
        if issue_items:
            body_parts.append(
                "<div class='dk-reflection-subtitle'>问题清单</div>"
                f"<ul class='dk-reflection-list'>{''.join(issue_items)}</ul>"
            )
        if not issue_items and not hit_items:
            body_parts.append(
                "<div class='dk-reflection-subtitle'>校验结果</div>"
                "<ul class='dk-reflection-list'><li>本轮未发现新的物理问题，报告通过校验。</li></ul>"
            )

        panels.append(
            f"<details class='dk-reflection-accordion {box_class}'{open_attr}>"
            "<summary class='dk-reflection-summary'>"
            f"<span class='dk-reflection-summary-text'><span>{icon}</span><span>第 {esc(round_num)} 轮反思</span></span>"
            f"<span class='dk-reflection-status'>{status_text}</span>"
            "</summary>"
            f"<div class='dk-reflection-body'>{''.join(body_parts)}</div>"
            "</details>"
        )

    return "<div class='dk-reflection-wrap'>" + "".join(panels) + "</div>"


def image_to_data_url(image_path: str) -> str:
    file_path = Path(image_path)
    mime_type = mimetypes.guess_type(file_path.name)[0] or "image/png"
    encoded = base64.b64encode(file_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def extract_dashscope_text(response) -> str:
    if hasattr(response, "output"):
        output = response.output
    elif isinstance(response, dict):
        output = response.get("output", {})
    else:
        output = {}

    choices = getattr(output, "choices", None)
    if choices is None and isinstance(output, dict):
        choices = output.get("choices", [])
    if not choices:
        return "模型未返回可解析内容。"

    message = choices[0].get("message") if isinstance(choices[0], dict) else getattr(choices[0], "message", None)
    if not message:
        return "模型未返回可解析内容。"

    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        if parts:
            return "\n".join(parts)
    return str(content) if content else "模型未返回可解析内容。"


def extract_core_scenario_summary(image_result: str) -> str:
    match = re.search(
        r"\[SCENE_SUMMARY\]\s*(.*?)\s*\[/SCENE_SUMMARY\]",
        image_result,
        flags=re.S,
    )
    if match:
        return match.group(1).strip()
    return ""


def strip_scene_summary_block(image_result: str) -> str:
    cleaned = re.sub(
        r"\[SCENE_SUMMARY\]\s*.*?\s*\[/SCENE_SUMMARY\]",
        "",
        image_result,
        flags=re.S,
    )
    return cleaned.strip()


def extract_risk_level(image_result: str) -> str:
    prefixes = ["风险等级：", "风险等级:"]
    for raw_line in image_result.splitlines():
        line = raw_line.strip()
        for prefix in prefixes:
            if line.startswith(prefix):
                return line[len(prefix):].strip()
    return ""


def highlight_risk_text(text: str) -> str:
    keywords = [
        "锈蚀",
        "腐蚀",
        "泄漏",
        "渗漏",
        "裂纹",
        "未佩戴护目镜",
        "未佩戴",
        "护目镜",
        "手轮颜色不一致",
        "警示标识缺失",
        "阀门异常",
        "保温破损",
        "管线裸露",
    ]
    highlighted = html.escape(text)
    for keyword in sorted(set(keywords), key=len, reverse=True):
        escaped_keyword = html.escape(keyword)
        highlighted = highlighted.replace(
            escaped_keyword,
            f"<span class='dk-risk-highlight'>{escaped_keyword}</span>",
        )
    return highlighted


def format_image_report(image_result: str) -> str:
    cleaned_report = strip_scene_summary_block(image_result)
    risk_level = extract_risk_level(cleaned_report)
    badge_html = ""
    if "高风险" in risk_level:
        badge_html = "<div class='dk-risk-badge-row'><span class='dk-risk-badge high'>🚨 高风险</span></div>"
    elif "中风险" in risk_level:
        badge_html = "<div class='dk-risk-badge-row'><span class='dk-risk-badge medium'>⚠️ 中风险</span></div>"
    elif "低风险" in risk_level:
        badge_html = "<div class='dk-risk-badge-row'><span class='dk-risk-badge low'>✅ 低风险</span></div>"

    paragraphs = []
    for raw_line in cleaned_report.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("风险等级：") or line.startswith("风险等级:"):
            continue
        paragraphs.append(f"<p>{highlight_risk_text(line)}</p>")

    if not paragraphs:
        paragraphs.append("<p>暂无可展示的识别结果。</p>")

    return (
        "<div class='dk-image-report'>"
        "<div class='dk-image-report-body'>"
        f"{badge_html}{''.join(paragraphs)}"
        "</div></div>"
    )


def compose_hazop_seed(core_summary: str, additional_context: str) -> str:
    summary_line = core_summary.strip() if core_summary else "未提取到核心异常场景总结。"
    context_line = additional_context.strip() if additional_context else "未提供额外上下文。"
    return (
        "请基于以下现场巡检信息执行 HAZOP 文本分析：\n"
        f"核心异常场景总结：{summary_line}\n"
        f"补充说明：{context_line}"
    )


def render_progress_panel(steps) -> str:
    icon_map = {
        "pending": "⬜",
        "running": "⏳",
        "done": "✅",
        "error": "❌",
    }
    items = []
    for step in steps:
        status = step.get("status", "pending")
        items.append(
            "<div class='dk-progress-item is-{status}'>"
            "<div class='dk-progress-icon'>{icon}</div>"
            "<div class='dk-progress-body'>"
            "<div class='dk-progress-title'>{title}</div>"
            "<div class='dk-progress-desc'>{desc}</div>"
            "</div></div>".format(
                status=html.escape(status),
                icon=icon_map.get(status, "⬜"),
                title=html.escape(step.get("title", "")),
                desc=html.escape(step.get("desc", "")),
            )
        )
    return "<div class='dk-progress'><div class='dk-progress-stack'>" + "".join(items) + "</div></div>"


def analyze_image_risk(image_path: str, additional_context: str):
    print("[analyze_image_risk] 收到图片分析请求")
    if not image_path:
        message = "请先上传现场照片。"
        print("[analyze_image_risk] 未上传图片，直接返回")
        result = (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
            gr.update(visible=False),
            message,
        )
        print("[analyze_image_risk] 返回完成：empty-state")
        return result

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        message = "未检测到 `DASHSCOPE_API_KEY`，请先在环境变量或 `.env` 中配置后再进行图片分析。"
        print("[analyze_image_risk] 缺少 DASHSCOPE_API_KEY，直接返回")
        result = (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
            gr.update(visible=False),
            message,
        )
        print("[analyze_image_risk] 返回完成：missing-api-key")
        return result

    try:
        from dashscope import MultiModalConversation
    except ImportError:
        message = "未安装 `dashscope`，请先执行 `pip install -r requirements.txt`。"
        print("[analyze_image_risk] 缺少 dashscope 依赖，直接返回")
        result = (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
            gr.update(visible=False),
            message,
        )
        print("[analyze_image_risk] 返回完成：missing-dashscope")
        return result

    user_text = "请分析这张化工现场照片中的安全风险。"
    if additional_context and additional_context.strip():
        user_text += f"\n补充说明：{additional_context.strip()}"

    try:
        print("[analyze_image_risk] 正在读取并编码图片")
        data_url = image_to_data_url(image_path)
        print("[analyze_image_risk] 正在调用 MultiModalConversation.call")
        response = MultiModalConversation.call(
            model="qwen-vl-max",
            api_key=api_key,
            stream=False,
            messages=[
                {"role": "system", "content": [{"text": IMAGE_SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {"image": data_url},
                        {"text": user_text},
                    ],
                },
            ],
        )
        print("[analyze_image_risk] 视觉模型返回成功，正在解析文本")
        result_text = extract_dashscope_text(response)
        core_summary = extract_core_scenario_summary(result_text)
        if not core_summary:
            core_summary = "未从视觉模型返回中提取到核心异常场景总结，请重试或补充现场说明。"
        _current_result["image_seed"] = core_summary
        status = "图片分析完成，可以将识别结果一键转入文本 HAZOP 分析。"
        result = (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(value=format_image_report(result_text), visible=True),
            gr.update(value=core_summary, visible=True),
            gr.update(value="识别完成后可一键回填到文本分析。", visible=True),
            gr.update(visible=True),
            status,
        )
        print("[analyze_image_risk] 返回完成：success")
        return result
    except Exception as exc:
        message = f"图片分析失败：{exc}"
        print(f"[analyze_image_risk] 执行异常：{exc}")
        result = (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
            gr.update(visible=False),
            message,
        )
        print("[analyze_image_risk] 返回完成：error")
        return result


def analyze_streaming(user_input: str):
    global _current_result
    print("[analyze_streaming] 收到分析请求")
    initial_steps = [
        {"title": "Step 1/3 上下文解析", "desc": "等待开始", "status": "pending"},
        {"title": "Step 2/3 RAG分析", "desc": "等待开始", "status": "pending"},
        {"title": "Step 3/3 物理反思", "desc": "等待开始", "status": "pending"},
    ]
    empty = (
        "请先输入要分析的化工异常场景。",
        "分析完成后，这里会展示参考案例、证据链和置信信息。",
        "分析完成后，这里会展示反思日志和物理校验过程。",
        "{}",
        None,
        render_progress_panel(initial_steps),
    )

    if not user_input or not user_input.strip():
        print("[analyze_streaming] 输入为空，返回空状态")
        yield empty
        return

    _current_result = {
        "report": None,
        "input": user_input,
        "scores": None,
        "reflection_history": [],
        "image_seed": _current_result.get("image_seed", ""),
    }

    report = None
    context = None
    reflection_history = []
    progress_steps = [
        {
            "title": "Step 1/3 上下文解析",
            "desc": "正在提取设备、参数、偏差方向和工艺条件...",
            "status": "running",
        },
        {"title": "Step 2/3 RAG分析", "desc": "等待开始", "status": "pending"},
        {"title": "Step 3/3 物理反思", "desc": "等待开始", "status": "pending"},
    ]
    progress = render_progress_panel(progress_steps)
    print("[analyze_streaming] 正在执行：Step 1/3 初始化进度面板并首次 yield")
    yield (
        "分析中，请稍候...",
        "正在整理知识溯源信息...",
        "正在准备反思日志...",
        "{}",
        None,
        progress,
    )

    try:
        print("[analyze_streaming] 正在执行：Step 1/3 上下文解析")
        ctx_result = ContextAgent().run(user_input.strip())
        context = ctx_result["context"]
        progress_steps[0] = {
            "title": "Step 1/3 上下文解析",
            "desc": f"已完成，识别到 {context['equipment']} / {context['parameter']} / {context['deviation_direction']}",
            "status": "done",
        }
        progress_steps[1] = {
            "title": "Step 2/3 RAG分析",
            "desc": "正在检索相似案例并生成结构化报告...",
            "status": "running",
        }
        progress = render_progress_panel(progress_steps)
        print("[analyze_streaming] Step 1/3 完成，准备推进到 Step 2/3")
    except Exception as exc:
        print(f"[analyze_streaming] Step 1/3 失败：{exc}")
        progress_steps[0] = {
            "title": "Step 1/3 上下文解析",
            "desc": f"执行失败：{exc}",
            "status": "error",
        }
        print("[analyze_streaming] 正在执行：Step 1/3 失败后的 yield")
        yield (
            f"上下文解析失败：{exc}",
            "未生成知识溯源。",
            "未生成反思日志。",
            "{}",
            None,
            render_progress_panel(progress_steps),
        )
        return

    print("[analyze_streaming] 正在执行：Step 2/3 开始前的过渡 yield")
    yield (
        "正在执行 RAG 检索与报告生成...",
        "正在汇总参考案例...",
        "等待反思日志...",
        "{}",
        None,
        progress,
    )

    correction_guidance = ""
    round_num = 0

    while round_num < config.MAX_REFLECTION_ROUNDS:
        print(f"[analyze_streaming] 进入反思循环，第 {round_num + 1} 轮")
        progress_steps[1] = {
            "title": "Step 2/3 RAG分析",
            "desc": f"第 {round_num + 1} 轮检索和报告生成中...",
            "status": "running",
        }
        progress = render_progress_panel(progress_steps)
        print(f"[analyze_streaming] 正在执行：第 {round_num + 1} 轮 Step 2/3 的 yield")
        yield (
            "正在执行 RAG 检索与报告生成...",
            "正在汇总参考案例...",
            "等待反思日志...",
            "{}",
            None,
            progress,
        )

        try:
            print(f"[analyze_streaming] 正在执行：第 {round_num + 1} 轮 RAGAgent.run")
            report = RAGAgent().run(
                context=context,
                correction_guidance=correction_guidance,
                reflection_rounds=round_num,
            )
            matched_count = len(report.get("analysis_metadata", {}).get("referenced_cases", []))
            print(f"[analyze_streaming] 第 {round_num + 1} 轮 Step 2/3 完成，匹配案例数：{matched_count}")
            progress_steps[1] = {
                "title": "Step 2/3 RAG分析",
                "desc": f"已完成，匹配 {matched_count} 个相似案例",
                "status": "done",
            }
        except Exception as exc:
            print(f"[analyze_streaming] 第 {round_num + 1} 轮 Step 2/3 失败：{exc}")
            progress_steps[1] = {
                "title": "Step 2/3 RAG分析",
                "desc": f"执行失败：{exc}",
                "status": "error",
            }
            print("[analyze_streaming] 正在执行：Step 2/3 失败后的 yield")
            yield (
                f"RAG 报告生成失败：{exc}",
                "未生成知识溯源。",
                "未生成反思日志。",
                "{}",
                None,
                render_progress_panel(progress_steps),
            )
            return

        progress_steps[2] = {
            "title": "Step 3/3 物理反思",
            "desc": f"第 {round_num + 1} 轮双层校验中，三个智能体正在接力思考...",
            "status": "running",
        }
        progress = render_progress_panel(progress_steps)
        print(f"[analyze_streaming] 正在执行：第 {round_num + 1} 轮 Step 3/3 的 yield")
        yield (
            "正在执行物理校验与反思修正...",
            render_sources_panel(report),
            "正在生成反思日志...",
            json.dumps(report, ensure_ascii=False, indent=2),
            None,
            progress,
        )

        try:
            print(f"[analyze_streaming] 正在执行：第 {round_num + 1} 轮 ReflectionAgent.run")
            ref_result = ReflectionAgent().run(report)
            record = {
                "round": round_num + 1,
                "passed": ref_result["passed"],
                "issues": ref_result["issues"],
                "fallacy_hits": ref_result.get("fallacy_hits", []),
            }
            reflection_history.append(record)
            print(
                f"[analyze_streaming] 第 {round_num + 1} 轮反思完成："
                f"passed={ref_result['passed']} issues={len(ref_result['issues'])}"
            )

            if ref_result["passed"]:
                progress_steps[2] = {
                    "title": "Step 3/3 物理反思",
                    "desc": "已完成，通过双层物理校验",
                    "status": "done",
                }
                print("[analyze_streaming] Step 3/3 通过，跳出循环")
                break

            correction_guidance = ref_result["correction_guidance"]
            progress_steps[2] = {
                "title": "Step 3/3 物理反思",
                "desc": f"发现 {len(ref_result['issues'])} 个问题，正在回传修正意见并重新生成...",
                "status": "running",
            }
            progress = render_progress_panel(progress_steps)
            print(f"[analyze_streaming] 第 {round_num + 1} 轮发现问题，正在执行修正提示 yield")
            yield (
                "正在根据反思结果修正报告...",
                render_sources_panel(report),
                render_reflection_panel(reflection_history),
                json.dumps(report, ensure_ascii=False, indent=2),
                None,
                progress,
            )
            progress_steps[1] = {
                "title": "Step 2/3 RAG分析",
                "desc": f"收到第 {round_num + 1} 轮修正意见，准备重新生成...",
                "status": "running",
            }
            round_num += 1
            print(f"[analyze_streaming] 进入下一轮，round_num={round_num}")
        except Exception as exc:
            print(f"[analyze_streaming] 第 {round_num + 1} 轮 Step 3/3 异常：{exc}")
            progress_steps[2] = {
                "title": "Step 3/3 物理反思",
                "desc": f"校验异常，已保留当前报告（{exc}）",
                "status": "error",
            }
            break

    if not report:
        print("[analyze_streaming] 未生成 report，返回失败态")
        yield (
            "分析失败，未生成报告。",
            "未生成知识溯源。",
            "未生成反思日志。",
            "{}",
            None,
            render_progress_panel(progress_steps),
        )
        return

    print("[analyze_streaming] 正在执行：生成最终展示内容")
    report.setdefault("analysis_metadata", {})
    report["analysis_metadata"]["reflection_rounds"] = len(reflection_history) or 1

    report_md = render_report(report)
    sources_md = render_sources_panel(report)
    reflection_md = render_reflection_panel(reflection_history)

    scores = None
    scores_md = "评估结果暂不可用。"
    try:
        scores = evaluate_report(report)
        scores_md = format_scores_markdown(scores)
    except Exception as exc:
        scores_md = f"评估失败：{exc}"

    raw_json = json.dumps(report, ensure_ascii=False, indent=2)
    export_path = None
    try:
        with open(EXPORT_PATH, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        export_path = EXPORT_PATH
    except Exception:
        export_path = None

    _current_result = {
        "report": report,
        "input": user_input,
        "scores": scores,
        "reflection_history": reflection_history,
        "image_seed": _current_result.get("image_seed", ""),
    }

    progress = render_progress_panel(progress_steps)
    print("[analyze_streaming] 正在执行：最终结果 yield")
    yield (
        report_md,
        sources_md,
        reflection_md,
        raw_json,
        export_path,
        progress,
    )


def send_image_seed_to_text(seed_text: str):
    if not seed_text or not seed_text.strip():
        return "", "暂无可回填内容，请先完成图片分析。"
    return seed_text, "已将图片识别结果写入文本分析输入框，可切换到“HAZOP 文本分析”继续分析。"


def submit_feedback(rating: int, comment: str):
    if not _current_result.get("report"):
        return "请先完成一次分析再提交反馈。"

    try:
        save_feedback(
            user_input=_current_result["input"],
            report=_current_result["report"],
            rating=rating,
            comment=comment,
            scores=_current_result.get("scores"),
        )
        return f"反馈已记录，评分：{rating}/5。"
    except Exception as exc:
        return f"反馈提交失败：{exc}"


def refresh_stats():
    return format_stats_markdown(get_stats())


def create_ui():
    with gr.Blocks(
        title="DK-SAR Chemical Safety Analysis",
        fill_height=True,
    ) as demo:
        gr.HTML(f"<style>{CUSTOM_CSS}</style>")
        gr.HTML(
            """
            <div class="dk-banner">
              <div class="dk-banner-row">
                <div>
                  <h2 class="dk-title">DK-SAR 化工安全分析系统</h2>
                  <div class="dk-subtitle">
                    Dual Knowledge-enhanced Self-Adaptive Reasoning · HAZOP Analysis Platform
                  </div>
                </div>
                <div class="dk-top-meta">支持文本分析 · 图片识别 · 物理反思校验</div>
              </div>
            </div>
            """
        )

        with gr.Tabs():
            with gr.Tab("HAZOP 文本分析"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=320):
                        gr.Markdown("<div class='dk-section-title'>输入场景</div>")
                        user_input = gr.Textbox(
                            label="化工异常场景描述",
                            placeholder="例如：3号反应釜温度持续升高，压力同步上升，冷却水温偏高，操作员已开启泄压阀。",
                            lines=7,
                        )
                        gr.Examples(
                            examples=EXAMPLES,
                            inputs=[user_input],
                            label="快速开始：选择示例场景",
                        )
                        analyze_btn = gr.Button("开始 HAZOP 分析", variant="primary", size="lg")

                        with gr.Accordion("使用建议", open=False):
                            gr.Markdown(
                                "\n".join(
                                    [
                                        "- 尽量写清设备、参数、偏差方向和现场操作。",
                                        "- 如果来自拍照识别，可直接使用右侧 Tab 回填后的文本。",
                                        "- 下面这些关键词也可以直接触发快速测试："
                                        f"{' / '.join(TEXT_TAB_EXAMPLES)}",
                                    ]
                                )
                            )

                        with gr.Accordion("用户反馈", open=False):
                            rating_slider = gr.Slider(1, 5, value=4, step=1, label="评分")
                            comment_box = gr.Textbox(
                                label="反馈意见",
                                placeholder="这次分析是否准确、哪里还可以更好。",
                                lines=2,
                            )
                            feedback_btn = gr.Button("提交反馈")
                            feedback_result = gr.Markdown()

                    with gr.Column(scale=2):
                        gr.Markdown("<div class='dk-section-title'>分析进度</div>")
                        progress_output = gr.HTML(render_progress_panel([
                            {"title": "Step 1/3 上下文解析", "desc": "等待开始", "status": "pending"},
                            {"title": "Step 2/3 RAG分析", "desc": "等待开始", "status": "pending"},
                            {"title": "Step 3/3 物理反思", "desc": "等待开始", "status": "pending"},
                        ]))
                        with gr.Tabs():
                            with gr.Tab("HAZOP 报告"):
                                report_output = gr.HTML()
                            with gr.Tab("知识溯源"):
                                sources_output = gr.HTML(
                                    render_sources_panel(None)
                                )
                            with gr.Tab("反思日志"):
                                reflection_output = gr.HTML(
                                    render_reflection_panel([])
                                )
                            with gr.Tab("原始 JSON"):
                                json_output = gr.Code(language="json", value="{}")

                        export_file = gr.File(label="导出 JSON 报告", interactive=False)
                        stats_output = gr.Markdown(value=format_stats_markdown(get_stats()))
                        refresh_btn = gr.Button("刷新统计", size="sm")

            with gr.Tab("现场巡检识别"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=320):
                        gr.Markdown(
                            "<div class='dk-helper'>上传设备、管道、阀门、作业环境等现场照片，系统自动识别安全风险</div>"
                        )
                        image_input = gr.Image(
                            label="上传现场照片",
                            type="filepath",
                            sources=["upload", "clipboard"],
                            height=360,
                        )
                        image_context = gr.Textbox(
                            label="补充说明",
                            placeholder="例如：这是 3 号车间冷却水管道附近，设备已运行 2 年。",
                            lines=2,
                        )
                        image_btn = gr.Button("开始图片风险识别", variant="primary", size="lg")
                        image_status = gr.Markdown("等待上传照片。")

                    with gr.Column(scale=2):
                        image_placeholder = gr.Markdown(
                            """
                            <div class="dk-empty-state">
                              <div>
                                <div class="dk-empty-state-icon">🖼️</div>
                                <div class="dk-empty-state-title">请先在左侧上传巡检照片</div>
                                <div class="dk-empty-state-desc">
                                  系统将自动识别现场风险，并关联 HAZOP 分析线索
                                </div>
                              </div>
                            </div>
                            """,
                            visible=True,
                        )
                        image_report_title = gr.Markdown(
                            "<div class='dk-section-title'>风险评估报告</div>",
                            visible=False,
                        )
                        image_output = gr.HTML(
                            value="",
                            visible=False,
                        )
                        image_seed = gr.Textbox(
                            label="核心异常场景总结（用于 HAZOP 草稿）",
                            lines=8,
                            interactive=False,
                            placeholder="系统提取的核心场景将显示在此处，您也可以手动修改...",
                            elem_classes=["dk-seed-box"],
                            visible=False,
                        )
                        transfer_status = gr.Markdown("", visible=False)
                        transfer_btn = gr.Button("➡️ 将识别结果转为 HAZOP 分析", variant="primary", visible=False)

        analyze_btn.click(
            fn=analyze_streaming,
            inputs=[user_input],
            outputs=[
                report_output,
                sources_output,
                reflection_output,
                json_output,
                export_file,
                progress_output,
            ],
            queue=True,
        )
        image_btn.click(
            fn=analyze_image_risk,
            inputs=[image_input, image_context],
            outputs=[
                image_placeholder,
                image_report_title,
                image_output,
                image_seed,
                transfer_status,
                transfer_btn,
                image_status,
            ],
            queue=False,
        )
        transfer_btn.click(
            fn=send_image_seed_to_text,
            inputs=[image_seed],
            outputs=[user_input, transfer_status],
            queue=False,
        )
        feedback_btn.click(
            fn=submit_feedback,
            inputs=[rating_slider, comment_box],
            outputs=[feedback_result],
            queue=False,
        )
        refresh_btn.click(fn=refresh_stats, outputs=[stats_output], queue=False)

        gr.HTML(
            """
            <div class="dk-footer">
              技术架构：LangGraph 多智能体编排 · 双知识库 RAG/反思校验 · DashScope 多模态视觉分析
            </div>
            """
        )

    return demo


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  DK-SAR 系统启动中...")
    print("=" * 60)
    initialize_system()
    create_ui().queue(default_concurrency_limit=1, status_update_rate=1).launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=THEME,
        show_error=True,
        debug=True,
    )
