"""
从CSB PDF报告中批量提取结构化HAZOP案例
使用：python extract_cases.py
需要：pip install PyPDF2 dashscope
"""
import os
import json
import time
from pathlib import Path

# pip install PyPDF2 如果没装
try:
    from PyPDF2 import PdfReader
except ImportError:
    print("请先安装: pip install PyPDF2")
    exit(1)

from dashscope import Generation

PDF_DIR = Path("csb_reports")
OUTPUT_DIR = Path("csb_cases")
OUTPUT_DIR.mkdir(exist_ok=True)

API_KEY = os.getenv("DASHSCOPE_API_KEY")


def extract_text_from_pdf(pdf_path, max_pages=15):
    """从PDF提取前N页文本"""
    try:
        reader = PdfReader(str(pdf_path))
        text = ""
        for i, page in enumerate(reader.pages[:max_pages]):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        print(f"  PDF读取失败: {e}")
        return ""


def extract_case_with_ai(case_name, pdf_text):
    """用通义千问从PDF文本中提取结构化案例"""
    
    # 截取前4000字符（避免超token限制）
    text_snippet = pdf_text[:4000] if len(pdf_text) > 4000 else pdf_text
    
    prompt = f"""你是一位资深化工安全工程师和HAZOP分析专家。

以下是美国CSB（Chemical Safety Board）事故调查报告的内容摘录。
请从中提取结构化的HAZOP案例数据。

事故名称：{case_name}

报告内容：
{text_snippet}

请输出严格的JSON格式（不要输出其他任何内容），包含以下结构：

{{
  "case_id": "",
  "basic_info": {{
    "incident_name": "事故英文名称",
    "incident_name_cn": "事故中文名称",
    "date": "事故日期",
    "location": "地点",
    "company": "公司名",
    "fatalities": 0,
    "injuries": 0
  }},
  "process_context": {{
    "process_type": "工艺类型（反应/蒸馏/储运/换热/管道/粉尘处理等）",
    "equipment": ["关键设备"],
    "materials": ["涉及物料"],
    "operating_conditions": "工况描述"
  }},
  "failure_mechanism": {{
    "root_cause": "根本原因（一句话）",
    "deviation_type": "HAZOP偏差类型（温度偏高/压力偏高/流量偏低/泄漏等）",
    "guide_word": "IEC 61882引导词（More/Less/No/Reverse/Part of等）",
    "parameter": "偏差参数",
    "causal_chain": "因果链：原因→中间事件→后果",
    "physical_principles": "涉及的物理原理"
  }},
  "protection_system": {{
    "existing_safeguards": ["事故前已有的保护措施"],
    "safeguard_failures": "保护措施为什么失效了",
    "csb_recommendations": ["CSB的改进建议（取前3条最重要的）"]
  }},
  "embedding_text": "用100词左右的英文概括事故核心机理，包含设备类型、物料、偏差、根因和后果，用于语义检索"
}}

只输出JSON，不要输出```json标记或其他文字。"""

    try:
        response = Generation.call(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            api_key=API_KEY,
            result_format="message"
        )
        
        result = response.output.choices[0].message.content.strip()
        
        # 清理可能的markdown标记
        if result.startswith("```json"):
            result = result[7:]
        if result.startswith("```"):
            result = result[3:]
        if result.endswith("```"):
            result = result[:-3]
        result = result.strip()
        
        case_data = json.loads(result)
        return case_data
        
    except json.JSONDecodeError as e:
        print(f"  JSON解析失败: {e}")
        # 保存原始输出供调试
        debug_path = OUTPUT_DIR / f"debug_{case_name[:30]}.txt"
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(result if 'result' in dir() else "no result")
        return None
    except Exception as e:
        print(f"  AI抽取失败: {e}")
        return None


def main():
    print("=" * 60)
    print("  从CSB PDF报告中提取结构化HAZOP案例")
    print("=" * 60)
    
    # 检查API Key
    global API_KEY
    if not API_KEY:
        API_KEY = input("请输入DASHSCOPE_API_KEY: ").strip()
    
    # 找到所有PDF
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    pdfs = [p for p in pdfs if p.stat().st_size > 10000]  # 过滤太小的文件
    
    print(f"\n找到 {len(pdfs)} 个PDF文件")
    print(f"预计耗时: {len(pdfs) * 5 // 60} 分钟\n")
    
    all_cases = []
    success = 0
    fail = 0
    
    for i, pdf_path in enumerate(pdfs):
        case_name = pdf_path.stem.replace("_", " ")
        print(f"[{i+1}/{len(pdfs)}] {case_name}")
        
        # 检查是否已处理过
        output_file = OUTPUT_DIR / f"P{i+1:03d}_{pdf_path.stem}.json"
        if output_file.exists():
            print(f"  已处理过，跳过")
            with open(output_file, "r", encoding="utf-8") as f:
                all_cases.append(json.load(f))
            success += 1
            continue
        
        # 提取PDF文本
        print(f"  读取PDF...")
        pdf_text = extract_text_from_pdf(pdf_path)
        
        if not pdf_text or len(pdf_text) < 100:
            print(f"  PDF文本太短，跳过")
            fail += 1
            continue
        
        print(f"  提取到 {len(pdf_text)} 字符，调用AI分析...")
        
        # AI提取
        case_data = extract_case_with_ai(case_name, pdf_text)
        
        if case_data:
            case_data["case_id"] = f"P{i+1:03d}"
            case_data["source_pdf"] = pdf_path.name
            
            # 保存单个文件
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(case_data, f, ensure_ascii=False, indent=2)
            
            all_cases.append(case_data)
            success += 1
            print(f"  ✅ 提取成功: {case_data.get('basic_info', {}).get('incident_name_cn', '未知')}")
        else:
            fail += 1
            print(f"  ❌ 提取失败")
        
        time.sleep(3)  # API限流
    
    # 保存合并文件
    with open(OUTPUT_DIR / "all_cases.json", "w", encoding="utf-8") as f:
        json.dump(all_cases, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"全部完成！")
    print(f"成功: {success} 个")
    print(f"失败: {fail} 个")
    print(f"案例保存在: {OUTPUT_DIR.absolute()}")
    print(f"合并文件: {OUTPUT_DIR / 'all_cases.json'}")
    print(f"{'=' * 60}")
    
    print(f"\n下一步：将 csb_cases/all_cases.json 的内容合并到你项目的 data/cases.json 中")


if __name__ == "__main__":
    main()
