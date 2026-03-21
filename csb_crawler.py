"""
CSB事故报告批量下载 + AI结构化抽取脚本
使用方法：
1. 开VPN
2. pip install requests beautifulsoup4 dashscope
3. 在PowerShell中设置代理（端口号换成你VPN的）：
   $env:HTTP_PROXY = "http://127.0.0.1:7890"
   $env:HTTPS_PROXY = "http://127.0.0.1:7890"
4. python csb_crawler.py
"""

import os
import json
import time
import requests
from bs4 import BeautifulSoup
from pathlib import Path

# ============================================================
# 配置
# ============================================================
OUTPUT_DIR = Path("csb_reports")       # PDF保存目录
CASES_DIR = Path("csb_cases")          # 结构化JSON保存目录
OUTPUT_DIR.mkdir(exist_ok=True)
CASES_DIR.mkdir(exist_ok=True)

BASE_URL = "https://www.csb.gov"
COMPLETED_URL = f"{BASE_URL}/investigations/completed-investigations/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

# ============================================================
# 第一步：获取所有已完成调查的链接
# ============================================================
def get_investigation_links():
    """获取所有已完成调查案例的详情页链接"""
    print("正在获取CSB已完成调查列表...")
    resp = requests.get(COMPLETED_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    
    links = []
    # CSB的案例列表通常在主内容区的链接中
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # 调查案例的URL格式通常包含 /investigations/ 或特定模式
        if "/investigations/" in href and href != "/investigations/" and "completed" not in href:
            full_url = href if href.startswith("http") else BASE_URL + href
            name = a.get_text(strip=True)
            if name and len(name) > 3:  # 过滤空链接
                links.append({"name": name, "url": full_url})
    
    # 去重
    seen = set()
    unique_links = []
    for link in links:
        if link["url"] not in seen:
            seen.add(link["url"])
            unique_links.append(link)
    
    print(f"找到 {len(unique_links)} 个调查案例")
    return unique_links


def find_pdf_links(case_url):
    """从案例详情页找到PDF下载链接"""
    try:
        resp = requests.get(case_url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        
        pdf_links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.lower().endswith(".pdf"):
                full_url = href if href.startswith("http") else BASE_URL + href
                text = a.get_text(strip=True)
                # 优先选择包含"report"/"final"的PDF
                pdf_links.append({"url": full_url, "text": text})
        
        return pdf_links
    except Exception as e:
        print(f"  获取PDF链接失败: {e}")
        return []


def download_pdf(url, filename):
    """下载PDF文件"""
    filepath = OUTPUT_DIR / filename
    if filepath.exists():
        print(f"  已存在，跳过: {filename}")
        return filepath
    
    try:
        resp = requests.get(url, headers=HEADERS, timeout=60, stream=True)
        resp.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  下载成功: {filename} ({filepath.stat().st_size // 1024}KB)")
        return filepath
    except Exception as e:
        print(f"  下载失败: {e}")
        return None


# ============================================================
# 第二步：批量下载PDF
# ============================================================
def batch_download():
    """批量下载CSB调查报告PDF"""
    links = get_investigation_links()
    
    downloaded = []
    for i, case in enumerate(links):
        print(f"\n[{i+1}/{len(links)}] {case['name']}")
        
        # 找PDF链接
        pdf_links = find_pdf_links(case["url"])
        
        if not pdf_links:
            print("  未找到PDF")
            continue
        
        # 优先选择包含"report"或"final"的PDF
        best_pdf = None
        for pdf in pdf_links:
            text_lower = pdf["text"].lower()
            if any(kw in text_lower for kw in ["final report", "investigation report", "report"]):
                best_pdf = pdf
                break
        if not best_pdf:
            best_pdf = pdf_links[0]  # 取第一个
        
        # 生成文件名
        safe_name = case["name"][:80].replace("/", "-").replace("\\", "-").replace(" ", "_")
        filename = f"{safe_name}.pdf"
        
        filepath = download_pdf(best_pdf["url"], filename)
        if filepath:
            downloaded.append({
                "name": case["name"],
                "url": case["url"],
                "pdf_url": best_pdf["url"],
                "pdf_path": str(filepath)
            })
        
        time.sleep(1)  # 礼貌延迟，不要太快
    
    # 保存下载记录
    with open(OUTPUT_DIR / "download_log.json", "w", encoding="utf-8") as f:
        json.dump(downloaded, f, ensure_ascii=False, indent=2)
    
    print(f"\n下载完成！共下载 {len(downloaded)} 个PDF")
    return downloaded


# ============================================================
# 第三步：用AI从PDF中提取结构化案例
# ============================================================
def extract_case_with_ai(case_name, pdf_text_or_summary):
    """用通义千问从报告内容中提取结构化HAZOP案例"""
    
    # 使用dashscope API
    from dashscope import Generation
    
    prompt = f"""你是一位资深化工安全工程师和HAZOP分析专家。

请基于以下CSB事故调查报告的信息，提取结构化的HAZOP案例数据。

事故名称：{case_name}

报告内容摘要：
{pdf_text_or_summary[:3000]}

请输出严格的JSON格式，包含以下四级层次结构：

{{
  "case_id": "P0XX",
  "basic_info": {{
    "incident_name": "事故英文名称",
    "incident_name_cn": "事故中文名称",
    "date": "事故日期",
    "location": "地点",
    "company": "公司名",
    "fatalities": 0,
    "injuries": 0,
    "csb_report_no": "CSB报告编号"
  }},
  "process_context": {{
    "process_type": "工艺类型（反应/蒸馏/储运/换热/管道等）",
    "equipment": ["关键设备列表"],
    "materials": ["涉及物料"],
    "operating_conditions": "工况描述"
  }},
  "failure_mechanism": {{
    "root_cause": "根本原因",
    "deviation_type": "HAZOP偏差类型",
    "guide_word": "IEC 61882引导词",
    "parameter": "偏差参数（温度/压力/流量/液位/成分等）",
    "causal_chain": "因果链（原因→中间事件→后果）",
    "physical_principles": "涉及的物理原理"
  }},
  "protection_system": {{
    "existing_safeguards": ["事故前已有的保护措施"],
    "safeguard_failures": "保护措施为什么失效了",
    "csb_recommendations": ["CSB的改进建议"]
  }},
  "embedding_text": "用一段100词左右的英文概括事故的核心失效机理和HAZOP相关信息，用于语义检索。重点包含设备类型、物料、偏差类型、根因和后果。"
}}

只输出JSON，不要输出其他内容。"""

    try:
        response = Generation.call(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            result_format="message"
        )
        
        result = response.output.choices[0].message.content
        # 清理JSON
        result = result.strip()
        if result.startswith("```json"):
            result = result[7:]
        if result.startswith("```"):
            result = result[3:]
        if result.endswith("```"):
            result = result[:-3]
        
        case_data = json.loads(result.strip())
        return case_data
    
    except Exception as e:
        print(f"  AI抽取失败: {e}")
        return None


def batch_extract_without_pdf():
    """
    不需要下载PDF！直接基于案例名称让AI生成结构化案例。
    因为CSB案例都是公开的知名事故，AI训练数据中包含这些信息。
    """
    
    # 50个最典型的CSB案例
    cases = [
        # 反应类
        "T2 Laboratories Inc. Reactive Chemical Explosion (2007)",
        "Morton International Inc. Runaway Chemical Reaction (1998)",
        "Bayer CropScience Pesticide Waste Tank Explosion (2008)",
        "Synthron Chemical Explosion (2006)",
        "First Chemical Corp. Reactive Chemical Explosion (2002)",
        "Concept Sciences Hydroxylamine Explosion (1999)",
        "KMCO LLC Fatal Fire and Explosion (2019)",
        "Catalyst Systems Reactive Chemical Explosion (2002)",
        "BP Amoco Thermal Decomposition Incident (2001)",
        "Formosa Plastics Propylene Explosion (2005)",
        "Formosa Plastics Vinyl Chloride Explosion (2004)",
        "Sterigenics Ethylene Oxide Explosion (2004)",
        "Dow Louisiana Operations Explosions (2023)",
        "Shell Polymers Monaca Chemical Release (2023)",
        "Bio-Lab Lake Charles Chemical Fire (2020)",
        
        # 储运类
        "Caribbean Petroleum (CAPECO) Refinery Tank Explosion and Fire (2009)",
        "Imperial Sugar Company Dust Explosion and Fire (2008)",
        "Intercontinental Terminals Company (ITC) Tank Fire (2019)",
        "Marcus Oil and Chemical Tank Explosion (2003)",
        "Motiva Enterprises Sulfuric Acid Tank Explosion (2001)",
        "Barton Solvents Explosions and Fire (2007)",
        "Allied Terminals Fertilizer Tank Collapse (2008)",
        "Packaging Corporation Storage Tank Explosion (2002)",
        "West Fertilizer Explosion and Fire (2013)",
        "Hoeganaes Corporation Fatal Flash Fires (2011)",
        
        # 炼化类
        "BP America Texas City Refinery Explosion (2005)",
        "Chevron Richmond Refinery Fire (2012)",
        "Tesoro Anacortes Refinery Fatal Explosion and Fire (2010)",
        "ExxonMobil Torrance Refinery Explosion (2015)",
        "Philadelphia Energy Solutions Refinery Fire and Explosions (2019)",
        "Husky Energy Superior Refinery Explosion and Fire (2018)",
        "Valero McKee Refinery Propane Fire (2007)",
        "Giant Industries Refinery Explosions and Fire (2001)",
        "Williams Olefins Plant Explosion and Fire (2013)",
        "Silver Eagle Refinery Flash Fire and Explosion (2009)",
        "TPC Port Neches Explosions and Fire (2019)",
        "Tosco Avon Refinery Petroleum Naphtha Fire (1999)",
        "CITGO Refinery Hydrofluoric Acid Release (2009)",
        "Marathon Martinez Renewable Fuels Fire (2022)",
        "ExxonMobil Baton Rouge Chemical Release and Fire (2016)",
        
        # 有毒气体/窒息类
        "Union Carbide Corp. Nitrogen Asphyxiation Incident (1998)",
        "Valero Delaware City Refinery Asphyxiation Incident (2005)",
        "DuPont La Porte Facility Toxic Chemical Release (2014)",
        "DuPont Belle Toxic Chemical Releases (2010)",
        "Foundation Food Group Fatal Chemical Release (2021)",
        "Georgia-Pacific Corp. Hydrogen Sulfide Poisoning (2002)",
        "Aghorn Operating H2S Release (2019)",
        "DPC Enterprises Festus Chlorine Release (2002)",
        "MFG Chemical Inc. Toxic Gas Release (2004)",
        "MGPI Processing Toxic Chemical Release (2016)",
        
        # 热工作/维修/其他
        "E.I. DuPont De Nemours Fatal Hotwork Explosion (2010)",
        "Kleen Energy Natural Gas Explosion (2010)",
        "Packaging Corporation Hot Work Explosion (2011)",
        "Evergreen Packaging Paper Mill Fire During Hot Work (2017)",
        "Praxair Flammable Gas Cylinder Fire (2005)",
        "Goodyear Heat Exchanger Rupture (2008)",
        "D.D. Williamson Catastrophic Vessel Failure (2003)",
        "Loy Lange Box Company Pressure Vessel Explosion (2017)",
        "Sonat Exploration Catastrophic Vessel Overpressurization (1998)",
        "Carbide Industries Fire and Explosion (2003)",
    ]
    
    print(f"准备生成 {len(cases)} 个结构化案例...")
    print("使用通义千问API基于公开信息生成（不需要下载PDF）\n")
    
    all_cases = []
    for i, case_name in enumerate(cases):
        print(f"[{i+1}/{len(cases)}] {case_name}")
        
        case_data = extract_case_with_ai(case_name, f"这是美国CSB公开调查的真实化工事故案例：{case_name}。请基于你对这个事故的了解提取结构化信息。")
        
        if case_data:
            case_data["case_id"] = f"P{i+1:03d}"
            all_cases.append(case_data)
            
            # 保存单个文件
            with open(CASES_DIR / f"P{i+1:03d}.json", "w", encoding="utf-8") as f:
                json.dump(case_data, f, ensure_ascii=False, indent=2)
            
            print(f"  ✅ 生成成功")
        else:
            print(f"  ❌ 生成失败")
        
        time.sleep(2)  # API限流
    
    # 保存合并文件
    with open(CASES_DIR / "all_cases.json", "w", encoding="utf-8") as f:
        json.dump(all_cases, f, ensure_ascii=False, indent=2)
    
    print(f"\n完成！共生成 {len(all_cases)} 个结构化案例")
    print(f"保存在 {CASES_DIR} 目录下")


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  CSB事故报告采集 & 结构化案例生成")
    print("=" * 60)
    print()
    print("请选择模式：")
    print("  1. 直接用AI生成结构化案例（推荐，快速，不需要下载PDF）")
    print("  2. 先下载PDF，再用AI抽取（更准确，但需要VPN和更多时间）")
    print()
    
    choice = input("请输入 1 或 2：").strip()
    
    if choice == "2":
        print("\n--- 模式2：下载PDF ---")
        downloaded = batch_download()
        print(f"\nPDF已下载到 {OUTPUT_DIR} 目录")
        print("接下来你可以用AI逐个处理这些PDF")
    else:
        print("\n--- 模式1：AI直接生成 ---")
        # 检查API Key
        if not os.getenv("DASHSCOPE_API_KEY"):
            api_key = input("请输入你的DASHSCOPE_API_KEY：").strip()
            os.environ["DASHSCOPE_API_KEY"] = api_key
        
        batch_extract_without_pdf()
