# 🚀 PyCharm 使用指南 — 手把手教你把项目跑起来

## ── 第一步：把代码放进你的 dk-sar 文件夹 ──────────────────

1. 下载 ZIP 压缩包，解压
2. 把解压出来的所有文件复制到你的 dk-sar 文件夹（C:\用户\19227\桌面\dk-sar）
3. **覆盖**原来的空文件

最终文件夹里应该有这些文件：
```
dk-sar/
├── agents/          ← 文件夹
├── utils/           ← 文件夹
├── data/            ← 文件夹
├── app.py           ← 网页界面主程序
├── main.py          ← 多智能体编排逻辑
├── config.py        ← 配置文件
├── requirements.txt ← 依赖包列表
├── .env.example     ← API Key 模板
└── README.md
```

---

## ── 第二步：在 PyCharm 中打开项目 ──────────────────────────

1. 打开 PyCharm
2. 点击菜单 **File → Open**
3. 选择你的 dk-sar 文件夹，点击 **OK**
4. 如果弹出"Trust Project?"，点击 **Trust Project**

---

## ── 第三步：配置 Python 解释器（虚拟环境）────────────────

1. 点击菜单 **File → Settings**（快捷键 Ctrl+Alt+S）
2. 左侧找到 **Project: dk-sar → Python Interpreter**
3. 点击右边的齿轮图标 ⚙️ → **Add Interpreter → Add Local Interpreter**
4. 选择 **Virtualenv Environment → New**
5. 路径选 dk-sar 文件夹下的 venv，点击 **OK**
6. 等 PyCharm 创建完虚拟环境

---

## ── 第四步：安装依赖包 ──────────────────────────────────────

**方法A（推荐）：用 PyCharm Terminal**

1. 点击底部 **Terminal** 标签（或快捷键 Alt+F12）
2. 确认路径是你的 dk-sar 文件夹
3. 输入以下命令，回车：

```
pip install -r requirements.txt
```

等待安装完成（大约需要 3-5 分钟，会有很多滚动文字，正常的）

**方法B：PyCharm 自动提示安装**

打开 requirements.txt 文件，PyCharm 会在顶部提示 "Install requirements"，直接点击即可

---

## ── 第五步：配置 API Key ────────────────────────────────────

### 申请 API Key（免费）
1. 打开浏览器，访问：https://bailian.aliyun.com
2. 用手机号注册阿里云账号
3. 进入"百炼"控制台 → API-KEY → 创建 API Key
4. 复制那串 sk-xxxxxxxxxx 的字符串

### 在 PyCharm 中配置
1. 在 dk-sar 文件夹中，右键 `.env.example` → **Copy**，然后 **Paste**
2. 把复制出来的文件重命名为 `.env`（去掉 .example）
3. 双击打开 `.env`，把内容改为：
```
DASHSCOPE_API_KEY=sk-你的真实APIKey粘贴在这里
```
4. 保存（Ctrl+S）

⚠️ 注意：.env 文件里有你的 API Key，不能上传到 GitHub！
   （项目里的 .gitignore 已经自动帮你排除了）

---

## ── 第六步：运行系统 ─────────────────────────────────────────

1. 在 PyCharm 左侧项目树中，找到 **app.py**
2. 右键 → **Run 'app'**
   或者直接点击 app.py 代码窗口右上角的绿色 ▶ 按钮

3. 底部 Run 窗口会显示启动日志：
```
DK-SAR 系统初始化
正在加载 Embedding 模型（首次会下载约300MB）...
向量索引构建完成
系统就绪 ✅
Running on http://localhost:7860
```

4. 浏览器自动打开，或手动访问 http://localhost:7860

---

## ── 第七步：上传到 GitHub Desktop ──────────────────────────

1. 打开 **GitHub Desktop**
2. 菜单 **File → Add Local Repository**
3. 选择你的 dk-sar 文件夹
4. 底部 Summary 填写：`feat: 初始化DK-SAR多智能体系统`
5. 点击 **Commit to main**
6. 点击右上角 **Publish repository**
7. 取消勾选 "Keep this code private"（设为公开），点击 **Publish Repository**

---

## ── 常见问题 ────────────────────────────────────────────────

**Q: 运行报错 "No module named 'xxx'"**
A: 在 PyCharm Terminal 运行 `pip install -r requirements.txt`

**Q: 报错 "API Key 无效"**
A: 检查 .env 文件是否存在，Key 是否正确，注意不要有空格

**Q: 首次运行很慢**
A: 正常！第一次会下载 sentence-transformers 模型（约300MB），有网就行

**Q: 找不到 .env 文件**
A: Windows 文件夹默认隐藏 . 开头的文件，在 PyCharm 里可以看到

**Q: 端口 7860 被占用**
A: 在 app.py 最后一行把 server_port=7860 改成 7861 或其他数字
