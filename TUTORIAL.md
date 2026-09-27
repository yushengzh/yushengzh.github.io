# 个人主页编辑与发布教程

新主页只有三个文件是真正需要改的：`index.html`（内容）、`styles.css`（样式）、
`assets/`（放 CV、图片、PDF）。**没有构建步骤，没有依赖**，改完 push 就上线。

- 仓库本地位置：`/home/yusheng.zhao/wmlm/yushengzh.github.io`
- 线上地址：<https://yushengzh.github.io>

---

## 1. 目录结构

```
yushengzh.github.io/
├── index.html              ← 页面内容，平时主要改这个
├── styles.css              ← 配色 / 字号 / 栏宽，想换风格改这个
├── favicon.svg             ← 浏览器标签页小图标
├── assets/
│   └── cv.pdf              ← 导航栏 "CV" 指向的文件，目前是 2022 年旧简历，请替换
├── archive/2022-jemdoc/    ← 旧主页整体归档（旧简历、成绩单、课程笔记、论文 PDF）
│   └── README.md
├── README.md
├── TUTORIAL.md             ← 本文件
└── .nojekyll               ← 告诉 GitHub Pages 跳过额外处理，不要删
```

---

## 2. 本地预览

**方式 A（最快）**：直接把 `index.html` 拖进浏览器。

**方式 B（路径和线上完全一致）**：在仓库目录启动一个本地服务器：

```bash
cd /home/yusheng.zhao/wmlm/yushengzh.github.io
python3 -m http.server 8000
```

然后浏览器打开 <http://localhost:8000>。

如果是在集群上操作（没有图形界面），在自己电脑的终端里加一条 SSH 端口转发：

```bash
ssh -L 8000:localhost:8000 <你的用户名>@<集群地址>
```

连上之后在服务器上跑上面的 `python3 -m http.server 8000`，本地浏览器访问
<http://localhost:8000> 就能看到。改完 HTML/CSS 直接刷新页面，必要时
`Ctrl/Cmd + Shift + R` 强制刷新。

**方式 C**：VS Code 装 Live Server 插件，右键 `index.html` → *Open with Live Server*，
保存即自动刷新。

---

## 3. 怎么改内容（`index.html`）

页面从上到下是：导航栏 → 自我介绍（hero）→ Research → Selected work → Notes →
Contact → 页脚。每块前面都有 `<!-- 注释 -->` 说明，注释不会被显示出来。

### 3.1 改自我介绍

找到 `<p class="lede">` 那一段，直接改文字。建议保持 2–3 句。链接写法是
`<a href="网址">显示文字</a>`。

### 3.2 加 / 删一条研究兴趣

`<section id="research">` 里的每个 `<li> ... </li>` 是一条。复制一整块就能新增一条，
删掉整块就少一条。格式约定是「加粗短语 + 一句话」：

```html
      <li>
        <strong>短语标题。</strong>
        一句话说明。
      </li>
```

### 3.3 加一篇论文

`<section id="work">` 里，每个 `<li>` 是一篇。复制这个模板改内容：

```html
      <li>
        <a class="title" href="https://arxiv.org/abs/XXXX.XXXXX">论文标题</a>
        <p class="meta">Yusheng Zhao, 合作者, <em>会议/期刊名</em>, 年份. <a href="链接">[PDF]</a> <a href="链接">[Code]</a></p>
      </li>
```

几点提示：

- `href` 可以直接填 arXiv / DOI 外部链接；如果 PDF 放在仓库里，就填相对路径，
  例如把文件放到 `assets/paper.pdf`，然后 `href="assets/paper.pdf"`。
- `[PDF]`、`[Code]` 是惯例写法，也可以换成 `PDF`、`Code`。
- 不想显示某个链接就删掉对应的 `<a>...</a>`。

### 3.4 Notes（课程 / 阅读笔记）

目前 Notes 指向归档里的旧笔记页，路径形如
`archive/2022-jemdoc/assets/records/ML_notes.html`。以后自己写的笔记，建议
新建 `notes/` 目录，把 HTML 或 PDF 放进去，然后在这里加一条：

```html
      <li><a href="notes/文件名.html">标题</a> — 一句话说明。</li>
```

### 3.5 联系方式

在 `<section id="contact">` 里，用 `·` 分隔若干链接。不需要的链接直接删掉
（记得连前后的 `·` 一起删）。邮箱要改文字和 `mailto:` 两处。

### 3.6 加头像照片（可选，默认不放照片，更接近极简风格）

1. 把照片压到 300 KB 以内，命名成 `photo.jpg`，放进 `assets/`。
2. 在 `<main class="wrap">` 里、`<p class="lede">` 之前插入一行：

```html
  <img class="portrait" src="assets/photo.jpg" alt="Yusheng Zhao" width="96" height="96" />
```

3. 在 `styles.css` 末尾追加：

```css
.portrait {
  width: 96px;
  height: 96px;
  border-radius: 50%;
  object-fit: cover;
  display: block;
  margin-bottom: 1.75rem;
}
```

### 3.7 改标题与描述（影响 Google 结果和分享卡片）

`index.html` 顶部 `<head>` 里：

- `<title>Yusheng Zhao</title>`：浏览器标签页标题。
- `<meta name="description" ...>`：搜索引擎摘要。
- `og:title` / `og:description`：微信、Twitter 等分享时的标题和描述。

### 3.8 换配色 / 栏宽 / 字号

只改 `styles.css` 开头的变量就能整体换风格：

```css
:root {
  --bg: #ffffff;      /* 背景色 */
  --fg: #121212;      /* 正文色 */
  --muted: #6d6d6d;   /* 次级文字（小节标题、注释性文字） */
  --rule: #e6e6e6;    /* 分隔线 */
  --width: 41rem;     /* 内容栏宽，41rem 约 656px；想更宽改成 46rem */
  --sans: ...;        /* 字体栈，想换衬线体可加入 Georgia, "Songti SC" */
}
```

`@media (prefers-color-scheme: dark)` 那一段是深色模式，跟随系统设置自动切换。
浏览器里想预览深色效果：开发者工具 → 渲染 → 模拟 CSS 媒体特性 →
prefers-color-scheme: dark。

---

## 4. 发布

### 方式一：命令行（推荐，在这台机器上用）

```bash
cd /home/yusheng.zhao/wmlm/yushengzh.github.io

git status                 # 先看看自己改了什么
git add -A
git commit -m "update homepage"
git push
```

`git push` 之后就自动上线了。如果提示需要登录，执行 `gh auth login` 走一遍浏览器授权。

### 方式二：直接在 GitHub 网页上改

打开 <https://github.com/yushengzh/yushengzh.github.io>，点进文件 → 右上角铅笔图标 →
编辑 → 页面底部 **Commit changes**。适合改错别字、换个链接这类小改动。
注意网页编辑器不会帮你检查 HTML，改完最好再看一眼页面。

---

## 5. 上线是怎么发生的

1. `git push` 到 `master` 分支。
2. GitHub Pages 自动触发构建，仓库 **Actions** 标签页里能看到 `pages-build-deployment`。
3. 一般 30–60 秒后生效，访问 <https://yushengzh.github.io>。
4. 构建状态也可以在这里看：
   <https://github.com/yushengzh/yushengzh.github.io/deployments>

Pages 设置在仓库 **Settings → Pages**：

- **Build and deployment → Source** 选 *Deploy from a branch*，分支 `master`、
  目录 `/ (root)`。
- 关键约束：**私有仓库开启 Pages 需要 GitHub Pro**。GitHub Free 账号下仓库必须
  是 public 才能发布。如果 Settings → Pages 报错或无法启用，就是这个原因——
  要么把仓库设为 public（个人主页本来也适合公开），要么升级账号。

---

## 6. 反悔 / 恢复

- 撤销某次改动：`git revert <commit>`，或者直接在网页上把文字改回来再 commit。
- 想恢复旧版主页：把 `archive/2022-jemdoc/` 里的文件移回仓库根目录再 push，
  细节见 `archive/README.md`。
- 所有历史提交都在 git 里，任何一版都能找回来，这次改版不会让旧内容丢失。

---

## 7. 常见问题

| 现象 | 原因 / 处理 |
| --- | --- |
| 改完线上没变化 | 先强制刷新（`Ctrl/Cmd + Shift + R`），再等 1 分钟看 Actions 是否构建完成 |
| 图片 / 链接 404 | Linux 区分大小写，`Photo.JPG` 不等于 `photo.jpg`；路径要和文件实际位置一致 |
| 中文显示乱码 | 文件必须以 UTF-8 保存；`index.html` 已声明 `<meta charset="utf-8">` |
| 页面加载慢 | 多半是放了原图或超大 PDF，图片压到 300 KB 以内再上传 |
| 不想公开邮箱 | 删掉 mailto 只留 GitHub，或把邮箱写成 `yszhao0717 [at] gmail.com` |

---

## 8. 以后想扩展（可选）

- **自定义域名**：买域名后在 Settings → Pages → Custom domain 填上，
  再到 DNS 服务商加一条指向 `yushengzh.github.io` 的 CNAME 记录。
- **写博客**：新建 `posts/` 目录手写 HTML 页面即可，保持零依赖；
  文章多了再考虑 Hugo / Astro 之类的静态站点生成器，代价是引入构建步骤。
- **访问统计**：GoatCounter 或 Cloudflare Web Analytics，加一行 `<script>` 就行。
- **中文版页面**：复制一份 `index.html` 改名 `index-cn.html`，两页互相加语言切换链接。
