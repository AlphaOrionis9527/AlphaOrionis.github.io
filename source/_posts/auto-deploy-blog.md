---
title: 如何让 AI 助手帮你自动发布博客
date: 2026-02-14 17:15:00
tags:
  - 技术教程
  - Hexo
  - GitHub
  - 自动化
categories:
  - 技术手册
---

你是否想过让你的 AI 助手帮你管理博客？本文将介绍如何实现这一点。

## 背景

我的博客使用 Hexo 框架，托管在 GitHub Pages，域名是 alphaorionis.top。之前每次发布文章都需要手动执行一堆命令，非常麻烦。

## 实现方案

### 1. 准备 GitHub Token

在 GitHub 设置中生成一个 Personal Access Token，给予 `repo` 权限。

### 2. 博客结构

博客使用双分支策略：
- `main` 分支：存放生成的静态 HTML 文件
- `source` 分支：存放 Hexo 源文件（文章、配置等）

### 3. GitHub Actions 自动部署

创建 `.github/workflows/deploy.yml`，当 source 分支有推送时，自动生成并部署到 main 分支。

### 4. 手动部署命令

由于自动部署配置较为复杂，也可以手动部署：

```bash
cd ~/my-blog
hexo clean && hexo generate
cd public
git add .
git commit -m "Deploy $(date)"
git push -f origin main
```

## 总结

通过这种方式，AI 助手可以直接通过 GitHub API 创建文章、提交代码，实现博客的自动化管理。

---

*本文由 AI 助手协助编写并自动发布*
