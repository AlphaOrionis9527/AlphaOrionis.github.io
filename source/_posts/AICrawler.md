---
title: AI爬虫工作流-使RAG内容及时更新补充
tags: Technical Manual
---
# AI爬虫工作流

根据网站内容自动生成几个QA，自动以日期格式保存为Markdown格式。用于扩充RAG或微调资料。

以下是预览内容，从我的思维导图文件中转换而来。具体流程设计参考Xmind思维导图。对应的工作流可以从思维导图文件中橙色标记位置找到并另存到本地去掉'.bak'后使用。
#### 工作流思维导图下载
下载地址: [百度网盘](https://pan.baidu.com/s/1D0AANztCE5e8JLplRchN9w) 提取码: wmb3

## 前置条件

### Docker

### DeepSeek官网获取API-KEY，本身支持FunctionCall功能

### 暂时未支持OpenAI-Compatible

### LLM支持FunctionCall功能

### DeepSeek官网API


## n8n工作流本地部署

### 复制命令到cmd（Windows）或zsh（macOS）

## crawl4ai本地部署

### 复制命令到cmd或zsh


## 工作流Crawl2md

### 导入工作流

### AI环节涉及的DeepSeek提示词已填好

### 初始爬取限制2任务可修改


## 实现作用

### 爬取互联网内容到本地

### 转换成Markdown文件

### 形成QA问答内容

### 投喂本地RAG知识库

### 文件名已优化为当前日期格式，以免过长无法正常获取文件

### 实现RAG知识内容不断更新，避免过时

