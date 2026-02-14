---
title: AI终端-Powershell-MCP-Server
tags: Technical Manual
---

以下是针对 Windows11 系统实践场景的操作。
macOS通过PD虚拟机安装Windows即可轻松实现，将操作放到虚拟环境能够防止一些AI误操作风险。

## 一、功能：
### 以下是PowerShell在办公场景中的实用应用列表：

``` matlab
📁 文件管理自动化

批量重命名文件 - 统一命名规范，如"合同_2025_001.pdf"
文件分类整理 - 按日期、类型自动分文件夹
查找重复文件 - 清理重复的文档和图片
文件夹结构创建 - 快速创建项目文件夹模板
大文件查找 - 找出占用空间大的文件进行清理

📊 数据处理和报表

Excel数据批量处理 - 合并多个Excel文件
CSV文件转换 - 格式转换和数据清洗
文件统计报告 - 生成文件数量、大小统计
日志分析 - 提取关键信息生成报告
数据备份验证 - 检查备份文件完整性

📧 办公自动化

邮件附件批量下载 - 从Outlook提取附件
文档格式批量转换 - Word转PDF、图片格式转换
打印任务管理 - 批量打印文档
快捷方式批量创建 - 为常用文件夹创建桌面快捷方式
软件安装检查 - 检查办公软件版本和更新

🔍 信息收集和监控

网络连接监控 - 检查办公网络状态
系统性能监控 - 监控电脑运行状态
磁盘空间警报 - 空间不足时自动提醒
文件变更监控 - 监控重要文件夹的变化
设备状态检查 - 打印机、网络设备状态

📅 时间和任务管理

定时提醒 - 会议、截止日期提醒
工作日志生成 - 自动记录每日文件操作
任务计划执行 - 定时备份、清理等操作
考勤辅助 - 记录开关机时间
项目进度跟踪 - 文件修改时间统计

🛠️ 系统维护

临时文件清理 - 定期清理系统垃圾
软件卸载 - 批量卸载不需要的程序
注册表清理 - 清理无效注册表项
启动项管理 - 优化开机启动速度
系统信息收集 - 生成电脑配置报告

📋 文档和内容处理

文档内容搜索 - 在多个文档中搜索关键词
图片批处理 - 压缩、调整尺寸、水印添加
文本提取 - 从PDF、图片中提取文字
模板应用 - 批量应用文档模板
版本控制 - 文档版本管理和比较

🔐 安全和合规

文件权限管理 - 批量设置文件访问权限
敏感信息检查 - 扫描文档中的敏感数据
密码生成 - 生成安全密码
文件加密备份 - 重要文件的安全备份
访问日志分析 - 分析文件访问记录

📈 效率提升工具

快速启动菜单 - 创建常用程序启动脚本
工作环境配置 - 一键配置工作桌面布局
批量操作模板 - 保存常用操作为脚本
数据验证 - 批量验证数据格式和完整性
报告自动生成 - 定期生成各类统计报告

这些场景都能显著提高办公效率，减少重复性工作，让文员可以专注于更有价值的任务。你对哪个场景最感兴趣？我可以详细展示具体的实现方法。
```
---

## 二、场景：
### 通过PowerShell工具，你可以做很多有趣和实用的事情！以下是一些有趣的应用场景：

``` matlab
通过PowerShell工具，你可以做很多有趣和实用的事情！以下是一些有趣的应用场景：
系统管理和监控

实时监控CPU、内存、磁盘使用情况
查看和管理正在运行的进程和服务
监控网络连接状态和流量
检查系统启动时间和运行状态
管理Windows防火墙规则

文件和数据处理

批量重命名文件和文件夹
查找重复文件并清理磁盘空间
批量处理图片、文档等文件属性
创建文件备份和同步脚本
分析文件夹大小和文件分布统计

网络和安全

扫描本地网络中的设备
检查开放端口和网络服务
监控网络延迟和连接质量
获取WiFi密码和网络配置
检查系统安全更新状态

自动化任务

定时清理临时文件和缓存
自动化软件安装和配置
批量创建用户账户和权限设置
自动备份重要文件到指定位置
定时发送系统状态报告

趣味功能

创建ASCII艺术和文字画
生成随机密码和UUID
播放系统声音或文字转语音
控制系统音量和电源设置
创建桌面壁纸幻灯片

开发辅助

批量处理代码文件格式化
检查端口占用和服务状态
管理环境变量和路径设置
快速搭建本地测试环境
监控文件变化并触发操作

数据分析

分析日志文件和错误报告
统计文件类型和大小分布
监控系统性能趋势
生成各种格式的报告
处理CSV和JSON数据

这些都可以通过PowerShell实现，既实用又有趣！你对哪个方面比较感兴趣？我可以帮你实际演示一些功能。
```
---

## 三、代码：
### 代码文件，用于复制后在VScode或Pycharm中新建Python文件powershell.py后粘贴：

``` bash
import subprocess
import json
import sys
import os
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("PowerShell Toolkit")

@mcp.tool()
def powershell_command(command: str, timeout: int = 30, encoding: str = 'utf-8'):
    """
    Execute PowerShell command and return the result.
    
    Args:
        command (str): PowerShell command to execute
        timeout (int): Command timeout in seconds (default: 30)
        encoding (str): Output encoding (default: utf-8)
        
    Returns:
        dict: Contains success status, output, and error information
    """
    try:
        # 检测系统和PowerShell版本
        powershell_executable = _get_powershell_executable()
        
        # 构建PowerShell命令，设置输出编码
        if sys.platform == "win32":
            # Windows系统，解决中文编码问题
            powershell_cmd = [
                powershell_executable,
                '-NoProfile',
                '-OutputFormat', 'Text',
                '-Command',
                f'[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; {command}'
            ]
        else:
            # 非Windows系统
            powershell_cmd = [
                powershell_executable,
                '-NoProfile',
                '-Command',
                command
            ]
        
        # 设置环境变量以支持UTF-8
        env = os.environ.copy()
        if sys.platform == "win32":
            env['PYTHONIOENCODING'] = 'utf-8'
        
        # 执行命令
        result = subprocess.run(
            powershell_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding=encoding,
            errors='replace',  # 处理编码错误
            env=env
        )
        
        # 构建返回结果
        response = {
            "success": result.returncode == 0,
            "return_code": result.returncode,
            "stdout": result.stdout.strip() if result.stdout else "",
            "stderr": result.stderr.strip() if result.stderr else "",
            "command": command,
            "powershell_version": powershell_executable
        }
        
        # 格式化输出
        if response["success"]:
            if response["stdout"]:
                return f"✅ Command executed successfully:\n\n{response['stdout']}"
            else:
                return "✅ Command executed successfully (no output)"
        else:
            error_msg = response["stderr"] if response["stderr"] else "Unknown error"
            return f"❌ Command failed (Exit Code: {response['return_code']}):\n\n{error_msg}"
            
    except subprocess.TimeoutExpired:
        return f"⏰ Command timed out after {timeout} seconds"
    except FileNotFoundError:
        return f"❌ PowerShell not found. Please ensure PowerShell is installed and in PATH.\nTried: {powershell_executable}"
    except Exception as e:
        return f"❌ Unexpected error: {str(e)}"

def _get_powershell_executable():
    """
    检测可用的PowerShell可执行文件
    """
    if sys.platform == "win32":
        # Windows系统，优先使用PowerShell 7，然后是Windows PowerShell
        candidates = [
            'pwsh',  # PowerShell 7+
            'powershell'  # Windows PowerShell 5.x
        ]
    else:
        # Linux/macOS系统
        candidates = [
            'pwsh',  # PowerShell Core
            'powershell'
        ]
    
    for candidate in candidates:
        try:
            subprocess.run([candidate, '-Version'], 
                         capture_output=True, 
                         timeout=5)
            return candidate
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    
    # 如果都找不到，返回默认值
    return 'powershell' if sys.platform == "win32" else 'pwsh'

@mcp.tool()
def get_powershell_info():
    """
    获取PowerShell版本和系统信息
    
    Returns:
        dict: PowerShell和系统信息
    """
    try:
        powershell_executable = _get_powershell_executable()
        
        # 获取PowerShell版本信息
        version_cmd = [
            powershell_executable,
            '-NoProfile',
            '-Command',
            '$PSVersionTable | ConvertTo-Json'
        ]
        
        result = subprocess.run(
            version_cmd,
            capture_output=True,
            text=True,
            timeout=10,
            encoding='utf-8',
            errors='replace'
        )
        
        if result.returncode == 0:
            try:
                version_info = json.loads(result.stdout)
                return f"✅ PowerShell Information:\n\n{json.dumps(version_info, indent=2, ensure_ascii=False)}"
            except json.JSONDecodeError:
                return f"✅ PowerShell Version (raw):\n\n{result.stdout}"
        else:
            return f"❌ Failed to get PowerShell version: {result.stderr}"
            
    except Exception as e:
        return f"❌ Error getting PowerShell info: {str(e)}"

if __name__ == "__main__":
    mcp.run()
    
```

---

## 四、环境：
### 安装好Python环境，建议3.12或以上。窗口键+R，打开cmd命令输入：
``` bash
pip install mcp[cli]
```
安装好MCP的这个SDK工具后会出现以下返回值：
``` matlab
C:\Users\alphaorionis>pip install "mcp[cli]"


#下面是返回值：

Defaulting to user installation because normal site-packages is not writeable
Collecting mcp[cli]
  Downloading mcp-1.6.0-py3-none-any.whl.metadata (20 kB)
Collecting anyio>=4.5 (from mcp[cli])
  Downloading anyio-4.9.0-py3-none-any.whl.metadata (4.7 kB)
Collecting httpx-sse>=0.4 (from mcp[cli])
  Using cached httpx_sse-0.4.0-py3-none-any.whl.metadata (9.0 kB)
Collecting httpx>=0.27 (from mcp[cli])
  Using cached httpx-0.28.1-py3-none-any.whl.metadata (7.1 kB)
Collecting pydantic-settings>=2.5.2 (from mcp[cli])
  Downloading pydantic_settings-2.9.1-py3-none-any.whl.metadata (3.8 kB)
Collecting pydantic<3.0.0,>=2.7.2 (from mcp[cli])
  Downloading pydantic-2.11.3-py3-none-any.whl.metadata (65 kB)
Collecting sse-starlette>=1.6.1 (from mcp[cli])
  Downloading sse_starlette-2.2.1-py3-none-any.whl.metadata (7.8 kB)
Collecting starlette>=0.27 (from mcp[cli])
  Downloading starlette-0.46.2-py3-none-any.whl.metadata (6.2 kB)
Collecting uvicorn>=0.23.1 (from mcp[cli])
  Downloading uvicorn-0.34.2-py3-none-any.whl.metadata (6.5 kB)
Collecting python-dotenv>=1.0.0 (from mcp[cli])
  Downloading python_dotenv-1.1.0-py3-none-any.whl.metadata (24 kB)
Collecting typer>=0.12.4 (from mcp[cli])
  Downloading typer-0.15.2-py3-none-any.whl.metadata (15 kB)
Requirement already satisfied: idna>=2.8 in c:\users\alphaorionis\appdata\local\packages\pythonsoftwarefoundation.python.3.13_qbz5n2kfra8p0\localcache\local-packages\python313\site-packages (from anyio>=4.5->mcp[cli]) (3.10)
Requirement already satisfied: sniffio>=1.1 in c:\users\alphaorionis\appdata\local\packages\pythonsoftwarefoundation.python.3.13_qbz5n2kfra8p0\localcache\local-packages\python313\site-packages (from anyio>=4.5->mcp[cli]) (1.3.1)
Requirement already satisfied: certifi in c:\users\alphaorionis\appdata\local\packages\pythonsoftwarefoundation.python.3.13_qbz5n2kfra8p0\localcache\local-packages\python313\site-packages (from httpx>=0.27->mcp[cli]) (2024.12.14)
Collecting httpcore==1.* (from httpx>=0.27->mcp[cli])
  Downloading httpcore-1.0.8-py3-none-any.whl.metadata (21 kB)
Requirement already satisfied: h11<0.15,>=0.13 in c:\users\alphaorionis\appdata\local\packages\pythonsoftwarefoundation.python.3.13_qbz5n2kfra8p0\localcache\local-packages\python313\site-packages (from httpcore==1.*->httpx>=0.27->mcp[cli]) (0.14.0)
Collecting annotated-types>=0.6.0 (from pydantic<3.0.0,>=2.7.2->mcp[cli])
  Using cached annotated_types-0.7.0-py3-none-any.whl.metadata (15 kB)
Collecting pydantic-core==2.33.1 (from pydantic<3.0.0,>=2.7.2->mcp[cli])
  Downloading pydantic_core-2.33.1-cp313-cp313-win_amd64.whl.metadata (6.9 kB)
Requirement already satisfied: typing-extensions>=4.12.2 in c:\users\alphaorionis\appdata\local\packages\pythonsoftwarefoundation.python.3.13_qbz5n2kfra8p0\localcache\local-packages\python313\site-packages (from pydantic<3.0.0,>=2.7.2->mcp[cli]) (4.12.2)
Collecting typing-inspection>=0.4.0 (from pydantic<3.0.0,>=2.7.2->mcp[cli])
  Downloading typing_inspection-0.4.0-py3-none-any.whl.metadata (2.6 kB)
Collecting click>=8.0.0 (from typer>=0.12.4->mcp[cli])
  Using cached click-8.1.8-py3-none-any.whl.metadata (2.3 kB)
Collecting shellingham>=1.3.0 (from typer>=0.12.4->mcp[cli])
  Using cached shellingham-1.5.4-py2.py3-none-any.whl.metadata (3.5 kB)
Collecting rich>=10.11.0 (from typer>=0.12.4->mcp[cli])
  Downloading rich-14.0.0-py3-none-any.whl.metadata (18 kB)
Collecting colorama (from click>=8.0.0->typer>=0.12.4->mcp[cli])
  Using cached colorama-0.4.6-py2.py3-none-any.whl.metadata (17 kB)
Collecting markdown-it-py>=2.2.0 (from rich>=10.11.0->typer>=0.12.4->mcp[cli])
  Using cached markdown_it_py-3.0.0-py3-none-any.whl.metadata (6.9 kB)
Collecting pygments<3.0.0,>=2.13.0 (from rich>=10.11.0->typer>=0.12.4->mcp[cli])
  Using cached pygments-2.19.1-py3-none-any.whl.metadata (2.5 kB)
Collecting mdurl~=0.1 (from markdown-it-py>=2.2.0->rich>=10.11.0->typer>=0.12.4->mcp[cli])
  Using cached mdurl-0.1.2-py3-none-any.whl.metadata (1.6 kB)
Downloading anyio-4.9.0-py3-none-any.whl (100 kB)
Using cached httpx-0.28.1-py3-none-any.whl (73 kB)
Downloading httpcore-1.0.8-py3-none-any.whl (78 kB)
Using cached httpx_sse-0.4.0-py3-none-any.whl (7.8 kB)
Downloading pydantic-2.11.3-py3-none-any.whl (443 kB)
Downloading pydantic_core-2.33.1-cp313-cp313-win_amd64.whl (2.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.0/2.0 MB 21.2 MB/s eta 0:00:00
Downloading pydantic_settings-2.9.1-py3-none-any.whl (44 kB)
Downloading python_dotenv-1.1.0-py3-none-any.whl (20 kB)
Downloading sse_starlette-2.2.1-py3-none-any.whl (10 kB)
Downloading starlette-0.46.2-py3-none-any.whl (72 kB)
Downloading typer-0.15.2-py3-none-any.whl (45 kB)
Downloading uvicorn-0.34.2-py3-none-any.whl (62 kB)
Downloading mcp-1.6.0-py3-none-any.whl (76 kB)
Using cached annotated_types-0.7.0-py3-none-any.whl (13 kB)
Using cached click-8.1.8-py3-none-any.whl (98 kB)
Downloading rich-14.0.0-py3-none-any.whl (243 kB)
Using cached shellingham-1.5.4-py2.py3-none-any.whl (9.8 kB)
Downloading typing_inspection-0.4.0-py3-none-any.whl (14 kB)
Using cached markdown_it_py-3.0.0-py3-none-any.whl (87 kB)
Using cached pygments-2.19.1-py3-none-any.whl (1.2 MB)
Using cached colorama-0.4.6-py2.py3-none-any.whl (25 kB)
Using cached mdurl-0.1.2-py3-none-any.whl (10.0 kB)
Installing collected packages: typing-inspection, shellingham, python-dotenv, pygments, pydantic-core, mdurl, httpx-sse, httpcore, colorama, anyio, annotated-types, starlette, pydantic, markdown-it-py, httpx, click, uvicorn, sse-starlette, rich, pydantic-settings, typer, mcp
  WARNING: The script dotenv.exe is installed in 'C:\Users\alphaorionis\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\Scripts' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The script pygmentize.exe is installed in 'C:\Users\alphaorionis\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\Scripts' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The script markdown-it.exe is installed in 'C:\Users\alphaorionis\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\Scripts' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The script httpx.exe is installed in 'C:\Users\alphaorionis\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\Scripts' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The script uvicorn.exe is installed in 'C:\Users\alphaorionis\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\Scripts' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The script typer.exe is installed in 'C:\Users\alphaorionis\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\Scripts' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The script mcp.exe is installed in 'C:\Users\alphaorionis\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\Scripts' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
Successfully installed annotated-types-0.7.0 anyio-4.9.0 click-8.1.8 colorama-0.4.6 httpcore-1.0.8 httpx-0.28.1 httpx-sse-0.4.0 markdown-it-py-3.0.0 mcp-1.6.0 mdurl-0.1.2 pydantic-2.11.3 pydantic-core-2.33.1 pydantic-settings-2.9.1 pygments-2.19.1 python-dotenv-1.1.0 rich-14.0.0 shellingham-1.5.4 sse-starlette-2.2.1 starlette-0.46.2 typer-0.15.2 typing-inspection-0.4.0 uvicorn-0.34.2

C:\Users\alphaorionis>

```

返回以上信息，代表安装成功。

配置好后，此时需要新开一个CMD，输入MCP验证是否生效。

代码环境搭建完成。

## 五、配置json文件：
### 配置json文件内容如下：
注意修改文件路径。
可以通过Claude调用，也可以用cursor，或者VScode+Cline，或者CherryStudio应用。

``` bash

{
  "mcpServers": {
    "powershell": {
      "command": "python",
      "args": [
        "C:\\Users\\root\\Desktop\\MCP\\powershell.py"（这里写你自己的路径）
      ]
    }
  }
}

```

打开Claude，工具生效。


## 总结：
powershell基本上可以对你的电脑做你任何想做的事情。

如：
根据文件做个小报告。
桌面上有哪些文件？整理XXX文件到XXX文件夹。
操作具体Excel、CSV，根据内容生成专业的HTML可视化分析报告。
轻量级数据清洗。
等等等等……

数据可视化示例：
可以用具体的表，在开源网站[vchart](https://visactor.io/vchart/)中选择一个视图样式，将视图右边的代码复制粘贴给AI，让AI模仿这个数据展示形式，结合你的数据，做出可视化。
原理：让AI使用Powershell读取Excel，操作Excel。

操作其他文档大同小异。

个人认为，使用体验总体要比filesystem-mcp可用性高。
我是放在Mac下通过PD带Windows使用的。
放在虚拟机Windows时能防止一些危险操作，以保证定期备份系统随时可以在一分钟内恢复。
这个很考验大模型的编码能力。
