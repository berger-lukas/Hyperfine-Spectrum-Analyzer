# v7 新用户安装与配置指南

[English](INSTALL.md) · [项目首页](../README.md) · [升级与发布](RELEASING.md)

## 一、下载完整版本

推荐使用 **64 位 Python 3.11**。本发布版的数值库采用开发环境验证过的版本；不要直接搭配 Python 3.13 或更高版本。Python 3.12 已列入自动测试矩阵，实际验证状态见 [VALIDATION.md](VALIDATION.md)。

在 GitHub 切换到 `v7-release` 分支，点击 **Code → Download ZIP**，完整解压到有写入权限的文件夹。若版本发布在 fork，请使用该 fork 的地址。不要只下载 `_7.py`。

以下文件必须保留在一起：

```text
plotcomparison_2026_7.py        启动入口
spectrum_workspace_v7.py       工作区、编辑器、外部程序运行
spectrum_config_v7.py          配置管理
spectrum_intensity_v7.py       强度显示
assets/                       完整样式和交互文件夹
requirements-v7.txt           v7 依赖清单
config.json 和 molecules/     上游公开示例
```

三个 `spectrum_*_v7.py` 是软件自带源码，不需要通过 pip 安装。请勿复制旧电脑的 `venv`；应在新电脑重新创建虚拟环境。原始 `plotcomparison.py` 保留，但启动它会进入老版本。

## 二、安装 Python 依赖并启动

先从 [Python 官网](https://www.python.org/downloads/)安装 Python 3.11。Linux 用户通过发行版提供的方式安装 Python 3.11 和相应 venv 支持，不要替换系统 Python。

以下命令均在项目文件夹中执行。

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-v7.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe plotcomparison_2026_7.py
```

直接指定虚拟环境的 Python 可以避免 PowerShell 激活脚本的权限问题。在 VS Code 中也要选择这个解释器。

### macOS / Linux 终端

```bash
python3.11 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements-v7.txt
./.venv/bin/python -m pip check
./.venv/bin/python plotcomparison_2026_7.py
```

Apple Silicon 用户使用原生或 universal Python，避免混用 Intel 与 ARM 环境。可用以下命令检查 Python 的架构：

```bash
python3.11 -c "import platform; print(platform.python_version(), platform.machine())"
```

在浏览器打开 **http://127.0.0.1:8053/**。终端必须保持运行；Ctrl+C 停止服务。端口被占用时，在启动命令末尾加 `--port 8054`，然后打开相应地址。

## 三、新建光谱配置

1. 点击 **New Configuration**，给配置命名。
2. 选择实验谱文件：第一列频率 **MHz**、第二列强度，必须有表头且数据为数字。新配置自动识别分隔符。无表头数据请先添加列名。
3. 添加一个或多个 CAT 文件。可以继续追加，文件可以位于不同文件夹。
4. 保存并打开。以后使用 **Edit Configuration** 修改。

上游公开示例可以用于浏览、指认；它不保证附带执行 SPFIT 所需的完整输入文件。自己的实验请新建配置，避免覆盖公开示例。

同一实验谱可以关联多个 CAT；**活动 CAT** 决定 SPFIT/SPCAT 的工作目录与文件基名。例如 `sample.cat` 对应同目录下的 `sample.par/.var/.int/.lin`。程序无需复制到每个分子文件夹。

迁移到其他电脑时，要重新选择数据文件，或修改 JSON 路径。Windows 的 `E:/...` 不能直接用于 Mac；区分大小写的文件系统还要求文件名大小写完全一致。相对路径以 JSON 所在目录为基准，不随启动终端的位置变化。

## 四、一步步配置 SPFIT / SPCAT

**浏览现有 CAT、拟合实验峰和指认不要求安装 SPFIT/SPCAT。** 只有需要在 Fitting Space 执行它们时才需要安装。它们是独立可执行程序，不属于 pip 依赖，本发布包不附带二进制文件。

### 1. 下载哪一版

访问[科隆大学 SPIN 下载说明](https://spin.astro.uni-koeln.de/chapter/Prerequisites/)，分别下载两个程序：

| 电脑 | 对应选项 |
|---|---|
| Windows | SPFIT / SPCAT for Windows |
| M 系列 Mac | SPFIT / SPCAT for macOS (ARM) |
| Intel Mac | SPFIT / SPCAT for macOS (Intel) |
| 兼容的 Ubuntu 系统 | SPFIT / SPCAT for Ubuntu |

Mac 在“苹果菜单 → 关于本机”查看芯片。Linux 用 `uname -m` 检查架构；不能认为 Ubuntu 预编译文件适用于所有发行版或 ARM 机器。如果二进制不兼容，可以从该安装页或 [CDMS 源码页面](https://cdms.astro.uni-koeln.de/classic/pickett)获取源码，在目标电脑安装 C 编译器和 make 后，在源码目录执行 `make spfit`、`make spcat`。

[Kisiel 程序资源页](http://info.ifpan.edu.pl/~kisiel/asym/asym.htm#pickett)也可作为参考。下载时确认系统标签，Windows `.exe` 不能在 Mac/Linux 原生运行。

### 2. 放在哪里，如何确认能启动

Windows 示例：将 `spfit.exe`、`spcat.exe` 放到 `C:\Tools\Pickett`。在 PowerShell 中分别运行：

```powershell
& 'C:\Tools\Pickett\spfit.exe'
& 'C:\Tools\Pickett\spcat.exe'
```

每次出现文件名输入提示后，按回车退出，再测试另一个程序。这里只验证可以启动，不会验证分子模型。

Mac/Linux 示例：将文件名为 `spfit`、`spcat` 的程序放到 `~/Tools/Pickett/`。先授予执行权限，再分别启动：

```bash
chmod +x "$HOME/Tools/Pickett/spfit" "$HOME/Tools/Pickett/spcat"
"$HOME/Tools/Pickett/spfit"
"$HOME/Tools/Pickett/spcat"
```

同样，出现文件名提示后按回车退出。若 macOS 阻止打开下载的程序，确认来源后，在系统“隐私与安全性”中为该程序单独批准；不要关闭系统整体保护。`Bad CPU type` 或 `Exec format error` 通常提示系统/架构不匹配，需要重新选择版本或本机编译。

### 3. 推荐：环境变量只设置一次

Windows 搜索“编辑账户的环境变量”，新建两个**用户变量**：

| 变量名 | 示例值 |
|---|---|
| `SPFIT_PATH` | `C:\Tools\Pickett\spfit.exe` |
| `SPCAT_PATH` | `C:\Tools\Pickett\spcat.exe` |

变量值是完整程序路径，不只是文件夹。设置后**完全退出并重开 VS Code 和终端**。

也可以只在当前 PowerShell 会话设置，并从同一会话启动软件：

```powershell
$env:SPFIT_PATH = 'C:\Tools\Pickett\spfit.exe'
$env:SPCAT_PATH = 'C:\Tools\Pickett\spcat.exe'
.\.venv\Scripts\python.exe plotcomparison_2026_7.py
```

Mac/Linux 在启动软件的终端中执行：

```bash
export SPFIT_PATH="$HOME/Tools/Pickett/spfit"
export SPCAT_PATH="$HOME/Tools/Pickett/spcat"
./.venv/bin/python plotcomparison_2026_7.py
```

如需持久保存，把两行 `export` 加入自己 shell 的启动文件，例如交互式 zsh 的 `~/.zshrc`、bash 的 `~/.bashrc`，再新开终端。从 Finder 或桌面启动的 VS Code 不一定继承这些变量；此时从配置好的终端启动软件，或采用下面的 JSON 方式。

### 4. 备选：在 JSON 中填写完整路径

目前配置弹窗只管理名称、实验谱和 CAT，不包含程序路径输入框。用文本编辑器打开对应 `config*.json`，在顶层加入以下字段，保留原有数据设置，注意逗号：

Windows：

```json
"spfit_path": "C:/Tools/Pickett/spfit.exe",
"spcat_path": "C:/Tools/Pickett/spcat.exe"
```

Mac：

```json
"spfit_path": "/Users/你的用户名/Tools/Pickett/spfit",
"spcat_path": "/Users/你的用户名/Tools/Pickett/spcat"
```

Linux 通常使用 `/home/你的用户名/...`。以上是字段片段，不是完整 JSON。路径含空格也可以使用；本发布版支持 `~` 展开以及相对 JSON 目录的程序路径。为方便排错，优先填写完整路径。更改安装设置后重启服务。

查找优先级固定为：**JSON → SPFIT_PATH/SPCAT_PATH → 系统 PATH**。高优先级路径失效时会明确报错，不会自动尝试其他版本。旧电脑 JSON 里的绝对路径必须更新或删除。系统 PATH 加的是程序所在**文件夹**；专用变量填的是**程序文件**。仅设置 shell alias 不够。

### 5. 在 Fitting Space 首次运行

1. 选择活动 CAT，进入 **FITTING SPACE**，核对工作目录和文件基名。
2. 如果要使用当前指认，点击 **Write Assignments to Working LIN**。主页面普通 LIN 导出只是带时间戳的归档，不代替工作 LIN。
3. 检查并保存编辑器中的输入文件。SPFIT 需要同名 `.par/.lin`，SPCAT 需要 `.var/.int`。
4. 点击 **Run SPFIT**，查看控制台和 FIT 文件。检查行数、Bad Line、拒绝行、RMS、参数及不确定度。
5. 审核拟合后点击 **Run SPCAT**。也可用 **SPFIT → SPCAT** 顺序执行；发现执行失败、Bad Line 或缺少 FIT COMPLETE 时，链式运行停止。但软件不能代替科学质量判断。
6. 点击 **Refresh CAT** 或 **Refresh All CATs**，再用 **Back to Assigner Space** 返回。SPCAT 运行结束不会自动覆盖主图中的预测。

运行前会在分子目录的 `.assigner-history/` 保留输入与旧输出备份，并写入 `run.log`。SPFIT/SPCAT 会更新工作文件，请使用有写入权限的工作目录。命令框只接受 `spfit` / `spcat` 及可选的活动文件基名，不是完整终端，也不支持交互式 stdin 输入。

## 五、常见问题

| 问题 | 处理方法 |
|---|---|
| 页面还是旧版 | 确认运行 `_7.py`，停止旧服务，检查端口，重启后 Ctrl+F5 或 Cmd+Shift+R。 |
| 缺少 `spectrum_workspace_v7` | 三个自带模块必须和主脚本在一起，不要用 pip 搜索安装。 |
| 缺少 Dash 等库 | 用启动软件的同一个 Python 安装 `requirements-v7.txt`。 |
| 安装 NumPy/SciPy 时编译失败 | 检查 Python 版本、位数及架构；参考环境为 Python 3.11，不能复制外机 venv。 |
| 找不到 SPFIT/SPCAT | 先独立启动验证，再确认完整路径和环境变量；重启 IDE。 |
| 环境变量正确仍报旧路径 | JSON 优先级更高，删除或修改其中旧地址。 |
| 权限不足 / WinError 5 | 检查程序执行权限、分子目录及 `.assigner-history` 写权限，改用可写工作副本。 |
| 架构错误 / WinError 193 | 检查系统和芯片版本，确认下载的是二进制而非网页。 |
| 缺少输入文件 | PAR/VAR/INT/LIN 必须位于活动 CAT 同目录且同名，注意大小写。 |
| 修改 CAT 后仍显示旧谱 | 点击 Refresh CAT；空文件或无效 CAT 会被拒绝，原图保留。 |
| 编辑器拒绝保存 | 文件已被外部修改；先保留草稿，再重新加载并合并更改。 |
| CAT 刷新后有 missing/ambiguous | 检查对应跃迁，解决后才能写工作 LIN，不能按旧行号继续使用。 |

## 六、显示及格式边界

- 实验谱要求表头、前两列数值、MHz 频率，不自动转换单位。
- 当前 CAT 读取沿用数字量子数的固定列宽格式，不承诺支持所有字母扩展编码。
- LIN 导入主要对应本程序导出格式，不保证兼容所有外部 LIN 变体。
- “Use original experimental intensity”恢复原始数值，不代表增加物理校准。可在 JSON 中用 `intensity_unit` 设置纵轴单位标签。
- 每个 CAT 在实验谱完整频段内独立选最强峰作为显示基准。缩放不改变基准；无频段内正强度跃迁时模拟强度显示为零。Loomis–Wood 仍按谱带归一化。
- 每个实验谱只使用一个编辑标签页。本程序仅监听本机，不用于远程多用户部署。
- 外部修改实验谱文件后需重启；外部修改 CAT 用刷新按钮。Stop 可能留下不完整外部输出，必要时从备份恢复。
