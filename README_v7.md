# Hyperfine Interactive Spectrum Assigner v7

本发布候选版：**7.0.0-rc.1**。启动入口为 `plotcomparison_2026_7.py`。
原始 `plotcomparison.py` 保留。原本地 `_6.py` 不参与这次发布，也没有被覆盖。

- [完整中文安装与 SPFIT/SPCAT 配置指南](docs/INSTALL_zh-CN.md)
- [English setup guide](docs/INSTALL.md)
- [版本升级说明](CHANGELOG.md)
- [已有用户迁移、GitHub 分支上传与版本发布建议](docs/RELEASING.md)
- [测试记录与平台验证边界](docs/VALIDATION.md)

使用完整项目及 `requirements-v7.txt`，不要只复制主脚本。推荐 Python 3.11。
新增的三个自带模块和完整 `assets/` 文件夹必须与入口脚本一起保留。
个人配置、指认、自动保存和分子工作文件不包含在发布包中。

打开 http://127.0.0.1:8053/；端口占用时通过 `--port 8054` 启动。
本轮只整理、修复并验证已有功能，没有新增程序路径设置窗口。
