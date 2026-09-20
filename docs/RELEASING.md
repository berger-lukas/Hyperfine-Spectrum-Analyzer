# Upgrade, review and publish / 升级与发布

## For existing users / 已有用户升级

1. Stop the old server and back up the complete project plus molecular working
   directories. Include configs, `autosave/`, `assignments/`, PAR/VAR/INT/LIN,
   CAT/FIT/OUT and `.assigner-history/`. Do not rely on Git for ignored files.
2. Extract the complete new version into a separate writable directory. Keep
   the old installation available for rollback. Create a new Python 3.11 venv
   and install `requirements-v7.txt`; do not upgrade a working scientific
   environment in place merely to try this release.
3. Copy personal JSON configurations as `config-<name>.json`, preserving all
   scientific settings. Update paths if the data moved. Leave the supplied
   public `config.json` intact. Use a `name` field or Edit Configuration to give
   each a readable name.
4. V7 workspace identity is based on the absolute configuration path. Moving or
   renaming a config changes its identity; copying `autosave/v7/` alone does not
   automatically migrate its state. Prefer keeping the config path unchanged
   for an in-place update after backup. When moving installations, explicitly
   export/import your working LIN and check assignments before continuing;
   LIN does not carry every Gaussian-fit context or UI setting.
5. Set SPFIT/SPCAT paths on the new computer. A stale JSON override wins over
   environment/PATH settings. See the [English](INSTALL.md) or
   [Chinese](INSTALL_zh-CN.md) walkthrough.
6. Start `_7.py`, choose the correct port, hard-refresh the browser and verify
   a known spectrum/assignment subset before scientific work. The original
   `plotcomparison.py` still starts the original UI.

中文要点：先完整备份，再在独立目录试用；复制整个版本而不是仅 `_7.py`。
新电脑重建虚拟环境，重新配置数据和程序路径。配置移动会改变自动保存标识，
不能假设原自动保存目录会自动关联。通过工作 LIN 迁移后要复核量子数、行数与不确定度。
不要在服务器运行中覆盖代码。回退时停止 v7，恢复备份并启动原环境。

## Release branch / 发布分支

This preparation uses branch **`v7-release`**, based on upstream main commit
`2c59ff9e5ce9988898fc29a9d5bda1a39c9824c8`. The candidate version is
**`7.0.0-rc.1`**. Keep main and the original entry point intact during review.

An upload-ready source ZIP is an alternative to Git. It includes tracked source,
docs, tests, public examples and CI, but excludes `.git`, virtual environments,
personal configurations, assignments, caches and molecular backups. A sibling
SHA-256 file can be used to verify the archive's bytes.

### Push with repository write access

From the prepared repository, after reviewing `git status` and the commit:

```bash
git push -u origin v7-release
```

If a branch with that name already exists and the push is rejected, inspect the
remote history first. Do not use force-push to overwrite someone else's work.

### If you do not have upstream write access

Use GitHub's **Fork** button to create your own fork. In the prepared repository,
add the fork (replace YOUR_ACCOUNT) and push the existing branch:

```bash
git remote add myfork https://github.com/YOUR_ACCOUNT/Hyperfine-Spectrum-Analyzer.git
git push -u myfork v7-release
```

Then open a pull request from your fork's `v7-release` to the original project's
`main`, if you want to propose the upgrade to its maintainer. Uploading a branch
does not grant permission to merge it into someone else's main branch.

Prefer Git/GitHub Desktop for preserving file structure and reviewing diffs.
If using the web uploader, switch to the intended new branch, extract the source
ZIP, and upload its **contents** to the repository root. Do not upload a venv,
the nested ZIP itself as source, or your personal working folder.

### Suggested pull request title

```text
Add v7 spectrum workspaces and integrated Pickett fitting workflow
```

### Suggested pull request description

The original workflow requires editing a JSON path to switch spectra and leaving
the application to run SPFIT/SPCAT. This branch adds named spectrum workspaces,
configuration editing and a Fitting Space with protected file editing, external
program execution and explicit catalog refresh. The original entry point and
public examples remain available.

Simulation normalization now references the measured frequency band. An optional
display switch restores original experimental intensities. Bilingual setup and
migration guides cover Windows, macOS and Linux, including native SPFIT/SPCAT
installation. Validation and remaining platform limitations are recorded in
`docs/VALIDATION.md`; CI tests synthetic workflows without private data.

## Before a stable release / 正式发布前

- Review the staged diff and ensure no personal paths, spectra, assignments,
  credentials, proprietary inputs or local binaries are included. The tracked
  example `config.json` is not protected by ignore rules; keep it public.
- Confirm tests from a freshly created venv. Review the GitHub Actions matrix
  on Windows, macOS and Linux. A configured workflow is not a passed workflow.
- On each target OS, test first launch, new/edit configuration, editor save,
  native SPFIT/SPCAT startup, a copied scientific case and CAT refresh. CI's
  default suite does not distribute or execute external Pickett binaries.
- Have a user review their assignments and FIT results before adopting the
  release for production spectroscopy. Record discovered limits.
- Update `VERSION`, `CHANGELOG.md`, the README version and validation record
  together. Keep `rc` until the remaining checks are complete.
- Only then create an annotated release tag and a GitHub Release. Do not reuse
  or move an already-published version tag. Attach the source ZIP if useful and
  link to the installation guide; no need to bundle external executables.

中文建议：本次先用 `7.0.0-rc.1` 供测试。跨平台 CI 与各系统真实程序检查通过后，
再发布 `7.0.0`。后续修复用 `7.0.1`，新增兼容功能用 `7.1.0`；有不兼容的数据或
接口变更时再考虑主版本。每次更新记录“改了什么、影响哪些旧配置、如何迁移、
如何回退、实际测试了什么”。不要用一次覆盖式上传替代版本历史。
