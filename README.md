# Y-B

PyTorch / torch_npu 安装包与兼容性补丁脚本。

## 补丁脚本

[apply_torch_npu_patches.py](apply_torch_npu_patches.py)

## 安装包下载

以下 3 个原始 `.whl` 文件保存在 [Releases](https://github.com/zhongsuny2-prog/Y-B/releases/tag/torch-npu-2.15.0-dev20261005) 中，文件名标记为 Python 3.12、Linux aarch64。

- [torch-2.15.0.dev20261005+cpu-cp312-cp312-manylinux_2_28_aarch64.whl](https://github.com/zhongsuny2-prog/Y-B/releases/download/torch-npu-2.15.0-dev20261005/torch-2.15.0.dev20261005%2Bcpu-cp312-cp312-manylinux_2_28_aarch64.whl) — 162.93 MiB
- [torch_npu-2.15.0+git281a08e-cp312-cp312-linux_aarch64.whl](https://github.com/zhongsuny2-prog/Y-B/releases/download/torch-npu-2.15.0-dev20261005/torch_npu-2.15.0%2Bgit281a08e-cp312-cp312-linux_aarch64.whl) — 43.71 MiB
- [torchvision-0.30.0.dev20261005+cpu-cp312-cp312-manylinux_2_28_aarch64.whl](https://github.com/zhongsuny2-prog/Y-B/releases/download/torch-npu-2.15.0-dev20261005/torchvision-0.30.0.dev20261005%2Bcpu-cp312-cp312-manylinux_2_28_aarch64.whl) — 1.98 MiB

因 `torch` 安装包超过 GitHub 普通仓库单文件 100 MiB 限制，安装包作为 Release 附件保存。克隆仓库后，请从上方链接另行下载安装包。

Release 说明中提供 SHA-256 校验值。
