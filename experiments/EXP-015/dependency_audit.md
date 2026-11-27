# 环境审查

使用 `.local/round014-venv`（venv --system-site-packages；包从既有环境只读继承，并非完整独立依赖闭包），没有升级或卸载旧环境。Python 3.11.9、PyTorch 2.4.1+cu118、torchvision 0.19.1+cu118、NumPy 1.26.4、SpikingJelly 0.0.0.0.14、timm 1.0.22，driver 550.144.03，nvcc 11.8。当前环境 CuPy 不存在，本地常见环境及缓存未发现可复用安装；官方 LIF 构造直接抛出 `ImportError: CuPy is not installed!`。README 要求 cupy-cuda12x，并提示 numpy np.int 兼容问题。驱动存在不代表 CuPy、torch、nvcc 兼容；未修改驱动、旧环境或下载大型依赖。

CPU 测试显式使用 torch backend 的合成 LIF 子类；权重和 monitor 使用原官方实现。CuPy 与 torch 的前向及梯度对照尚未做，不能把 torch 路径等同论文后端，不能绕过 GPU benchmark 进入门槛。完整依赖闭包、官方单 GPU 前反向、训练参数组、AMP 和非零惩罚资源测量均未通过实测。预训练权重：UNKNOWN，官方 README 和固定代码树未发现可核验权重链接或模型文件。
