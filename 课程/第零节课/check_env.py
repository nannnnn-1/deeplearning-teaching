"""
深度学习培训 · 课前环境检查脚本
================================
课前只需要装好 Miniconda + PyTorch + 最基础的几个库。
其他课程库会在第一节课统一安装。

运行方式: python check_env.py
预期: 所有项显示 ✓
"""

import sys


def check_python():
    version = sys.version_info
    ok = (3, 11) <= (version.major, version.minor) <= (3, 13)
    status = "✓" if ok else "✗"
    extra = "  (推荐 3.12)" if not ok else ""
    print(f"{status} Python {version.major}.{version.minor}.{version.micro}{extra}")
    return ok


def check_torch():
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  → CUDA 可用: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            print(f"  → MPS 可用 (Apple Silicon)")
        else:
            print(f"  → 使用 CPU 运行")
        return True
    except ImportError:
        print("✗ PyTorch 未安装")
        print("  安装命令: pip install torch torchvision torchaudio")
        return False


def check_package(name, import_name=None, install_hint=None):
    import_name = import_name or name
    try:
        mod = __import__(import_name)
        ver = getattr(mod, "__version__", "unknown")
        print(f"✓ {name:20s} {ver}")
        return True
    except ImportError:
        print(f"✗ {name:20s} 未安装")
        if install_hint:
            print(f"  安装命令: {install_hint}")
        return False


def main():
    print("=" * 50)
    print("深度学习培训 · 课前环境检查")
    print("=" * 50)
    print("（第一节课会带大家安装 transformers、ultralytics 等剩余库）")
    print()

    results = []
    results.append(("Python", check_python()))
    results.append(("PyTorch", check_torch()))

    # 课前只需要这几个基础库
    basics = [
        ("torchvision", "torchvision", "pip install torchvision"),
        ("matplotlib", "matplotlib", "pip install matplotlib"),
        ("jupyterlab", "jupyterlab", "pip install jupyterlab"),
        ("numpy", "numpy", "pip install numpy"),
        ("PIL", "PIL", "pip install pillow"),
    ]

    for name, imp, hint in basics:
        results.append((name, check_package(name, imp, hint)))

    print()
    print("=" * 50)
    ok = sum(1 for _, r in results if r)
    total = len(results)
    print(f"检查结果: {ok}/{total} 项通过")
    if ok == total:
        print("🎉 课前基础环境就绪，第一节课见！")
    else:
        print("⚠️  请对照《环境安装指南》补装未通过的项")
    print("=" * 50)


if __name__ == "__main__":
    main()
