#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Install Script for Sunshine01 Plugin
快速安装缺失的依赖包
"""

import sys
import subprocess
import importlib

def install_package(package):
    """安装单个包"""
    try:
        print(f"正在安装 {package}...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", package
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ {package} 安装成功")
            return True
        else:
            print(f"✗ {package} 安装失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ {package} 安装出错: {e}")
        return False

def check_package(package):
    """检查包是否已安装"""
    try:
        importlib.import_module(package)
        return True
    except ImportError:
        return False

def main():
    """主函数"""
    print("Sunshine01 Plugin 快速安装程序")
    print("=" * 40)
    
    # 必需的依赖包
    required_packages = [
        ("tqdm", "进度条显示"),
        ("scipy", "科学计算"),
        ("matplotlib", "可视化"),
    ]
    
    print("检查缺失的依赖包...")
    missing_packages = []
    
    for package, description in required_packages:
        if check_package(package):
            print(f"✓ {package} ({description}) - 已安装")
        else:
            print(f"✗ {package} ({description}) - 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n需要安装 {len(missing_packages)} 个包:")
        for package in missing_packages:
            print(f"  - {package}")
        
        response = input("\n是否现在安装这些包? (y/n): ").lower().strip()
        if response in ['y', 'yes', '是']:
            print("\n开始安装...")
            failed_packages = []
            
            for package in missing_packages:
                if not install_package(package):
                    failed_packages.append(package)
            
            if failed_packages:
                print(f"\n✗ 以下包安装失败: {', '.join(failed_packages)}")
                print("请手动安装这些包:")
                for package in failed_packages:
                    print(f"  pip install {package}")
            else:
                print("\n✓ 所有包安装成功！")
                print("现在可以重新启动QGIS并加载插件了。")
        else:
            print("安装已取消")
    else:
        print("\n✓ 所有必需的包都已安装！")
        print("插件应该可以正常加载了。")
    
    print("\n" + "=" * 40)
    print("安装完成！")

if __name__ == "__main__":
    main() 