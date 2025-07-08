#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sunshine Analysis Plugin Dependency Installer
安装插件所需的Python依赖包
"""

import sys
import subprocess
import importlib

def check_python_version():
    """检查Python版本"""
    if sys.version_info < (3, 7):
        print("✗ Python版本过低，需要Python 3.7或更高版本")
        print(f"当前版本: {sys.version}")
        return False
    else:
        print(f"✓ Python版本: {sys.version}")
        return True

def install_package(package, description=""):
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
    """主安装函数"""
    print("Sunshine Analysis Plugin 依赖安装程序")
    print("=" * 50)
    
    # 检查Python版本
    if not check_python_version():
        sys.exit(1)
    
    # 必需的依赖包
    required_packages = [
        ("numpy>=1.20.0", "numpy", "数值计算"),
        ("scipy>=1.7.0", "scipy", "科学计算"),
        ("geopandas>=0.10.0", "geopandas", "地理数据处理"),
        ("pyproj>=3.0.0", "pyproj", "坐标转换"),
        ("matplotlib>=3.3.0", "matplotlib", "可视化"),
        ("tqdm>=4.60.0", "tqdm", "进度条"),
    ]
    
    # 可选的依赖包
    optional_packages = [
        ("astral>=2.2", "astral", "精确天文计算"),
        ("cupy>=9.0.0", "cupy", "GPU加速"),
    ]
    
    print("\n=== 检查必需的依赖包 ===")
    missing_required = []
    
    for package_spec, package_name, description in required_packages:
        if check_package(package_name):
            print(f"✓ {package_name} ({description}) - 已安装")
        else:
            print(f"✗ {package_name} ({description}) - 未安装")
            missing_required.append((package_spec, package_name, description))
    
    # 安装缺失的必需包
    if missing_required:
        print(f"\n需要安装 {len(missing_required)} 个必需的包:")
        for package_spec, package_name, description in missing_required:
            print(f"  - {package_name}: {description}")
        
        response = input("\n是否现在安装这些包? (y/n): ").lower().strip()
        if response in ['y', 'yes', '是']:
            print("\n开始安装...")
            failed_packages = []
            
            for package_spec, package_name, description in missing_required:
                if not install_package(package_spec, description):
                    failed_packages.append(package_name)
            
            if failed_packages:
                print(f"\n✗ 以下包安装失败: {', '.join(failed_packages)}")
                print("请手动安装这些包，或检查网络连接和权限")
            else:
                print("\n✓ 所有必需的包安装成功！")
        else:
            print("安装已取消")
    else:
        print("\n✓ 所有必需的包都已安装！")
    
    # 检查可选的依赖包
    print("\n=== 检查可选的依赖包 ===")
    for package_spec, package_name, description in optional_packages:
        if check_package(package_name):
            print(f"✓ {package_name} ({description}) - 已安装")
        else:
            print(f"- {package_name} ({description}) - 未安装（可选）")
    
    # 提供安装可选包的选择
    missing_optional = []
    for package_spec, package_name, description in optional_packages:
        if not check_package(package_name):
            missing_optional.append((package_spec, package_name, description))
    
    if missing_optional:
        print(f"\n发现 {len(missing_optional)} 个可选包未安装:")
        for package_spec, package_name, description in missing_optional:
            print(f"  - {package_name}: {description}")
        
        response = input("\n是否安装可选包? (y/n): ").lower().strip()
        if response in ['y', 'yes', '是']:
            print("\n开始安装可选包...")
            for package_spec, package_name, description in missing_optional:
                install_package(package_spec, description)
    
    print("\n" + "=" * 50)
    print("依赖检查完成！")
    print("\n如果所有必需的包都已安装，插件应该可以正常工作。")
    print("要测试插件功能，请运行: python test_plugin.py")

if __name__ == "__main__":
    main() 