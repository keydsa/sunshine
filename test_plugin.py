#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sunshine Analysis Plugin Test Script
测试插件的基本功能
"""

import sys
import os
import math
from datetime import datetime

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(__file__))

try:
    from sunshine_analysis import SunshineAnalyzer
    print("✓ 成功导入SunshineAnalyzer")
except ImportError as e:
    print(f"✗ 导入SunshineAnalyzer失败: {e}")
    sys.exit(1)

def test_azimuth_calculation():
    """测试日出方位角计算"""
    print("\n=== 测试日出方位角计算 ===")
    
    analyzer = SunshineAnalyzer()
    
    # 测试不同位置的日出方位角
    test_cases = [
        (39.9042, 116.4074, "2025-01-27", "北京"),
        (31.2304, 121.4737, "2025-01-27", "上海"),
        (23.1291, 113.2644, "2025-01-27", "广州"),
        (30.2741, 120.1551, "2025-01-27", "杭州"),
    ]
    
    for lat, lon, date_str, city in test_cases:
        try:
            azimuth = analyzer.calculate_sunrise_azimuth(lat, lon, date_str)
            print(f"✓ {city} ({lat:.4f}, {lon:.4f}) 在 {date_str} 的日出方位角: {azimuth:.1f}°")
        except Exception as e:
            print(f"✗ {city} 计算失败: {e}")
    
    # 测试不同日期的方位角变化
    print("\n--- 测试不同日期的方位角变化 ---")
    lat, lon = 39.9042, 116.4074  # 北京
    dates = ["2025-03-21", "2025-06-21", "2025-09-23", "2025-12-21"]  # 春分、夏至、秋分、冬至
    
    for date_str in dates:
        try:
            azimuth = analyzer.calculate_sunrise_azimuth(lat, lon, date_str)
            print(f"✓ 北京在 {date_str} 的日出方位角: {azimuth:.1f}°")
        except Exception as e:
            print(f"✗ {date_str} 计算失败: {e}")

def test_endpoint_calculation():
    """测试终点坐标计算"""
    print("\n=== 测试终点坐标计算 ===")
    
    analyzer = SunshineAnalyzer()
    
    # 测试从北京出发的不同方位角和距离
    start_lat, start_lon = 39.9042, 116.4074  # 北京
    test_cases = [
        (90, 10000, "正东10km"),
        (180, 10000, "正南10km"),
        (270, 10000, "正西10km"),
        (0, 10000, "正北10km"),
    ]
    
    for azimuth, distance, description in test_cases:
        try:
            end_lon, end_lat = analyzer.calculate_endpoint(start_lon, start_lat, azimuth, distance)
            print(f"✓ {description}: ({end_lat:.4f}, {end_lon:.4f})")
        except Exception as e:
            print(f"✗ {description} 计算失败: {e}")

def test_dependencies():
    """测试依赖包"""
    print("\n=== 测试依赖包 ===")
    
    dependencies = [
        ("numpy", "数值计算"),
        ("geopandas", "地理数据处理"),
        ("gdal", "栅格数据处理"),
        ("matplotlib", "可视化"),
        ("scipy", "科学计算"),
        ("pyproj", "坐标转换"),
    ]
    
    for package, description in dependencies:
        try:
            __import__(package)
            print(f"✓ {package} ({description}) - 已安装")
        except ImportError:
            print(f"✗ {package} ({description}) - 未安装")
    
    # 测试可选依赖
    optional_deps = [
        ("astral", "精确天文计算"),
        ("cupy", "GPU加速"),
    ]
    
    print("\n--- 可选依赖 ---")
    for package, description in optional_deps:
        try:
            __import__(package)
            print(f"✓ {package} ({description}) - 已安装")
        except ImportError:
            print(f"- {package} ({description}) - 未安装（可选）")

def test_analyzer_initialization():
    """测试分析器初始化"""
    print("\n=== 测试分析器初始化 ===")
    
    try:
        analyzer = SunshineAnalyzer()
        print("✓ SunshineAnalyzer 初始化成功")
        
        # 测试日志功能
        analyzer.log("测试日志消息")
        print("✓ 日志功能正常")
        
    except Exception as e:
        print(f"✗ 分析器初始化失败: {e}")

def main():
    """主测试函数"""
    print("Sunshine Analysis Plugin 测试")
    print("=" * 50)
    
    # 测试依赖包
    test_dependencies()
    
    # 测试分析器初始化
    test_analyzer_initialization()
    
    # 测试方位角计算
    test_azimuth_calculation()
    
    # 测试终点坐标计算
    test_endpoint_calculation()
    
    print("\n" + "=" * 50)
    print("测试完成！")
    print("\n如果所有测试都通过，说明插件基本功能正常。")
    print("要测试完整功能，请在QGIS中加载DEM和点数据文件。")

if __name__ == "__main__":
    main() 