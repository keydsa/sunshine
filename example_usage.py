#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sunshine Analysis Plugin Usage Example
插件使用示例
"""

import sys
import os
from datetime import datetime

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(__file__))

try:
    from sunshine_analysis import SunshineAnalyzer
except ImportError as e:
    print(f"无法导入SunshineAnalyzer: {e}")
    print("请先运行 install_dependencies.py 安装依赖")
    sys.exit(1)

def example_azimuth_calculation():
    """示例：计算日出方位角"""
    print("=== 日出方位角计算示例 ===")
    
    analyzer = SunshineAnalyzer()
    
    # 示例位置
    locations = [
        (39.9042, 116.4074, "北京"),
        (31.2304, 121.4737, "上海"),
        (23.1291, 113.2644, "广州"),
        (30.2741, 120.1551, "杭州"),
    ]
    
    # 示例日期
    dates = [
        ("2025-03-21", "春分"),
        ("2025-06-21", "夏至"),
        ("2025-09-23", "秋分"),
        ("2025-12-21", "冬至"),
    ]
    
    print("不同城市在不同节气的日出方位角:")
    print("-" * 60)
    
    for lat, lon, city in locations:
        print(f"\n{city} ({lat:.4f}, {lon:.4f}):")
        for date_str, season in dates:
            try:
                azimuth = analyzer.calculate_sunrise_azimuth(lat, lon, date_str)
                print(f"  {season} ({date_str}): {azimuth:.1f}°")
            except Exception as e:
                print(f"  {season} ({date_str}): 计算失败 - {e}")

def example_endpoint_calculation():
    """示例：计算射线终点"""
    print("\n=== 射线终点计算示例 ===")
    
    analyzer = SunshineAnalyzer()
    
    # 从北京出发的不同方向
    start_lat, start_lon = 39.9042, 116.4074  # 北京
    
    directions = [
        (0, "正北"),
        (45, "东北"),
        (90, "正东"),
        (135, "东南"),
        (180, "正南"),
        (225, "西南"),
        (270, "正西"),
        (315, "西北"),
    ]
    
    distances = [1000, 5000, 10000]  # 1km, 5km, 10km
    
    print(f"从北京 ({start_lat:.4f}, {start_lon:.4f}) 出发的射线终点:")
    print("-" * 60)
    
    for azimuth, direction in directions:
        print(f"\n{direction}方向 ({azimuth}°):")
        for distance in distances:
            try:
                end_lon, end_lat = analyzer.calculate_endpoint(start_lon, start_lat, azimuth, distance)
                print(f"  {distance}m: ({end_lat:.4f}, {end_lon:.4f})")
            except Exception as e:
                print(f"  {distance}m: 计算失败 - {e}")

def example_visibility_analysis():
    """示例：可见性分析（需要DEM和点数据）"""
    print("\n=== 日出可见性分析示例 ===")
    print("注意：此示例需要DEM和点数据文件")
    print("在实际使用中，请准备以下文件：")
    print("1. DEM文件（.tif, .tiff, .asc等格式）")
    print("2. 点数据文件（.shp, .gpkg, .geojson等格式）")
    print("3. 确保两个文件使用相同的坐标系")
    
    # 示例参数
    example_params = {
        'dem_path': 'path/to/your/dem.tif',
        'points_path': 'path/to/your/points.shp',
        'output_path': 'path/to/output/result.gpkg',
        'date': datetime(2025, 6, 21),  # 夏至
        'max_dist': 50000,  # 50km
        'initial_step': 100,  # 100m
        'batch_size': 500,  # 500个点一批
    }
    
    print("\n示例参数:")
    for key, value in example_params.items():
        print(f"  {key}: {value}")
    
    print("\n使用步骤:")
    print("1. 在QGIS中启动插件")
    print("2. 选择DEM文件和点数据文件")
    print("3. 设置输出路径")
    print("4. 选择分析日期")
    print("5. 调整参数（可选）")
    print("6. 点击'开始分析'")
    print("7. 等待分析完成，结果将自动加载到QGIS")

def example_parameter_tuning():
    """示例：参数调优建议"""
    print("\n=== 参数调优建议 ===")
    
    scenarios = [
        {
            'name': '小数据集 (<1000点)',
            'max_dist': 20000,
            'initial_step': 50,
            'batch_size': 200,
            'description': '适合快速测试和精确分析'
        },
        {
            'name': '中等数据集 (1000-10000点)',
            'max_dist': 50000,
            'initial_step': 100,
            'batch_size': 500,
            'description': '平衡精度和速度的推荐设置'
        },
        {
            'name': '大数据集 (>10000点)',
            'max_dist': 50000,
            'initial_step': 200,
            'batch_size': 1000,
            'description': '适合大规模分析，优先考虑速度'
        },
        {
            'name': '高精度分析',
            'max_dist': 100000,
            'initial_step': 25,
            'batch_size': 200,
            'description': '最高精度设置，但计算时间较长'
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        print(f"  最大分析距离: {scenario['max_dist']}m")
        print(f"  初始步长: {scenario['initial_step']}m")
        print(f"  批次大小: {scenario['batch_size']}个点")
        print(f"  说明: {scenario['description']}")

def main():
    """主函数"""
    print("Sunshine Analysis Plugin 使用示例")
    print("=" * 60)
    
    # 基本功能示例
    example_azimuth_calculation()
    example_endpoint_calculation()
    
    # 高级功能示例
    example_visibility_analysis()
    example_parameter_tuning()
    
    print("\n" + "=" * 60)
    print("示例演示完成！")
    print("\n要开始实际使用：")
    print("1. 确保已安装所有依赖包")
    print("2. 在QGIS中加载插件")
    print("3. 准备DEM和点数据文件")
    print("4. 按照界面提示进行操作")
    print("\n如有问题，请查看README.md文件或运行test_plugin.py进行测试")

if __name__ == "__main__":
    main() 