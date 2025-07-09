# -*- coding: utf-8 -*-
"""
/***************************************************************************
 Sunshine Analysis Module
                                 A QGIS plugin
 日出方位角计算和日出可见性分析核心功能
                             -------------------
        begin                : 2025-07-07
        git sha              : $Format:%H$
        copyright            : (C) 2025 by Gao Jiayao
        email                : 2177904925@qq.com
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""

import math
import numpy as np
import geopandas as gpd
from osgeo import gdal
import matplotlib.pyplot as plt
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # 创建一个简单的进度条替代
    class tqdm:
        def __init__(self, total=None, desc=None, **kwargs):
            self.total = total
            self.desc = desc
            self.n = 0
        
        def update(self, n=1):
            self.n += n
            if self.desc:
                print(f"{self.desc}: {self.n}/{self.total}")
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            pass

import multiprocessing as mp
import warnings
from pyproj import CRS
try:
    import matplotlib.colors as mcolors
    MCOLORS_AVAILABLE = True
except ImportError:
    MCOLORS_AVAILABLE = False
    mcolors = None

try:
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    cKDTree = None

import os
import time
import sys
import multiprocessing.sharedctypes as sharedctypes
from multiprocessing import Array
import matplotlib
from datetime import datetime
from qgis.core import QgsRasterLayer, QgsVectorLayer, QgsFeature, QgsFields, QgsField, QgsWkbTypes, QgsVectorFileWriter
from PyQt5.QtCore import QVariant
import tempfile
from qgis.core import QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsProject

# 尝试导入astral库用于精确的天文计算
try:
    from astral import LocationInfo
    from astral.sun import sun
    ASTRAL_AVAILABLE = True
except ImportError:
    ASTRAL_AVAILABLE = False

# 尝试导入cupy用于GPU加速
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("[SunshineAnalysis] CuPy导入成功，GPU可用")
    # GPU预热，提前初始化CUDA上下文
    try:
        cp.zeros(1)
        print("[SunshineAnalysis] GPU预热完成，CUDA上下文已初始化")
    except Exception as e:
        print(f"[SunshineAnalysis] GPU预热失败: {e}")
except Exception as e:
    print(f"[SunshineAnalysis] CuPy导入失败: {e}")
    cp = None
    GPU_AVAILABLE = False

warnings.filterwarnings('ignore')


class SunshineAnalyzer:
    """日出分析器类"""
    
    def __init__(self):
        """初始化分析器"""
        self.dem_data = None
        self.dem_transform = None
        self.dem_array = None
        self.dem_crs = None
        
    def log(self, message, level=0):
        """记录日志"""
        from qgis.core import QgsMessageLog
        QgsMessageLog.logMessage(message, "Sunshine Analysis", level=level)
    
    def create_shared_array(self, shape, dtype):
        """创建共享内存数组"""
        size = int(np.prod(shape))
        ctype = np.ctypeslib.as_ctypes_type(dtype)
        shared_array = Array(ctype, size, lock=False)
        return np.frombuffer(shared_array, dtype=dtype).reshape(shape)
    
    def calculate_endpoint(self, lon_start, lat_start, azimuth_deg, distance, R=6371000):
        """计算给定起点、方位角和距离的终点坐标"""
        lon1 = math.radians(lon_start)
        lat1 = math.radians(lat_start)
        azimuth = math.radians(azimuth_deg)

        # 使用Haversine公式计算终点坐标
        angular_distance = distance / R
        lat2 = math.asin(math.sin(lat1) * math.cos(angular_distance) +
                         math.cos(lat1) * math.sin(angular_distance) * math.cos(azimuth))

        lon2 = lon1 + math.atan2(math.sin(azimuth) * math.sin(angular_distance) * math.cos(lat1),
                                 math.cos(angular_distance) - math.sin(lat1) * math.sin(lat2))

        return math.degrees(lon2), math.degrees(lat2)
    
    def calculate_sunrise_azimuth(self, lat, lon, date=None):
        """计算日出方位角（支持日期参数）"""
        if date is None:
            # 如果没有提供日期，使用平均值
            if lat > 0:  # 北半球
                return 90 - 30 * math.cos(math.radians(lat))
            else:  # 南半球
                return 90 + 30 * math.cos(math.radians(lat))
        
        # 如果提供了日期，尝试使用精确计算
        if ASTRAL_AVAILABLE:
            try:
                # 解析日期字符串
                if isinstance(date, str):
                    date_obj = datetime.strptime(date, "%Y-%m-%d").date()
                else:
                    date_obj = date
                
                # 创建位置信息
                location = LocationInfo("", "", "UTC", lat, lon)
                
                # 计算指定日期的日出信息
                sunrise_info = sun(location.observer, date=date_obj, tzinfo=location.timezone)
                
                # 返回日出方位角
                return sunrise_info['azimuth']
                
            except Exception as e:
                self.log(f"精确计算日出方位角失败: {e}，使用简化计算", level=1)
                # 回退到简化计算
                pass
        
        # 简化计算：基于日期和纬度
        if isinstance(date, str):
            try:
                date_obj = datetime.strptime(date, "%Y-%m-%d")
            except:
                date_obj = datetime.now()
        else:
            date_obj = date or datetime.now()
        
        # 计算一年中的天数（1-365）
        day_of_year = date_obj.timetuple().tm_yday
        
        # 计算太阳赤纬（简化版）
        # 春分约在3月21日（第80天），秋分约在9月23日（第266天）
        # 夏至约在6月21日（第172天），冬至约在12月21日（第355天）
        
        # 使用正弦函数近似太阳赤纬变化
        declination = 23.45 * math.sin(math.radians(360 * (day_of_year - 80) / 365))
        
        # 计算日出方位角
        # 在春分/秋分时，日出在正东（90度）
        # 在夏至时，北半球日出偏北，南半球日出偏南
        # 在冬至时，北半球日出偏南，南半球日出偏北
        
        if lat > 0:  # 北半球
            # 北半球：夏季偏北，冬季偏南
            azimuth_offset = -declination * math.cos(math.radians(lat))
        else:  # 南半球
            # 南半球：夏季偏南，冬季偏北
            azimuth_offset = declination * math.cos(math.radians(abs(lat)))
        
        return 90 + azimuth_offset
    
    def load_dem(self, dem_path):
        """加载DEM数据"""
        ds = gdal.Open(dem_path)
        if ds is None:
            raise ValueError(f"无法打开DEM文件: {dem_path}")

        transform = ds.GetGeoTransform()
        band = ds.GetRasterBand(1)
        dem_array = band.ReadAsArray()

        # 获取坐标系
        proj = ds.GetProjection()
        crs = CRS.from_wkt(proj) if proj else None

        self.log(f"DEM加载成功: 尺寸={dem_array.shape}, 分辨率={transform[1]}度")
        
        self.dem_data = ds
        self.dem_transform = transform
        self.dem_array = dem_array
        self.dem_crs = crs
        
        return ds, transform, dem_array, crs
    
    def batch_bilinear_interpolation(self, dem_array, transform, lons, lats):
        """批量进行双线性插值"""
        lons = np.array(lons)
        lats = np.array(lats)
        xs = (lons - transform[0]) / transform[1]
        ys = (lats - transform[3]) / transform[5]

        # 获取整数部分和小数部分
        x1s = np.floor(xs).astype(int)
        y1s = np.floor(ys).astype(int)
        dxs = xs - x1s
        dys = ys - y1s

        # 边界检查
        valid_mask = (x1s >= 0) & (y1s >= 0) & (x1s < dem_array.shape[1] - 1) & (y1s < dem_array.shape[0] - 1)

        # 初始化高程数组
        elevations = np.zeros_like(lons)

        # 有效点处理
        if np.any(valid_mask):
            x1s_valid = x1s[valid_mask]
            y1s_valid = y1s[valid_mask]
            dxs_valid = dxs[valid_mask]
            dys_valid = dys[valid_mask]

            # 获取四个邻近点的高程
            z11 = dem_array[y1s_valid, x1s_valid]
            z12 = dem_array[y1s_valid, x1s_valid + 1]
            z21 = dem_array[y1s_valid + 1, x1s_valid]
            z22 = dem_array[y1s_valid + 1, x1s_valid + 1]

            # 双线性插值
            elevations[valid_mask] = (z11 * (1 - dxs_valid) * (1 - dys_valid) +
                                      z12 * dxs_valid * (1 - dys_valid) +
                                      z21 * (1 - dxs_valid) * dys_valid +
                                      z22 * dxs_valid * dys_valid)

        return elevations
    
    def calculate_sunrise_score(self, lon, lat, dem_array, transform, date=None, max_dist=50000, initial_step=100):
        """计算单个点的日出可见性评分"""
        lon = float(lon)
        lat = float(lat)
        # 计算日出方位角
        azimuth = self.calculate_sunrise_azimuth(lat, lon, date)
        
        # 初始化评分
        score = 5.0  # 满分5分
        current_dist = initial_step
        
        # 沿着日出方向进行射线追踪
        while current_dist <= max_dist:
            # 计算当前射线终点
            end_lon, end_lat = self.calculate_endpoint(lon, lat, azimuth, current_dist)
            
            # 获取起点和终点的高程
            start_elevation = self.batch_bilinear_interpolation(dem_array, transform, [lon], [lat])[0]
            end_elevation = self.batch_bilinear_interpolation(dem_array, transform, [end_lon], [end_lat])[0]
            
            # 计算视线高度
            # 使用线性插值计算视线高度
            sight_height = start_elevation + (end_elevation - start_elevation) * (current_dist / max_dist)
            
            # 计算地球曲率影响
            earth_curvature = current_dist * current_dist / (2 * 6371000)  # 地球曲率修正
            
            # 计算实际视线高度
            actual_sight_height = start_elevation + earth_curvature
            
            # 检查是否被遮挡
            if end_elevation > actual_sight_height:
                # 计算遮挡程度
                obstruction_height = end_elevation - actual_sight_height
                obstruction_ratio = obstruction_height / (end_elevation - start_elevation + 1e-6)
                
                # 根据遮挡程度扣分
                if obstruction_ratio > 0.8:
                    score -= 2.0  # 严重遮挡
                elif obstruction_ratio > 0.5:
                    score -= 1.5  # 部分遮挡
                elif obstruction_ratio > 0.2:
                    score -= 1.0  # 轻微遮挡
                else:
                    score -= 0.5  # 轻微影响
                
                # 如果完全被遮挡，直接返回0分
                if score <= 0:
                    return 0.0
            
            # 增加步长（自适应步长）
            if current_dist < 1000:
                current_dist += initial_step
            elif current_dist < 5000:
                current_dist += initial_step * 2
            else:
                current_dist += initial_step * 5
        
        return max(0.0, score)
    
    def process_batch(self, args):
        """处理一批点数据"""
        batch_indices, shared_dem, transform, date, max_dist, initial_step = args
        
        # 从共享内存恢复DEM数组
        dem_array = np.frombuffer(shared_dem, dtype=np.float32).reshape(self.dem_array.shape)
        
        scores = []
        for idx, point in batch_indices:
            lon, lat = point.x, point.y
            score = self.calculate_sunrise_score(lon, lat, dem_array, transform, date, max_dist, initial_step)
            scores.append(score)
        
        return batch_indices, scores
    
    def gpu_calculate_sunrise_scores(self, points, dem_array, transform, date=None, max_dist=50000, initial_step=100):
        """
        使用GPU批量计算日出可见性评分
        """
        if not GPU_AVAILABLE or cp is None:
            raise RuntimeError("GPU计算不可用，请安装CuPy")

        # 日志：检查points类型
        print(f"[SunshineAnalysis] points类型: {type(points)}, 第一个元素: {type(points[0])}")
        print(f"[SunshineAnalysis] 示例点: {points[0]}")

        # 将数据转移到GPU
        dem_gpu = cp.asarray(dem_array)
        transform_gpu = cp.array(transform)

        # 结果数组
        scores = cp.zeros(len(points))

        # 地球参数
        R = 6371000.0
        sun_elevation = -0.833  # 太阳中心高度角

        # 将点坐标转换为GPU数组，强制float32
        try:
            lons = cp.array([float(p.x()) for p in points], dtype=cp.float32)
            lats = cp.array([float(p.y()) for p in points], dtype=cp.float32)
        except Exception as e:
            print(f"[SunshineAnalysis] 点坐标转换失败: {e}")
            raise

        # 计算日出方位角 (改进版，GPU实现)
        if date is None:
            # 简化计算
            azimuths = cp.where(lats > 0,
                                90 - 30 * cp.cos(cp.radians(lats)),
                                90 + 30 * cp.cos(cp.radians(lats)))
        else:
            # 使用改进的方位角计算
            if ASTRAL_AVAILABLE:
                try:
                    # 解析日期字符串
                    if isinstance(date, str):
                        date_obj = datetime.strptime(date, "%Y-%m-%d").date()
                    else:
                        date_obj = date
                    
                    # 计算一年中的天数
                    day_of_year = date_obj.timetuple().tm_yday
                    
                    # 计算太阳赤纬
                    declination = 23.45 * cp.sin(cp.radians(360 * (day_of_year - 80) / 365))
                    
                    # 计算方位角偏移
                    azimuth_offset = cp.where(lats > 0,
                                             -declination * cp.cos(cp.radians(lats)),
                                             declination * cp.cos(cp.radians(cp.abs(lats))))
                    
                    azimuths = 90 + azimuth_offset
                    
                except Exception as e:
                    self.log(f"GPU精确计算失败: {e}，使用默认值", level=1)
                    azimuths = cp.full(len(points), 90.0)
            else:
                # 简化计算
                if isinstance(date, str):
                    try:
                        date_obj = datetime.strptime(date, "%Y-%m-%d")
                        day_of_year = date_obj.timetuple().tm_yday
                        declination = 23.45 * math.sin(math.radians(360 * (day_of_year - 80) / 365))
                        
                        # 为每个点计算方位角
                        azimuths = cp.zeros(len(points))
                        for i in range(len(points)):
                            lat_val = cp.asnumpy(lats[i])
                            if lat_val > 0:  # 北半球
                                azimuth_offset = -declination * math.cos(math.radians(lat_val))
                            else:  # 南半球
                                azimuth_offset = declination * math.cos(math.radians(abs(lat_val)))
                            azimuths[i] = 90 + azimuth_offset
                            
                    except:
                        azimuths = cp.full(len(points), 90.0)
                else:
                    azimuths = cp.full(len(points), 90.0)

        # 获取目标点高程
        xs = (lons - transform_gpu[0]) / transform_gpu[1]
        ys = (lats - transform_gpu[3]) / transform_gpu[5]

        x1s = cp.floor(xs).astype(int)
        y1s = cp.floor(ys).astype(int)
        dxs = xs - x1s
        dys = ys - y1s

        # 边界检查
        valid_mask = (x1s >= 0) & (y1s >= 0) & (x1s < dem_gpu.shape[1] - 1) & (y1s < dem_gpu.shape[0] - 1)

        # 获取四个邻近点的高程
        z11 = dem_gpu[y1s, x1s]
        z12 = dem_gpu[y1s, x1s + 1]
        z21 = dem_gpu[y1s + 1, x1s]
        z22 = dem_gpu[y1s + 1, x1s + 1]

        # 双线性插值
        h_start = (z11 * (1 - dxs) * (1 - dys) +
                   z12 * dxs * (1 - dys) +
                   z21 * (1 - dxs) * dys +
                   z22 * dxs * dys)

        # 初始化
        max_elev_angles = cp.full(len(points), -90.0)
        distances = cp.full(len(points), initial_step, dtype=cp.float32)
        steps = cp.full(len(points), initial_step, dtype=cp.float32)
        sample_counts = cp.zeros(len(points), dtype=cp.int32)

        # 迭代计算
        for _ in range(1000):  # 最大迭代次数
            # 计算采样点坐标
            lon_rad = cp.radians(lons)
            lat_rad = cp.radians(lats)
            azimuth_rad = cp.radians(azimuths)
            angular_distances = distances / R

            # 计算终点坐标
            sin_lat1 = cp.sin(lat_rad)
            cos_lat1 = cp.cos(lat_rad)
            sin_angular = cp.sin(angular_distances)
            cos_angular = cp.cos(angular_distances)

            lat2 = cp.arcsin(sin_lat1 * cos_angular +
                             cos_lat1 * sin_angular * cp.cos(azimuth_rad))

            lon2 = lon_rad + cp.arctan2(cp.sin(azimuth_rad) * sin_angular * cos_lat1,
                                        cos_angular - sin_lat1 * cp.sin(lat2))

            lon_p = cp.degrees(lon2)
            lat_p = cp.degrees(lat2)

            # 获取采样点高程 (简化版，实际应使用双线性插值)
            xs_p = (lon_p - transform_gpu[0]) / transform_gpu[1]
            ys_p = (lat_p - transform_gpu[3]) / transform_gpu[5]

            x1s_p = cp.floor(xs_p).astype(int)
            y1s_p = cp.floor(ys_p).astype(int)
            dxs_p = xs_p - x1s_p
            dys_p = ys_p - y1s_p

            # 边界检查
            valid_mask_p = (x1s_p >= 0) & (y1s_p >= 0) & (x1s_p < dem_gpu.shape[1] - 1) & (y1s_p < dem_gpu.shape[0] - 1)

            # 获取四个邻近点的高程
            z11_p = cp.where(valid_mask_p, dem_gpu[y1s_p, x1s_p], 0)
            z12_p = cp.where(valid_mask_p, dem_gpu[y1s_p, x1s_p + 1], 0)
            z21_p = cp.where(valid_mask_p, dem_gpu[y1s_p + 1, x1s_p], 0)
            z22_p = cp.where(valid_mask_p, dem_gpu[y1s_p + 1, x1s_p + 1], 0)

            # 双线性插值
            h_p = (z11_p * (1 - dxs_p) * (1 - dys_p) +
                   z12_p * dxs_p * (1 - dys_p) +
                   z21_p * (1 - dxs_p) * dys_p +
                   z22_p * dxs_p * dys_p)

            # 地球曲率修正
            h_corr = - (distances ** 2) / (2 * R * 0.17)
            h_p_adjusted = h_p + h_corr

            # 计算视线仰角
            elev_angles = cp.degrees(cp.arctan2(h_p_adjusted - h_start, distances))

            # 更新最大仰角
            max_elev_angles = cp.maximum(max_elev_angles, elev_angles)

            # 提前终止条件
            terminate_mask = (max_elev_angles > sun_elevation + 5.0) | (distances > max_dist)

            # 动态调整步长
            steps = cp.minimum(initial_step * (1 + distances / 10000), max_dist / 10)
            distances += steps
            sample_counts += 1

            # 检查是否所有点都已完成
            if cp.all(terminate_mask):
                break

        # 计算评分
        diff = max_elev_angles - sun_elevation
        scores = cp.zeros(len(points))
        scores = cp.where(max_elev_angles <= sun_elevation, 5.0, scores)
        scores = cp.where((diff < 0.5) & (scores == 0), 4.0, scores)
        scores = cp.where((diff < 1.0) & (scores == 0), 3.0, scores)
        scores = cp.where((diff < 2.0) & (scores == 0), 2.0, scores)
        scores = cp.where((diff < 5.0) & (scores == 0), 1.0, scores)

        return cp.asnumpy(scores)
    
    def analyze_sunrise_visibility(self, dem, points, output, date=None, max_distance=50000, initial_step=100, batch_size=500, use_gpu=False, progress_callback=None, status_callback=None):
        """
        dem: 可以是QgsRasterLayer对象或文件路径
        points: 可以是QgsVectorLayer对象或文件路径
        output: 输出点矢量文件（GPKG/SHP）路径
        """
        # 1. 加载DEM和点图层
        if isinstance(dem, QgsRasterLayer):
            dem_layer = dem
        else:
            dem_layer = QgsRasterLayer(dem, "DEM")
        if not dem_layer.isValid():
            raise ValueError("DEM图层无效")
        if isinstance(points, QgsVectorLayer):
            points_layer = points
        else:
            points_layer = QgsVectorLayer(points, "Points", "ogr")
        if not points_layer.isValid():
            raise ValueError("点图层无效")
        # 2. 自动转换到WGS84
        wgs84 = QgsCoordinateReferenceSystem('EPSG:4326')
        # 点图层转换
        if points_layer.crs() != wgs84:
            temp_points_path = tempfile.mktemp(suffix='.gpkg')
            QgsVectorFileWriter.writeAsVectorFormatV2(points_layer, temp_points_path, QgsProject.instance().transformContext(), wgs84, driverName='GPKG')
            points_layer_wgs = QgsVectorLayer(temp_points_path, "PointsWGS84", "ogr")
            if not points_layer_wgs.isValid():
                raise ValueError("点图层WGS84转换失败")
            points_layer_for_analysis = points_layer_wgs
        else:
            points_layer_for_analysis = points_layer
        # DEM转换
        if dem_layer.crs() != wgs84:
            temp_dem_path = tempfile.mktemp(suffix='.tif')
            import processing
            processing.run("gdal:warpreproject", {
                'INPUT': dem_layer,
                'SOURCE_CRS': dem_layer.crs().authid(),
                'TARGET_CRS': 'EPSG:4326',
                'RESAMPLING': 0,
                'NODATA': None,
                'TARGET_RESOLUTION': None,
                'OPTIONS': '',
                'DATA_TYPE': 0,
                'TARGET_EXTENT': None,
                'TARGET_EXTENT_CRS': None,
                'MULTITHREADING': False,
                'EXTRA': '',
                'OUTPUT': temp_dem_path
            })
            dem_layer_wgs = QgsRasterLayer(temp_dem_path, "DEMWGS84")
            if not dem_layer_wgs.isValid():
                raise ValueError("DEM WGS84转换失败")
            dem_layer_for_analysis = dem_layer_wgs
        else:
            dem_layer_for_analysis = dem_layer
        # 3. 后续分析全部用WGS84数据
        # 4. 输出结果后再投影回原始点图层坐标系
        output_wgs = output
        if points_layer.crs() != wgs84:
            output_wgs = tempfile.mktemp(suffix='.gpkg')
        # 4. 遍历每个点，计算得分和方位角
        provider = dem_layer_for_analysis.dataProvider()
        extent = dem_layer_for_analysis.extent()
        width = dem_layer_for_analysis.width()
        height = dem_layer_for_analysis.height()
        block = provider.block(1, extent, width, height)
        ba = block.data()
        # 自动检测数据类型
        qgis_dtype = provider.dataType(1)
        if qgis_dtype == 6:  # Qgis.Float32
            dtype = np.float32
        elif qgis_dtype == 2:  # Qgis.UInt16
            dtype = np.uint16
        elif qgis_dtype == 1:  # Qgis.Int16
            dtype = np.int16
        elif qgis_dtype == 5:  # Qgis.Byte
            dtype = np.uint8
        else:
            dtype = np.float32  # 默认
        arr = np.frombuffer(ba, dtype=np.float32)
        if arr.size == width * height:
            dem_array = arr.reshape((height, width))
        else:
            arr = np.frombuffer(ba, dtype=np.int16)
            if arr.size == width * height:
                dem_array = arr.reshape((height, width))
                print('[SunshineAnalysis] 自动检测到DEM为Int16类型')
                if status_callback:
                    status_callback('自动检测到DEM为Int16类型')
            else:
                raise ValueError(f'栅格数据像元数不匹配，期望{width*height}，实际{arr.size}，请检查DEM数据格式')
        transform = [
            extent.xMinimum(),
            (extent.xMaximum() - extent.xMinimum()) / width,
            0,
            extent.yMaximum(),
            0,
            -(extent.yMaximum() - extent.yMinimum()) / height
        ]
        features = list(points_layer_for_analysis.getFeatures())
        total = len(features)
        # 构建输出字段
        fields = QgsFields()
        for f in points_layer_for_analysis.fields():
            fields.append(f)
        fields.append(QgsField('score', QVariant.Double))
        fields.append(QgsField('azimuth', QVariant.Double))
        # 写入前如有同名文件先删除
        if os.path.exists(output_wgs):
            os.remove(output_wgs)
        if output_wgs.lower().endswith('.shp'):
            driver_name = 'ESRI Shapefile'
        else:
            driver_name = 'GPKG'
        writer = QgsVectorFileWriter(
            output_wgs,
            'utf-8',
            fields,
            QgsWkbTypes.Point,
            points_layer_for_analysis.crs(),
            driver_name
        )
        # 检查是否使用GPU加速
        print(f"[SunshineAnalysis] use_gpu参数: {use_gpu}, GPU_AVAILABLE: {GPU_AVAILABLE}")
        if use_gpu and GPU_AVAILABLE:
            try:
                print("[SunshineAnalysis] 正在使用GPU分支进行分析...")
                if status_callback:
                    status_callback("使用GPU加速计算...")
                # 准备点数据
                points_data = []
                for feat in features:
                    geom = feat.geometry()
                    pt = geom.asPoint()
                    points_data.append(pt)
                # 使用GPU批量计算
                scores = self.gpu_calculate_sunrise_scores(
                    points_data, dem_array, transform, date, max_distance, initial_step
                )
                # 写入结果
                for i, (feat, score) in enumerate(zip(features, scores)):
                    geom = feat.geometry()
                    pt = geom.asPoint()
                    lon, lat = pt.x(), pt.y()
                    azimuth = self.calculate_sunrise_azimuth(lat, lon, date)
                    # 强制类型转换，防止空值
                    try:
                        score_val = float(score)
                    except Exception:
                        score_val = -1.0
                    try:
                        azimuth_val = float(azimuth)
                    except Exception:
                        azimuth_val = -1.0
                    new_feat = QgsFeature(fields)
                    new_feat.setGeometry(geom)
                    attrs = list(feat.attributes())
                    attrs.append(score_val)
                    attrs.append(azimuth_val)
                    new_feat.setAttributes(attrs)
                    writer.addFeature(new_feat)
                    if progress_callback:
                        progress_callback(int((i+1)/total*100))
            except Exception as e:
                print(f"[SunshineAnalysis] GPU计算失败: {e}，回退到CPU计算")
                if status_callback:
                    status_callback(f"GPU计算失败: {e}，回退到CPU计算")
                # 回退到CPU计算
                use_gpu = False
        # CPU计算（如果GPU不可用或失败）
        if not use_gpu or not GPU_AVAILABLE:
            print(f"[SunshineAnalysis] 正在使用CPU分支进行分析... use_gpu: {use_gpu}, GPU_AVAILABLE: {GPU_AVAILABLE}")
            for i, feat in enumerate(features):
                geom = feat.geometry()
                pt = geom.asPoint()
                lon, lat = pt.x(), pt.y()
                azimuth = self.calculate_sunrise_azimuth(lat, lon, date)
                score = self.calculate_sunrise_score(
                    lon, lat, dem_array, transform, date, max_distance, initial_step
                )
                # 强制类型转换，防止空值
                try:
                    score_val = float(score)
                except Exception:
                    score_val = -1.0
                try:
                    azimuth_val = float(azimuth)
                except Exception:
                    azimuth_val = -1.0
                new_feat = QgsFeature(fields)
                new_feat.setGeometry(geom)
                attrs = list(feat.attributes())
                attrs.append(score_val)
                attrs.append(azimuth_val)
                new_feat.setAttributes(attrs)
                writer.addFeature(new_feat)
                if progress_callback:
                    progress_callback(int((i+1)/total*100))
        del writer
        # 如果需要再投影回原始坐标系
        if points_layer.crs() != wgs84:
            final_output = output
            import processing
            processing.run("native:reprojectlayer", {
                'INPUT': output_wgs,
                'TARGET_CRS': points_layer.crs().authid(),
                'OUTPUT': final_output
            })
            # 清理临时文件
            if os.path.exists(output_wgs):
                os.remove(output_wgs)
            if 'temp_points_path' in locals() and os.path.exists(temp_points_path):
                os.remove(temp_points_path)
            if 'temp_dem_path' in locals() and os.path.exists(temp_dem_path):
                os.remove(temp_dem_path)
            output_path = final_output
        else:
            output_path = output_wgs
        # 检查输出有效性
        test_layer = QgsVectorLayer(output_path, "test", "ogr")
        if not test_layer.isValid():
            raise ValueError(f"输出文件无效: {output_path}")
        if status_callback:
            status_callback(f"分析完成，结果已保存到: {output_path}")
        if progress_callback:
            progress_callback(100)
        return output_path 