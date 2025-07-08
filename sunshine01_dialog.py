# -*- coding: utf-8 -*-
"""
/***************************************************************************
 SunshineDialog
                                 A QGIS plugin
 日出方位角计算和日出可见性分析对话框
                             -------------------
        begin                : 2025-01-27
        git sha              : $Format:%H$
        copyright            : (C) 2025 by Sunshine Analysis
        email                : sunshine@example.com
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

import os
import sys
from datetime import datetime, date
from pathlib import Path

from qgis.PyQt import uic
from qgis.PyQt import QtWidgets
from qgis.PyQt.QtCore import QDate, QThread, pyqtSignal, Qt
from qgis.PyQt.QtWidgets import QFileDialog, QMessageBox, QProgressBar, QVBoxLayout, QWidget
from qgis.core import QgsProject, QgsVectorLayer, QgsRasterLayer, QgsMessageLog
from qgis.utils import iface

# 导入我们的分析功能
from .sunshine_analysis import SunshineAnalyzer

# This loads your .ui file so that PyQt can populate your plugin with the elements from Qt Designer
FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'sunshine01_dialog_base.ui'))


class AnalysisThread(QThread):
    """后台分析线程"""
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, analyzer, params):
        super().__init__()
        self.analyzer = analyzer
        self.params = params
    
    def run(self):
        try:
            self.status.emit("开始分析...")
            
            # 检查是否使用GPU
            use_gpu = self.params.get('use_gpu', False)
            if use_gpu:
                self.status.emit("使用GPU加速计算...")
            
            result = self.analyzer.analyze_sunrise_visibility(
                progress_callback=self.progress.emit,
                status_callback=self.status.emit,
                **self.params
            )
            self.finished.emit(True, "分析完成")
        except Exception as e:
            self.finished.emit(False, f"分析失败: {str(e)}")


class SunshineDialog(QtWidgets.QDialog, FORM_CLASS):
    def __init__(self, parent=None):
        """Constructor."""
        super(SunshineDialog, self).__init__(parent)
        # Set up the user interface from Designer through FORM_CLASS.
        # After self.setupUi() you can access any designer object by doing
        # self.<objectname>, and you can use autoconnect slots - see
        # http://qt-project.org/doc/qt-4.8/designer-using-a-ui-file.html
        # #widgets-and-dialogs-with-auto-connect
        self.setupUi(self)
        
        # 初始化分析器
        self.analyzer = SunshineAnalyzer()
        self.analysis_thread = None
        
        # 初始化界面
        self.init_ui()
        
        # 设置默认日期为今天
        if hasattr(self, 'dateEdit'):
            self.dateEdit.setDate(QDate.currentDate())
        
        # 连接信号
        self.connect_signals()
        
        # 填充图层下拉框
        self.populate_layer_combos()
    
    def connect_signals(self):
        """连接信号和槽"""
        try:
            # 文件选择按钮
            if hasattr(self, 'btnSelectDEM'):
                self.btnSelectDEM.clicked.connect(self.select_dem_file)
            if hasattr(self, 'btnSelectPoints'):
                self.btnSelectPoints.clicked.connect(self.select_points_file)
            if hasattr(self, 'btnSelectOutput'):
                self.btnSelectOutput.clicked.connect(self.select_output_file)
            
            # 分析按钮
            if hasattr(self, 'btnAnalyze'):
                self.btnAnalyze.clicked.connect(self.start_analysis)
            
            # 参数变化
            if hasattr(self, 'spinMaxDistance'):
                self.spinMaxDistance.valueChanged.connect(self.update_parameters)
            if hasattr(self, 'spinInitialStep'):
                self.spinInitialStep.valueChanged.connect(self.update_parameters)
            if hasattr(self, 'spinBatchSize'):
                self.spinBatchSize.valueChanged.connect(self.update_parameters)
            if hasattr(self, 'checkGPU'):
                self.checkGPU.stateChanged.connect(self.on_gpu_check_changed)
            
            # 坐标系选择
            if hasattr(self, 'btnSelectCRS'):
                self.btnSelectCRS.clicked.connect(self.select_custom_crs)
            if hasattr(self, 'comboCRS'):
                self.comboCRS.currentIndexChanged.connect(self.on_crs_combo_changed)
                
        except Exception as e:
            print(f"信号连接错误: {e}")
            import traceback
            traceback.print_exc()
    
    def init_ui(self):
        """初始化界面"""
        try:
            # 设置窗口标题
            self.setWindowTitle("日出分析工具")
            
            # 设置默认输出路径
            default_output = os.path.join(os.path.expanduser("~"), "sunrise_analysis.gpkg")
            if hasattr(self, 'lineEditOutput'):
                self.lineEditOutput.setText(default_output)
            
            # 设置参数范围
            if hasattr(self, 'spinMaxDistance'):
                self.spinMaxDistance.setRange(1000, 100000)
                self.spinMaxDistance.setValue(50000)
                self.spinMaxDistance.setSuffix(" 米")
            
            if hasattr(self, 'spinInitialStep'):
                self.spinInitialStep.setRange(10, 1000)
                self.spinInitialStep.setValue(100)
                self.spinInitialStep.setSuffix(" 米")
            
            if hasattr(self, 'spinBatchSize'):
                self.spinBatchSize.setRange(100, 10000)
                self.spinBatchSize.setValue(500)
            
            # 添加进度条
            self.progressBar = QProgressBar()
            self.progressBar.setVisible(False)
            if hasattr(self, 'layout'):
                self.layout().addWidget(self.progressBar)
            
            # 初始化坐标系选择
            self.selected_custom_crs = None
            if hasattr(self, 'btnSelectCRS'):
                self.btnSelectCRS.setEnabled(False)  # 初始状态禁用
            
            # 初始化GPU加速
            self.gpu_available = self.check_gpu_availability()
            if hasattr(self, 'checkGPU'):
                self.checkGPU.setEnabled(self.gpu_available)
                if not self.gpu_available:
                    self.checkGPU.setToolTip("GPU加速不可用，请安装CuPy和CUDA")
                
        except Exception as e:
            print(f"UI初始化错误: {e}")
            import traceback
            traceback.print_exc()
    
    def select_dem_file(self):
        """选择DEM文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择DEM文件",
            "",
            "Raster files (*.tif *.tiff *.asc *.img);;All files (*.*)"
        )
        if file_path:
            self.lineEditDEM.setText(file_path)
            self.validate_inputs()
    
    def select_points_file(self):
        """选择点数据文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择点数据文件",
            "",
            "Vector files (*.shp *.gpkg *.geojson);;All files (*.*)"
        )
        if file_path:
            self.lineEditPoints.setText(file_path)
            self.validate_inputs()
    
    def select_output_file(self):
        """选择输出文件"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "选择输出文件",
            self.lineEditOutput.text(),
            "GeoPackage (*.gpkg);;Shapefile (*.shp);;GeoJSON (*.geojson)"
        )
        if file_path:
            self.lineEditOutput.setText(file_path)
    
    def on_crs_combo_changed(self, index):
        """坐标系下拉框变化时的处理"""
        if hasattr(self, 'btnSelectCRS'):
            # 只有当选择"自定义..."时才启用选择按钮
            self.btnSelectCRS.setEnabled(index == 3)  # 3是"自定义..."的索引
    
    def select_custom_crs(self):
        """选择自定义坐标系"""
        from qgis.gui import QgsProjectionSelectionDialog
        
        dialog = QgsProjectionSelectionDialog(self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            crs = dialog.crs()
            if crs.isValid():
                # 更新下拉框显示选中的坐标系
                crs_description = f"{crs.description()} ({crs.authid()})"
                self.comboCRS.setItemText(3, crs_description)
                # 保存选中的坐标系
                self.selected_custom_crs = crs
    
    def validate_inputs(self):
        """验证输入文件"""
        dem_path = self.lineEditDEM.text()
        points_path = self.lineEditPoints.text()
        
        # 检查DEM文件
        if dem_path and not os.path.exists(dem_path):
            QMessageBox.warning(self, "警告", f"DEM文件不存在: {dem_path}")
            return False
        
        # 检查点数据文件
        if points_path and not os.path.exists(points_path):
            QMessageBox.warning(self, "警告", f"点数据文件不存在: {points_path}")
            return False
        
        # 检查输出目录
        output_path = self.lineEditOutput.text()
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir, exist_ok=True)
                except Exception as e:
                    QMessageBox.warning(self, "警告", f"无法创建输出目录: {e}")
                    return False
        
        return True
    
    def update_parameters(self):
        """更新参数显示"""
        max_dist = self.spinMaxDistance.value()
        initial_step = self.spinInitialStep.value()
        batch_size = self.spinBatchSize.value()
        
        # 获取GPU状态
        gpu_status = "启用" if (hasattr(self, 'checkGPU') and self.checkGPU.isChecked()) else "禁用"
        gpu_available = "可用" if self.gpu_available else "不可用"
        
        # 更新参数说明
        self.labelParamInfo.setText(
            f"最大分析距离: {max_dist:,} 米\n"
            f"初始步长: {initial_step} 米\n"
            f"批次大小: {batch_size} 个点\n"
            f"GPU加速: {gpu_status} ({gpu_available})"
        )
    

    
    def populate_layer_combos(self):
        """填充DEM和点图层下拉框"""
        if hasattr(self, 'comboDEM'):
            self.comboDEM.clear()
            for layer in QgsProject.instance().mapLayers().values():
                if isinstance(layer, QgsRasterLayer):
                    self.comboDEM.addItem(layer.name(), layer.id())
        if hasattr(self, 'comboPoints'):
            self.comboPoints.clear()
            for layer in QgsProject.instance().mapLayers().values():
                if isinstance(layer, QgsVectorLayer) and layer.geometryType() == 0:
                    self.comboPoints.addItem(layer.name(), layer.id())
    
    def get_selected_dem(self):
        """优先返回下拉框选择的DEM图层，否则返回文件路径"""
        if hasattr(self, 'comboDEM') and self.comboDEM.currentIndex() >= 0:
            layer_id = self.comboDEM.itemData(self.comboDEM.currentIndex())
            if layer_id:
                layer = QgsProject.instance().mapLayer(layer_id)
                if layer:
                    return layer
        if hasattr(self, 'lineEditDEM'):
            path = self.lineEditDEM.text().strip()
            if path:
                return path
        return None

    def get_selected_points(self):
        """优先返回下拉框选择的点图层，否则返回文件路径"""
        if hasattr(self, 'comboPoints') and self.comboPoints.currentIndex() >= 0:
            layer_id = self.comboPoints.itemData(self.comboPoints.currentIndex())
            if layer_id:
                layer = QgsProject.instance().mapLayer(layer_id)
                if layer:
                    return layer
        if hasattr(self, 'lineEditPoints'):
            path = self.lineEditPoints.text().strip()
            if path:
                return path
        return None

    def get_output_path(self):
        """输出路径为空时自动生成GeoTIFF临时文件"""
        if hasattr(self, 'lineEditOutput'):
            path = self.lineEditOutput.text().strip()
            if path:
                return path
        import tempfile
        return tempfile.mktemp(suffix='.gpkg')
    
    def get_output_crs(self):
        """获取输出坐标系"""
        from qgis.core import QgsCoordinateReferenceSystem
        
        crs_index = self.comboCRS.currentIndex()
        
        if crs_index == 0:  # 与项目坐标系一致
            project_crs = QgsProject.instance().crs()
            return project_crs if project_crs.isValid() else QgsCoordinateReferenceSystem('EPSG:4326')
        
        elif crs_index == 1:  # 与输入点图层一致
            points_layer = self.get_selected_points()
            if points_layer and hasattr(points_layer, 'crs') and points_layer.crs().isValid():
                return points_layer.crs()
            else:
                return QgsCoordinateReferenceSystem('EPSG:4326')
        
        elif crs_index == 2:  # WGS84
            return QgsCoordinateReferenceSystem('EPSG:4326')
        
        elif crs_index == 3:  # 自定义
            if hasattr(self, 'selected_custom_crs') and self.selected_custom_crs:
                return self.selected_custom_crs
            else:
                return QgsCoordinateReferenceSystem('EPSG:4326')
        
        else:
            return QgsCoordinateReferenceSystem('EPSG:4326')
    
    def check_gpu_availability(self):
        """检查GPU加速是否可用"""
        try:
            import cupy as cp
            # 尝试创建一个简单的GPU数组来测试
            test_array = cp.array([1, 2, 3])
            del test_array
            return True
        except ImportError:
            return False
        except Exception:
            return False
    
    def on_gpu_check_changed(self, state):
        """GPU复选框状态改变时的处理"""
        if state == 2:  # 选中
            if not self.gpu_available:
                QMessageBox.warning(
                    self,
                    "GPU加速不可用",
                    "GPU加速功能不可用。\n\n"
                    "要启用GPU加速，请安装以下组件：\n"
                    "1. CUDA Toolkit (https://developer.nvidia.com/cuda-downloads)\n"
                    "2. CuPy: pip install cupy-cuda11x (根据您的CUDA版本选择)\n\n"
                    "安装完成后重启QGIS即可使用GPU加速功能。"
                )
                self.checkGPU.setChecked(False)
        else:
            # 取消选中
            pass
    
    def start_analysis(self):
        """启动分析"""
        # 获取输入
        dem_input = self.get_selected_dem()
        points_input = self.get_selected_points()
        output_path = self.get_output_path()
        # 组装参数
        params = {
            'dem': dem_input,
            'points': points_input,
            'output': output_path,
            # 其他参数...
            'date': self.dateEdit.date().toPyDate() if hasattr(self, 'dateEdit') else None,
            'max_distance': self.spinMaxDistance.value() if hasattr(self, 'spinMaxDistance') else 50000,
            'initial_step': self.spinInitialStep.value() if hasattr(self, 'spinInitialStep') else 100,
            'batch_size': self.spinBatchSize.value() if hasattr(self, 'spinBatchSize') else 500,
            'use_gpu': self.checkGPU.isChecked() if hasattr(self, 'checkGPU') else False,
        }
        # 启动线程
        self.progressBar.setVisible(True)
        self.progressBar.setValue(0)
        self.analysis_thread = AnalysisThread(self.analyzer, params)
        self.analysis_thread.progress.connect(self.progressBar.setValue)
        self.analysis_thread.status.connect(self.update_status)
        self.analysis_thread.finished.connect(self.analysis_finished)
        self.analysis_thread.start()
    
    def update_status(self, message):
        """更新状态信息"""
        self.labelStatus.setText(message)
        QgsMessageLog.logMessage(message, "Sunshine Analysis")
    
    def analysis_finished(self, success, message):
        """分析完成回调"""
        self.progressBar.setVisible(False)
        QMessageBox.information(self, "分析完成" if success else "分析失败", message)
        if success:
            # 自动加载输出点矢量图层到QGIS
            output_path = self.get_output_path()
            iface.addVectorLayer(output_path, "分析结果", "ogr")
    
    def closeEvent(self, event):
        """关闭事件"""
        if self.analysis_thread and self.analysis_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "确认",
                "分析正在进行中，确定要退出吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.analysis_thread.terminate()
                self.analysis_thread.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
