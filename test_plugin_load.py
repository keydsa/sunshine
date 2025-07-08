#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试插件加载
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试基本导入"""
    print("测试基本导入...")
    
    try:
        # 测试PyQt导入
        from qgis.PyQt.QtCore import QCoreApplication
        print("✓ QCoreApplication 导入成功")
        
        # 测试QGIS核心导入
        from qgis.core import QgsMessageLog
        print("✓ QGIS核心模块导入成功")
        
        # 测试插件主模块导入
        from sunshine01 import Sunshine01
        print("✓ 插件主模块导入成功")
        
        # 测试对话框导入
        from sunshine01_dialog import SunshineDialog
        print("✓ 对话框模块导入成功")
        
        # 测试分析模块导入
        from sunshine_analysis import SunshineAnalyzer
        print("✓ 分析模块导入成功")
        
        print("\n所有导入测试通过！")
        return True
        
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_locale_handling():
    """测试locale处理"""
    print("\n测试locale处理...")
    
    try:
        from qgis.PyQt.QtCore import QCoreApplication
        
        # 测试locale方法是否存在
        if hasattr(QCoreApplication, 'locale'):
            print("✓ QCoreApplication.locale() 方法存在")
            locale = QCoreApplication.locale().name()[:2]
            print(f"  当前locale: {locale}")
        else:
            print("⚠ QCoreApplication.locale() 方法不存在，将使用默认locale")
            
        print("✓ Locale处理测试通过")
        return True
        
    except Exception as e:
        print(f"✗ Locale处理失败: {e}")
        return False

def test_plugin_initialization():
    """测试插件初始化"""
    print("\n测试插件初始化...")
    
    try:
        # 创建模拟的iface对象
        class MockIface:
            def addToolBar(self, name):
                return MockToolBar()
            
            def addPluginToMenu(self, menu, action):
                pass
                
            def removePluginMenu(self, menu, action):
                pass
                
            def removeToolBarIcon(self, action):
                pass
                
            def addToolBarIcon(self, action):
                pass
                
            def mainWindow(self):
                return None
                
            def messageBar(self):
                return MockMessageBar()
        
        class MockToolBar:
            def setObjectName(self, name):
                pass
        
        class MockMessageBar:
            def pushMessage(self, title, text, level, duration):
                pass
        
        # 测试插件初始化
        from sunshine01 import Sunshine01
        iface = MockIface()
        plugin = Sunshine01(iface)
        print("✓ 插件初始化成功")
        
        # 测试翻译功能
        translated = plugin.tr("测试")
        print(f"✓ 翻译功能正常: {translated}")
        
        return True
        
    except Exception as e:
        print(f"✗ 插件初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== 插件加载测试 ===\n")
    
    # 运行所有测试
    tests = [
        test_imports,
        test_locale_handling,
        test_plugin_initialization
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"测试 {test.__name__} 出现异常: {e}")
            results.append(False)
    
    # 总结
    print("\n=== 测试总结 ===")
    passed = sum(results)
    total = len(results)
    
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有测试通过！插件应该可以正常加载。")
    else:
        print("⚠ 部分测试失败，请检查上述错误信息。") 