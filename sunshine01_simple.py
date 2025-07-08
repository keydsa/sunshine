# -*- coding: utf-8 -*-
"""
/***************************************************************************
 Sunshine01 Simple Version
                                 A QGIS plugin
 日出方位角计算和日出可见性分析 - 简化版
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
from qgis.PyQt.QtCore import QCoreApplication, QTranslator
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QMessageBox
from qgis.utils import iface


class Sunshine01Simple:
    """QGIS Plugin Implementation - Simple Version."""

    def __init__(self, iface):
        """Constructor."""
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        
        # 初始化本地化
        locale = QCoreApplication.locale().name()[:2]
        locale_path = os.path.join(
            self.plugin_dir,
            'i18n',
            'sunshine01_{}.qm'.format(locale)
        )

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)
            QCoreApplication.installTranslator(self.translator)

        # 声明实例属性
        self.actions = []
        self.menu = self.tr('&Sunshine Analysis')

    def tr(self, message):
        """获取翻译字符串"""
        return QCoreApplication.translate('Sunshine01', message)

    def add_action(
        self,
        icon_path,
        text,
        callback,
        enabled_flag=True,
        add_to_menu=True,
        add_to_toolbar=True,
        status_tip=None,
        whats_this=None,
        parent=None
    ):
        """添加工具栏图标"""
        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            self.iface.addToolBarIcon(action)

        if add_to_menu:
            self.iface.addPluginToMenu(self.menu, action)

        self.actions.append(action)
        return action

    def initGui(self):
        """创建菜单条目和工具栏图标"""
        icon_path = ':/plugins/sunshine01/icon.png'
        self.add_action(
            icon_path,
            text=self.tr(u'日出分析工具 (简化版)'),
            callback=self.run,
            parent=self.iface.mainWindow(),
            status_tip=self.tr(u'计算日出方位角和可见性分析'),
            whats_this=self.tr(u'日出分析工具：计算指定日期的日出方位角，分析地形对日出可见性的影响')
        )

    def unload(self):
        """从QGIS GUI中移除插件菜单项和图标"""
        for action in self.actions:
            self.iface.removePluginMenu(self.tr(u'&Sunshine Analysis'), action)
            self.iface.removeToolBarIcon(action)

    def run(self):
        """运行方法"""
        try:
            # 简单的测试功能
            QMessageBox.information(
                iface.mainWindow(),
                "Sunshine Analysis",
                "插件加载成功！\n\n这是一个简化版本，用于测试插件是否能正常加载。\n\n完整功能请使用主版本。"
            )
        except Exception as e:
            QMessageBox.critical(
                iface.mainWindow(),
                "错误",
                f"插件运行出错：{str(e)}"
            ) 