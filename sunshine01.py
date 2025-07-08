# -*- coding: utf-8 -*-
"""
/***************************************************************************
 Sunshine01
                                 A QGIS plugin
 日出方位角计算和日出可见性分析
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

# standard library
import logging
import os
from pathlib import Path

# PyQGIS
from qgis.core import (
    QgsFeature,
    QgsField,
    QgsGeometry,
    QgsMessageLog,
    QgsPoint,
    QgsProject,
    QgsVectorLayer,
    QgsRasterLayer,
    QgsProcessingFeedback,
    QgsCoordinateReferenceSystem
)
from qgis.PyQt.QtCore import (
    QCoreApplication,
    QFile,
    QFileInfo,
    Qt,
    QTranslator,
    QVariant,
)
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QFileDialog, QMenu
from qgis.utils import iface

# project package
from .sunshine01_dialog import SunshineDialog

# ############################################################################
# ########## Globals ###############
# ##################################

logger = logging.getLogger(__name__)


# ############################################################################
# ########## Classes ###############
# ##################################

class Sunshine01:
    """QGIS Plugin Implementation."""

    def __init__(self, iface):
        """Constructor.

        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgsInterface
        """
        # Save reference to the QGIS interface
        self.iface = iface
        # initialize plugin directory
        self.plugin_dir = os.path.dirname(__file__)
        # initialize locale
        try:
            locale = QCoreApplication.locale().name()[:2]
        except AttributeError:
            # Fallback for older PyQt versions
            locale = 'en'
        
        locale_path = os.path.join(
            self.plugin_dir,
            'i18n',
            'sunshine01_{}.qm'.format(locale)
        )

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)
            QCoreApplication.installTranslator(self.translator)

        # Declare instance attributes
        self.actions = []
        self.menu = self.tr('&Sunshine Analysis')
        
        # 创建工具栏
        self.toolbar = self.iface.addToolBar('Sunshine01')
        self.toolbar.setObjectName('Sunshine01')
        self.toolbar.setWindowTitle(self.tr('Sunshine Analysis'))
        
        # 确保工具栏可见
        self.toolbar.setVisible(True)

    @staticmethod
    def log(
        message: str,
        application: str = "Sunshine Analysis",
        log_level: int = 0,
        push: bool = False,
    ):
        """Send messages to QGIS messages windows and to the user as a message bar.

        :param message: message to display
        :type message: str
        :param application: name of the application sending the message, defaults to "Sunshine Analysis"
        :type application: str, optional
        :param log_level: message level. Possible values: 0 (info), 1 (warning), \
            2 (critical), 3 (success), 4 (none - grey). Defaults to 0 (info)
        :type log_level: int, optional
        :param push: also display the message in the QGIS message bar in addition to the log, defaults to False
        :type push: bool, optional
        """
        # send it to QGIS messages panel
        QgsMessageLog.logMessage(
            message=message, tag=application, notifyUser=push, level=log_level
        )

        # optionally, display message on QGIS Message bar (above the map canvas)
        if push:
            iface.messageBar().pushMessage(
                title=application, text=message, level=log_level, duration=(log_level+1)*3
            )

    # noinspection PyMethodMayBeStatic
    def tr(self, message):
        """Get the translation for a string using Qt translation API.

        We implement this ourselves since we do not inherit QObject.

        :param message: String for translation.
        :type message: str, QString

        :returns: Translated version of message.
        :rtype: QString
        """
        # noinspection PyTypeChecker,PyArgumentList,PyCallByClass
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
        parent=None,
        object_name=None
    ):
        """Add a toolbar icon to the toolbar.

        :param icon_path: Path to the icon for this action. Can be a resource
            path (e.g. ':/plugins/foo/bar.png') or a normal file system path.
        :type icon_path: str

        :param text: Text that should be shown in menu items for this action.
        :type text: str

        :param callback: Function to be called when the action is triggered.
        :type callback: function

        :param enabled_flag: A flag indicating if the action should be enabled
            by default. Defaults to True.
        :type enabled_flag: bool

        :param add_to_menu: Flag indicating whether the action should also
            be added to the menu. Defaults to True.
        :type add_to_menu: bool

        :param add_to_toolbar: Flag indicating whether the action should also
            be added to the toolbar. Defaults to True.
        :type add_to_toolbar: bool

        :param status_tip: Optional text to show in a popup when mouse pointer
            hovers over the action.
        :type status_tip: str

        :param parent: Parent widget for the new action. Defaults None.
        :type parent: QWidget

        :param whats_this: Optional text to show in the status bar when the
            mouse pointer hovers over the action.

        :returns: The action that was created. Note that the action is also
            added to self.actions list.
        :rtype: QAction
        """

        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            # Adds plugin icon to Plugins toolbar
            self.iface.addToolBarIcon(action)
            # Also add to our custom toolbar
            self.toolbar.addAction(action)

        if add_to_menu:
            self.iface.addPluginToMenu(
                self.menu,
                action)

        self.actions.append(action)

        return action

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""

        # 添加日出方位角计算功能
        icon_path = os.path.join(self.plugin_dir, 'icon.png')
        
        # 检查图标文件是否存在，如果不存在则使用默认图标
        if not os.path.exists(icon_path):
            icon_path = os.path.join(self.plugin_dir, 'icon.svg')
            if not os.path.exists(icon_path):
                # 如果都没有，使用空字符串（QGIS会使用默认图标）
                icon_path = ''
        
        self.add_action(
            icon_path,
            text=self.tr(u'日出分析工具'),
            callback=self.run,
            parent=self.iface.mainWindow(),
            status_tip=self.tr(u'计算日出方位角和可见性分析'),
            whats_this=self.tr(u'日出分析工具：计算指定日期的日出方位角，分析地形对日出可见性的影响')
        )

    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginMenu(
                self.tr(u'&Sunshine Analysis'),
                action)
            self.iface.removeToolBarIcon(action)
        
        # Remove the toolbar
        if hasattr(self, 'toolbar'):
            self.toolbar.deleteLater()

    def run(self):
        """Run method that performs all the real work"""
        # Create the dialog with elements (after translation) and keep reference
        self.dlg = SunshineDialog()

        # show the dialog
        self.dlg.show()
        # Run the dialog event loop
        result = self.dlg.exec_()
        # See if OK was pressed
        if result:
            # Do something useful here
            self.log("日出分析工具已启动", push=True)
