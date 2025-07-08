# -*- coding: utf-8 -*-
"""
/***************************************************************************
 Sunshine01 Simple
                                 A QGIS plugin
 日出方位角计算和日出可见性分析 - 简化版
                             -------------------
        begin                : 2025-01-27
        copyright            : (C) 2025 by Sunshine Analysis
        email                : sunshine@example.com
        git sha              : $Format:%H$
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
 This script initializes the plugin, making it known to QGIS.
"""


# noinspection PyPep8Naming
def classFactory(iface):  # pylint: disable=invalid-name
    """Load Sunshine01Simple class from file sunshine01_simple.

    :param iface: A QGIS interface instance.
    :type iface: QgsInterface
    """
    #
    from .sunshine01_simple import Sunshine01Simple
    return Sunshine01Simple(iface) 