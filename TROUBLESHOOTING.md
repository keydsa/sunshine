# Sunshine01 Plugin 故障排除指南

## 插件加载错误解决方案

### 错误信息
```
Python错误 : 无法加载插件"sunshine01"因在调用其classFactory()方法时发生错误
```

### 解决步骤

#### 1. 检查基本文件结构
确保以下文件存在且完整：
```
sunshine01/
├── __init__.py                    # 插件初始化文件
├── sunshine01.py                  # 主插件模块
├── sunshine01_dialog.py           # 对话框实现
├── sunshine01_dialog_base.ui      # 用户界面定义
├── sunshine_analysis.py           # 核心分析功能
├── metadata.txt                   # 插件元数据
└── icon.png                       # 插件图标
```

#### 2. 运行调试脚本
在插件目录中运行调试脚本：
```bash
cd sunshine01
python debug_import.py
```

这将检查：
- Python依赖包是否正确安装
- 模块导入是否正常
- 基本功能是否可用

#### 3. 检查Python依赖
运行依赖安装脚本：
```bash
python install_dependencies.py
```

确保以下包已安装：
- numpy
- geopandas
- gdal
- matplotlib
- scipy
- pyproj
- tqdm

#### 4. 测试简化版本
如果完整版本有问题，可以测试简化版本：

1. 备份当前的 `__init__.py`
2. 复制 `__init__simple.py` 为 `__init__.py`
3. 重启QGIS
4. 测试简化版本是否能加载

#### 5. 检查QGIS版本兼容性
确保：
- QGIS版本 >= 3.0
- Python版本 >= 3.7
- 插件目录路径正确

#### 6. 查看详细错误信息
在QGIS中：
1. 打开 "视图" → "消息日志"
2. 查看Python错误标签页
3. 寻找具体的错误信息

#### 7. 常见问题及解决方案

##### 问题1：模块导入错误
**错误信息**：`ModuleNotFoundError: No module named 'xxx'`

**解决方案**：
```bash
pip install numpy geopandas gdal matplotlib scipy pyproj tqdm
```

##### 问题2：UI文件加载错误
**错误信息**：`No module named 'PyQt5'` 或类似

**解决方案**：
- 确保QGIS使用正确的Python环境
- 检查PyQt版本兼容性

##### 问题3：类名不匹配
**错误信息**：`NameError: name 'MyPlugin' is not defined`

**解决方案**：
- 检查 `__init__.py` 中的类名是否正确
- 确保类名与主文件中的类名一致

##### 问题4：文件路径错误
**错误信息**：`FileNotFoundError`

**解决方案**：
- 检查插件目录路径
- 确保所有文件都在正确位置
- 检查文件权限

#### 8. 手动测试步骤

1. **测试Python环境**：
```python
import sys
print(sys.version)
import qgis
print("QGIS Python环境正常")
```

2. **测试基本导入**：
```python
from qgis.PyQt.QtWidgets import QDialog
print("PyQt导入正常")
```

3. **测试插件目录**：
```python
import os
plugin_dir = "path/to/sunshine01"
print(f"插件目录存在: {os.path.exists(plugin_dir)}")
```

#### 9. 临时解决方案

如果问题持续存在，可以：

1. **使用简化版本**：
   - 使用 `sunshine01_simple.py` 作为主文件
   - 修改 `__init__.py` 导入简化版本

2. **分步测试**：
   - 先测试基本功能
   - 逐步添加复杂功能
   - 逐个模块测试

3. **环境隔离**：
   - 创建新的Python虚拟环境
   - 重新安装所有依赖
   - 测试插件功能

#### 10. 获取帮助

如果问题仍然存在：

1. **收集信息**：
   - QGIS版本
   - Python版本
   - 操作系统
   - 错误日志

2. **运行完整测试**：
```bash
python test_plugin.py
```

3. **检查系统要求**：
   - 内存：建议8GB以上
   - 存储：建议SSD
   - CPU：建议多核

#### 11. 预防措施

1. **定期备份**：
   - 备份插件文件
   - 记录配置更改

2. **版本管理**：
   - 使用版本控制
   - 记录依赖版本

3. **测试环境**：
   - 在测试环境中先验证
   - 逐步部署到生产环境

### 联系支持

如果以上步骤都无法解决问题，请提供：
1. 完整的错误日志
2. 系统环境信息
3. 已尝试的解决步骤
4. 插件版本信息 