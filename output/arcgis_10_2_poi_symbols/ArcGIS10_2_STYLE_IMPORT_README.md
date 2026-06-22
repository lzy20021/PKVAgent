# ArcGIS 10.2 POI Symbol Pack

这些 PNG/BMP 是为 ArcGIS 10.2 Picture Marker Symbol 准备的符号源文件。
注意：在“打开 Esri 样式 (*.style)”对话框里看不到 PNG/BMP 是正常的；那个对话框只能打开已经生成好的 .style。

建议导入尺寸：
- 1:84000 到 1:120000：使用 96px 或 128px PNG，ArcMap 符号大小设为 13-16 pt。
- 1:253000：使用 64px 或 96px PNG，ArcMap 符号大小设为 10-13 pt。
- 1:532000：使用 48px 或 64px PNG，ArcMap 符号大小设为 8-10 pt。

ArcMap 10.2 生成 .style 的步骤：
1. 打开 ArcMap，进入 Customize > Style Manager。
2. 点击 Styles > Create New Style，保存为 MapGenAI_POI_Classic.style。
3. 在左侧展开新建 style，点击 Marker Symbols。
4. 在右侧空白处右键 New > Marker Symbol。
5. Type 选择 Picture Marker Symbol，点击 Picture，选择本目录对应 PNG。
6. Name 分别填写：便利店、学校、停车场、酒店、银行、医院、餐饮、交通设施。
7. Category 建议填写 MapGenAI POI；Tags 可填写 poi, classic, amenity。
8. 保存后，该 .style 即可在 ArcGIS 10.2 的 Style References 中引用。

如果要自动生成 .style，请把整个目录复制到安装了 ArcGIS Desktop 10.2 的电脑上，运行 build_style_arcobjects_10_2.bat。

符号对应：
- 便利店: convenience_store.png
- 学校: school.png
- 停车场: parking.png
- 酒店: hotel.png
- 银行: bank.png
- 医院: hospital.png
- 餐饮: restaurant.png
- 交通设施: transit.png
