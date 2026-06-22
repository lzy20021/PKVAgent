你刚才打开的是“Esri 样式 (*.style)”对话框，所以它不会显示 PNG/BMP。

本目录里的图片只是符号源文件。要得到 ArcGIS 10.2 能直接打开的 .style：

1. 把整个 arcgis_10_2_poi_symbols 文件夹复制到安装了 ArcGIS Desktop 10.2 的电脑上。
2. 双击运行 build_style_arcobjects_10_2.bat。
3. 成功后会在本目录生成：
   MapGenAI_POI_Classic.style
4. 回到 ArcMap/ArcCatalog，在你截图的这个窗口里选择 MapGenAI_POI_Classic.style。

如果 bat 失败，说明该机器没有 ArcGIS Desktop 10.2 的 ArcObjects .NET 组件，
或者没有可用的 Basic/Standard/Advanced 许可。此时只能在 ArcMap 里用
Customize > Style Manager > Create New Style，再逐个 New > Marker Symbol，
Type 选择 Picture Marker Symbol，导入本目录的 PNG/BMP。

八类符号：
- 便利店: convenience_store.bmp / convenience_store.png
- 学校: school.bmp / school.png
- 停车场: parking.bmp / parking.png
- 酒店: hotel.bmp / hotel.png
- 银行: bank.bmp / bank.png
- 医院: hospital.bmp / hospital.png
- 餐饮: restaurant.bmp / restaurant.png
- 交通设施: transit.bmp / transit.png
