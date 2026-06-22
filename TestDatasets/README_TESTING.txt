TestDatasets 使用说明（POI 八类 + 智能体综合）

1) 生成/更新数据（依赖仓库内 amap_poi_dataset_expanded/all_poi_expanded.csv）

   单城市（自动挑一座同时具备八类的城市）：
   python scripts/build_testdataset_poi8_agent.py

   指定一个城市：
   python scripts/build_testdataset_poi8_agent.py --city 北京市

   一次生成多个区域（逗号分隔）：
   python scripts/build_testdataset_poi8_agent.py --cities 广州市,上海市,成都市,太原市

   一键生成脚本内置的约 20 个代表性城市（推荐人工回归前执行）：
   python scripts/build_testdataset_poi8_agent.py --all-regions

   输出目录：TestDatasets/poi_8types_<城市名>/
   - points.geojson   约 120 点（每类 15，可用 --per-type 调整），字段含 demo_type、point_symbol、query_category
   - manifest.json    元数据

2) 无界面端到端测试（模拟「当前图层」+ 自然语言指令，走完整综合流水线）

   默认使用「最新修改过」的 poi_8types_*/points.geojson：
   python scripts/run_agent_test_poi8_current_layer.py --scale 250000

   指定某一区域的 points.geojson：
   python scripts/run_agent_test_poi8_current_layer.py --geojson E:\GradOpening\MapGenAI\TestDatasets\poi_8types_北京市\points.geojson --scale 220000

   每次运行会在该 geojson 同目录下 agent_runs/run_<时间戳>/ 生成 TEST_REPORT.txt、summary.json 及对比图 compare_before_after_*.png

3) 在 MapGenAI 图形界面人工测试（推荐流程）

   启动程序：
   python main.py

   在软件中打开工作区到 TestDatasets，或直接拖入/打开：
   E:\GradOpening\MapGenAI\TestDatasets\poi_8types_<城市名>\points.geojson

   将 points 设为当前图层后，在 AI 助手输入（比例尺可改）：
   请你对我的当前图层做综合，目标比例尺是1:250000

   对比综合前后：地图工具栏点「对比」，左侧选原始 points 图层，右侧选生成的 points_gen_1_* 图层，拖动中线查看。

4) 符号说明
   - 综合过程会保留各点的 point_symbol 属性；导出 PNG 时若存在该列则按符号分色绘制（见 src/agent/core.py 生成脚本）
   - 主窗口地图渲染由图层样式「点符号」与属性 point_symbol 自动匹配（native 画布）
