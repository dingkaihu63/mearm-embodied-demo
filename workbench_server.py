"""
MeArm 工作台服务器 — 实时调试与观测仪表盘
============================================
入口脚本（兼容旧启动方式），实际代码已拆分至 mearm_controller/ 包。

用法:
  python workbench_server.py                        # 模拟模式 (无串口)
  python workbench_server.py --port COM3            # 连接 Arduino
  python workbench_server.py --port COM3 --cam 1    # 指定摄像头

然后打开浏览器访问: http://localhost:5000
"""

from mearm_controller.server import main

if __name__ == "__main__":
    main()
