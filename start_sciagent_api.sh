#!/bin/bash

# 设置环境变量
export DASHSCOPE_API_KEY='sk-4c812957a66e4d22afc3d0e3a79f5e00'
export SEMANTIC_SCHOLAR_API_KEY='UaqkBHopjV5UFKKVBkfYe5toY0OsyrqI6VsqSvCg'
export HF_HOME='/home/ZHWang/workspace'

# 切换到正确的目录
cd /home/ZHWang/workspace/SciAgent/SciAgentsDiscovery/API2.0

# 启动API服务
python sciagent_api.py