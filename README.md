# SciAgent-APIBGI-

## 快速开始

### 环境要求

- Python 3.8+
- FastAPI
- AutoGen
- 相关依赖包

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动服务

```bash
chmod +x start_sciagent_api.sh
./start_sciagent_api.sh
```

或者直接运行：

```bash
python sciagent_api.py
```

### 访问界面

打开浏览器访问：`http://localhost:8000/sciagent_web.html`

## API端点

- `POST /chat` - 发送消息到聊天会话
- `GET /session/{session_id}/status` - 获取会话状态
- `GET /session/{session_id}/poll` - 轮询会话输出
- `DELETE /session/{session_id}` - 删除会话
- `GET /sessions` - 列出所有会话

## 使用说明

1. 在Web界面中输入您的研究问题或关键词
2. 系统会自动调用多个AI代理进行协作
3. 生成包含完整研究提案的响应
4. 支持实时查看处理过程和结果

## 技术特点

- **异步处理**：支持长时间运行的AI对话
- **会话管理**：完整的会话生命周期管理
- **错误处理**：完善的错误处理和恢复机制
- **实时更新**：WebSocket风格的实时消息推送

## 贡献

欢迎提交Issue和Pull Request来改进项目。

## 许可证

[添加您的许可证信息]# SciAgent-APIBGI-
