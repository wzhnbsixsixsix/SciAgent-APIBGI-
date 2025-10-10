from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import uuid
import asyncio
import threading
import queue
import json
import os
import sys
from datetime import datetime, timedelta

# 保存原始的logging模块
import logging as std_logging
std_logging.basicConfig(level=std_logging.INFO)
logger = std_logging.getLogger(__name__)

# 添加SciAgent路径
sys.path.append('/home/ZHWang/workspace/SciAgent/SciAgentsDiscovery')

# 导入SciAgent模块
from ScienceDiscovery import *

app = FastAPI(title="SciAgent API", version="1.0.0")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求和响应模型
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    clear_history: bool = False

class ChatResponse(BaseModel):
    session_id: str
    response: str
    status: str
    timestamp: str

class SessionStatus(BaseModel):
    session_id: str
    status: str  # "active", "waiting_for_input", "completed", "error"
    created_at: str
    last_activity: str
    message_count: int

# 会话管理类
class ChatSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.message_count = 0
        self.status = "active"
        self.conversation_history = []
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.chat_thread = None
        self.user_proxy = None
        self.manager = None
        self.waiting_for_input = False
        self.current_prompt = ""
        
    def update_activity(self):
        self.last_activity = datetime.now()
        
    def is_expired(self, timeout_minutes: int = 30) -> bool:
        return datetime.now() - self.last_activity > timedelta(minutes=timeout_minutes)

# 全局会话存储
sessions: Dict[str, ChatSession] = {}

# 自定义UserProxyAgent，支持API交互
class APIUserProxyAgent:
    def __init__(self, session: ChatSession):
        self.session = session
        self.name = "user"
        
    def get_human_input(self, prompt: str = "") -> str:
        """获取人类输入，通过API队列机制"""
        self.session.waiting_for_input = True
        self.session.current_prompt = prompt
        self.session.status = "waiting_for_input"
        
        # 将提示放入输出队列
        self.session.output_queue.put({
            "type": "input_request",
            "prompt": prompt,
            "timestamp": datetime.now().isoformat()
        })
        
        # 等待输入
        try:
            user_input = self.session.input_queue.get(timeout=300)  # 5分钟超时
            self.session.waiting_for_input = False
            self.session.status = "active"
            return user_input
        except queue.Empty:
            self.session.status = "timeout"
            return "exit"  # 超时退出

def create_custom_user_proxy(session: ChatSession):
    """创建自定义的用户代理"""
    # 修改原有的user agent
    original_user = user
    
    # 重写human_input_mode的行为
    def custom_get_human_input(prompt=""):
        return APIUserProxyAgent(session).get_human_input(prompt)
    
    custom_user = autogen.UserProxyAgent(
        name="user",
        is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
        human_input_mode="ALWAYS",  # 改为ALWAYS以确保及时响应
        system_message="user. You are a human admin. You pose the task.",
        llm_config=False,
        code_execution_config=False,
    )
    
    # 替换获取输入的方法
    custom_user.get_human_input = custom_get_human_input
    
    # 重新注册函数到新的用户代理
    @custom_user.register_for_execution()
    def generate_path_custom(keyword_1: Annotated[Union[str, None], 'the first node in the knowledge graph. None for random selection.'],
                            keyword_2: Annotated[Union[str, None], 'the second node in the knowledge graph. None for random selection.'],
                           ) -> str:
        
        path_list_for_vis, path_list_for_vis_string = create_path(G, embedding_tokenizer,
                                        embedding_model, node_embeddings , generate_graph_expansion=None,
                                        randomness_factor=0.2, num_random_waypoints=4, shortest_path=False,
                                        second_hop=False, data_dir='./', save_files=False, verbatim=True,
                                        keyword_1 = keyword_1, keyword_2=keyword_2,)

        return path_list_for_vis_string
    
    @custom_user.register_for_execution()
    def rate_novelty_feasibility_custom(hypothesis: Annotated[str, 'the research hypothesis.']) -> str:
        res = novelty_admin.initiate_chat(
        novelty_assistant,
            clear_history=True,
            silent=False,
            max_turns=10,
        message=f'''Rate the following research hypothesis\n\n{hypothesis}. \n\nCall the function three times at most, but not in parallel. Wait for the results before calling the next function. ''',
            summary_method="reflection_with_llm",
            summary_args={"summary_prompt" : "Return all the results of the analysis as is."}
        )

        return res.summary
    
    return custom_user

def run_chat_session(session: ChatSession, initial_message: str, clear_history: bool = False):
    """在后台线程中运行聊天会话"""
    try:
        # 创建自定义用户代理
        custom_user = create_custom_user_proxy(session)
        
        # 重新创建groupchat和manager，使用自定义用户代理
        custom_groupchat = autogen.GroupChat(
            agents=[custom_user, planner, assistant, ontologist, scientist,
                    hypothesis_agent, outcome_agent, mechanism_agent, 
                    design_principles_agent, unexpected_properties_agent, 
                    comparison_agent, novelty_agent, critic_agent], 
            messages=[], 
            max_round=50, 
            admin_name='user', 
            send_introductions=True, 
            allow_repeat_speaker=True,
            speaker_selection_method='auto',
        )
        
        custom_manager = autogen.GroupChatManager(
            groupchat=custom_groupchat, 
            llm_config=gpt4turbo_config, 
            system_message='you dynamically select a speaker.'
        )
        
        session.user_proxy = custom_user
        session.manager = custom_manager
        
        # 添加消息监听器来实时输出对话内容
        def message_handler(recipient, messages, sender, config):
            """处理对话消息"""
            try:
                # 获取最新消息
                if messages and len(messages) > 0:
                    latest_message = messages[-1]
                    # 确保消息内容是JSON安全的
                    content = latest_message.get("content", "") if isinstance(latest_message, dict) else str(latest_message)
                    # 清理可能导致JSON解析问题的字符
                    content = content.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                    
                    session.output_queue.put({
                        "type": "message",
                        "sender": sender.name if hasattr(sender, 'name') else str(sender),
                        "recipient": recipient.name if hasattr(recipient, 'name') else str(recipient),
                        "content": content,
                        "timestamp": datetime.now().isoformat()
                    })
            except Exception as e:
                logger.error(f"Message handler error: {e}")
            
            # 必须返回 False, None 以继续消息流
            return False, None
        
        # 注册消息处理器到所有代理
        agents_to_monitor = [custom_user, custom_manager, planner, assistant, ontologist, scientist,
                           hypothesis_agent, outcome_agent, mechanism_agent, 
                           design_principles_agent, unexpected_properties_agent, 
                           comparison_agent, novelty_agent, critic_agent]
        
        for agent in agents_to_monitor:
            try:
                agent.register_reply(
                    [autogen.Agent, None],
                    reply_func=message_handler,
                    config={"callback": None}
                )
            except Exception as e:
                logger.warning(f"Failed to register message handler for {agent.name}: {e}")
        
        # 开始对话
        result = custom_user.initiate_chat(
            recipient=custom_manager,
            message=initial_message,
            clear_history=clear_history
        )
        
        # 对话完成 - 安全地处理结果
        session.status = "completed"
        try:
            # 尝试提取有用的结果信息
            if hasattr(result, 'summary'):
                result_content = str(result.summary)
            elif hasattr(result, 'chat_history'):
                result_content = f"Chat completed with {len(result.chat_history)} messages"
            else:
                result_content = "Chat session completed successfully"
        except:
            result_content = "Chat session completed"
            
        session.output_queue.put({
            "type": "chat_completed",
            "result": result_content,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Chat session error: {e}")
        session.status = "error"
        session.output_queue.put({
            "type": "error",
            "error": str(e).replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t'),
            "timestamp": datetime.now().isoformat()
        })

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """发送消息到聊天会话"""
    try:
        # 获取或创建会话
        if request.session_id and request.session_id in sessions:
            session = sessions[request.session_id]
        else:
            session_id = str(uuid.uuid4())
            session = ChatSession(session_id)
            sessions[session_id] = session
        
        session.update_activity()
        session.message_count += 1
        
        # 如果会话正在等待输入，提供输入
        if session.waiting_for_input:
            session.input_queue.put(request.message)
            
            # 等待响应
            try:
                response_data = session.output_queue.get(timeout=30)
                return ChatResponse(
                    session_id=session.session_id,
                    response=json.dumps(response_data),
                    status=session.status,
                    timestamp=datetime.now().isoformat()
                )
            except queue.Empty:
                return ChatResponse(
                    session_id=session.session_id,
                    response="Processing...",
                    status=session.status,
                    timestamp=datetime.now().isoformat()
                )
        
        # 如果是新会话或需要清除历史，启动新的聊天线程
        if session.chat_thread is None or not session.chat_thread.is_alive() or request.clear_history:
            session.chat_thread = threading.Thread(
                target=run_chat_session,
                args=(session, request.message, request.clear_history)
            )
            session.chat_thread.start()
            
            # 等待初始响应或输入请求
            try:
                response_data = session.output_queue.get(timeout=30)
                return ChatResponse(
                    session_id=session.session_id,
                    response=json.dumps(response_data),
                    status=session.status,
                    timestamp=datetime.now().isoformat()
                )
            except queue.Empty:
                return ChatResponse(
                    session_id=session.session_id,
                    response="Chat session started, processing...",
                    status=session.status,
                    timestamp=datetime.now().isoformat()
                )
        
        return ChatResponse(
            session_id=session.session_id,
            response="Session already active",
            status=session.status,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/session/{session_id}/status", response_model=SessionStatus)
async def get_session_status(session_id: str):
    """获取会话状态"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    return SessionStatus(
        session_id=session.session_id,
        status=session.status,
        created_at=session.created_at.isoformat(),
        last_activity=session.last_activity.isoformat(),
        message_count=session.message_count
    )

@app.get("/session/{session_id}/poll")
async def poll_session(session_id: str):
    """轮询会话输出"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    session.update_activity()
    
    try:
        # 非阻塞获取输出
        response_data = session.output_queue.get_nowait()
        return {
            "session_id": session_id,
            "data": response_data,
            "status": session.status,
            "timestamp": datetime.now().isoformat()
        }
    except queue.Empty:
        return {
            "session_id": session_id,
            "data": None,
            "status": session.status,
            "timestamp": datetime.now().isoformat()
        }

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """删除会话"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    # 停止聊天线程
    if session.chat_thread and session.chat_thread.is_alive():
        # 发送退出信号
        if session.waiting_for_input:
            session.input_queue.put("exit")
    
    del sessions[session_id]
    return {"message": "Session deleted"}

@app.get("/sessions")
async def list_sessions():
    """列出所有会话"""
    return {
        "sessions": [
            {
                "session_id": session.session_id,
                "status": session.status,
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "message_count": session.message_count
            }
            for session in sessions.values()
        ]
    }

@app.on_event("startup")
async def startup_event():
    """启动时的初始化"""
    logger.info("SciAgent API服务启动")
    logger.info(f"图谱节点数: {G.number_of_nodes()}")
    logger.info(f"图谱边数: {G.number_of_edges()}")

@app.on_event("shutdown")
async def shutdown_event():
    """关闭时的清理"""
    logger.info("SciAgent API服务关闭")
    # 清理所有会话
    for session in sessions.values():
        if session.chat_thread and session.chat_thread.is_alive():
            if session.waiting_for_input:
                session.input_queue.put("exit")

# 定期清理过期会话
async def cleanup_expired_sessions():
    """清理过期会话"""
    while True:
        try:
            expired_sessions = [
                session_id for session_id, session in sessions.items()
                if session.is_expired()
            ]
            
            for session_id in expired_sessions:
                logger.info(f"清理过期会话: {session_id}")
                session = sessions[session_id]
                if session.chat_thread and session.chat_thread.is_alive():
                    if session.waiting_for_input:
                        session.input_queue.put("exit")
                del sessions[session_id]
                
        except Exception as e:
            logger.error(f"清理会话时出错: {e}")
        
        await asyncio.sleep(300)  # 每5分钟清理一次

# 启动清理任务
@app.on_event("startup")
async def start_cleanup_task():
    asyncio.create_task(cleanup_expired_sessions())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)