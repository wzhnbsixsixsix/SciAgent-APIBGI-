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
import io
import hashlib
from contextlib import redirect_stdout, asynccontextmanager
from datetime import datetime, timedelta

# 阿里云百炼API配置
DASHSCOPE_API_KEY = 'sk-4c812957a66e4d22afc3d0e3a79f5e00'
os.environ['DASHSCOPE_API_KEY'] = DASHSCOPE_API_KEY

os.environ['HF_HOME'] = '/home/ZHWang/workspace'

# Semantic Scholar API配置（保持不变）
SemanticScholar_api_key = 'UaqkBHopjV5UFKKVBkfYe5toY0OsyrqI6VsqSvCg'
os.environ['SEMANTIC_SCHOLAR_API_KEY'] = SemanticScholar_api_key

# 切换到正确的工作目录
os.chdir('/home/ZHWang/workspace/SciAgent/SciAgentsDiscovery')

# 输出目录配置
data_dir_output = './graph_giant_component_LLMdiscovery_example/'

# 设置正确的图文件路径
sys.path.append('.')  # 添加当前目录到路径

# 保存原始的logging模块
import logging as std_logging
std_logging.basicConfig(
    level=std_logging.DEBUG,  # 改为DEBUG级别
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = std_logging.getLogger(__name__)

# 添加SciAgent路径
sys.path.append('/home/ZHWang/workspace/SciAgent/SciAgentsDiscovery')

# 导入SciAgent模块
from ScienceDiscovery import *

# 使用新的lifespan事件处理器替代on_event
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时的初始化
    logger.info("SciAgent API服务启动")
    logger.info(f"图谱节点数: {G.number_of_nodes()}")
    logger.info(f"图谱边数: {G.number_of_edges()}")
    
    # 启动清理任务
    cleanup_task = asyncio.create_task(cleanup_expired_sessions())
    
    yield
    
    # 关闭时的清理
    logger.info("SciAgent API服务关闭")
    cleanup_task.cancel()
    # 清理所有会话
    for session in sessions.values():
        if session.chat_thread and session.chat_thread.is_alive():
            if session.waiting_for_input:
                session.input_queue.put("exit")

app = FastAPI(title="SciAgent API", version="1.0.0", lifespan=lifespan)

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
        
        # 添加消息去重机制
        self.sent_messages = set()  # 已发送的消息ID集合
        self.message_counter = 0    # 消息计数器
        self.last_message_hash = None  # 最后一条消息的哈希
        
    def update_activity(self):
        self.last_activity = datetime.now()
        
    def is_expired(self, timeout_minutes: int = 30) -> bool:
        return datetime.now() - self.last_activity > timedelta(minutes=timeout_minutes)
    
    def add_unique_message(self, message_data):
        """添加消息到队列，确保不重复"""
        try:
            # 生成消息哈希
            content = str(message_data.get('content', ''))
            sender = str(message_data.get('sender', ''))
            msg_type = str(message_data.get('type', ''))
            
            # 限制内容长度用于哈希计算
            hash_content = content[:1000] if len(content) > 1000 else content
            message_hash = hashlib.md5(f"{msg_type}_{sender}_{hash_content}".encode()).hexdigest()
            
            # 检查是否已存在
            if message_hash not in self.sent_messages:
                # 添加哈希和消息ID
                message_data['message_hash'] = message_hash
                if 'message_id' not in message_data:
                    message_data['message_id'] = len(self.sent_messages) + 1
                
                # 添加到队列
                self.output_queue.put(message_data)
                self.sent_messages.add(message_hash)
                
                # 限制sent_messages集合大小，防止内存泄漏
                if len(self.sent_messages) > 1000:
                    # 移除最旧的一半消息哈希
                    old_hashes = list(self.sent_messages)[:500]
                    for old_hash in old_hashes:
                        self.sent_messages.discard(old_hash)
                
                logger.debug(f"Added unique message: type={msg_type}, hash={message_hash[:8]}")
            else:
                logger.debug(f"Skipped duplicate message: type={msg_type}, hash={message_hash[:8]}")
                
        except Exception as e:
            logger.error(f"Error adding message: {e}")
            # 如果出错，仍然尝试添加消息（不去重）
            try:
                self.output_queue.put(message_data)
            except:
                pass

# 全局会话存储
sessions: Dict[str, ChatSession] = {}

# 自定义UserProxyAgent，支持API交互
class APIUserProxyAgent:
    def __init__(self, session: ChatSession):
        self.session = session
        self.name = "user"
        
    def get_human_input(self, prompt: str = "") -> str:
        """获取人类输入，通过API队列机制"""
        logger.info(f"系统请求用户输入: session_id={self.session.session_id}")
        logger.info(f"输入提示: {prompt}")
        
        self.session.waiting_for_input = True
        self.session.current_prompt = prompt
        self.session.status = "waiting_for_input"
        
        # 使用去重方法发送输入请求
        input_request = {
            "type": "input_request",
            "prompt": prompt,
            "timestamp": datetime.now().isoformat()
        }
        self.session.add_unique_message(input_request)
        logger.info(f"已发送输入请求到队列")
        
        # 等待输入
        try:
            logger.info(f"等待用户输入，超时时间: 300秒")
            user_input = self.session.input_queue.get(timeout=300)  # 5分钟超时
            self.session.waiting_for_input = False
            self.session.status = "active"
            
            # 记录用户输入
            logger.info(f"收到用户输入: session_id={self.session.session_id}, input_length={len(user_input)}")
            logger.debug(f"用户输入内容: {user_input}")
            
            return user_input
        except queue.Empty:
            logger.warning(f"用户输入超时: session_id={self.session.session_id}")
            self.session.status = "timeout"
            return "exit"  # 超时退出

def create_custom_user_proxy(session: ChatSession):
    """创建自定义的用户代理"""
    # 修改原有的user agent
    original_user = user
    
    # 重写human_input_mode的行为
    def custom_get_human_input(prompt=""):
        # 删除内容完整性检测，不再调用 check_completion_status
        
        # 检查是否是代理间的自动化流程
        if "critic_agent" in prompt.lower() or "next speaker" in prompt.lower():
            logger.info("检测到代理间自动化流程，自动继续")
            return ""  # 自动继续，不等待用户输入
        
        # 对于用户输入请求，先检查是否有简单的继续信号
        logger.info(f"用户输入请求: {prompt}")
        
        # 获取用户输入
        user_input = APIUserProxyAgent(session).get_human_input(prompt)
        
        # 处理空输入或简单的继续信号
        if user_input.strip() in ["", " ", "continue", "继续", "next", "下一步"]:
            logger.info("检测到继续信号，自动继续对话")
            return ""  # 返回空字符串表示继续
        
        return user_input
    
    # 改进的终止条件判断函数
    def improved_is_termination_msg(x):
        """改进的终止消息判断"""
        content = x.get("content", "").strip()
        
        # 转换为小写进行检查
        content_lower = content.lower()
        
        # 只检查 TERMINATE 关键词
        if "terminate" in content_lower:
            logger.info("检测到TERMINATE关键词")
            # 发送指定格式的chat_completed消息
            session.add_unique_message({
                "type": "chat_completed",
                "result": "研究分析已完成",
                "timestamp": datetime.now().isoformat(),
                "message_id": f"terminate_{session.message_counter}"
            })
            session.status = "completed"
            session.waiting_for_input = False
            return True
                
        return False
    
    custom_user = autogen.UserProxyAgent(
        name="user",
        is_termination_msg=improved_is_termination_msg,
        human_input_mode="ALWAYS",  # 保持ALWAYS以确保及时响应
        system_message="user. You are a human admin. You pose the task. When you want to end the session, please respond with 'TERMINATE'.",
        llm_config=False,
        code_execution_config=False,
    )
    
    # 替换获取输入的方法
    custom_user.get_human_input = custom_get_human_input
    
    # 关键修复：清除原有的函数注册并重新注册
    # 清除planner和assistant的原有函数映射
    if hasattr(planner, '_function_map'):
        planner._function_map.clear()
    if hasattr(assistant, '_function_map'):
        assistant._function_map.clear()
    
    # 定义统一的函数实现
    def generate_path_impl(keyword_1: Annotated[Union[str, None], 'the first node in the knowledge graph. None for random selection.'],
                          keyword_2: Annotated[Union[str, None], 'the second node in the knowledge graph. None for random selection.'],
                         ) -> str:
        
        path_list_for_vis, path_list_for_vis_string = create_path(G, embedding_tokenizer,
                                        embedding_model, node_embeddings , generate_graph_expansion=None,
                                        randomness_factor=0.2, num_random_waypoints=4, shortest_path=False,
                                        second_hop=False, data_dir='./', save_files=False, verbatim=True,
                                        keyword_1 = keyword_1, keyword_2=keyword_2,)

        return path_list_for_vis_string
    
    def rate_novelty_feasibility_impl(hypothesis: Annotated[str, 'the research hypothesis.']) -> str:
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
    
    # 注册到custom_user执行
    @custom_user.register_for_execution()
    def generate_path_custom(keyword_1: Annotated[Union[str, None], 'the first node in the knowledge graph. None for random selection.'],
                            keyword_2: Annotated[Union[str, None], 'the second node in the knowledge graph. None for random selection.'],
                           ) -> str:
        return generate_path_impl(keyword_1, keyword_2)
    
    @custom_user.register_for_execution()
    def rate_novelty_feasibility_custom(hypothesis: Annotated[str, 'the research hypothesis.']) -> str:
        return rate_novelty_feasibility_impl(hypothesis)
    
    # 重新注册到planner和assistant，同时注册LLM调用和执行
    @planner.register_for_llm(description='''This function can be used to create a knowledge path. The function may either take two keywords as the input or randomly assign them and then returns a path between these nodes. 
The path contains several concepts (nodes) and the relationships between them (edges). The function returns the path.
Do not use this function if the path is already provided. If neither path nor the keywords are provided, select None for the keywords so that a path will be generated between randomly selected nodes.''')
    @custom_user.register_for_execution()
    def generate_path(keyword_1: Annotated[Union[str, None], 'the first node in the knowledge graph. None for random selection.'],
                        keyword_2: Annotated[Union[str, None], 'the second node in the knowledge graph. None for random selection.'],
                     ) -> str:
        return generate_path_impl(keyword_1, keyword_2)
    
    @assistant.register_for_llm(description='''This function can be used to create a knowledge path. The function may either take two keywords as the input or randomly assign them and then returns a path between these nodes. 
The path contains several concepts (nodes) and the relationships between them (edges). The function returns the path.
Do not use this function if the path is already provided. If neither path nor the keywords are provided, select None for the keywords so that a path will be generated between randomly selected nodes.''')
    @custom_user.register_for_execution()
    def generate_path_assistant(keyword_1: Annotated[Union[str, None], 'the first node in the knowledge graph. None for random selection.'],
                        keyword_2: Annotated[Union[str, None], 'the second node in the knowledge graph. None for random selection.'],
                     ) -> str:
        return generate_path_impl(keyword_1, keyword_2)
    
    @planner.register_for_llm(description='''Use this function to rate the novelty and feasibility of a research idea against the literature. The function uses semantic scholar to access the literature articles.  
The function will return the novelty and feasibility rate from 1 to 10 (lowest to highest). The input to the function is the hypothesis with its details.''')
    @custom_user.register_for_execution()
    def rate_novelty_feasibility(hypothesis: Annotated[str, 'the research hypothesis.']) -> str:
        return rate_novelty_feasibility_impl(hypothesis)
    
    @assistant.register_for_llm(description='''Use this function to rate the novelty and feasibility of a research idea against the literature. The function uses semantic scholar to access the literature articles.  
The function will return the novelty and feasibility rate from 1 to 10 (lowest to highest). The input to the function is the hypothesis with its details.''')
    @custom_user.register_for_execution()
    def rate_novelty_feasibility_assistant(hypothesis: Annotated[str, 'the research hypothesis.']) -> str:
        return rate_novelty_feasibility_impl(hypothesis)
    
    return custom_user

# 删除整个 check_completion_status 函数
# def check_completion_status(session: ChatSession) -> bool:
#     """检查研究提案是否完成"""
#     try:
#         # 获取最近的消息历史
#         recent_messages = []
#         temp_queue = queue.Queue()
#         
#         # 从输出队列中提取消息进行分析
#         while not session.output_queue.empty():
#             try:
#                 msg = session.output_queue.get_nowait()
#                 recent_messages.append(msg)
#                 temp_queue.put(msg)  # 保存消息以便放回队列
#             except queue.Empty:
#                 break
#         
#         # 将消息放回队列
#         while not temp_queue.empty():
#             session.output_queue.put(temp_queue.get())
#         
#         # 检查是否包含研究提案的所有必要部分
#         required_sections = [
#             "hypothesis", "result", "mechanism", "design", 
#             "unexpected", "comparison", "novelty"
#         ]
#         
#         found_sections = set()
#         message_content = ""
#         
#         for msg in recent_messages:
#             if isinstance(msg, dict) and msg.get("type") in ["message", "agent_message"]:
#                 content = msg.get("content", "").lower()
#                 message_content += content + " "
#                 
#                 for section in required_sections:
#                     if section in content:
#                         found_sections.add(section)
#         
#         # 如果找到了大部分必要部分，认为提案基本完成
#         completion_ratio = len(found_sections) / len(required_sections)
#         
#         # 同时检查消息长度，确保有足够的内容
#         has_sufficient_content = len(message_content) > 1000
#         
#         return completion_ratio >= 0.7 and has_sufficient_content
#         
#     except Exception as e:
#         logger.error(f"Error checking completion status: {e}")
#         return False

def run_chat_session(session: ChatSession, initial_message: str, clear_history: bool = False):
    """在后台线程中运行聊天会话"""
    import time
    start_time = time.time()
    timeout_seconds = 600  # 10分钟超时
    
    # 保存原始stdout
    original_stdout = sys.stdout
    
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
            max_round=35,  # 减少到35轮
            admin_name='user', 
            send_introductions=True, 
            allow_repeat_speaker=True,
            speaker_selection_method='auto',
        )
        
        # 改进的系统消息，删除 TASK COMPLETED 相关内容
        # 保持和原来一样的简单系统消息
        improved_system_message = 'you dynamically select a speaker.'
        
        custom_manager = autogen.GroupChatManager(
            groupchat=custom_groupchat, 
            llm_config=gpt4turbo_config, 
            system_message=improved_system_message
        )
        
        session.user_proxy = custom_user
        session.manager = custom_manager
        
        # 创建stdout捕获器
        class StdoutCapture:
            def __init__(self, session):
                self.session = session
                self.original_stdout = original_stdout  # 使用函数作用域的变量
                self.in_tool_call_block = False
                self.buffer = ""  # 添加缓冲区
                self.content_buffer = []  # 新增：内容缓冲区
                self.last_flush_time = time.time()  # 新增：上次刷新时间
                self.buffer_timeout = 2.0  # 新增：缓冲超时时间（秒）
                
            def write(self, text):
                # 同时写入原始stdout和捕获处理
                self.original_stdout.write(text)
                self.original_stdout.flush()
                
                # 将文本添加到缓冲区
                self.buffer += text
                
                # 检查是否包含完整的消息
                lines = text.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                        
                    # 捕获 Next speaker 信息
                    if "Next speaker:" in line:
                        # 先刷新之前的内容缓冲区
                        self._flush_content_buffer()
                        
                        system_message = {
                            "type": "system_info",
                            "content": line,
                            "timestamp": datetime.now().isoformat()
                        }
                        self.session.add_unique_message(system_message)
                        
                        # 如果下一个发言者是user，预先发送input_request
                        if "Next speaker: user" in line:
                            input_request = {
                                "type": "input_request",
                                "prompt": "System is waiting for your input...",
                                "timestamp": datetime.now().isoformat()
                            }
                            self.session.add_unique_message(input_request)
                            self.session.waiting_for_input = True
                            self.session.status = "waiting_for_input"
                    
                    # 捕获代理名称和消息内容
                    elif line.endswith(":") and len(line.split()) == 1:
                        # 可能是代理名称 - 先刷新之前的内容
                        self._flush_content_buffer()
                        agent_name = line.replace(":", "").strip()
                        if agent_name in ["planner", "assistant", "ontologist", "scientist", 
                                        "hypothesis_agent", "outcome_agent", "mechanism_agent",
                                        "design_principles_agent", "unexpected_properties_agent",
                                        "comparison_agent", "novelty_agent", "critic_agent"]:
                            # 这是一个代理开始发言的标记
                            pass
                    
                    # 捕获实际的消息内容（非系统信息）
                    elif line and not line.startswith("Next speaker:") and not line.startswith("*****"):
                        # 检查是否是有意义的内容
                        if len(line) > 10 and not line.startswith("2025-"):  # 过滤掉日志时间戳
                            # 添加到内容缓冲区而不是立即发送
                            self.content_buffer.append(line)
                            self.last_flush_time = time.time()
                    
                    # 工具调用相关处理
                    if "***** Suggested tool call" in line:
                        self._flush_content_buffer()  # 先刷新内容缓冲区
                        self.in_tool_call_block = True
                        tool_call_message = {
                            "type": "tool_call_suggestion",
                            "content": line,
                            "timestamp": datetime.now().isoformat()
                        }
                        self.session.add_unique_message(tool_call_message)
                    elif self.in_tool_call_block and line.startswith("Arguments:"):
                        args_message = {
                            "type": "tool_call_arguments",
                            "content": line,
                            "timestamp": datetime.now().isoformat()
                        }
                        self.session.add_unique_message(args_message)
                    elif "******************************************************************************" in line:
                        if self.in_tool_call_block:
                            end_message = {
                                "type": "tool_call_end",
                                "content": "Tool call suggestion completed",
                                "timestamp": datetime.now().isoformat()
                            }
                            self.session.add_unique_message(end_message)
                            self.in_tool_call_block = False
                
                # 检查是否需要刷新缓冲区（基于时间或大小）
                current_time = time.time()
                if (len(self.content_buffer) >= 5 or  # 缓冲区有5行或更多
                    (self.content_buffer and current_time - self.last_flush_time > self.buffer_timeout)):  # 超时
                    self._flush_content_buffer()
                            
            def _flush_content_buffer(self):
                """刷新内容缓冲区，将缓冲的内容作为一个消息发送"""
                if self.content_buffer:
                    combined_content = '\n'.join(self.content_buffer)
                    content_message = {
                        "type": "agent_message",
                        "content": combined_content,
                        "timestamp": datetime.now().isoformat()
                    }
                    self.session.add_unique_message(content_message)
                    self.content_buffer.clear()
                    
            def flush(self):
                self.original_stdout.flush()
                # 刷新时也清空内容缓冲区
                self._flush_content_buffer()
        
        # 创建stdout捕获器实例
        stdout_capture = StdoutCapture(session)
        
        # 添加消息监听器来实时输出对话内容
        def message_handler(recipient, messages, sender, config):
            """处理对话消息"""
            try:
                # 检查超时
                if time.time() - start_time > timeout_seconds:
                    logger.warning("Chat session timeout reached")
                    # 发送超时完成消息
                    session.output_queue.put({
                        "type": "chat_completed",
                        "result": "Session timeout - automatically completed",
                        "timestamp": datetime.now().isoformat(),
                        "message_id": f"timeout_{session.message_counter}"
                    })
                    session.status = "completed"
                    return True, "TIMEOUT: Session has exceeded maximum duration"
                
                # 获取最新消息
                if messages and len(messages) > 0:
                    latest_message = messages[-1]
                    
                    # 检查消息内容是否包含完成信号
                    if isinstance(latest_message, dict):
                        content = latest_message.get("content", "").lower()
                        
                        # 检查是否包含终止信号
                        if "terminate" in content:
                            logger.info(f"在消息中检测到终止信号: terminate")
                            # 立即发送chat_completed消息
                            session.output_queue.put({
                                "type": "chat_completed", 
                                "result": "研究分析已完成",
                                "timestamp": datetime.now().isoformat(),
                                "message_id": f"msg_completion_{session.message_counter}"
                            })
                            session.status = "completed"
                            return True, "TERMINATE"
                        
                        return False, None
                        
            except Exception as e:
                logger.error(f"Message handler error: {e}")
                return False, None
        
        # 重定向stdout
        sys.stdout = stdout_capture
        
        # 开始对话
        result = custom_user.initiate_chat(
            recipient=custom_manager,
            message=initial_message,
            clear_history=clear_history
        )
        
        # 对话完成 - 确保发送chat_completed消息
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
            
        # 确保发送最终的chat_completed消息
        final_completion_msg = {
            "type": "chat_completed",
            "result": result_content,
            "timestamp": datetime.now().isoformat(),
            "message_id": f"final_completion_{session.message_counter}",
            "session_status": "completed"
        }
        session.output_queue.put(final_completion_msg)
        logger.info(f"发送最终完成消息: {final_completion_msg}")
        
    except Exception as e:
        logger.error(f"Chat session error: {e}")
        session.status = "error"
        # 即使出错也发送完成消息
        error_completion_msg = {
            "type": "chat_completed",
            "result": f"Session completed with error: {str(e)}",
            "error": str(e).replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t'),
            "timestamp": datetime.now().isoformat(),
            "message_id": f"error_completion_{session.message_counter}",
            "session_status": "error"
        }
        session.output_queue.put(error_completion_msg)
        logger.info(f"发送错误完成消息: {error_completion_msg}")
    finally:
        # 恢复原始stdout
        sys.stdout = original_stdout

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

@app.get("/session/{session_id}/poll")
async def poll_session(session_id: str):
    """轮询会话状态和消息"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    session.update_activity()
    
    # 记录会话状态和队列大小
    logger.debug(f"轮询会话: session_id={session_id}, status={session.status}, "
                f"output_queue_size={session.output_queue.qsize()}, "
                f"input_queue_size={session.input_queue.qsize()}, "
                f"waiting_for_input={session.waiting_for_input}")
    
    # 尝试从输出队列获取消息
    try:
        message_data = session.output_queue.get_nowait()
        logger.debug(f"从队列获取消息: session_id={session_id}, type={message_data.get('type', 'unknown')}")
        return {
            "session_id": session_id,
            "data": message_data,
            "status": session.status,
            "timestamp": datetime.now().isoformat()
        }
    except queue.Empty:
        logger.debug(f"队列为空: session_id={session_id}, status={session.status}")
        
        # 如果会话正在等待用户输入，返回input_request
        if session.waiting_for_input and session.status == "waiting_for_input":
            logger.info(f"会话等待用户输入，返回input_request: session_id={session_id}")
            return {
                "session_id": session_id,
                "data": {
                    "type": "input_request",
                    "prompt": session.current_prompt,
                    "timestamp": datetime.now().isoformat()
                },
                "status": session.status,
                "timestamp": datetime.now().isoformat()
            }
        
        # 如果会话已完成，返回完成状态
        if session.status == "completed":
            return {
                "session_id": session_id,
                "data": {
                    "type": "chat_completed",
                    "result": "Session completed - no more messages",
                    "timestamp": datetime.now().isoformat(),
                    "message_id": f"empty_completion_{session.message_counter}",
                    "session_status": "completed"
                },
                "status": session.status,
                "timestamp": datetime.now().isoformat()
            }
        
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
        
        await asyncio.sleep(600)  # 每10分钟清理一次

# 启动清理任务
@app.on_event("startup")
async def start_cleanup_task():
    asyncio.create_task(cleanup_expired_sessions())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
