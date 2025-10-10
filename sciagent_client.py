import requests
import json
import time
from typing import Optional

class SciAgentClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session_id: Optional[str] = None
    
    def start_chat(self, message: str, clear_history: bool = False) -> dict:
        """开始聊天会话"""
        response = requests.post(
            f"{self.base_url}/chat",
            json={
                "message": message,
                "session_id": self.session_id,
                "clear_history": clear_history
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            self.session_id = data["session_id"]
            return data
        else:
            raise Exception(f"请求失败: {response.status_code} - {response.text}")
    
    def send_message(self, message: str) -> dict:
        """发送消息"""
        if not self.session_id:
            return self.start_chat(message)
        
        return self.start_chat(message, clear_history=False)
    
    def poll_response(self) -> dict:
        """轮询响应"""
        if not self.session_id:
            raise Exception("没有活动会话")
        
        response = requests.get(f"{self.base_url}/session/{self.session_id}/poll")
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"轮询失败: {response.status_code} - {response.text}")
    
    def get_status(self) -> dict:
        """获取会话状态"""
        if not self.session_id:
            raise Exception("没有活动会话")
        
        response = requests.get(f"{self.base_url}/session/{self.session_id}/status")
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"获取状态失败: {response.status_code} - {response.text}")
    
    def interactive_chat(self):
        """交互式聊天"""
        print("SciAgent 交互式聊天客户端")
        print("输入 'quit' 退出")
        print("-" * 50)
        
        # 开始会话
        initial_message = input("请输入您的研究问题: ")
        
        try:
            # 发送初始消息
            response = self.start_chat(initial_message, clear_history=True)
            print(f"会话ID: {response['session_id']}")
            print(f"状态: {response['status']}")
            
            # 处理响应
            response_data = json.loads(response['response'])
            if response_data.get('type') == 'input_request':
                print(f"\n系统提示: {response_data.get('prompt', '')}")
            
            while True:
                # 轮询新的响应
                poll_data = self.poll_response()
                
                if poll_data['data']:
                    data = poll_data['data']
                    
                    if data['type'] == 'input_request':
                        # 系统请求输入
                        user_input = input(f"\n{data.get('prompt', '请输入: ')}")
                        if user_input.lower() == 'quit':
                            break
                        
                        # 发送用户输入
                        self.send_message(user_input)
                    
                    elif data['type'] == 'chat_completed':
                        print("\n对话完成!")
                        print(f"结果: {data.get('result', '')}")
                        break
                    
                    elif data['type'] == 'error':
                        print(f"\n错误: {data.get('error', '')}")
                        break
                
                # 检查会话状态
                status = self.get_status()
                if status['status'] in ['completed', 'error', 'timeout']:
                    break
                
                time.sleep(2)  # 等待2秒再轮询
                
        except KeyboardInterrupt:
            print("\n用户中断")
        except Exception as e:
            print(f"\n错误: {e}")
        
        # 清理会话
        if self.session_id:
            try:
                requests.delete(f"{self.base_url}/session/{self.session_id}")
                print("会话已清理")
            except:
                pass

def main():
    client = SciAgentClient()
    client.interactive_chat()

if __name__ == "__main__":
    main()