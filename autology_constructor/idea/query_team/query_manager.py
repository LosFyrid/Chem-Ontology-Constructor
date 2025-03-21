from typing import Dict, List, Optional
from pydantic import BaseModel, Field
import uuid
from datetime import datetime, timedelta
from queue import PriorityQueue
import threading
import time
import hashlib



class Query(BaseModel):
    """查询请求"""
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    natural_query: str
    
    # 元数据
    originating_team: str  # dreamer, critic等
    originating_agent: str  # 发起查询的agent
    priority: str = "normal"  # high, normal, low
    
    # 回调信息
    callback_id: Optional[str] = None  # 用于异步回调的ID
    
    # 状态跟踪
    created_at: datetime = Field(default_factory=datetime.now)
    status: str = "pending"  # pending, processing, completed, failed

class QueryCache:
    def __init__(self, ttl: int = 3600):  # 默认缓存1小时
        self.cache = {}  # 查询哈希到结果的映射
        self.timestamps = {}  # 查询哈希到时间戳的映射
        self.ttl = ttl  # 缓存生存时间(秒)
    
    def _generate_key(self, query: Query) -> str:
        """生成更稳定的缓存键"""
        # 使用内容哈希而非对象ID
        natural_query_hash = hashlib.md5(query.natural_query.encode()).hexdigest()
            
        # 使用查询类型、内容和本体哈希构建缓存键
        return f"{query.natural_query}:{natural_query_hash}"
    
    def get(self, query: Query) -> Optional[Dict]:
        """获取缓存的查询结果"""
        key = self._generate_key(query)
        if key in self.cache:
            # 检查是否过期
            timestamp = self.timestamps[key]
            if datetime.now() - timestamp < timedelta(seconds=self.ttl):
                return self.cache[key]
            else:
                # 过期删除
                del self.cache[key]
                del self.timestamps[key]
        return None
    
    def set(self, query: Query, result: Dict) -> None:
        """缓存查询结果"""
        key = self._generate_key(query)
        self.cache[key] = result
        self.timestamps[key] = datetime.now()
    
    def clear(self) -> None:
        """清空缓存"""
        self.cache.clear()
        self.timestamps.clear()
    
    def cleanup(self) -> int:
        """清理过期缓存，返回清理的数量"""
        expired_keys = []
        now = datetime.now()
        
        for key, timestamp in self.timestamps.items():
            if now - timestamp > timedelta(seconds=self.ttl):
                expired_keys.append(key)
                
        for key in expired_keys:
            del self.cache[key]
            del self.timestamps[key]
            
        return len(expired_keys)


class QueryQueueManager:
    def __init__(self):
        self.pending_queries = PriorityQueue()  # 优先级队列
        self.active_queries = {}  # 正在处理的查询
        self.completed_queries = {}  # 已完成的查询
        self.callbacks = {}
        self.retries = {}  # 查询ID到重试次数的映射
        self.max_retries = 3  # 最大重试次数
        self.failed_queries = {}  # 失败的查询
        self.cache = QueryCache()
        
    def enqueue(self, query: Query) -> str:
        """添加查询到队列，先检查缓存"""
        # 检查缓存
        cached_result = self.cache.get(query)
        if cached_result:
            # 直接存储为完成结果
            query.status = "completed"
            self.completed_queries[query.query_id] = (query, cached_result)
            return query.query_id
            
        # 无缓存时正常入队
        priority = {"high": 1, "normal": 2, "low": 3}.get(query.priority, 2)
        self.pending_queries.put((priority, query))
        return query.query_id
        
    def get_next_query(self) -> Optional[Query]:
        """获取下一个要处理的查询"""
        if self.pending_queries.empty():
            return None
        _, query = self.pending_queries.get()
        self.active_queries[query.query_id] = query
        query.status = "processing"
        return query
        
    def store_result(self, query_id: str, result: Dict) -> None:
        """存储查询结果并缓存"""
        if query_id in self.active_queries:
            query = self.active_queries.pop(query_id)
            query.status = "completed"
            self.completed_queries[query_id] = (query, result)
            
            # 缓存结果(非错误结果才缓存)
            if not result.get("error"):
                self.cache.set(query, result)
            
    def get_result(self, query_id: str) -> Optional[Dict]:
        """获取查询结果，如果存在"""
        if query_id in self.completed_queries:
            return self.completed_queries[query_id][1]
        return None
    
    def register_callback(self, query_id: str, callback_fn: callable) -> None:
        """注册查询完成时的回调函数"""
        self.callbacks[query_id] = callback_fn
        
    def mark_failed(self, query_id: str, error_message: str) -> None:
        """标记查询失败"""
        if query_id in self.active_queries:
            query = self.active_queries.pop(query_id)
            query.status = "failed"
            self.failed_queries[query_id] = query  # 添加到失败查询字典
            self.completed_queries[query_id] = (query, {"error": error_message})
    
    def _trigger_callback(self, query_id: str) -> None:
        """触发查询回调"""
        if query_id in self.callbacks and query_id in self.completed_queries:
            try:
                callback = self.callbacks.pop(query_id)
                query, result = self.completed_queries[query_id]
                callback(query, result)
            except Exception as e:
                print(f"回调执行错误: {str(e)}")
    def retry_query(self, query_id: str) -> bool:
        """重试失败的查询"""
        if query_id in self.failed_queries:
            query = self.failed_queries[query_id]
            current_retries = self.retries.get(query_id, 0)
            
            if current_retries < self.max_retries:
                # 更新重试计数
                self.retries[query_id] = current_retries + 1
                
                # 重新入队
                query.status = "pending"
                priority = {"high": 1, "normal": 2, "low": 3}.get(query.priority, 2)
                self.pending_queries.put((priority, query))
                
                # 从失败列表中移除
                del self.failed_queries[query_id]
                return True
                
        return False

class QueryManager:
    """独立的查询管理器，负责处理和管理所有查询请求"""
    
    def __init__(self):
        """初始化查询管理器"""
        self.query_manager = QueryQueueManager()
        self._query_worker = None
        self._stop_worker = False
        
        # 初始化LangGraph
        self.query_graph = create_query_graph()
        self.graph_saver = MemorySaver()
        self._subscribers = {}  # 用于存储订阅查询结果的回调
    
    def start_worker(self):
        """启动查询处理工作线程"""
        if self._query_worker is None or not self._query_worker.is_alive():
            self._stop_worker = False
            self._query_worker = threading.Thread(target=self._process_queries)
            self._query_worker.daemon = True
            self._query_worker.start()
    
    def stop_worker(self):
        """停止查询处理工作线程"""
        self._stop_worker = True
        if self._query_worker:
            self._query_worker.join(timeout=2.0)
            
    def _process_queries(self):
        """查询处理线程的主循环"""
        while not self._stop_worker:
            query = self.query_manager.get_next_query()
            if query:
                try:
                    # 调用LangGraph执行查询
                    result = self._execute_query_with_langgraph(query)
                    self.query_manager.store_result(query.query_id, result)
                    # 触发回调
                    self._notify_subscribers(query.query_id)
                except Exception as e:
                    self.handle_error(e)
                    self.query_manager.mark_failed(query.query_id, str(e))
            else:
                # 没有查询时短暂休眠，避免CPU空转
                time.sleep(0.1)
    
    def handle_error(self, error: Exception):
        """处理错误"""
        print(f"查询错误: {str(error)}")
        
    def _execute_query_with_langgraph(self, query: Query) -> Dict:
        """使用LangGraph执行查询"""
        # 与原代码相同，但不依赖于Dreamer状态
        
    def submit_query(self, query_text: str, 
                    query_context: Dict = None,
                    priority: str = "normal") -> str:
        """提交查询并返回查询ID"""
        # 创建查询实例
        query = Query(
            query_id=str(uuid.uuid4()),
            natural_query=query_text,
            query_strategy="tool_sequence",
            templated_query="",
            **query_context,  # 额外的上下文信息
            priority=priority
        )
        
        return self.query_manager.enqueue(query)
    
    def subscribe(self, query_id: str, callback: callable) -> None:
        """订阅查询结果"""
        self._subscribers[query_id] = callback
        
    def _notify_subscribers(self, query_id: str) -> None:
        """当查询完成时通知订阅者"""
        self.query_manager._trigger_callback(query_id)
        
        # 获取查询结果
        result = self.query_manager.get_result(query_id)
        if result and query_id in self._subscribers:
            # 通知订阅者
            try:
                self._subscribers[query_id](query_id, result)
                # 移除订阅
                del self._subscribers[query_id]
            except Exception as e:
                print(f"通知订阅者错误: {str(e)}")
                
    def get_result(self, query_id: str) -> Optional[Dict]:
        """获取查询结果"""
        return self.query_manager.get_result(query_id)
    
    def retry_failed_queries(self) -> List[str]:
        """重试所有失败的查询"""
        retried_ids = []
        for query_id in list(self.query_manager.failed_queries.keys()):
            if self.query_manager.retry_query(query_id):
                retried_ids.append(query_id)
        return retried_ids