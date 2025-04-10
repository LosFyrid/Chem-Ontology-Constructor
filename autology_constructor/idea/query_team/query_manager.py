from enum import Enum
from typing import Dict, List, Optional, Callable, Any
from pydantic import BaseModel, Field
import uuid
from datetime import datetime, timedelta
from queue import PriorityQueue
import threading
import time
import hashlib
# Assuming owlready2 is available in the environment
# from owlready2 import World, ThingClass # For type hinting if needed
from .query_transformers import QueryToStateTransformer, StateToQueryTransformer

class QueryStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class Query(BaseModel):
    """查询请求"""
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    natural_query: str
    
    # 元数据
    originating_team: str  # dreamer, critic等
    originating_agent: str  # 发起查询的agent
    priority: str = "normal"  # high, normal, low
    
    # 查询上下文
    query_context: Dict[str, Any] = Field(default_factory=dict)
    
    # 回调信息
    callback_id: Optional[str] = None  # 用于异步回调的ID
    
    # 状态跟踪
    created_at: datetime = Field(default_factory=datetime.now)
    status: QueryStatus = QueryStatus.PENDING
    result: Optional[Dict] = None
    error: Optional[str] = None

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
            query.status = QueryStatus.COMPLETED
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
        query.status = QueryStatus.PROCESSING
        return query
        
    def store_result(self, query_id: str, result: Dict) -> None:
        """存储查询结果并缓存"""
        if query_id in self.active_queries:
            query = self.active_queries.pop(query_id)
            query.status = QueryStatus.COMPLETED
            self.completed_queries[query_id] = (query, result)
            
            # 缓存结果(非错误结果才缓存)
            if not result.get("error"):
                self.cache.set(query, result)
            
    def get_result(self, query_id: str) -> Optional[Dict]:
        """获取查询结果，如果存在"""
        if query_id in self.completed_queries:
            return self.completed_queries[query_id][1]
        return None
    
    def register_callback(self, query_id: str, callback_fn: Callable) -> None:
        """注册查询完成时的回调函数"""
        self.callbacks[query_id] = callback_fn
        
    def mark_failed(self, query_id: str, error_message: str) -> None:
        """标记查询失败"""
        if query_id in self.active_queries:
            query = self.active_queries.pop(query_id)
            query.status = QueryStatus.FAILED
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
                query.status = QueryStatus.PENDING
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
        self._query_worker = None
        self._stop_worker = False
        self.query_manager = QueryQueueManager()
        self._subscribers = {}
        self._query_to_state = QueryToStateTransformer()
        self._state_to_query = StateToQueryTransformer()
        self.class_name_cache: List[str] = [] # Add class name cache

        # 推迟 LangGraph 初始化，避免循环导入
        self.query_graph = None
        # self.graph_saver = MemorySaver() # Removed MemorySaver as it wasn't imported
    
    def _initialize_graph(self):
        """Initializes the LangGraph query graph if not already done."""
        if self.query_graph is None:
            from .query_workflow import create_query_graph # Local import
            self.query_graph = create_query_graph()

    def update_class_name_cache(self, ontology: Any):
        """Manually update the class name cache from the ontology."""
        if ontology and hasattr(ontology, 'classes'):
            try:
                self.class_name_cache = sorted([cls.name for cls in ontology.classes()])
                print(f"Class name cache updated with {len(self.class_name_cache)} classes.")
            except Exception as e:
                print(f"Error updating class name cache: {e}")
        else:
            print("Warning: Ontology object invalid or missing 'classes' attribute during cache update.")
            self.class_name_cache = []

    def start_worker(self):
        """启动查询处理工作线程"""
        self._initialize_graph() # Ensure graph is initialized before starting worker
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
                    # Ensure graph is ready
                    if self.query_graph is None:
                         print("Error: Query graph not initialized. Cannot process query.")
                         self.query_manager.mark_failed(query.query_id, "Query graph not initialized")
                         continue # Skip this query
                         
                    # 调用LangGraph执行查询
                    result = self._execute_query_with_langgraph(query)
                    self.query_manager.store_result(query.query_id, result)
                    # 触发回调
                    self._notify_subscribers(query.query_id)
                except Exception as e:
                    self.handle_error(e)
                    self.query_manager.mark_failed(query.query_id, str(e))
                    # Also notify subscribers about the failure
                    self._notify_subscribers(query.query_id) # Notify even on failure
            else:
                # 没有查询时短暂休眠，避免CPU空转
                time.sleep(0.1)
    
    def handle_error(self, error: Exception):
        """处理错误"""
        # Consider adding more sophisticated logging here
        print(f"Query processing error: {str(error)}")
        
    def _execute_query_with_langgraph(self, query: Query) -> Dict:
        """使用LangGraph执行查询"""
        # 将Query转换为QueryState
        query_state = self._query_to_state.transform(query)
        
        # Add the class name cache to the initial state
        query_state["available_classes"] = self.class_name_cache

        # 执行查询工作流
        # Ensure graph is initialized (double check)
        if self.query_graph is None:
             raise RuntimeError("Query graph not initialized before invoking.")
             
        final_state = self.query_graph.invoke(query_state)

        # 将最终的QueryState转换回Query对象（更新状态/结果）
        self._state_to_query.transform(final_state, query)

        return final_state # Return the final state dictionary
    
    def submit_query(self, query_text: str, 
                    query_context: Dict = None,
                    priority: str = "normal") -> str:
        """提交查询并返回查询ID"""
        # 创建查询实例
        query = Query(
            natural_query=query_text,
            originating_team=query_context.get("originating_team", "unknown"),
            originating_agent=query_context.get("originating_agent", "unknown"),
            priority=priority,
            query_context=query_context or {}
        )
        
        return self.query_manager.enqueue(query)
    
    def subscribe(self, query_id: str, callback: Callable) -> None:
        """订阅查询结果"""
        self._subscribers[query_id] = callback
        
    def _notify_subscribers(self, query_id: str) -> None:
        """当查询完成或失败时通知订阅者"""
        # Trigger internal callback if any (seems duplicated, review QueryQueueManager)
        # self.query_manager._trigger_callback(query_id)
        
        # Check for registered subscribers
        if query_id in self._subscribers:
            callback = self._subscribers.pop(query_id) # Remove subscriber after notifying
            query_obj, result_dict = None, None
            
            # Retrieve query and result/error
            if query_id in self.query_manager.completed_queries:
                 query_obj, result_dict = self.query_manager.completed_queries[query_id]
            elif query_id in self.query_manager.failed_queries: # Check failed dict too
                 # This path might be redundant if mark_failed also puts it in completed_queries
                 query_obj = self.query_manager.failed_queries[query_id]
                 result_dict = {"error": query_obj.error if query_obj.error else "Marked as failed"}
            
            if callback and query_obj and result_dict is not None:
                try:
                    # Pass query_id and result_dict (which contains 'error' on failure)
                    callback(query_id, result_dict)
                except Exception as e:
                    print(f"Error executing subscriber callback for query {query_id}: {str(e)}")
            elif callback:
                 # If query info not found but callback exists, notify about the issue
                 try:
                     callback(query_id, {"error": f"Query state for {query_id} not found after processing."}) 
                 except Exception as e:
                     print(f"Error executing subscriber callback (query not found) for query {query_id}: {str(e)}")
                
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