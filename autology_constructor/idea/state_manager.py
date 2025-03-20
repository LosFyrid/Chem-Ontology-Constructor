from typing import Dict, List, TypedDict, Literal, Annotated, Optional, Any
from langgraph.graph.message import AnyMessage, add_messages
from pydantic import BaseModel, Field
import uuid
from datetime import datetime, timedelta
from queue import PriorityQueue
import threading
import time
import hashlib

# 导入LangGraph相关组件
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

# 导入Query工作流
from .query_team.query_workflow import create_query_graph, QueryState

class DreamerState(TypedDict):
    """Dreamer团队状态"""
    # 输入
    ontology: Any  # 主要本体
    additional_ontologies: Optional[List[Any]]  # 用于跨领域分析的其他本体
    
    # 分析
    analysis_type: str  # "single_domain" 或 "cross_domain"
    domain_analysis: Optional[Dict]  # 领域结构分析结果
    gap_analysis: Dict  # 研究空白分析
    research_ideas: List[Dict]  # 生成的研究创意
    
    # 评价与改进
    critic_feedback: Optional[Dict]  # 来自Critic Team的反馈
    idea_versions: Optional[List[Dict]]  # 记录创意的不同版本
    
    # 查询管理
    pending_queries: Optional[List[Dict]]  # 等待Query Team处理的查询
    query_results: Optional[Dict]  # Query Team返回的查询结果
    information_needs: Optional[List[Dict]]  # 已识别的信息需求
    
    # 工作流状态管理
    stage: str  # 当前阶段
    previous_stage: Optional[str]  # 上一阶段
    status: str  # 状态：initialized, processing, waiting_for_query, waiting_for_critic, error, completed
    
    # 系统
    messages: Annotated[List[AnyMessage], add_messages]  # 系统消息


class StateManager:
    def __init__(self):
        """初始化Dreamer团队状态管理器"""
        self.state: DreamerState = {
            "ontology": None,
            "additional_ontologies": None,
            "analysis_type": "single_domain",
            "domain_analysis": None,
            "gap_analysis": {},
            "research_ideas": [],
            "critic_feedback": None,
            "idea_versions": [],
            "pending_queries": [],
            "query_results": {},
            "information_needs": [],
            "stage": "initialized",
            "previous_stage": None,
            "status": "initialized",
            "messages": []
        }
        self.query_manager = QueryQueueManager()
        self._query_worker = None
        self._stop_worker = False
        
        # 初始化LangGraph
        self.query_graph = create_query_graph()
        self.graph_saver = MemorySaver()
    
    def start_query_worker(self):
        """启动查询处理工作线程"""
        if self._query_worker is None or not self._query_worker.is_alive():
            self._stop_worker = False
            self._query_worker = threading.Thread(target=self._process_queries)
            self._query_worker.daemon = True
            self._query_worker.start()
    
    def stop_query_worker(self):
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
                    self._notify_query_completed(query.query_id)
                except Exception as e:
                    self.handle_error(e)
                    self.query_manager.mark_failed(query.query_id, str(e))
            else:
                # 没有查询时短暂休眠，避免CPU空转
                time.sleep(0.1)
                
    def handle_error(self, error: Exception):
        """处理错误"""
        # 记录错误到状态
        self.state["messages"].append(f"Error: {str(error)}")
        self.state["status"] = "error"
    
    def _execute_query_with_langgraph(self, query: Query) -> Dict:
        """使用LangGraph执行查询"""
        try:
            # 为查询创建一个唯一的线程ID
            thread_id = str(uuid.uuid4())
            
            # 准备查询状态，确保与Query类字段匹配
            initial_state = {
                "query": query.natural_query,
                "source_ontology": query.ontology,
                "query_type": query.query_type,
                "templated_query": query.templated_query,
                "query_strategy": query.query_strategy,
                "additional_ontology": query.additional_ontology,
                "originating_team": query.originating_team,
                "originating_stage": query.originating_stage,
                "query_results": {},
                "status": "initialized",
                "stage": "querying",
                "previous_stage": None,
                "messages": []
            }
            
            # 执行查询工作流
            for _ in self.query_graph.stream(initial_state, thread_id):
                pass  # 我们不需要处理中间事件
            
            # 获取最终状态
            final_state = self.query_graph.get_state(thread_id)
            
            if final_state.get("status") in ["error", "warning"]:
                return {
                    "error": final_state.get("error", "未知错误"),
                    "messages": final_state.get("messages", []),
                    "status": final_state.get("status")
                }
            
            return {
                "results": final_state.get("query_results", {}),
                "messages": final_state.get("messages", []),
                "status": "success"
            }
        except Exception as e:
            return {"error": str(e), "status": "error"}
    
    def submit_query(self, query_text: str,
                    priority: str = "normal", 
                    originating_stage: str = None) -> str:
        """提交查询并返回查询ID"""
        if originating_stage is None:
            originating_stage = self.state.get("stage", "unknown")
            
        # 创建Query实例，确保所有必需字段都有值
        query = Query(
            query_id=str(uuid.uuid4()),
            natural_query=query_text,
            query_strategy="tool_sequence",  # 默认策略
            templated_query="",  # 初始为空，由agent处理
            ontology=self.state.get("ontology"),
            additional_ontology=self.state.get("additional_ontologies", [None])[0],
            originating_team="dreamer",
            originating_stage=originating_stage,
            priority=priority
        )
        
        query_id = self.query_manager.enqueue(query)
        
        # 更新状态
        pending_queries = self.state.get("pending_queries", [])
        pending_queries.append({"query_id": query_id, "query": query_text})
        
        self.update_state({
            "pending_queries": pending_queries,
            "status": "waiting_for_query"
        })
        
        return query_id
    
    def check_query_result(self, query_id: str) -> Optional[Dict]:
        """检查查询结果，如果完成则返回结果"""
        return self.query_manager.get_result(query_id)
    
    def _notify_query_completed(self, query_id: str) -> None:
        """当查询完成时通知相关组件"""
        self.query_manager._trigger_callback(query_id)
        
        # 获取查询结果
        result = self.query_manager.get_result(query_id)
        if result:
            # 更新状态
            self.process_query_results({query_id: result})
    
    def submit_query_with_callback(self, query_text: str, query_type: str, 
                                  callback_fn: callable,
                                  priority: str = "normal") -> str:
        """提交查询并注册回调函数"""
        query_id = self.submit_query(query_text, query_type, priority)
        self.query_manager.register_callback(query_id, callback_fn)
        return query_id
    
    def retry_failed_queries(self) -> List[str]:
        """重试所有失败的查询"""
        retried_ids = []
        for query_id in list(self.query_manager.failed_queries.keys()):
            if self.query_manager.retry_query(query_id):
                retried_ids.append(query_id)
                self.add_message({"role": "system", "content": f"正在重试查询 {query_id}"})
        return retried_ids

    def process_query_results(self, results: Dict[str, Dict]) -> None:
        """处理查询结果并更新状态"""
        if not results:
            return
            
        # 更新状态中的查询结果
        query_results = self.state.get("query_results", {})
        query_results.update(results)
        
        # 查找并移除已完成的查询
        pending_queries = self.state.get("pending_queries", [])
        completed_query_ids = set(results.keys())
        pending_queries = [q for q in pending_queries if q.get("query_id") not in completed_query_ids]
        
        # 更新状态
        self.update_state({
            "query_results": query_results,
            "pending_queries": pending_queries,
            "status": "processing" if not pending_queries else "waiting_for_query"
        })
    
    def update_state(self, updates: Dict) -> None:
        """更新Dreamer团队状态"""
        for key, value in updates.items():
            if key in self.state:
                self.state[key] = value
    
    def add_message(self, message: Dict) -> None:
        """添加消息到状态"""
        if not isinstance(message, dict) or "role" not in message or "content" not in message:
            raise ValueError("消息必须包含role和content字段")
        self.state["messages"].append(message)

def create_state_manager() -> StateManager:
    """创建并返回一个StateManager实例"""
    return StateManager()


class Query(BaseModel):
    """查询请求"""
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    natural_query: str
    
    # 元数据
    originating_team: str  # dreamer, critic等
    originating_stage: str  # 发起查询的阶段
    priority: str = "normal"  # high, normal, low
    
    # 回调信息
    callback_id: Optional[str] = None  # 用于异步回调的ID
    
    # 状态跟踪
    created_at: datetime = Field(default_factory=datetime.now)
    status: str = "pending"  # pending, processing, completed, failed

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
    



class QueryCache:
    def __init__(self, ttl: int = 3600):  # 默认缓存1小时
        self.cache = {}  # 查询哈希到结果的映射
        self.timestamps = {}  # 查询哈希到时间戳的映射
        self.ttl = ttl  # 缓存生存时间(秒)
    
    def _generate_key(self, query: Query) -> str:
        """生成更稳定的缓存键"""
        # 使用内容哈希而非对象ID
        onto_hash = "none"
        target_hash = "none"
        
        if query.ontology:
            # 获取本体内容的简单表示
            onto_repr = str(type(query.ontology).__name__)
            if hasattr(query.ontology, "get_iri"):
                onto_repr += ":" + str(query.ontology.get_iri())
            onto_hash = hashlib.md5(onto_repr.encode()).hexdigest()
        
        if query.additional_ontology:
            # 获取目标本体内容的简单表示
            target_repr = str(type(query.additional_ontology).__name__)
            if hasattr(query.additional_ontology, "get_iri"):
                target_repr += ":" + str(query.additional_ontology.get_iri())
            target_hash = hashlib.md5(target_repr.encode()).hexdigest()
            
        # 使用查询类型、内容和本体哈希构建缓存键
        return f"{query.natural_query}:{onto_hash}:{target_hash}"
    
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