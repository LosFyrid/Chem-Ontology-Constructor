from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# 全局LLM实例（可配置参数和复用实例）
llm_instance = ChatOpenAI(temperature=0.7)

def invoke_llm(prompt: ChatPromptTemplate, **kwargs) -> str:
    """
    调用 LLM，传入 prompt 和格式化参数，返回 LLM 响应内容（str）
    """
    try:
        msg = prompt.format_messages(**kwargs)
        response = llm_instance.invoke(msg)
        return response.content.strip()
    except Exception as e:
        # 可以在这里加入日志记录或错误上报
        raise RuntimeError(f"LLM 调用失败：{e}") 