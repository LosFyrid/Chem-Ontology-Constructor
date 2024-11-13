from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph import create_react_agent
class AgentTemplate:
    """A template class for creating language model agents with tools.
    
    This class provides functionality to create and configure an agent workflow
    that combines a language model with a set of tools. The workflow allows
    the agent to use tools to accomplish tasks through a series of steps.

    Attributes:
        name (str): The name identifier for this agent template.
        tools (list): The list of tools available to the agent.
        system_prompt (str): The system prompt that defines the agent's behavior.
        model: The language model to use for the agent.
    """

    def __init__(self, name, tools, system_prompt, model):
        self.name = name
        self.tools = tools
        self.system_prompt = system_prompt
        self.model = model.bind_tools(self.tools)

    def create_agent(self):
        """Create an agent."""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK, another assistant with different tools "
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any of the other assistants have the final answer or deliverable,"
                    " prefix your response with FINAL ANSWER so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        prompt = prompt.partial(system_message=self.system_prompt)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in self.tools]))
        return prompt | self.model

    def create_react_agent(self, **kwargs):
        """Creates a ReAct agent using the configured model and tools.
        
        Args:
            **kwargs: 额外的参数将被传递给langgraph的create_react_agent函数
        
        Returns:
            A ReAct agent that can be used to process messages.
        """
        return create_react_agent(
            llm=self.model,
            tools=self.tools,
            system_message=self.system_prompt,
            **kwargs
        )

