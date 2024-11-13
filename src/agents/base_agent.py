from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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

    def create_react_agent(self):
        """Creates an agent workflow with the configured model and tools.

        This method constructs a workflow graph that coordinates between the language
        model and tools. The workflow handles the back-and-forth between model
        invocations and tool executions.

        Returns:
            A compiled workflow graph that can be invoked to run the agent.
        """
        # Define tool node
        tool_node = ToolNode(self.tools)

        # Define graph
        workflow = StateGraph(MessagesState)

        # Define nodes
        workflow.add_node("agent", self.call_model)
        workflow.add_node("tools", tool_node)

        # Set entry point
        workflow.add_edge("agent", "tools")

        # Add conditional edges
        workflow.add_conditional_edges("agent", self.should_continue)

        # Add regular edge
        workflow.add_edge("tools", "agent")

        return workflow.compile()

    def call_model(self, state: MessagesState):
        """Invokes the language model with the current message state.

        This method handles calling the language model with the current conversation
        state and processes its response.

        Args:
            state (MessagesState): The current message state containing the full
                conversation history.

        Returns:
            dict: A dictionary containing the updated messages including the model's
                response.
        """
        messages = state['messages']
        response = self.model.invoke(messages)
        return {"messages": [response]}

    def should_continue(self, state: MessagesState):
        """Determines if the agent should continue processing or terminate.

        This method checks the last message to determine if there are pending tool
        calls that need to be executed, or if the agent should terminate.

        Args:
            state (MessagesState): The current message state containing the full
                conversation history.

        Returns:
            str: Returns "tools" if the last message contains tool calls that need
                to be executed, or "END" if the agent should terminate.
        """
        messages = state['messages']
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return "END"

