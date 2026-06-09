from ..LLLM.agent_context import Event, ExecutionContext, Message, AgentToolResult
from ..LLLM.tool_common import ToolCall


def test_execution_context_records_and_flattens_events() -> None:
    context = ExecutionContext()

    user_message = context.add_user_message("hello")
    agent_items = [
        Message(role="assistant", content="checking"),
        ToolCall(tool_call_id="call_0_0", name="lookup", arguments={"q": "x"}),
        AgentToolResult(
            tool_call_id="call_0_0",
            name="lookup",
            status="success",
            content=["found"],
        ),
    ]
    context.add_event(
        Event(
            execution_id=context.execution_id,
            author="assistant",
            content=agent_items,
        )
    )
    context.final_result = "done"

    assert user_message == Message(role="user", content="hello")
    assert [event.author for event in context.events] == ["user", "assistant"]
    assert context.items() == [user_message, *agent_items]
    assert context.messages() == [
        Message(role="user", content="hello"),
        Message(role="assistant", content="checking"),
    ]
    assert context.final_result == "done"
