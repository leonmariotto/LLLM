"""
LLM communication layer for agent.

Support internal agent, could support remote agent.
This separate interface allow to isolate the agent implementation from
the LLM implementation.
"""


# class LlmRequest(BaseModel):
#     """Request object for LLM calls."""
#
#     model_config = {"arbitrary_types_allowed": True}
#
#     instructions: List[str] = Field(default_factory=list)
#     contents: List[ContentItem] = Field(default_factory=list)
#     tools: List[BaseTool] = Field(default_factory=list)
#     tool_choice: Optional[str] = None
#     model_id: Optional[str] = None
#
#     def append_instructions(self, text: str) -> None:
#         """Append a single instruction string to the instructions list."""
#         self.instructions.append(text)
#
#
# class LlmResponse(BaseModel):
#     """Response object from LLM calls."""
#
#     content: List[ContentItem] = Field(default_factory=list)
#     error_message: Optional[str] = None
#     usage_metadata: Dict[str, Any] = Field(default_factory=dict)
#
#
# def build_messages(request: LlmRequest) -> List[Dict[str, Any]]:
#     """
#     Convert LlmRequest to messages list format.
#     """
#     messages: List[Dict[str, Any]] = []
#
#     for instruction in request.instructions:
#         messages.append({"role": "system", "content": instruction})
#
#     for item in request.contents:
#         if isinstance(item, Message):
#             messages.append({"role": item.role, "content": item.content})
#
#         elif isinstance(item, ToolCall):
#             tool_call_dict: Dict[str, Any] = {
#                 "id": item.tool_call_id,
#                 "type": "function",
#                 "function": {
#                     "name": item.name,
#                     "arguments": json.dumps(item.arguments),
#                 },
#             }
#             if messages and messages[-1]["role"] == "assistant":
#                 messages[-1].setdefault("tool_calls", []).append(tool_call_dict)
#             else:
#                 messages.append(
#                     {
#                         "role": "assistant",
#                         "content": None,
#                         "tool_calls": [tool_call_dict],
#                     }
#                 )
#
#         else:  # isinstance(item, ToolResult):
#             messages.append(
#                 {
#                     "role": "tool",
#                     "tool_call_id": item.tool_call_id,
#                     "content": str(item.content[0]) if item.content else "",
#                 }
#             )
#
#     return messages
#
#
# # TODO wtf ? this should not be re-defined. It should appear only once, maybe in a
# # model_common.py file?...
# # So far defined in Qwen2, GeneratorWithTool and here.
# @dataclass(frozen=True)
# class AssistantOutput:
#     """Parsed assistant completion with optional tool calls."""
#
#     content: str
#     tool_calls: tuple[ToolCall, ...] = ()
#
#
# class TextGenerator(Protocol):
#     """Underlying completion generator used for each assistant turn."""
#
#     tokenizer: Any
#
#     def generate_from_tokens(
#         self,
#         prompt_tokens: list[int],
#         *,
#         stop_at_eos: bool = True,
#         max_generated_token: int = 20,
#         cache_length: int | None = None,
#         temperature: float = 0.0,
#         top_k: int | None = None,
#         top_p: float | None = None,
#         include_prompt: bool = True,
#     ) -> str: ...
#
#
# class ToolTokenizer(Protocol):
#     """Tokenizer operations required by the model-agnostic tool loop."""
#
#     def apply_chat_template(
#         self,
#         messages: Sequence[ToolMessage],
#         *,
#         tools: Sequence[dict[str, object]] | None = None,
#         tokenize: bool = True,
#         add_generation_prompt: bool = False,
#     ) -> dict[str, list[int]] | str: ...
#
#     def parse_assistant_output(self, completion: str) -> AssistantOutput: ...
#
#
# class LlmClient:
#     """
#     Client for LLM.
#     Use Generator. Do not handle tool call, agent handle it.
#     Take a LlmRequest, parse output and return a LlmResponse.
#     """
#
#     def __init__(self, generator: TextGenerator):
#         self.generator = generator
#         self.tokenizer = cast(ToolTokenizer, generator.tokenizer)
#
#     def generate(self, request: LlmRequest) -> LlmResponse:
#         """
#         Looks like GeneratorWithTool use the following steps:
#         messages = _copy_messages(..)
#         prompt_token = _encode_history(messages)
#         Let's see what we need here.
#         LlmResponse contain a list of event, i don't know if my model is able
#         to output more than one event at a time...
#         """
#         try:
#             messages = build_messages(request)
#             tools = (
#                [t.tool_definition for t in request.tools] if request.tools else None
#             )
#             prompt_token = self.tokenizer.apply_chat_template(
#                messages,
#                tools=tools,
#                tokenize=True,
#                add_generation_prompt=True,
#             )
#             completion = self.generator.generate_from_tokens(
#                prompt_tokens,
#                stop_at_eos=stop_at_eos,
#                max_generated_token=max_generated_token,
#                cache_length=cache_length,
#                temperature=temperature,
#                top_k=top_k,
#                top_p=top_p,
#                include_prompt=False,
#             )
#             parsed_output: AssistantOutput = self.tokenizer.parse_assistant_output(
#                completion
#             )
#             return self._parse_response(response)
#         except Exception as e:
#             return LlmResponse(error_message=str(e))
#
#     def _parse_response(self, response: AssistantOutput) -> LlmResponse:
#         """
#         Create a LlmResponse from the LLM produced AssistantOutput.
#         """
