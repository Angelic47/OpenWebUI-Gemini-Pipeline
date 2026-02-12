"""MCP tool patch for the Gemini Pipeline."""

import inspect
import logging
from typing import Any, Callable


class MCPPatcher:
    """Patches MCP tools for compatibility with Gemini API."""

    def __init__(self):
        self.log = logging.getLogger("google_ai.pipe")

    def create_gemini_compatible_tool(self, tool_def: dict) -> Callable:
        """
        为 MCP 工具创建一个 Google SDK 兼容的包装函数。
        动态修改 __name__, __doc__ (包含Args描述) 和 __signature__。
        """
        import json

        original_func = tool_def["callable"]
        spec = tool_def.get("spec", {})

        # 1. 获取基本信息
        tool_name = spec.get("name", "unknown_tool")
        tool_desc = spec.get("description", "")

        # 2. 准备构建 Docstring 的 Args 部分
        spec_params = spec.get("parameters", {})
        props = spec_params.get("properties", {})
        required = spec_params.get("required", [])

        args_doc = []

        # 3. 动态重建函数签名 (inspect.Signature) 并构建 Docstring
        parameters = []

        for param_name, param_info in props.items():
            # 推断参数类型
            param_type = Any
            type_str = param_info.get("type", "string")
            param_desc = param_info.get("description", "")

            # 类型映射
            if type_str == "string":
                param_type = str
            elif type_str == "integer":
                param_type = int
            elif type_str == "number":
                param_type = float
            elif type_str == "boolean":
                param_type = bool
            elif type_str == "object":
                param_type = dict
            elif type_str == "array":
                # 关键修复：处理数组内部类型
                items_def = param_info.get("items", {})
                item_type_str = items_def.get(
                    "type", "string"
                )  # 默认为 string 以防万一

                if item_type_str == "string":
                    param_type = list[str]
                elif item_type_str == "integer":
                    param_type = list[int]
                elif item_type_str == "number":
                    param_type = list[float]
                elif item_type_str == "boolean":
                    param_type = list[bool]
                elif item_type_str == "object":
                    # 对于复杂的嵌套对象数组，List[dict] 通常能满足 Gemini 的基本校验
                    # 它会生成 items: { type: object }
                    param_type = list[dict]
                else:
                    # 如果实在不知道是什么，默认为 List[str] 或 List[Any]
                    # Google SDK 对 List[Any] 的处理可能有限，List[str] 比较安全
                    param_type = list
            else:
                param_type = Any
                self.log.warning(
                    f"Unknown parameter type '{type_str}' for param '{param_name}' in tool '{tool_name}'"
                )

            # 确定默认值
            default = inspect.Parameter.empty
            is_optional = param_name not in required
            if is_optional:
                default = param_info.get("default", None)

            # --- 构建 Args Docstring 行 ---
            # 格式: param_name (type): description
            # 示例: destination (str): The destination city.
            type_name = type_str
            if is_optional:
                type_name += ", optional"

            arg_line = f"    {param_name} ({type_name}): {param_desc}"
            if is_optional and default is not None:
                arg_line += f" Defaults to {default}."
            args_doc.append(arg_line)

            # --- 构建 inspect.Parameter ---
            param = inspect.Parameter(
                name=param_name,
                kind=inspect.Parameter.KEYWORD_ONLY,
                default=default,
                annotation=param_type,
            )
            parameters.append(param)

        # 4. 组合最终的 Docstring
        # 格式:
        # Description...
        #
        # Args:
        #     param1 (type): desc...
        #     param2 (type): desc...
        final_doc = tool_desc
        if args_doc:
            final_doc += "\n\nArgs:\n" + "\n".join(args_doc)

        # 5. 定义包装函数
        async def wrapper(*args, **kwargs):
            self.log.debug(
                f"Gemini: Invoking tool '{tool_name}' with args: {args}, kwargs: {kwargs}"
            )
            result = None
            if inspect.iscoroutinefunction(original_func):
                result = await original_func(*args, **kwargs)
            else:
                result = original_func(*args, **kwargs)
            self.log.debug(f"Tool '{tool_name}' raw result: {result}")

            final_result = result
            try:
                if isinstance(result, list):
                    # 尝试提取 text 类型的多段内容
                    texts = []
                    for item in result:
                        if isinstance(item, dict) and item.get("type") == "text":
                            content = item.get("text", "")
                            if content:
                                texts.append(content)

                    # 如果提取到了文本，就用换行符连接
                    if texts:
                        final_result = "\n\n".join(texts)
                    else:
                        # 如果是列表但没找到 text 字段，转 JSON 字符串
                        final_result = json.dumps(result, ensure_ascii=False)

                elif isinstance(result, dict):
                    # 如果是字典，转 JSON 字符串
                    final_result = json.dumps(result, ensure_ascii=False)

                # 如果本来就是 str/int/float，保持原样即可
            except Exception as e:
                self.log.warning(
                    f"Error processing tool result: {e}, using str() fallback"
                )
                final_result = str(result)

            self.log.debug(f"Gemini: Tool '{tool_name}' returned: {final_result}")
            return final_result

        # 6. 应用元数据
        wrapper.__name__ = tool_name
        wrapper.__qualname__ = tool_name
        wrapper.__doc__ = final_doc  # 设置包含 Args 的完整 docstring

        # 设置签名
        new_sig = inspect.Signature(parameters=parameters)
        wrapper.__signature__ = new_sig

        self.log.debug(
            f"Created Gemini-compatible tool: {tool_name} with signature {new_sig}"
        )

        return wrapper

    def patch_mcp_tools(
        self,
        __metadata__: dict[str, Any],
        __tools__: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """
        Patch MCP tools from metadata into __tools__.
        
        Returns:
            Updated __tools__ dictionary or None
        """
        # 确保 tool_ids 和 __tools__ 是字典
        tool_ids = __metadata__.get("tool_ids", [])
        if tool_ids is None:
            tool_ids = []
        if __tools__ is None and len(tool_ids) > 0:
            __tools__ = {}

        # 检查是否有 MCP 工具需要处理
        mcp_tool_ids = [tid for tid in tool_ids if tid.startswith("server:mcp:")]

        if mcp_tool_ids:
            self.log.debug(
                f"Detected {len(mcp_tool_ids)} MCP tool IDs in metadata: {mcp_tool_ids}"
            )
            # 获取 metadata 中的工具和客户端
            metadata_tools = __metadata__.get("tools", {})
            mcp_clients = __metadata__.get("mcp_clients", {})

            for mcp_tool_id in mcp_tool_ids:
                # 提取服务器ID，格式: server:mcp:server_id
                mcp_server_id = mcp_tool_id[len("server:mcp:") :]

                # 获取对应的 MCPClient
                mcp_client = mcp_clients.get(mcp_server_id)
                if not mcp_client:
                    self.log.warning(
                        f"MCP Client '{mcp_server_id}' not found, skipping tool processing"
                    )
                    continue

                self.log.debug(f"Processing MCP tools for server '{mcp_tool_id}'")

                # 遍历 metadata 中的工具，找到属于这个客户端且类型为 mcp 的工具
                mcp_tools_added = 0
                for mcp_tool_name, mcp_tool_def in metadata_tools.items():
                    # 检查工具类型和客户端匹配
                    if not isinstance(mcp_tool_def, dict):
                        continue

                    tool_type = mcp_tool_def.get("type")
                    tool_client = mcp_tool_def.get("client")

                    # 必须是 MCP 类型且属于当前客户端
                    if tool_type != "mcp" or tool_client is not mcp_client:
                        continue

                    # 检查工具是否已经在 __tools__ 中
                    if mcp_tool_name not in __tools__:
                        __tools__[mcp_tool_name] = mcp_tool_def
                        __tools__[mcp_tool_name]["callable"] = (
                            self.create_gemini_compatible_tool(mcp_tool_def)
                        )
                        mcp_tools_added += 1
                        self.log.debug(
                            f"Added MCP tool '{mcp_tool_name}' from server '{mcp_tool_id}'"
                        )

                self.log.info(
                    f"Added {mcp_tools_added} tools for MCP server '{mcp_tool_id}'"
                )

            # 记录修复结果
            total_tools = len(__tools__)
            self.log.info(
                f"MCP tool patch applied. Total {total_tools} tools available: {list(__tools__.keys())}"
            )

        return __tools__
