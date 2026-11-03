"""Long-term memory controls for the local single-user workbench."""

import math
from datetime import UTC, datetime

import gradio as gr
from api.memory import (
    delete_memory,
    list_memories,
    load_memory,
    recall_memory,
    save_memory,
)
from api.status import load_health


def build_memory_controls():
    with gr.Accordion('长期记忆', open=False):
        gr.Markdown('保存你确认的偏好与案例，供后续会话召回。当前工作台采用本机单用户模式。')
        rows = gr.State([])
        selected = gr.Dropdown(label='已保存的记忆', choices=[], interactive=True)
        refresh = gr.Button('刷新记忆和服务状态')
        health = gr.JSON(label='记忆与工具路由状态')
        kind = gr.Dropdown([('偏好', 'preference'), ('确认案例', 'case')], value='preference', label='类型')
        key = gr.Textbox(label='记忆标识', max_lines=1)
        load = gr.Button('读取标识的当前版本')
        version = gr.Number(label='当前版本（新标识为 0）', value=0, precision=0, interactive=False)
        value = gr.Textbox(label='已确认内容', lines=3)
        source = gr.Textbox(label='确认来源', placeholder='例如：用户明确确认使用 eV 单位')
        ttl = gr.Number(label='从保存日起有效天数（1–365）', value=30, precision=0)
        with gr.Row():
            save = gr.Button('保存 / 更新', variant='primary')
            delete = gr.Button('删除当前版本')
        state = gr.Textbox(label='记忆操作结果', interactive=False)
        query = gr.Textbox(label='输入问题，预览会被召回的记忆')
        recall = gr.Button('预览召回')
        preview = gr.JSON(label='召回内容与预算')

    def refresh_rows():
        try:
            items = list_memories()
            choices = [(f"{item['kind']} / {item['key']} · v{item['version']}", str(index))
                       for index, item in enumerate(items)]
            status = load_health()
            return items, gr.update(choices=choices, value=None), {
                'research_memory': status.get('research_memory'), 'tool_routing': status.get('tool_routing'),
                'status': status.get('status'),
            }, f'已加载 {len(items)} 条有效记忆。'
        except ValueError as error:
            return [], gr.update(choices=[], value=None), {}, str(error)

    def fields(item):
        expires = item.get('expires')
        days = max(1, min(365, math.ceil((expires - datetime.now(UTC).timestamp()) / 86400))) if expires else 30
        return (item['kind'], item['key'], item['version'], item.get('value', ''), item.get('source', ''), days,
                '已读取当前版本；保存和删除会核对版本。状态：' + item.get('status', 'active'))

    def select_row(index, items):
        if index is None:
            return (gr.skip(),) * 7
        try:
            item = items[int(index)]
            return fields(load_memory(item['kind'], item['key']))
        except (ValueError, IndexError) as error:
            return (gr.skip(),) * 6 + (str(error),)

    def load_entry(memory_kind, memory_key):
        try:
            return fields(load_memory(memory_kind, memory_key))
        except ValueError as error:
            return (gr.skip(),) * 6 + (str(error),)

    def save_entry(memory_kind, memory_key, text, origin, expected, days):
        try:
            result = save_memory(memory_kind, memory_key, text, origin, expected, days)
            return result['version'], '已保存；下一次聊天会按查询和预算召回。可刷新列表查看。'
        except (ValueError, TypeError, OverflowError) as error:
            return gr.skip(), str(error)

    def delete_entry(memory_kind, memory_key, expected):
        try:
            delete_memory(memory_kind, memory_key, expected)
            return int(expected) + 1, '', '', '已删除；后续召回排除本条。重新保存将使用新的版本。'
        except (ValueError, TypeError, OverflowError) as error:
            return gr.skip(), gr.skip(), gr.skip(), str(error)

    def preview_recall(question):
        try:
            return recall_memory(question)
        except ValueError as error:
            return {'error': str(error)}

    refresh.click(refresh_rows, outputs=[rows, selected, health, state])
    targets = [kind, key, version, value, source, ttl, state]
    selected.change(select_row, inputs=[selected, rows], outputs=targets)
    load.click(load_entry, inputs=[kind, key], outputs=targets)
    # Changing identity fields invalidates the displayed revision until it is loaded again.
    for identity_field in [kind, key]:
        identity_field.input(lambda: (0, '标识已更改；已有标识请先读取当前版本。'), outputs=[version, state])
    save.click(save_entry, inputs=[kind, key, value, source, version, ttl], outputs=[version, state])
    delete.click(delete_entry, inputs=[kind, key, version], outputs=[version, value, source, state])
    recall.click(preview_recall, inputs=[query], outputs=[preview])
