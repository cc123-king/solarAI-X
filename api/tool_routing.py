"""Rank registered MCP tools while retaining dependencies for chained analysis."""

from research_memory import similarities

DEPENDENCIES = {'draw_curve', 'draw_table', 'search_embedded_text'}


def route_tools(query, tools, mode='all', encoder=None, top_k=3):
    if mode not in {'all', 'lexical', 'semantic'}:
        raise ValueError('SOLAR_TOOL_ROUTING must be all, lexical or semantic')
    if mode == 'all' or len(tools) <= top_k:
        return tools, {'method': 'all', 'registered': len(tools), 'selected': len(tools)}
    if mode == 'semantic' and encoder is None:
        raise ValueError('Semantic routing requires an explicitly configured local encoder')
    descriptions = [f"{tool.name.replace('_', ' ')} {tool.description or ''}" for tool in tools]
    scores = similarities(query, descriptions, encoder if mode == 'semantic' else None)
    order = sorted(range(len(tools)), key=lambda i: (-scores[i], tools[i].name))
    # An unsupported query retains the catalog instead of guessing a route.
    if not any(score > 0 for score in scores):
        return tools, {'method': 'all_no_match', 'registered': len(tools), 'selected': len(tools)}
    indices = set(order[:top_k]) | {i for i, tool in enumerate(tools) if tool.name in DEPENDENCIES}
    selected = [tool for i, tool in enumerate(tools) if i in indices]
    return selected, {'method': mode, 'registered': len(tools), 'selected': len(selected),
                      'names': [tool.name for tool in selected]}
