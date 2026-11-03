"""Explicit memory operations; these requests never call the chat model."""

from urllib.parse import quote

import config
import requests

from api.status import response_error_message


def memory_request(method: str, path: str = '', **kwargs):
    try:
        response = requests.request(method, f'{config.API_BASE_URL}/memory{path}', timeout=(5, 60), **kwargs)
        if not response.ok:
            raise ValueError(response_error_message(response))
        return response.json()
    except requests.RequestException as error:
        raise ValueError('长期记忆服务暂不可用，请检查后端连接后重试。') from error


def list_memories():
    return memory_request('GET')


def load_memory(kind, key):
    return memory_request('GET', '/entry', params={'kind': kind, 'key': key})


def save_memory(kind, key, value, source, version, ttl_days):
    return memory_request('PUT', json={'kind': kind, 'key': key, 'value': value, 'source': source,
                                      'expected_version': int(version), 'ttl_days': int(ttl_days)})


def delete_memory(kind, key, version):
    return memory_request('DELETE', f'/{quote(kind, safe="")}/{quote(key, safe="")}',
                          params={'expected_version': int(version)})


def recall_memory(query):
    return memory_request('GET', '/recall', params={'query': query})
