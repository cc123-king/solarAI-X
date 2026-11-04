"""The Gradio chat generator must preserve failures instead of blanking them."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize('mode', ['error', 'interrupted', 'empty', 'tools_disabled'])
def test_chat_generator_failure_and_mode_contract(mode):
    code = '''
import sys
from unittest.mock import patch
sys.path.insert(0, 'web')
from api import chat
chat.current_session.update(conversation_id='synthetic-test', messages=[], selected_model='agent-base')
mode=sys.argv[1]
captured=[]
def events(self):
 captured.append(self.params)
 if mode=='error': yield {'type':'error','content':'SYNTHETIC_FAILURE'}
 elif mode=='interrupted': yield {'type':'content','content':'partial answer'}
 else:
  yield {'type':'content','content':'Hello'}
  yield {'type':'done','content':''}
with patch.object(chat.SSEClient,'events',events):
 result=list(chat.send_message('' if mode=='empty' else 'Hello', [], False))
if mode=='error':
 assert 'SYNTHETIC_FAILURE' in result[-1][0][-1]['content']
 assert result[-1][1] and not chat.current_session['messages']
elif mode=='interrupted':
 assert result[-1][1] and result[-1][0][-1]['error']
 assert not chat.current_session['messages']
elif mode=='empty': assert result[-1][1] and not captured
else:
 assert captured[0]['tools_enabled'] is False
 assert result[-1][0][-1]['content']=='Hello'
print('contract-passed')
'''
    result = subprocess.run([sys.executable, '-c', code, mode], cwd=Path(__file__).resolve().parents[1],
                            env=dict(os.environ, PYTHONUTF8='1'), capture_output=True, text=True,
                            encoding='utf-8', check=False, timeout=15)
    assert result.returncode == 0, result.stderr
    assert 'contract-passed' in result.stdout
