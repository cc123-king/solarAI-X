"""Response-only loss masking and group isolation without opening held-out tasks."""

import importlib.util
import json
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1]/'scripts/train_lora.py'
spec = importlib.util.spec_from_file_location('solar_train_lora', path)
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


def test_training_data_requires_disjoint_group_manifest(tmp_path):
    def row(group, prompt):
        return {'usage': 'training_seed', 'fixture_group': group,
                'messages': [{'role': 'system', 'content': 'Use the tool.'}, {'role': 'user', 'content': prompt},
                             {'role': 'assistant', 'content': 'tool output'}]}
    train, validation, manifest = tmp_path/'train.jsonl', tmp_path/'validation.jsonl', tmp_path/'groups.json'
    train.write_text(json.dumps(row('train-1', 'one'))+'\n', encoding='utf-8')
    validation.write_text(json.dumps(row('validation-1', 'two'))+'\n', encoding='utf-8')
    manifest.write_text(json.dumps({'fixture_groups': ['development-1', 'sealed-group-only']}), encoding='utf-8')
    assert len(trainer.validate_data(train, validation, manifest)[0]) == 1
    manifest.write_text(json.dumps({'fixture_groups': ['train-1']}), encoding='utf-8')
    with pytest.raises(ValueError, match='重叠'):
        trainer.validate_data(train, validation, manifest)


def test_response_loss_mask_never_truncates_target():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return [1, 2, 3, 4, 5] if messages[-1]['role'] == 'assistant' else [1, 2, 3]
    row = {'messages': [{'role': 'user'}, {'role': 'assistant'}]}
    encoded = trainer.encode_example(Tokenizer(), row, 10)
    assert encoded['labels'] == [-100, -100, -100, 4, 5]
    with pytest.raises(ValueError, match='max_length'):
        trainer.encode_example(Tokenizer(), row, 4)
