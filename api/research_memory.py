"""Explicit long-term research hints for the local, single-user workbench."""

import json
import math
import re
import sqlite3
import time
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path


class MemoryConflict(ValueError):
    pass


def terms(text):
    runs = re.findall(r'[\u4e00-\u9fff]+', text)
    return set(re.findall(r'[a-z0-9_-]+', text.casefold()) +
               [run[i:i+2] for run in runs for i in range(max(1, len(run)-1))])


class LocalEncoder:
    """An explicitly configured local model; never fetch weights or call an API."""

    def __init__(self, path):
        if not Path(path).is_dir():
            raise ValueError('SOLAR_ENCODER_PATH must point to a local model directory')
        model_type = import_module('sentence_transformers').SentenceTransformer
        self.model = model_type(str(Path(path).resolve()), local_files_only=True, device='cpu')

    def __call__(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()


def similarities(query, texts, encoder=None):
    if not encoder:
        query_terms = terms(query)
        return [len(query_terms & terms(text)) / max(1, len(query_terms | terms(text))) for text in texts]
    vectors = encoder([query, *texts])
    if len(vectors) != len(texts) + 1 or not vectors or not vectors[0]:
        raise ValueError('Invalid embedding count or dimensions')
    dimension = len(vectors[0])
    if any(len(v) != dimension or any(not math.isfinite(float(x)) for x in v) for v in vectors):
        raise ValueError('Invalid embedding values')
    def cosine(v):
        q = vectors[0]
        numerator = sum(float(a)*float(b) for a, b in zip(q, v))
        denominator = math.sqrt(sum(float(x)**2 for x in q) * sum(float(x)**2 for x in v))
        return numerator / denominator if denominator else 0.0
    return [cosine(v) for v in vectors[1:]]


class ResearchMemory:
    def __init__(self, path, namespace='local', encoder=None, clock=time.time):
        self.path, self.namespace, self.encoder, self.clock = Path(path), namespace, encoder, clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS research_memory (
                namespace TEXT, kind TEXT, key TEXT, value TEXT, source TEXT,
                version INTEGER, expires REAL, deleted INTEGER, updated REAL,
                PRIMARY KEY(namespace,kind,key))''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def remember(self, *, kind, key, value, source, expected_version=0, ttl_days=30):
        if (kind not in {'preference', 'case'} or not 1 <= len(key) <= 80 or not 1 <= len(value) <= 1500
                or not 1 <= len(source) <= 200 or not 1 <= ttl_days <= 365 or expected_version < 0):
            raise ValueError('Invalid explicit memory')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            current = db.execute('SELECT version FROM research_memory WHERE namespace=? AND kind=? AND key=?',
                                 (self.namespace, kind, key)).fetchone()
            if (current['version'] if current else 0) != expected_version:
                raise MemoryConflict('Memory changed; reload its version before updating')
            if current is None and db.execute('SELECT count(*) FROM research_memory WHERE namespace=?',
                                               (self.namespace,)).fetchone()[0] >= 1024:
                raise ValueError('Local memory capacity reached')
            version = expected_version + 1
            db.execute('INSERT OR REPLACE INTO research_memory VALUES(?,?,?,?,?,?,?,?,?)',
                       (self.namespace, kind, key, value, source, version,
                        self.clock() + ttl_days*86400, 0, self.clock()))
        return {'kind': kind, 'key': key, 'version': version}

    def list(self):
        with self.connect() as db:
            return [dict(row) for row in db.execute('''SELECT kind,key,value,source,version,expires
                FROM research_memory WHERE namespace=? AND deleted=0 AND expires>?
                ORDER BY updated DESC LIMIT 256''', (self.namespace, self.clock()))]

    def forget(self, kind, key, expected_version):
        with self.connect() as db:
            result = db.execute('''UPDATE research_memory SET deleted=1,value='',source='',version=version+1
                WHERE namespace=? AND kind=? AND key=? AND version=? AND deleted=0''',
                (self.namespace, kind, key, expected_version))
            if result.rowcount != 1:
                raise MemoryConflict('Memory missing or changed')
        return {'deleted': True}

    def recall(self, query, max_chars=4000):
        rows = self.list()
        preferences = [r for r in rows if r['kind'] == 'preference'][:3]
        cases = [r for r in rows if r['kind'] == 'case']
        scores = similarities(query, [r['value'] for r in cases], self.encoder) if cases else []
        threshold = 0.35 if self.encoder else 0.0
        selected = sorted(zip(cases, scores), key=lambda pair: pair[1], reverse=True)
        candidates = preferences + [row for row, score in selected if score > threshold][:5]
        accepted, chars = [], 0
        for row in candidates:
            cost = len(json.dumps(row, ensure_ascii=False))
            if chars + cost <= max_chars:
                accepted.append(row)
                chars += cost
        return {'memories': accepted, 'method': 'local_semantic' if self.encoder else 'lexical',
                'chars': chars, 'max_chars': max_chars}
