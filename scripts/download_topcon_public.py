"""Retrieve the licensed 368 KB CSV from a 6 GB archive using HTTP byte ranges."""

import argparse
import hashlib
import json
import struct
import urllib.request
import zlib
from pathlib import Path

URL = ('https://data.mendeley.com/public-files/datasets/s2vnyfgc4s/files/'
       '7f16bb7d-04d5-4219-82b6-56a1297aca44/file_downloaded')
ARCHIVE_BYTES = 6055350940
CSV_SHA256 = '14479b042a29137d107aa7198eea4d82ab2804177c2e583074bf674cb54287d2'


def byte_range(start, end):
    request = urllib.request.Request(URL, headers={'User-Agent': 'SolarAI-X-public-data',
        'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'})
    with urllib.request.urlopen(request, timeout=30) as response:
        if (response.status != 206
                or response.headers.get('Content-Range') != f'bytes {start}-{end}/{ARCHIVE_BYTES}'):
            raise RuntimeError('Server did not honor the bounded range; full archive download refused.')
        raw = response.read(end-start+2)
        if len(raw) != end-start+1:
            raise RuntimeError('Incomplete byte range')
        return raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    target = args.output/'dataset/regression.csv'
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != CSV_SHA256:
            raise RuntimeError('Existing dataset differs; choose a new output directory.')
        print('Verified existing source CSV')
        return
    header = byte_range(8932, 8961)
    fields = struct.unpack('<4s5H3I2H', header)
    if fields[0] != b'PK\x03\x04' or fields[2] & 1 or fields[3] != 8:
        raise RuntimeError('Unexpected ZIP entry')
    name_size, extra_size = fields[-2:]
    if name_size > 100 or extra_size > 1000:
        raise RuntimeError('ZIP entry header exceeds budget')
    metadata = byte_range(8962, 8962+name_size+extra_size-1)
    if metadata[:name_size] != b'dataset/regression.csv':
        raise RuntimeError('ZIP entry name mismatch')
    start = 8962+name_size+extra_size
    decoder = zlib.decompressobj(-15)
    raw = decoder.decompress(byte_range(start, start+136046-1), 368189)
    if not decoder.eof or decoder.unconsumed_tail or decoder.unused_data:
        raise RuntimeError('Source entry decompression exceeds budget or has trailing data')
    if (len(raw) != 368188 or zlib.crc32(raw) != 1852582757
            or hashlib.sha256(raw).hexdigest() != CSV_SHA256):
        raise RuntimeError('Source CSV integrity mismatch')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    receipt = {'source_doi': '10.17632/s2vnyfgc4s.2', 'source_author': 'Chenhui Zhang',
        'license': 'CC-BY-4.0', 'csv_sha256': CSV_SHA256, 'csv_bytes': len(raw),
        'compressed_csv_bytes': 136046, 'full_archive_downloaded': False,
        'verification': 'entry CRC32 plus pinned extracted CSV SHA256',
        'archive_url': URL}
    (args.output/'source-receipt.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
