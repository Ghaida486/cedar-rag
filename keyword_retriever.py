"""Transparent word-overlap retrieval; standard library only."""
import argparse
from pathlib import Path
import re

STOPWORDS = set('a an the is are to of in at for and or may can i it be'.split())


def load_chunks():
    chunks = []
    for path in sorted((Path(__file__).parent / 'docs').glob('*.txt')):
        for paragraph in path.read_text().strip().split('\n\n'):
            match = re.fullmatch(r'\[([A-Z]\d+)\] (.+)', paragraph)
            if not match:
                raise ValueError(f'Malformed paragraph: {path}')
            chunks.append(dict(id=match[1], filename=path.name, text=match[2]))
    if len(chunks) != 6:
        raise ValueError('Expected exactly six chunks')
    return chunks


def words(text):
    return set(re.findall(r'[a-z0-9]+', text.lower())) - STOPWORDS


def retrieve(question, chunks):
    scored = []
    for chunk in chunks:
        matched = sorted(words(question) & words(chunk['text']))
        scored.append(dict(chunk, score=len(matched), matched=matched))
    return sorted(scored, key=lambda c: (-c['score'], c['id']))


def display(question, k=2):
    ranked = retrieve(question, load_chunks())
    print(f'Question: {question}\nk: {k}\nChunks: {len(ranked)}')
    for c in ranked:
        print(f"{c['id']} score={c['score']} matched={', '.join(c['matched']) or '(none)'}")
    selected = [c for c in ranked if c['score'] > 0][:k]
    print('Selected: ' + (', '.join(c['id'] for c in selected) or 'No matching passages.'))
    for c in selected:
        print(f"[{c['id']}] {c['filename']}\n{c['text']}")
    return ranked


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('question')
    parser.add_argument('-k', type=int, default=2)
    args = parser.parse_args()
    if args.k < 1:
        parser.error('k must be positive')
    display(args.question, args.k)
