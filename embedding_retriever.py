"""Real CPU embeddings and transparent similarity inspection."""
import argparse
from importlib.metadata import version
import numpy as np
from sentence_transformers import SentenceTransformer
from keyword_retriever import load_chunks

MODEL = 'sentence-transformers/all-MiniLM-L6-v2'


class Retriever:
    def __init__(self):
        self.chunks = load_chunks()
        self.model = SentenceTransformer(MODEL, device='cpu', cache_folder=str(__import__('pathlib').Path(__file__).parent / '.cache'))
        self.documents = self.model.encode_document([c['text'] for c in self.chunks], normalize_embeddings=True, convert_to_numpy=True)
        assert self.documents.shape == (6, 384)
        assert np.allclose(np.linalg.norm(self.documents, axis=1), 1, atol=1e-6)
        print('Model:', MODEL, 'device: cpu')
        for name in ['sentence-transformers', 'numpy', 'torch', 'transformers']:
            print(name, version(name))
        print('Document matrix shape:', self.documents.shape)
        print('All 384 coordinates are used; displayed coordinates have no individual semantic labels.')
        for c, vector in zip(self.chunks, self.documents):
            print(c['id'], 'first12:', fmt(vector[:12]), 'length:', f'{np.linalg.norm(vector):.8f}')

    def retrieve(self, question, k=2, inspect=False):
        query = self.model.encode_query(question, normalize_embeddings=True, convert_to_numpy=True)
        assert np.allclose(np.linalg.norm(query), 1, atol=1e-6)
        scores = self.documents @ query
        direct = scores / (np.linalg.norm(self.documents, axis=1) * np.linalg.norm(query))
        assert np.allclose(scores, direct, atol=1e-6)
        ranked = sorted([dict(c, score=float(s), index=i) for i, (c,s) in enumerate(zip(self.chunks, scores))], key=lambda c: (-c['score'], c['id']))
        print(f'\nQuestion: {question}\nk: {k}\nQuery shape: {query.shape}')
        print('Query first12:', fmt(query[:12]), 'length:', f'{np.linalg.norm(query):.8f}')
        print('Dot products agree with direct cosine within tolerance 1e-6.')
        for c in ranked:
            print(c['id'], f"score={c['score']:.8f}")
        print('Selected:', ', '.join(c['id'] for c in ranked[:k]))
        for c in ranked[:k]:
            print(f"[{c['id']}] {c['filename']}\n{c['text']}")
        if inspect:
            products = query.astype(np.float64) * self.documents[ranked[0]['index']].astype(np.float64)
            print('Top passage:', ranked[0]['id'])
            print('First12 products:', fmt(products[:12]))
            print('First12 subtotal:', f'{products[:12].sum():.10f}')
            print('Remaining372 subtotal:', f'{products[12:].sum():.10f}')
            print('Full similarity:', f'{products.sum():.10f}')
            shown = np.concatenate([self.documents[:, :12].ravel(), query[:12]])
            print('Displayed coordinate minimum/maximum:', float(shown.min()), float(shown.max()))
        return ranked


def fmt(vector):
    return ' '.join(f'{x:.4f}' for x in vector)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('question')
    parser.add_argument('-k', type=int, default=2)
    parser.add_argument('--inspect', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.k <= 6:
        parser.error('k must be between 1 and 6')
    Retriever().retrieve(args.question, args.k, args.inspect)
