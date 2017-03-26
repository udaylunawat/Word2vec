'''
Output Little-Endian bytes and labels file from gensim model
Also outputs necessary json config file portion
For use with TensorBoard
'''

import struct
from gensim.models import Word2Vec, Doc2Vec

#model = Word2Vec.load('thrones2vec.w2v')
# have to use python2 for some old models
model = Doc2Vec.load('thrones2vec.bin')

num_rows = len(model.wv.vocab)
dim = model.vector_size

tensor_out_fn = 'enwiki_wordvec_%d_%dd_tensors.bytes' % (num_rows, dim)
labels_out_fn = 'enwiki_wordvec_%d_%dd_labels.tsv' % (num_rows, dim)

tensor_out = open(tensor_out_fn, 'wb')

try:
    labels_out = open(labels_out_fn, 'w', encoding='utf-8')
except:
    labels_out = open(labels_out_fn, 'w')

labels_out.write('word\tcount\n')

for wd in model.wv.vocab:
    floatvals = model[wd].tolist()
    assert dim == len(floatvals)
    assert '\t' not in wd

    for f in floatvals:
        tensor_out.write(struct.pack('<f', f))

    try:
        labels_out.write('%s\t%s\n' % (wd, model.wv.vocab[wd].count))
    except:
        labels_out.write(('%s\t%s\n' % (wd, model.wv.vocab[wd].count)).encode('utf-8'))

tensor_out.close()
labels_out.close()

print('''{
  "embeddings": [
    {
      "tensorName": "EnWiki WordVec",
      "tensorShape": [%d, %d],
      "tensorPath": "%s",
      "metadataPath": "%s"
    }
  ],
  "modelCheckpointPath": "Demo datasets"
}''' % (num_rows, dim, tensor_out_fn, labels_out_fn))