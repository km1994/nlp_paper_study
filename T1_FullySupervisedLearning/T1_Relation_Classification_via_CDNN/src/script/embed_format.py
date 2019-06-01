'''
convert the pre-trained word embedding to .npy format
'''

import numpy as np

def convert_senna_embedding(senna_words_file, 
                            senna_embed_file, 
                            new_words_file,
                            new_embed_npfile,
                            word_dim):
  vocab = []
  # load senna vocab
  with open(senna_words_file) as f:
    for line in f:
      w = line.strip()
      vocab.append(w)
  
  # load senna embedding
  embed = []
  with open(senna_embed_file) as f:
    for line in f:
      vec = [float(x) for x in line.strip().split()]
      assert len(vec) == word_dim
      embed.append(vec)
  
  # write new words file
  with open(new_words_file, 'w') as f:
    for w in vocab:
      f.write('%s\n' % w)
  
  # save embed as .npy format
  embed = np.asarray(embed)
  embed = embed.astype(np.float32)
  np.save(new_embed_npfile, embed)

def convert_google_embedding( 
                            google_embed_file, 
                            new_words_file,
                            new_embed_npfile):
  import gensim
  model = gensim.models.KeyedVectors.load_word2vec_format(google_embed_file, 
                                                          binary=True)
  print('load finished.')
  
  embed = []
  with open(new_words_file, 'w') as f:
    for i, w in enumerate(model.index2word):
      if i%1000000 == 0:
        print(i)
      embed.append(model[w])
      f.write('%s\n'%w)
  
  embed = np.asarray(embed)
  print('converted to np array')
  embed = embed.astype(np.float32)
  print('converted to float32')
  np.save(new_embed_npfile, embed)

convert_senna_embedding('data/embedding/senna/words.lst',
                        'data/embedding/senna/embeddings.txt',
                        'data/senna_words.lst',
                        'data/senna_embed50.npy',
                        50)


convert_google_embedding('data/GoogleNews-vectors-negative300.bin',
                         'data/google_words.lst',
                         'data/google_embed300.npy')
  
