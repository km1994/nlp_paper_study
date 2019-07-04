import re
from collections import namedtuple

Raw_Example = namedtuple('Raw_Example', 'label entity1 entity2 sentence')
PositionPair = namedtuple('PosPair', 'first last')

pattern = r'\w+|[^\w\s]+' # nltk.tokenize.regexp.WordPunctTokenizer
regexp = re.compile(pattern)

def wordpunct_tokenizer(line):
  line = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", line)
  line = re.sub(r"\s{2,}", " ", line)
  # line = re.sub(r'\d+', '0', line)
  # line = re.sub(r'\d', '0', line)
  return regexp.findall(line)

def find_new_pos(entity, sent):
  ''' find new entity position in cleaned stence
  '''
  n = len(entity)
  for i in range(len(sent)):
    if sent[i:i+n]==entity:
      first, last = i, i+n-1
      return PositionPair(first, last) 

def clean_data(filename, new_file):
  '''clean the sentence in `filename`, change the cleaned entity position,
  and write the cleaned results in `new_file`
  '''
  data = []
  with open(filename) as f:
    for line in f:
      words = line.strip().lower().split()
      sent = words[5:]
      label = int(words[0])
      entity1 = PositionPair(int(words[1]), int(words[2]))
      entity2 = PositionPair(int(words[3]), int(words[4]))

      entity1 = sent[entity1.first: entity1.last+1]
      entity2 = sent[entity2.first: entity2.last+1]

      # cln_words = clean_str(line).split()
      cln_words = wordpunct_tokenizer(line.lower())
      cln_sent = cln_words[5:]

      entity1 = find_new_pos(entity1, cln_sent)
      entity2 = find_new_pos(entity2, cln_sent)

      if entity1 is None or entity2 is None:
        print(line)
        print(' '.join(cln_sent))

      example = Raw_Example(label, entity1, entity2, cln_sent)
      data.append(example)
  
  with open(new_file, 'w') as f:
    for example in data:
      entity1 = example.entity1
      entity2 = example.entity2

      label_pos = '%d %d %d %d %d' % (example.label, 
                                      entity1.first, entity1.last, 
                                      entity2.first, entity2.last)
      sent = ' '.join(example.sentence)

      f.write('%s %s\n' %(label_pos, sent))

  check_entity(filename, new_file)

def check_entity(file, new_file):
  '''check whether the cleaned entity pair is the same as 
  the original entity pair
  '''
  def load_entities(filename):
    data = []
    with open(filename) as f:
      for line in f:
        words = line.lower().strip().split()
        sent = words[5:]
        entity1 = PositionPair(int(words[1]), int(words[2]))
        entity2 = PositionPair(int(words[3]), int(words[4]))

        entity1 = ' '.join(sent[entity1.first: entity1.last+1])
        entity2 = ' '.join(sent[entity2.first: entity2.last+1])

        entity = entity1 + ' </> ' + entity2
        data.append(entity)
    return data

  data = load_entities(file)
  new_data = load_entities(new_file)

  assert len(data) == len(new_data)

  for i in range(len(data)):
    entity = data[i]
    new_entity = new_data[i]

    if entity != new_entity:
      print(i, entity, new_entity)



print('test data')
clean_data('data/test.txt', 'data/test.cln')
print('train data')
clean_data('data/train.txt', 'data/train.cln')


