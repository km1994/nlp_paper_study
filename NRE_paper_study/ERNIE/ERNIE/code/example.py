import torch
from knowledge_bert import BertTokenizer, BertModel, BertForMaskedLM

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('ernie_base')

# Tokenized input
text_a = "Who was Jim Henson ? "
text_b = "Jim Henson was a puppeteer ."

# Use TAGME
import tagme
# Set the authorization token for subsequent calls.
tagme.GCUBE_TOKEN = "<Your token goes here>"
text_a_ann = tagme.annotate(text_a)
text_b_ann = tagme.annotate(text_b)

# Read entity map
ent_map = {}
with open("kg_embed/entity_map.txt") as fin:
    for line in fin:
        name, qid = line.strip().split("\t")
        ent_map[name] = qid

def get_ents(ann):
    ents = []
    # Keep annotations with a score higher than 0.3
    for a in ann.get_annotations(0.3):
        if a.entity_title not in ent_map:
            continue
        ents.append([ent_map[a.entity_title], a.begin, a.end, a.score])
    return ents
        
ents_a = get_ents(text_a_ann)
ents_b = get_ents(text_b_ann)

# Tokenize
tokens_a, entities_a = tokenizer.tokenize(text_a, ents_a)
tokens_b, entities_b = tokenizer.tokenize(text_b, ents_b)

tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
ents = ["UNK"] + entities_a + ["UNK"] + entities_b + ["UNK"]
segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
input_mask = [1] * len(tokens)

# Mask a token that we will try to predict back with `BertForMaskedLM`
masked_index = 8
tokens[masked_index] = '[MASK]'

# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)

# Convert ents
entity2id = {}
with open("kg_embed/entity2id.txt") as fin:
    fin.readline()
    for line in fin:
        qid, eid = line.strip().split('\t')
        entity2id[qid] = int(eid)

indexed_ents = []
ent_mask = []
for ent in ents:
    if ent != "UNK" and ent in entity2id:
        indexed_ents.append(entity2id[ent])
        ent_mask.append(1)
    else:
        indexed_ents.append(-1)
        ent_mask.append(0)
ent_mask[0] = 1

print(indexed_tokens, indexed_ents)

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
ents_tensor = torch.tensor([indexed_ents])
segments_tensors = torch.tensor([segments_ids])
ent_mask = torch.tensor([ent_mask])

# Load pre-trained model (weights)
model, _ = BertForMaskedLM.from_pretrained('ernie_base')
model.eval()

vecs = []
vecs.append([0]*100)
with open("kg_embed/entity2vec.vec", 'r') as fin:
    for line in fin:
        vec = line.strip().split('\t')
        vec = [float(x) for x in vec]
        vecs.append(vec)
embed = torch.FloatTensor(vecs)
embed = torch.nn.Embedding.from_pretrained(embed)


# If you have a GPU, put everything on cuda
tokens_tensor = tokens_tensor.to('cuda')
ents_tensor = embed(ents_tensor+1).to('cuda')
ent_mask = ent_mask.to("cuda")
segments_tensors = segments_tensors.to('cuda')
model.to('cuda')

# Predict all tokens
with torch.no_grad():
    predictions = model(tokens_tensor, ents_tensor, ent_mask, segments_tensors)
    # confirm we were able to predict 'henson'
    predicted_index = torch.argmax(predictions[0, masked_index]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    assert predicted_token == 'henson'