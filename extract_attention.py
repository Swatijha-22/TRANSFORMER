# extract_attention.py
from transformers import BertTokenizer, BertModel
import torch

def get_attention(sentence):
    # Load pre-trained BERT (downloads ~440MB on first run)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained(
        'bert-base-uncased',
        output_attentions=True   # <-- this is the key flag
    )
    model.eval()

    # Tokenise the input
    inputs = tokenizer(sentence, return_tensors='pt')
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    # Forward pass — no gradient needed
    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.attentions is a tuple of 12 tensors
    # Each tensor shape: (batch=1, heads=12, seq_len, seq_len)
    attentions = outputs.attentions

    print(f"Sentence : {sentence}")
    print(f"Tokens   : {tokens}")
    print(f"Layers   : {len(attentions)}")
    print(f"Shape    : {attentions[0].shape}")
    # → torch.Size([1, 12, seq_len, seq_len])

    return tokens, attentions


if __name__ == '__main__':
    sentence = "The river bank was flooded after the storm."
    tokens, attentions = get_attention(sentence)