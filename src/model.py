import src.config as config
import torch
import transformers
import torch.nn as nn

class BertBaseUncased(nn.Module):
    def __init__(self):
        super(BertBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.dropout = nn.Dropout(0.3)
        self.l0 = nn.Linear(768, 2)

    def forward(self, ids, token_type_ids, attention_mask):
        # sequence_output [batch_size, sequence_len, hidden_size]
        # pooled_output [batch_size, 1, hidden_size]
        # [32, 128, 768]
        sequence_output, pooled_output = self.bert(
            input_ids=ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        drop = self.dropout(sequence_output)

        logits = self.l0(drop)
        logits_start, logits_end = logits[:, :, 0], logits[:, :, 1]
        # [batch_size, sequence_len, 1]

        logits_start = logits_start.squeeze(-1)
        logits_end = logits_end.squeeze(-1)
        # [batch_size, sequence_len]
        return logits_start, logits_end
