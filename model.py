from torch import nn
from torch import tensor
from torch import softmax
from transformers import BertPreTrainedModel, BertModel, BertForQuestionAnswering
class Start_Prob_Layer(nn.Module):
    def __init__(self, input_dim = 768, hidden_dim = 256, output_dim = 1, dropout_rate=0.):
        super(Start_Prob_Layer, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x):
        layer = self.layer(x)
        return layer

class End_Prob_Layer(nn.Module):
    def __init__(self, input_dim = 768, hidden_dim = 256, output_dim = 1, dropout_rate=0.):
        super(End_Prob_Layer, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x):
        layer = self.layer(x)
        return layer

class SQuAD_Bert(BertPreTrainedModel):
    def __init__(self, args, bert_config):
        super(SQuAD_Bert, self).__init__(bert_config)
        self.args = args
        self.bert = BertModel.from_pretrained(args.model_name_or_path, config=bert_config)
        self.start_layer = Start_Prob_Layer(input_dim = bert_config.hidden_size, dropout_rate=self.args.dropout_rate)
        self.end_layer = End_Prob_Layer(input_dim=bert_config.hidden_size, dropout_rate=self.args.dropout_rate)

    def forward(self, input_ids, attention_mask, segment_ids, start_positions, end_positions):
        outputs = self.bert(input_ids, attention_mask = attention_mask, token_type_ids = segment_ids)
        sequence_output = outputs[0]
        start_probs = self.start_layer(outputs[0])
        end_probs = self.end_layer(outputs[0])
        total_loss = 0
        ignored_index = start_probs.size(1)
        start_positions.clamp_(0, ignored_index)
        end_positions.clamp_(0, ignored_index)          # 将不合法的index归为ignore的范畴
        loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)  # 交叉熵损失只考虑了正确的label，结果是-log(softmax), 对多个求entorpy自动取平均
        start_loss = loss_fct(start_probs.view(-1, self.args.max_seq_length), start_positions)
        end_loss = loss_fct(end_probs.view(-1, self.args.max_seq_length), end_positions)
        loss = (start_loss * self.args.start_weight + end_loss) / (self.args.start_weight + 1)

        return loss, start_probs.view(-1, self.args.max_seq_length), end_probs.view(-1, self.args.max_seq_length)

