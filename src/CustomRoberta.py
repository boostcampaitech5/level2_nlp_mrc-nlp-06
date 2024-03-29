from typing import Optional, Union, Tuple
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel,RobertaModel,QuestionAnsweringModelOutput,RobertaForQuestionAnswering

"""
Custom Roberta for Question Answering
"""
class lstmLayer(nn.Module):
    def __init__(self, hidden_size, num_labels, drop_rate=0.1):
        super(lstmLayer, self).__init__()
        self.dropout = nn.Dropout(drop_rate)
        self.lstm_outputs = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.tanh = nn.Tanh()
        self.qa_outputs = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x,_ = self.lstm_outputs(x)
        x = self.tanh(x)
        return self.qa_outputs(x)

class bi_lstmLayer(nn.Module):
    def __init__(self, hidden_size, num_labels, drop_rate=0.1):
        super(bi_lstmLayer, self).__init__()
        self.dropout = nn.Dropout(drop_rate)
        self.lstm_outputs = nn.LSTM(hidden_size, hidden_size, bidirectional=True, batch_first=True)
        self.tanh = nn.Tanh()
        self.qa_outputs = nn.Linear(hidden_size*2, num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x,_ = self.lstm_outputs(x)
        x = self.tanh(x)
        return self.qa_outputs(x)

class mlp_Layer(nn.Module):
    def __init__(self, hidden_size, num_labels, drop_rate=0.1):
        super(mlp_Layer, self).__init__()
        self.dropout = nn.Dropout(drop_rate)
        self.linear_layer = nn.Linear(hidden_size, hidden_size)
        self.gelu = nn.GELU()
        self.qa_outputs = nn.Linear(hidden_size, num_labels)

    def forward(self, x): # linear layer 1 + fc 
        x = self.dropout(x)
        x = self.linear_layer(x)
        x = self.gelu(x)
        return self.qa_outputs(x)

class Conv_Layer(nn.Module):
    """Conv SDS layer for qa output"""

    def __init__(self,hidden_size, input_size):
        """
        Args:
            input_size (int): max sequence lengths
            hidden_dim (int): backbones's hidden dimension
        """

        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=input_size*2,
            kernel_size=3,
            padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=input_size*2,
            out_channels=input_size,
            kernel_size=1,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = x + torch.relu(out)
        out = self.layer_norm(out)
        return out

class SDS_Conv_Layer(nn.Module):
    """Conv SDS layer for qa output"""

    def __init__(
        self,hidden_size, input_size, num_labels):
        """
        Args:
            input_size (int): max sequence lengths
            hidden_dim (int): backbones's hidden dimension
        """

        super().__init__()
        convs = []
        for n in range(5):
            convs.append(Conv_Layer(hidden_size, input_size))
        self.convs = nn.Sequential(*convs)
        self.qa_outputs = nn.Linear(hidden_size, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convs(x)
        return self.qa_outputs(x)
           
def knowledge_distill(distill_model,start_logits, end_logits, input_ids,
                attention_mask,
                token_type_ids,
                position_ids,
                head_mask,
                inputs_embeds,
                output_attentions,
                output_hidden_states,
                return_dict):
    distill_model.eval()
    t_outputs = distill_model(input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,)

    t_start_logits, t_end_logits = t_outputs['start_logits'], t_outputs['end_logits']
    #t_start_logits.requires_grad_(True)
    #t_end_logits.requires_grad_(True)
    # Distill logits
    temperature = 1.0

    kl_loss = nn.KLDivLoss(reduction='none')
    kl_start = (kl_loss(
        F.log_softmax(start_logits/temperature, dim=1), target=F.softmax((t_start_logits/temperature)[:,:start_logits.size(1)], dim=1)
    ).sum(1)).mean(0)
    kl_end =  (kl_loss(
        F.log_softmax(end_logits/temperature, dim=1), target=F.softmax((t_end_logits/temperature)[:,:end_logits.size(1)], dim=1)
    ).sum(1)).mean(0)

    return (kl_start + kl_end)/2.0 * 0.5
class CustomRobertaForQuestionAnswering(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        # 추가 : classification 전에 어떤 layer 를 태울 것인지
        self.clf_layer = config.clf_layer

        # model
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        # clf layer
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        if config.clf_layer == 'bi_lstm':
            self.bi_lstm_outputs = bi_lstmLayer(config.hidden_size, config.num_labels)
        elif config.clf_layer == 'lstm':
            self.lstm = lstmLayer(config.hidden_size, config.num_labels)
        elif config.clf_layer == 'mlp':
            self.mlp_outputs = mlp_Layer(config.hidden_size, config.num_labels)
        elif config.clf_layer == 'SDS_cnn':
            self.sds_cnn_outputs = SDS_Conv_Layer(config.hidden_size, config.max_seq_len,config.num_labels)

        # self distillation
        self.distill_model = None
        self.distill = config.distill
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0] # last hidden states -> (8,384,1024) == (batch_size, max_seq_len, embedding_dim)
        if self.clf_layer == 'lstm':
            logits = self.lstm(sequence_output)
        elif self.clf_layer == 'bi_lstm':
            logits = self.bi_lstm_outputs(sequence_output)
        elif self.clf_layer == 'mlp':
            logits = self.mlp_outputs(sequence_output)
        elif self.clf_layer == 'SDS_cnn':
            logits =  self.sds_cnn_outputs(sequence_output)
        else:
            logits = self.qa_outputs(sequence_output) # (8,384, 2)
        start_logits, end_logits = logits.split(1, dim=-1) # (8,384, 1) * 2
        start_logits = start_logits.squeeze(-1).contiguous() # (8,384)
        end_logits = end_logits.squeeze(-1).contiguous()
        
        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.sqnueeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index) # 0~384 (max_seq_length)사이로 조정
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            if self.distill and total_loss is not None:
                total_loss = total_loss + knowledge_distill(self.distill_model,start_logits, end_logits,input_ids,
                    attention_mask,
                    token_type_ids,
                    position_ids,
                    head_mask,
                    inputs_embeds,
                    output_attentions,
                    output_hidden_states,
                    return_dict)

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
