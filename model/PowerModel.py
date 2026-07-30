import torch
import torch.nn as nn
import torch.nn.functional as F


class PowerEncoder(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers=2, dropout=0.1):
        super(PowerEncoder,self).__init__()
        self.input_proj = nn.Linear(num_features, hidden_size)
        self.lstm= nn.LSTM(hidden_size, hidden_size, num_layers = 2, dropout = dropout ,batch_first=True)
        self.droput = nn.Dropout(dropout)
        
    def forward(self,x):
        output = self.input_proj(x)
        encoder_outputs, (h_n, c_n) = self.lstm(output)
        return encoder_outputs, (h_n, c_n)


# Using Bahdanau attention to weight the output of encoder
class PowerAttention(nn.Module):
    def __init__(self,hidden_size):
        super(PowerAttention,self).__init__()
        self.Wa  = nn.Linear(hidden_size,hidden_size)
        self.Ua = nn.Linear(hidden_size,hidden_size)
        self.Va = nn.Linear(hidden_size,1)
        
    def forward(self,query,keys):
        scores  = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)
        return context, weights
       

class PowerDecoder(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers=2, dropout=0.1):
        super(PowerDecoder,self).__init__()
        self.input_proj = nn.Linear(num_features, hidden_size)
        self.num_features = num_features
        self.lstm = nn.LSTM(2*hidden_size, hidden_size,dropout  = dropout, num_layers = num_layers, batch_first=True)
        self.attention = PowerAttention(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, num_features)
     
    def forward(self, encoder_outputs, encoder_hidden, horizon_steps,  start_value,target_tensor=None):
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        decoder_input = start_value.view(batch_size, 1, self.num_features)
        decoder_hidden = encoder_hidden
        num_features = self.out.out_features
        decoder_outputs = []
        attentions = []

        for i in range(horizon_steps):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
               decoder_input = target_tensor[:, i:i + 1].unsqueeze(-1)
            else:
                # Without teacher forcing: use its own predictions as the next input
                decoder_input = decoder_output.detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.input_proj(input))
        h_n, _ = hidden
        query = h_n[-1].unsqueeze(1)  

        context, attn_weights = self.attention(query, encoder_outputs)

        lstm_input = torch.cat((embedded, context), dim=2)  
        output, hidden = self.lstm(lstm_input, hidden)
        output = self.out(output)

        return output, hidden, attn_weights
        
        
        


        
        