import numpy as np
import torch
import torch.nn.functional as F

class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, d_k, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.dropout = torch.nn.Dropout(dropout)
    def forward(self, q, k, v, mask=None):
        score = torch.matmul(q, k.transpose(-2,-1))
        score = score / np.sqrt(self.d_k)
        
        if mask is not None:
            mask = mask.unsqueeze(1)

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        self.d_model = d_model
        self.d_k = int(d_model / num_heads)
        self.h = num_heads
        if self.d_k*num_heads != self.d_model:
            raise Exception("d_model cannot be divided by num_heads.")
            
        self.fc_q = torch.nn.Linear(d_model, d_model)
        self.fc_k = torch.nn.Linear(d_model, d_model)
        self.fc_v = torch.nn.Linear(d_model, d_model)
        torch.nn.init.xavier_normal_(self.fc_q.weight)
        torch.nn.init.xavier_normal_(self.fc_k.weight)
        torch.nn.init.xavier_normal_(self.fc_v.weight)
        
        self.dropout = torch.nn.Dropout(dropout)
        
        self.out = torch.nn.Linear(d_model, d_model)
        
    def attention(self, q, k, v, d_k, mask=None):
        score = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_k)
        
        if mask is not None:
            mask = mask.unsqueeze(1)
            score.data.masked_fill(mask==0, -float('inf'))
            
        score = self.softmax(score)
        score = self.dropout(score)
        output = torch.matmul(score, v)
        
        return output
    
    def forward(self, q, k, v, mask=None):
#         (batch, time, hidden)
        batch_size = q.size(0)
        
#         (batch, head, time, hidden/head)
        q = (self.fc_q(q).view(batch_size, -1, self.h, self.d_k)).transpose(1,2)
        k = (self.fc_k(k).view(batch_size, -1, self.h, self.d_k)).transpose(1,2)
        v = (self.fc_v(v).view(batch_size, -1, self.h, self.d_k)).transpose(1,2)
        
#         (batch, head, time, hidden/head)
        outputs = self.attention(q, k, v, self.d_k, mask)
    
#         (batch, time, hidden)
        outputs = outputs.transpose(1,2).contiguous().view(batch_size, -1, self.d_model)

        output = self.out(outputs)
        
        return output
        

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        
        self.d_model = d_model
        
        pe = torch.zeros(size=(max_len, d_model))
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = np.sin(pos / 10000 ** (i / d_model))
                pe[pos, i+1] = np.cos(pos / 10000 ** (i / d_model))
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
        
    def forward(self, x):
        seq_len = x.size(1)
        output = x + self.pe[:, :seq_len, :].detach()
        
        return output
        
        
def create_mask(input_sequence, target_sequence, pad_token):
    src_mask = (input_sequence == pad_token)
    
    target_mask = (target_sequence == pad_token)
    
    size = target_sequence.size(1)
    
    time_mask = np.triu(np.ones((1, size, size)), k=1)
    
    target_mask = target_mask | time_mask
    
    return src_mask, target_mask
    
    
class FeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        
        self.fc1 = torch.nn.Linear(d_model, d_ff)
        self.fc2 = torch.nn.Linear(d_ff, d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        output = self.fc1(x)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        output = self.dropout(output)
        
        return output
        
        
class Norm(torch.nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(Norm, self).__init__()
        
        self.size = d_model
        
        self.alpha = torch.nn.Parameter(torch.ones(self.size))
        self.bias = torch.nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    
    def forward(self, x):
        output = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return output
        
        
class Encoder(torch.nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(Encoder, self).__init__()
        
        self.norm1 = Norm(d_model)
        self.norm2 = Norm(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.fc = FeedForward(d_model, d_model*4, dropout)
        
    def forward(self, x, src_mask=None):
        output = self.attention(x, x, x, src_mask)
        output = self.norm1(output + x)
        output_res = output
        output = self.fc(output)
        output = self.norm2(output + output_res)
        
        return output
        
        
class Decoder(torch.nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(Decoder, self).__init__()
        
        self.norm1 = Norm(d_model)
        self.norm2 = Norm(d_model)
        self.norm3 = Norm(d_model)
        self.attention1 = MultiHeadAttention(d_model, num_heads, dropout)
        self.attention2 = MultiHeadAttention(d_model, num_heads, dropout)
        self.fc = FeedForward(d_model, d_model*4, dropout)
    
    def forward(self, x, encoder_output, src_mask=None, target_mask=None):
        output = self.attention1(x, x, x, target_mask)
        output = self.norm1(output + x)
        output_res = output
        output = self.attention2(output, encoder_output, encoder_output, src_mask)
        output = self.norm2(output + output_res)
        output_res = output
        output = self.fc(output)
        output = self.norm3(output + output_res)
        
        return output
        
        
class EncoderStack(torch.nn.Module):
    def __init__(self, d_model, num_heads, num_layers, max_len, dropout=0.1, input_dropout=0.1):
        super(EncoderStack, self).__init__()
        
        self.max_len = max_len
        self.num_layers = num_layers
        self.pe = PositionalEncoding(d_model, max_len)
        self.layers = torch.nn.ModuleList([Encoder(d_model, num_heads, dropout) for l in range(num_layers)])
        self.input_dropout = torch.nn.Dropout(input_dropout)
        
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            torch.nn.BatchNorm2d(32),
            torch.nn.Hardtanh(0, 20, inplace=True),
            torch.nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            torch.nn.BatchNorm2d(32),
            torch.nn.Hardtanh(0, 20, inplace=True)
        )
        self.linear = torch.nn.Linear(4128,d_model)
        
    def forward(self, x, src_mask=None):
        output = x.unsqueeze(1)
        output = self.conv(output)
        output = output.transpose(1, 2)
        output = output.contiguous()
        sizes = output.size()
        output = output.view(sizes[0], sizes[1], sizes[2] * sizes[3])
        output = self.linear(output)
        
        output = self.input_dropout(output)
        output = self.pe(output[:,:self.max_len,:])
        
        for layer in range(self.num_layers):
            output = self.layers[layer](output, src_mask)
        
        return output
        
        
class DecoderStack(torch.nn.Module):
    def __init__(self, d_model, num_heads, num_layers, max_len, vocab_size, sos_id, eos_id, dropout=0.1, input_dropout=0.1):
        super(DecoderStack, self).__init__()
        
        self.num_layers = num_layers
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, max_len)
        self.layers = torch.nn.ModuleList([Decoder(d_model, num_heads, dropout) for l in range(num_layers)])
        self.sos_id = sos_id
        self.max_length = max_len
        self.fc = torch.nn.Linear(d_model, vocab_size)
        self.out = torch.nn.LogSoftmax(dim=2)
        self.input_dropout = torch.nn.Dropout(input_dropout)
        
    def decode(self, step_input, step_output):
        symbols = step_output.topk(1)[1]
        concat = torch.cat([step_input, symbols], dim=1)
        return concat
        
    def forward(self, x=None, encoder_output=None, src_mask=None, target_mask=None, is_predict=False):
#         when test, x=None / is_predict=True
#         when eval, x=script / is_predcit=True

        pred_outputs = []
        if x is None:
            batch_size = 1
            max_length = self.max_length
        else:
            batch_size = x.size(0)
            max_length = x.size(1)-1
        
        if is_predict == False:     
            output = self.embedding(x[:,:-1])
            output = self.pe(output)
            output = self.input_dropout(output)
            
            for layer in range(self.num_layers):
                output = self.layers[layer](output, encoder_output, src_mask, target_mask)
                
            output = self.fc(output)
            output = self.out(output)
        else:
#             make <sos> batch                          
            step_input = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1).cuda()
            
#             (batch, 1)
#             step_input => token
            for i in range(max_length):
#                 step_input = x[:, :i+1]
                step_output = step_input
                
                step_output = self.embedding(step_output)
#                 (batch, i+1, d_model)
                step_output = self.pe(step_output)
#                 (batch, i+1, d_model)  
                for layer in range(self.num_layers):
                    step_output = self.layers[layer](step_output, encoder_output, src_mask)

#                     (batch, i+1, d_model)
                
                step_output = self.fc(step_output)
#                 (batch, i+1, vocab)
                step_output = self.out(step_output)
#                 (batch, i+1, vocab)
#                 print(step_output.shape)

                step_output = step_output[:,i,:]
#                 (batch, vocab)

                pred_outputs.append(step_output)
#                 [(batch, vocab) * (i+1)]
                                    
                step_input = self.decode(step_input, step_output)
#                 (batch, i+2)

            output = torch.stack(pred_outputs, dim=1)
#             (batch, max_length, vocab)

        return output
        
        
class Transformer(torch.nn.Module):
    def __init__(self, d_model, num_heads, num_laysers, vocab_size, enc_max_len, dec_max_len, sos_id, eos_id, dropout=0.1, input_dropout=0.1):
        super(Transformer, self).__init__()
        
        self.encoder = EncoderStack(d_model, num_heads, num_laysers, enc_max_len, dropout, input_dropout)
        self.decoder = DecoderStack(d_model, num_heads, num_laysers, dec_max_len, vocab_size, sos_id, eos_id, dropout, input_dropout)
        
    def forward(self, x, target=None, src_mask=None, target_mask=None, is_predict=False):
        encoder_output = self.encoder(x, src_mask)
        decoder_output = self.decoder(target, encoder_output, src_mask, target_mask, is_predict)
        
        return decoder_output