# decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import json

with open("vocab.json", "r") as f:
    VOCAB = json.load(f)

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super(Attention, self).__init__()

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, encoder_outputs, hidden):
        batch_size, seq_len, _ = encoder_outputs.shape

        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)

        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, attention):
        super(Decoder, self).__init__()

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.attention = attention

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.embedding(input)
        a = self.attention(encoder_outputs, hidden[0])
        a = a.unsqueeze(1)
        weighted = torch.bmm(a, encoder_outputs)
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        output = torch.cat((embedded.squeeze(0), hidden[0].squeeze(0), weighted.squeeze(1)), dim=1)
        prediction = self.fc_out(output)
        return prediction, hidden

class Seq2SeqWithAttention(nn.Module):
    def __init__(self, cnn_model, attention, output_dim, emb_dim, enc_hid_dim, dec_hid_dim):
        super(Seq2SeqWithAttention, self).__init__()

        self.cnn_model = cnn_model
        self.attention = attention
        self.decoder = Decoder(output_dim, emb_dim, enc_hid_dim, dec_hid_dim, attention)

    def forward(self, x, target, teacher_forcing_ratio=0.5):
        encoder_outputs = self.cnn_model(x)

        batch_size, _, _ = encoder_outputs.shape
        trg_len = target.shape[1]
        trg_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(x.device)
        input = target[:, 0]

        hidden = (torch.zeros(1, batch_size, self.decoder.rnn.hidden_size).to(x.device),
                  torch.zeros(1, batch_size, self.decoder.rnn.hidden_size).to(x.device))

        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[:, t, :] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = target[:, t] if teacher_force and t < trg_len - 1 else top1

        return outputs
