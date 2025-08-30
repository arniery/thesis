# model_vc_clean.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearNorm(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super().__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.linear_layer.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super().__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=bias)
        nn.init.xavier_uniform_(self.conv.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, dim_neck, dim_emb, freq):
        super().__init__()
        self.dim_neck = dim_neck
        self.freq = freq

        convs = []
        for i in range(3):
            in_ch = 80 + dim_emb if i == 0 else 512
            convs.append(nn.Sequential(
                ConvNorm(in_ch, 512, kernel_size=5, padding=2, w_init_gain='relu'),
                nn.BatchNorm1d(512)
            ))
        self.convolutions = nn.ModuleList(convs)
        self.lstm = nn.LSTM(512, dim_neck, 2, batch_first=True, bidirectional=True)

    def forward(self, x, c_org):
        if x.dim() == 4:  # handle extra dims
            x = x.squeeze(1)

        # Expand speaker embedding across time
        c_org = c_org.unsqueeze(-1).expand(-1, -1, x.size(-1))
        x = torch.cat((x, c_org), dim=1)

        for conv in self.convolutions:
            x = F.relu(conv(x))

        x = x.transpose(1, 2)  # (B, T, C)
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        out_forward = outputs[:, :, :self.dim_neck]
        out_backward = outputs[:, :, self.dim_neck:]

        codes = []
        for i in range(0, outputs.size(1), self.freq):
            idx_forward = min(i + self.freq - 1, outputs.size(1) - 1)
            idx_backward = i
            codes.append(torch.cat((out_forward[:, idx_forward, :], out_backward[:, idx_backward, :]), dim=-1))

        return codes


class Decoder(nn.Module):
    def __init__(self, dim_neck, dim_emb, dim_pre):
        super().__init__()
        self.lstm1 = nn.LSTM(dim_neck * 2 + dim_emb, dim_pre, 1, batch_first=True)
        convs = []
        for _ in range(3):
            convs.append(nn.Sequential(
                ConvNorm(dim_pre, dim_pre, kernel_size=5, padding=2, w_init_gain='relu'),
                nn.BatchNorm1d(dim_pre)
            ))
        self.convolutions = nn.ModuleList(convs)
        self.lstm2 = nn.LSTM(dim_pre, 1024, 2, batch_first=True)
        self.linear_projection = LinearNorm(1024, 80)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = x.transpose(1, 2)
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        x, _ = self.lstm2(x)
        return self.linear_projection(x)


class Postnet(nn.Module):
    def __init__(self):
        super().__init__()
        convs = [nn.Sequential(ConvNorm(80, 512, kernel_size=5, padding=2, w_init_gain='tanh'),
                               nn.BatchNorm1d(512))]
        for _ in range(3):
            convs.append(nn.Sequential(ConvNorm(512, 512, kernel_size=5, padding=2, w_init_gain='tanh'),
                                       nn.BatchNorm1d(512)))
        convs.append(nn.Sequential(ConvNorm(512, 80, kernel_size=5, padding=2, w_init_gain='linear'),
                                   nn.BatchNorm1d(80)))
        self.convolutions = nn.ModuleList(convs)

    def forward(self, x):
        for conv in self.convolutions[:-1]:
            x = torch.tanh(conv(x))
        x = self.convolutions[-1](x)
        return x


class Generator(nn.Module):
    """Generator network (compatible with old checkpoints)."""
    def __init__(self, dim_neck, dim_emb, dim_pre, freq):
        super(Generator, self).__init__()
        
        self.encoder = Encoder(dim_neck, dim_emb, freq)
        self.decoder = Decoder(dim_neck, dim_emb, dim_pre)
        self.postnet = Postnet()

    def forward(self, x, c_org, c_trg=None):
        """
        x: (B, 80, T) mel-spectrogram
        c_org: (B, dim_emb) speaker embedding of source
        c_trg: (B, dim_emb) speaker embedding of target
        """

        # --- Encoder ---
        codes = self.encoder(x, c_org)  # list of code tensors

        # Expand codes along time
        tmp = []
        for code in codes:
            expand_len = int(x.size(2) / len(codes))
            tmp.append(code.unsqueeze(1).expand(-1, expand_len, -1))
        code_exp = torch.cat(tmp, dim=1)

        # Fix time dimension mismatch
        if code_exp.size(1) != x.size(2):
            code_exp = F.interpolate(
                code_exp.transpose(1, 2),  # (B, C, T)
                size=x.size(2),
                mode='linear',
                align_corners=False
            ).transpose(1, 2)

        # Concatenate with target speaker embedding if provided
        if c_trg is not None:
            encoder_outputs = torch.cat(
                (code_exp, c_trg.unsqueeze(1).expand(-1, code_exp.size(1), -1)),
                dim=-1
            )
        else:
            encoder_outputs = code_exp

        # --- Decoder ---
        mel_outputs = self.decoder(encoder_outputs)

        # --- Postnet ---
        mel_outputs_postnet = self.postnet(mel_outputs.transpose(2, 1))
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(2, 1)

        # Add channel dimension
        mel_outputs = mel_outputs.unsqueeze(1)
        mel_outputs_postnet = mel_outputs_postnet.unsqueeze(1)

        return mel_outputs, mel_outputs_postnet, torch.cat(codes, dim=-1)

