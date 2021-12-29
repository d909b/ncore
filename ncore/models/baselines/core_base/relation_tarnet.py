"""
Copyright (C) 2021  Patrick Schwab, GlaxoSmithKline plc, Sonali Parbhoo, Harvard University
Copyright (C) 2019  Patrick Schwab, ETH Zurich

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
 of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
import torch
import torch.nn as nn


class ModulatedLinear(nn.Linear):
    def __init__(self, in_features, out_features, num_treatments, bias=True, suppress_treatment_index=False,
                 with_modulators=True):
        super(ModulatedLinear, self).__init__(in_features, out_features, bias)
        self.embedding_w = nn.Embedding(num_treatments, out_features)
        self.embedding_b = nn.Embedding(num_treatments, out_features)
        nn.init.constant_(self.embedding_w.weight, 1.0)
        nn.init.constant_(self.embedding_b.weight, 0.0)
        self.suppress_treatment_index = suppress_treatment_index
        self.activation = nn.LeakyReLU()
        self.with_modulators = with_modulators
        self.num_treatments = num_treatments

    def forward(self, x):
        x, t = x
        x = super(ModulatedLinear, self).forward(x)
        if self.with_modulators:
            for treatment_offset in range(self.num_treatments):
                emb_w = self.embedding_w.weight[treatment_offset:treatment_offset+1]
                emb_b = self.embedding_b.weight[treatment_offset]
                ti = t[..., treatment_offset:treatment_offset+1]
                x = (x*emb_w)*ti + (1-ti)*x + emb_b*ti
        x = self.activation(x)

        if self.suppress_treatment_index:
            return x
        else:
            return x, t


class RelationTARNET(nn.Module):
    def __init__(self, input_dim, num_treatments,
                 num_base_layers, num_base_units,
                 num_head_layers, num_head_units,
                 with_modulators=True):
        super(RelationTARNET, self).__init__()
        sequence = []
        last_units = input_dim
        for _ in range(num_base_layers):
            dense = nn.Linear(last_units, num_base_units)
            activation = nn.LeakyReLU()
            sequence.append(dense)
            sequence.append(activation)
            last_units = num_base_units
        self.base_layer = nn.Sequential(*sequence)

        self.modulators = []
        self.head_layers = []
        for t_idx in range(num_treatments):
            sequence = []
            last_units = num_base_units
            for head_layer_idx in range(num_head_layers):
                is_last = head_layer_idx == num_head_layers - 1
                dense = ModulatedLinear(last_units, num_head_units, num_treatments,
                                        suppress_treatment_index=is_last,
                                        with_modulators=with_modulators)
                sequence.append(dense)
                last_units = num_head_units
            last_layer = nn.Linear(last_units, 1)
            sequence.append(last_layer)
            head_layer = nn.Sequential(*sequence)
            self.head_layers.append(head_layer)
        self.head_layers = nn.ModuleList(self.head_layers)

    def train_pure(self):
        for head_layer in self.head_layers:
            for dense in head_layer:
                if hasattr(dense, "embedding_w"):
                    dense.embedding_w.requires_grad_(False)
                    dense.embedding_b.requires_grad_(False)
                dense.weight.requires_grad = True
                dense.bias.requires_grad = True

    def train_mixed(self):
        for head_layer in self.head_layers:
            for dense in head_layer:
                if hasattr(dense, "embedding_w"):
                    dense.embedding_w.requires_grad_(True)
                    dense.embedding_b.requires_grad_(True)
                dense.weight.requires_grad = False
                dense.bias.requires_grad = False

    @staticmethod
    def convert_binary_list_to_int(binary_list, num_bits):
        mask = 2 ** torch.arange(num_bits-1, -1, -1).to(binary_list.device, binary_list.dtype)
        return torch.sum(mask * binary_list, -1)

    def forward(self, x):
        x, t = x
        x = self.base_layer(x)

        return_values = []
        for head_layer in self.head_layers:
            out_val = head_layer([x, t])
            return_values.append(out_val)
        x = torch.stack(return_values) * torch.unsqueeze(torch.transpose(t, 0, 1), dim=-1)
        x = torch.sum(x, dim=0) / (torch.sum(t, dim=-1, keepdim=True) + torch.finfo(torch.float32).eps)
        x = torch.squeeze(x, dim=-1)
        return x
