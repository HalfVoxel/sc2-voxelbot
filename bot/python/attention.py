import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import math


class SelfAttention2(nn.Module):
    def __init__(self, input_size, key_size=4, heads=1):
        assert input_size % heads == 0, f"input_size must be divisible by heads ({input_size} // {heads})"
        super(SelfAttention2, self).__init__()
        self.input_size = input_size
        self.key_size = key_size
        self.value_size = input_size // heads
        self.heads = heads

        # Input x Mkey, Input x Mval, Input x Mquery
        self.key = nn.Conv1d(input_size, heads * key_size, kernel_size=1)
        self.query = nn.Conv1d(input_size, heads * key_size, kernel_size=1)
        self.value = nn.Conv1d(input_size, heads * self.value_size, kernel_size=1)
        self.bias = nn.Conv1d(input_size, heads, kernel_size=1)

        # self.embedding = nn.Embedding(self.input_size, self.hidden_size)
        self.lin1 = nn.Linear(self.input_size, self.input_size)
        self.act1 = nn.ReLU()
        self.norm1 = nn.LayerNorm(self.input_size)
        self.norm2 = nn.LayerNorm(self.input_size)

    def forward(self, input, mask):
        batch_size = input.size()[0]
        sequence_length = input.size()[1]
        heads = self.heads

        if mask is not None:
            assert mask.dtype == torch.float
            assert mask.size() == (batch_size, sequence_length)
        assert input.size() == (batch_size, sequence_length, self.input_size)

        # assert input_mask.size() == (batch_size,)
        it = input.transpose(1,2)
        keys = self.key(it).view(batch_size, heads, self.key_size, sequence_length)
        queries = self.query(it).view(batch_size, heads, self.key_size, sequence_length)
        values = self.value(it).view(batch_size, heads, self.value_size, sequence_length)

        # Note: the 1 at the end is important for broadcasting
        # One bias for each result element (i) that is added to all attention scores for that element.
        # This affects the sigmoid cutoff that happens later
        biases = self.bias(it).view(batch_size, heads, 1, sequence_length)

        keys *= math.sqrt(1.0 / self.key_size)

        # Calculate all dot products between the queries and values
        # bmm = batch matrix multiplication
        # scores = Batch index x attention score that element i has for element j x result element index (i)
        scores = queries.transpose(2,3).matmul(keys) # + biases
        assert scores.size() == (batch_size, heads, sequence_length, sequence_length)

        # Apply a non-linearity and zero out any score for masked out items
        scores = scores.sigmoid()
        if mask is not None:
            scores = scores * mask.view(batch_size, 1, 1, sequence_length)
            scores = scores * mask.view(batch_size, 1, sequence_length, 1)
        # print(scores)

        # (BxHxVxS) x (BxHxSxS) = (BxHxVxS) -> (BxSxHV)
        attention = values.matmul(scores).view(batch_size, -1, sequence_length).transpose(1,2)
        # Residual normalization 1
        res = self.norm1(input + attention)
        # res = attention
        feedforward = self.act1(self.lin1(res))
        # Residual normalization 2
        result = self.norm2(res + feedforward)
        result = attention
        return result

class SelfAttention(nn.Module):
    def __init__(self, input_size, key_size=4, heads=1):
        assert input_size % heads == 0, f"input_size must be divisible by heads ({input_size} // {heads})"

        super(SelfAttention, self).__init__()
        self.input_size = input_size
        self.key_size = key_size

        # Input x Mkey, Input x Mval, Input x Mquery
        self.key = nn.Linear(input_size, key_size)
        self.query = nn.Linear(input_size, key_size)
        self.value_size = input_size
        self.value = nn.Linear(input_size, self.value_size)
        self.bias = nn.Linear(input_size, 1)

        # self.embedding = nn.Embedding(self.input_size, self.hidden_size)
        self.lin1 = nn.Linear(self.value_size, self.input_size)
        self.act1 = nn.ReLU()
        self.norm1 = nn.LayerNorm(self.input_size)
        self.norm2 = nn.LayerNorm(self.input_size, elementwise_affine=False)

    def forward(self, input, mask):
        batch_size = input.size()[0]
        sequence_length = input.size()[1]
        if mask is not None:
            assert mask.dtype == torch.float
            assert mask.size() == (batch_size, sequence_length), (mask.size(), (batch_size, sequence_length))
        assert input.size()[1:] == (sequence_length, self.input_size)

        # assert input_mask.size() == (batch_size,)

        keys = self.key(input)
        keys *= math.sqrt(1.0 / self.key_size)
        queries = self.query(input)
        values = self.value(input)
        biases = self.bias(input)
        # input_mask = input_mask.view((batch_size, 1))

        # Note: the 1 at the end is important for broadcasting
        # One bias for each result element (i) that is added to all attention scores for that element.
        # This affects the sigmoid cutoff that happens later
        assert biases.size() == (batch_size, sequence_length, 1)

        # Calculate all dot products between the queries and values
        # bmm = batch matrix multiplication
        # scores = Batch index x result element index (i) x attention score that element i has for element j
        scores = torch.bmm(keys, torch.transpose(queries, 1, 2)) + biases
        scores = scores.transpose(1,2)
        assert scores.size() == (batch_size, sequence_length, sequence_length)

        # Apply a non-linearity and zero out any score for masked out items
        scores = scores.sigmoid()
        if mask is not None:
            scores = scores * mask.view(batch_size, 1, sequence_length)
        # print(scores)

        # (BxSxS) x (BxSxN) = (BxSxN)
        attention = torch.bmm(scores, values)
        # Residual normalization 1
        res = self.norm1(input + attention)
        # res = attention
        feedforward = self.act1(self.lin1(res))
        # Residual normalization 2
        result = self.norm2(res + feedforward)
        # result = feedforward
        return result


class AttentionDecoder(nn.Module):
    def __init__(self, max_sequence_length, input_size, heads, value_size, output_sequence_length, key_size=4):
        super(AttentionDecoder, self).__init__()
        self.input_size = input_size
        self.heads = heads
        self.key_size = key_size
        self.max_sequence_length = max_sequence_length
        self.output_sequence_length = output_sequence_length

        self.key = nn.Linear(input_size, heads * key_size)
        self.value_size = value_size
        self.value = nn.Linear(input_size, heads * self.value_size)
        initial_query_data = torch.randn((self.output_sequence_length, self.heads, key_size), dtype=torch.float)
        self.queries = torch.nn.Parameter(data=initial_query_data, requires_grad=True)
        self.softmax = nn.Softmax(dim=3)
        self.output_size = self.heads * self.value_size

    def forward(self, input, mask):
        batch_size = input.size()[0]
        if mask is not None:
            assert mask.dtype == torch.uint8, "Mask must be an uint8 tensor"
            assert mask.size() == (batch_size, self.max_sequence_length), "Mask has the wrong size"
        expected_input_size = (batch_size, self.max_sequence_length, self.input_size)
        assert input.size(
        ) == expected_input_size, f"Input has the wrong size, found {input.size()} but expected {expected_input_size}"

        if mask is not None:
            floatMask = mask.to(dtype=torch.float)
        keys = self.key(input).view(batch_size, 1, self.max_sequence_length, self.heads, self.key_size, 1)
        keys *= math.sqrt(1.0 / self.key_size)

        values = self.value(input).view(batch_size, self.max_sequence_length, self.heads, self.value_size)  # BxSxHxV
        # print(keys)
        # print(input)
        # 1xOx1xHx1xK x Bx1xSxHxKx1 = BxOxSxHx1x1
        scores = torch.matmul(self.queries.view(1, self.output_sequence_length, 1, self.heads, 1, self.key_size), keys)
        scores = scores.view((batch_size, self.output_sequence_length, self.max_sequence_length, self.heads))

        if mask is not None:
            # Apply a non-linearity and mask items
            # Set scores to negative infinity (: is over all heads)
            # scores[~mask,:] = -math.inf
            scores = scores - 100000000 * (1 - floatMask).view(batch_size, 1, self.max_sequence_length, 1)
        scores = torch.transpose(scores, 2, 3)
        scores = self.softmax(scores)
        scores = torch.transpose(scores, 2, 3)

        # BxOxHx1xS x Bx1xHxSxV = BxOxHx1xV
        reshaped_scores = scores.transpose(2, 3).view(batch_size, self.output_sequence_length, self.heads, 1, self.max_sequence_length)
        reshaped_values = torch.transpose(values, 1, 2).view(batch_size, 1, self.heads, self.max_sequence_length, self.value_size)
        attention = torch.matmul(reshaped_scores, reshaped_values)
        attention = attention.view((batch_size, self.output_sequence_length, self.heads * self.value_size))

        return attention


class AttentionModule(nn.Module):
    def __init__(self, max_sequence_length, input_size, heads, value_size, key_size=4, self_attention_key_size=4, output_size=2, self_attention_heads=1):
        super(AttentionModule, self).__init__()
        self.input_size = input_size
        self.heads = heads
        self.key_size = key_size
        self.max_sequence_length = max_sequence_length
        self.output_size = output_size

        self.lin0 = nn.Linear(self.input_size, self.input_size)
        self.act0 = nn.ReLU()

        self.encoder = SelfAttention2(input_size, self_attention_key_size, heads=self_attention_heads)
        self.encoder2 = SelfAttention2(input_size, self_attention_key_size, heads=self_attention_heads)
        # self.encoder3 = SelfAttention(max_sequence_length, input_size, self_attention_key_size)
        # self.encoder4 = SelfAttention(max_sequence_length, input_size, self_attention_key_size)
        # self.encoder5 = SelfAttention(max_sequence_length, input_size, self_attention_key_size)
        # self.encoder6 = SelfAttention(max_sequence_length, input_size, self_attention_key_size)
        self.key = nn.Linear(input_size, heads * key_size)
        self.value_size = value_size
        self.value = nn.Linear(input_size, heads * self.value_size)
        self.queries = torch.nn.Parameter(data=torch.randn(
            (self.heads, key_size), dtype=torch.float), requires_grad=True)
        # self.queries2 = torch.nn.Parameter(data=torch.randn((input_size), device=device, dtype=torch.float), requires_grad=True)
        self.softmax = nn.Softmax(dim=2)
        self.lin1 = nn.Linear(self.heads * self.value_size, self.output_size)

    def forward(self, input, mask):
        batch_size = input.size()[0]
        assert mask.dtype == torch.uint8, "Mask must be an uint8 tensor"
        assert mask.size() == (batch_size, self.max_sequence_length), "Mask has the wrong size"
        expected_input_size = (batch_size, self.max_sequence_length, self.input_size)
        assert input.size(
        ) == expected_input_size, f"Input has the wrong size, found {input.size()} but expected {expected_input_size}"

        floatMask = mask.to(dtype=torch.float)
        encoded = self.act0(self.lin0(input))
        encoded = self.encoder(input, floatMask)
        encoded = self.encoder2(encoded, floatMask)
        # encoded = self.encoder3(encoded, floatMask)
        # encoded = self.encoder4(encoded, floatMask)
        # encoded = self.encoder5(encoded, mask)
        # encoded = self.encoder6(encoded, mask)
        keys = self.key(encoded).view(batch_size, self.max_sequence_length, self.heads, self.key_size)
        keys *= math.sqrt(1.0 / self.key_size)

        values = self.value(encoded).view(batch_size, self.max_sequence_length, self.heads, self.value_size)  # BxSxHxV
        # print(keys)
        # print(encoded)
        # BxSxHx1xK x HxKx1 = BxSxHx1x1
        # Bi Sj Hk Kl K2m H2n
        scores = torch.matmul(keys.view(batch_size, self.max_sequence_length, self.heads, 1,
                                        self.key_size), self.queries.view(self.heads, self.key_size, 1))
        scores = scores.view((batch_size, self.max_sequence_length, self.heads))
        assert scores.size() == (batch_size, self.max_sequence_length, self.heads), scores.size()

        # Apply a non-linearity and mask items
        # Set scores to negative infinity (: is over all heads)
        # scores[~mask,:] = -math.inf
        scores = scores - 100000000 * (1 - floatMask).view(batch_size, self.max_sequence_length, 1)
        scores = torch.transpose(scores, 1, 2)
        scores = self.softmax(scores)
        scores = torch.transpose(scores, 1, 2)
        # print(scores)
        # print(scores)

        # print(scores)

        # BxHx1xS x BxHxSxV = BxHx1xV
        attention = torch.matmul(torch.transpose(scores, 1, 2).view(batch_size, self.heads, 1, self.max_sequence_length), torch.transpose(values, 1, 2))
        attention = attention.view((batch_size, self.heads, self.value_size))
        assert attention.size() == (batch_size, self.heads, self.value_size)

        attention = attention.view(batch_size, -1)
        res = self.lin1(attention)
        res = res.view((-1, self.output_size))
        # res += self.queries2
        return res


def unit_test_attention():
    import random
    # Check that shuffling doesn't change anything
    m = SelfAttention2(input_size=4, key_size=5, heads=2)
    input = np.array([[1, 2, 3, 4], [3, 4, 5, 6], [0, 1, 0, 1], [10, 9, 8, 4]])
    mask = [1, 1, 1, 1]
    input2 = input[[0,3,2,1],:]
    maskTensor = torch.tensor(mask, dtype=torch.float).view(1, 4)
    shouldBeZero = m(torch.tensor(input, dtype=torch.float).view(1, 4, 4), maskTensor) - \
        m(torch.tensor(input2, dtype=torch.float).view(1, 4, 4), maskTensor)
    assert shouldBeZero.size() == (1, 4, m.input_size), shouldBeZero
    assert shouldBeZero.sum() < 0.0001, shouldBeZero

    # Check that masked out elements do not contribute to the result
    m = SelfAttention2(input_size=6, key_size=5, heads=2)
    input = [[1, 2, 3, 4, 5, 1], [3, 4, 5, 6, 7, 1], [0, 1, 0, 1, 2, 1], [10, 9, 8, 4, 2, 1]]
    mask = [1, 0, 0, 1]
    input2 = [[1, 2, 3, 4, 5, 1], [1, 4, 6, 1, 3, 1], [9, 2, 9, 2, 9, 1], [10, 9, 8, 4, 2, 1]]
    maskTensor = torch.tensor(mask, dtype=torch.float).view(1, 4)
    shouldBeZero = m(torch.tensor(input, dtype=torch.float).view(1, 4, 6), maskTensor) - \
        m(torch.tensor(input2, dtype=torch.float).view(1, 4, 6), maskTensor)
    assert shouldBeZero.size() == (1, 4, m.input_size), shouldBeZero
    assert shouldBeZero.sum() < 0.0001, shouldBeZero



    # Check that shuffling doesn't change anything
    m = AttentionModule(max_sequence_length=4, value_size=4, input_size=4, key_size=4, output_size=6, heads=2)
    input = [[1, 2, 3, 4], [3, 4, 5, 6], [0, 1, 0, 1], [10, 9, 8, 4]]
    mask = [1, 1, 1, 1]
    input2 = input[:]
    random.shuffle(input2)
    maskTensor = torch.tensor(mask, dtype=torch.uint8).view(1, 4)
    shouldBeZero = m(torch.tensor(input, dtype=torch.float).view(1, 4, 4), maskTensor) - \
        m(torch.tensor(input2, dtype=torch.float).view(1, 4, 4), maskTensor)
    assert shouldBeZero.size() == (1, m.output_size), shouldBeZero
    assert shouldBeZero.sum() < 0.0001, shouldBeZero

    # Check that masked out elements do not contribute to the result
    m = AttentionModule(max_sequence_length=4, value_size=4, input_size=6, key_size=4, output_size=6, heads=2)
    input = [[1, 2, 3, 4, 5, 1], [3, 4, 5, 6, 7, 1], [0, 1, 0, 1, 2, 1], [10, 9, 8, 4, 2, 1]]
    mask = [1, 0, 0, 1]
    input2 = [[1, 2, 3, 4, 5, 1], [1, 4, 6, 1, 3, 1], [9, 2, 9, 2, 9, 1], [10, 9, 8, 4, 2, 1]]
    maskTensor = torch.tensor(mask, dtype=torch.uint8).view(1, 4)
    shouldBeZero = m(torch.tensor(input, dtype=torch.float).view(1, 4, 6), maskTensor) - \
        m(torch.tensor(input2, dtype=torch.float).view(1, 4, 6), maskTensor)
    assert shouldBeZero.size() == (1, m.output_size), shouldBeZero
    assert shouldBeZero.sum() < 0.0001, shouldBeZero

    # Check that shuffling doesn't change anything
    m = AttentionDecoder(max_sequence_length=4, value_size=4, input_size=4, key_size=4, heads=2, output_sequence_length=3)
    input = [[1, 2, 3, 4], [3, 4, 5, 6], [0, 1, 0, 1], [10, 9, 8, 4]]
    mask = [1, 1, 1, 1]
    input2 = input[:]
    random.shuffle(input2)
    maskTensor = torch.tensor(mask, dtype=torch.uint8).view(1, 4)
    shouldBeZero = m(torch.tensor(input, dtype=torch.float).view(1, 4, 4), maskTensor) - \
        m(torch.tensor(input2, dtype=torch.float).view(1, 4, 4), maskTensor)
    assert shouldBeZero.size() == (1, 3, m.heads*m.value_size)
    assert shouldBeZero.sum() < 0.0001

    # Check that masked out elements do not contribute to the result
    m = AttentionDecoder(max_sequence_length=4, value_size=4, input_size=6, key_size=4, heads=2, output_sequence_length=3)
    input = [[1, 2, 3, 4, 5, 1], [3, 4, 5, 6, 7, 1], [0, 1, 0, 1, 2, 1], [10, 9, 8, 4, 2, 1]]
    mask = [1, 0, 0, 1]
    input2 = [[1, 2, 3, 4, 5, 1], [1, 4, 6, 1, 3, 1], [9, 2, 9, 2, 9, 1], [10, 9, 8, 4, 2, 1]]
    maskTensor = torch.tensor(mask, dtype=torch.uint8).view(1, 4)
    shouldBeZero = m(torch.tensor(input, dtype=torch.float).view(1, 4, 6), maskTensor) - \
        m(torch.tensor(input2, dtype=torch.float).view(1, 4, 6), maskTensor)
    assert shouldBeZero.size() == (1, 3, m.heads * m.value_size)
    assert shouldBeZero.sum() < 0.0001, shouldBeZero.sum()


unit_test_attention()
