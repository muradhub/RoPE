# RoPE
Rotationary Positional Embedding



Usage:
batch_size = 32
seq_length = 100
d_model = 512
inputs = np.random.uniform(size=(batch_size, seq_length, d_model))

rope = RoPE(d_model, seq_length)
output = rope(inputs)
