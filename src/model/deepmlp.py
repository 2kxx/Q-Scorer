from torch import nn

class ResidualMLPBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResidualMLPBlock, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()
        self.use_residual = input_dim == output_dim  # 只有维度匹配时可加残差

    def forward(self, x):
        out = self.linear(x)
        out = self.activation(out)
        if self.use_residual:
            out = out + x  # 残差连接
        return out


class DeepMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(DeepMLP, self).__init__()

        layers = []

        # 输入层
        layers.append(ResidualMLPBlock(input_dim, hidden_dims[0]))

        # 中间层
        for i in range(len(hidden_dims) - 1):
            layers.append(ResidualMLPBlock(hidden_dims[i], hidden_dims[i + 1]))

        # 输出层（不加激活和残差）
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

        self.model = nn.Sequential(*layers)


    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.model(x)
        x = self.output_layer(x).squeeze(dim=-1)
        return x