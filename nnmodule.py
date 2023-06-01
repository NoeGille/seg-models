import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """Split image into patches and then embed them"""

    def __init__(self, img_size:int=224, patch_size:int=16, in_channels:int=1, embed_dim:int=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # Define a conv layer to extract patches from the image
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        '''Transform the image into a tensor of patches'''
        # (n_samples, in_channels, img_size, img_size)
        x = self.proj(x) # (n_samples, embed_dim, n_patches**0.5, n_patches**0.5)
        x = x.flatten(2) # (n_samples, embed_dim, n_patches)
        x = x.transpose(1, 2)

        return x
    
class Attention(nn.Module):
    '''Attention mecanism'''

    def __init__(self, dim, n_heads, qkv_bias:bool=False, attn_p:float=0., proj_p:float=0):
        '''qkv_bias : If Ture then we include bias to the query, key adn value projections.

        attn_p : Dropout probability applied to the query, key and value tensors.
        proj_p : Dropout probability applied to the output tensor
        Note : Dropout is only applied during training, not during evaluation or prediction'''
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        # Input : embedding | Output : query, key and value vectors of the embedding
        # Note : We could write three seperate linear mapping that do the same thing
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        # Take the concatenates heads and map them into a new space
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)
        
    def forward(self, x):
        n_samples, n_tokens, dim = x.shape
        if dim != self.dim:
            raise ValueError
        
        qkv = self.qkv(x) # (n_samples, n_patches + 1, 3 * dim)
        qkv = qkv.reshape(
            n_samples, n_tokens, 3, self.n_heads, self.head_dim
        ) # (n_samples, n_patches + 1, 3, n_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) # (3, n_samples, n_heads, n_patches + 1, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1)
        dot_product = (
            q @ k_t
        ) * self.scale # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = dot_product.softmax(dim=-1) # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = self.attn_drop(attn)
        weighted_avg = attn @ v # (n_samples, n_heads, n_patches + 1, head_dim)
        # Flatten last 2 dimension <=> Concatenate each head output
        weighted_avg = weighted_avg.transpose(1, 2) # (n_samples, n_patches + 1, n_heads, head_dim)
        # head_dim = dim // n_heads => Get the same dimesion as input
        weighted_avg = weighted_avg.flatten(2) # (n_samples, n_patches + 1, dim)
        # Final linear projection and dropout
        x = self.proj(weighted_avg) # (n_samples, n_patches + 1, dim)
        x = self.proj_drop(x) # (n_samples, n_patches + 1, dim)

        return x
    
class MLP(nn.Module):
    '''MultiLayer Perception'''
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        '''One hidden layer'''
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        return self.dropout(self.fc2(self.dropout(self.activation(self.fc1(x)))))


class Block(nn.Module):

    def __init__(self, dim, n_heads, mlp_ratio, qkv_bias, p, attn_p):
        '''mlp_ratio : determine the hidden dimension size of the mlp module with respect to dim'''
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, n_heads, qkv_bias, attn_p, proj_p=p)
        self.mlp = MLP(
            in_features=dim, 
            hidden_features=int(dim * mlp_ratio), 
            out_features=dim
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class DownSampleBlock(nn.Module):
    '''Reduce the dimension of the image in input by 2'''
    def __init__(self, in_channels, out_channels, dilation=1):
        super(DownSampleBlock, self).__init__()
        # We keep the same dimension in input and ouput
        self.conv = DoubleConvolution(in_channels, out_channels, dilation=dilation)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2,2))

    def forward(self, x):
        x = self.conv(x)
        return self.pool(x), x

class DoubleConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super(DoubleConvolution, self).__init__()
        # We keep the same dimension in input and ouput

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1), padding=dilation, dilation=dilation)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1), padding=dilation, dilation=dilation)
        self.norm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
      x = self.relu(self.norm2(self.conv2(self.relu(self.norm1(self.conv1(x))))))
      return x

class UpSampleBlock(nn.Module):
    '''Increase the dimension of the input and reduce its number of channels
    out_channels reduces the number of channels of the input by 2 by default
    It can be set to a different value'''
    def __init__(self, in_channels, out_channels=None):
        super(UpSampleBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.up1 = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        self.conv1 = nn.Conv2d(in_channels=out_channels,
                                     out_channels=out_channels, kernel_size=(3, 3),
                                     stride=(1, 1), padding=(1,1))
        
    def forward(self, x):
        x = self.conv1(self.up1(x))
        return x

class ResidualConnection(nn.Module):
    '''Concatenate inputs of two blocs'''

    def __init__(self, in_channels, out_channels):
        '''in_channels has the same dimensions as out_channels'''

        super(ResidualConnection, self).__init__()
        self.conv = DoubleConvolution(in_channels * 2, out_channels)

    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
