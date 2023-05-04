from nnmodule import PatchEmbed, Block, ResidualConnection, UpSampleBlock, DoubleConvolution, DownSampleBlock
import torch.nn as nn
import torch


class UNet(nn.Module):

    NB_OF_FILTERS = 16

    def __init__(self, input_size, num_classes:int=10, depth:int=2):
        '''### Initialize a UNet model
        input_size : dimension of input
        num_classes : specify the number of classes in ouput
        depth : the number of blocks (depth of the model)'''
        super(UNet, self).__init__()
        channels = [input_size[-1]] + [self.NB_OF_FILTERS * (i + 1) for i in range(depth)]
        # first downsampling block
        self.dblocks = nn.ModuleList([DownSampleBlock(in_channels=channels[0], out_channels=channels[1])])
        self.bottleneck = DoubleConvolution(in_channels=channels[-1], out_channels=channels[-1])
        # Concatenate outputs from encoder and decoder to keep tracks of objects positions
        self.res_connect = nn.ModuleList([ResidualConnection(in_channels=channels[1], out_channels=num_classes)])
        # Last upsampling block
        self.ublocks = nn.ModuleList([UpSampleBlock(in_channels=channels[1])])

        for i in range(1,depth):
            # The number of channels double each time the depth increases
            self.dblocks.append(DownSampleBlock(in_channels=channels[i], out_channels=channels[i + 1]))
            self.res_connect.append(ResidualConnection(in_channels=channels[i + 1], out_channels=channels[i]))
            self.ublocks.append(UpSampleBlock(in_channels=channels[i + 1]))
        self.ublocks = self.ublocks[::-1]
        self.res_connect = self.res_connect[::-1]
        self.output = nn.Conv2d(in_channels=num_classes, out_channels=num_classes,
                               kernel_size=(3, 3), stride=(1, 1), padding=(1,1))

    def forward(self, x):
        depth = len(self.dblocks)

        # Encoder
        # Copy of output of each blocks before downsampling
        xs_down =[]
        for i, down_block in enumerate(self.dblocks):
            x, copy = down_block.forward(x)
            xs_down.append(copy)
        x = self.bottleneck.forward(x)
        xs_down = xs_down[::-1]
        # Decoder
        for i, up_block in enumerate(self.ublocks):
            x_up = up_block.forward(x)
            x = self.res_connect[i](x_up, xs_down[i])
        
        x = self.output(x)
        
        return x

class VisionTransformer(nn.Module):
    '''Vision transformer'''
    def __init__(self, img_size=384, patch_size=16, in_chans=3, n_classes=1000, 
                 embed_dim=768, depth=12, n_heads=4, mlp_ratio=4., qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_channels=in_chans,embed_dim=embed_dim)
        # Learnable parameter taht will represent the first token in the sequence. It has embed_dim elements.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=p)
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        # Transform input images into patch embedding
        n_samples = x.shape[0]
        x = self.patch_embed(x)
        # Replicates the class token over the sample dimension
        cls_token = self.cls_token.expand(n_samples, -1, -1) # (n_samples, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1) # (n_samples, 1 + n_patches, embed_dim)
        x = x + self.pos_embed # (n_samples, 1 + n_patches, embed_dim)
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_token_final = x[:, 0] # just the cls token
        x = self.head(cls_token_final)
        return x

class UNETR(nn.Module):
    '''UNETR model for 2D images'''
    def __init__(self, img_size=224, patch_size=16, in_chans=1, num_classes=10, 
                 embed_dim=768, depth=12, n_heads=4, mlp_ratio=4., qkv_bias=True, p=0., attn_p=0.):
        super(UNETR, self).__init__()
        self.depth = depth
        self.embed_dim = embed_dim
        # Encoder
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_channels=in_chans,embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=p)
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p
                )
                for _ in range(depth)
            ]
        )
        # TODO : Add residual connection
        
        
        # Decoder (uses deconv layers)
        # dim : 
        #   - input (768, W/16, H/16)
        #   - output (num_classes, W, H)
        self.up1 = nn.ConvTranspose2d(in_channels=embed_dim, out_channels=512, kernel_size=2, stride=2) # (512, W/8, H/8)
        self.residual1 = ResidualConnection(in_channels=512, out_channels=512)
        self.up2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2) # (256, W/4, H/4
        self.residual2 = ResidualConnection(in_channels=256, out_channels=256)
        self.up3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2) # (128, W/2, H/2)
        self.residual3 = ResidualConnection(in_channels=128, out_channels=128)
        self.up4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2) # (64, W, H)
        self.residual4 = ResidualConnection(in_channels=64, out_channels=64)
        # Note : these values are hardcoded for now and follow the paper of UNETR

        # Changing encoder output to match decoder input at each step of decoding
        self.z9 = nn.ModuleList(
            [UpSampleBlock(in_channels=768, out_channels=512)]
        )
        self.z6 = nn.ModuleList(
            [UpSampleBlock(in_channels=768, out_channels=512),
             UpSampleBlock(in_channels=512, out_channels=256)]
        )
        self.z3 = nn.ModuleList(
            [UpSampleBlock(in_channels=768, out_channels=512),
             UpSampleBlock(in_channels=512, out_channels=256),
             UpSampleBlock(in_channels=256, out_channels=128)]
        )
        self.double_conv1 = DoubleConvolution(in_channels=in_chans, out_channels=64)
        self.double_conv2 = DoubleConvolution(in_channels=64, out_channels=64)
        self.conv = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):

        inputs = x
        # Encoder

        # Transform input images into patch embedding
        n_samples = x.shape[0]
        x = self.patch_embed(x)
        # Replicates the class token over the sample dimension
        x = x + self.pos_embed # (n_samples, n_patches, embed_dim)
        x = self.pos_drop(x)

        # Keep in memory the output of each block
        encoder_output = torch.zeros((self.depth, n_samples, self.patch_embed.n_patches, self.embed_dim))
        for i, block in enumerate(self.blocks):
            x = block(x)
            encoder_output[i] = x
        print(encoder_output.shape)
        encoder_output = x.reshape(self.depth, n_samples, int(self.patch_embed.n_patches ** 0.5), int(self.patch_embed.n_patches **0.5), -1).permute(0, 1, 4, 2, 3) # (n_samples, embed_dim, W/16, H/16)
        print(encoder_output.shape)
        # Decoder
        x = self.up1(encoder_output[11])
        y = encoder_output[8]
        for deconv in self.z9:
            y = deconv(y)
        x = self.residual1(y, x)
        x = self.up2(x)
        y = encoder_output[5]
        for deconv in self.z6:
            y = deconv(y)
        x = self.residual2(y, x)
        x = self.up3(x)
        y = encoder_output[2]
        for deconv in self.z3:
            y = deconv(y)
        x = self.residual3(y, x)
        x = self.up4(x)
        x = self.residual4(self.double_conv1(inputs), x)
        x = self.double_conv2(x)
        x = self.conv(x)
        
        return x

