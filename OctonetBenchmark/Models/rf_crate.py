import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class ModReLU(nn.Module):
    def __init__(self, c: float = 1e-3):
        '''
        Martin Arjovsky, Amar Shah, and Yoshua Bengio. 2016. Unitary Evolution Recurrent Neural Networks. arXiv:1511.06464 [cs, stat
        '''
        super(ModReLU, self).__init__()
        self.b = nn.Parameter(torch.tensor(0.5))  # Initialize b with a default value, e.g., 0.5
        self.c = c

    def forward(self, z):
        modulus = torch.abs(z)
        scale = F.relu(modulus + self.b) / (modulus + self.c)
        return scale * z

class CReLU(nn.Module):
    def __init__(self):
        super(CReLU, self).__init__()

    def forward(self, z):
        real_relu = F.relu(z.real)
        imag_relu = F.relu(z.imag)
        return torch.complex(real_relu, imag_relu)
    
class ZReLU(nn.Module):
    def __init__(self):
        '''
        Nitzan Guberman. 2016. On Complex Valued Convolutional Neural Networks. arXiv:1602.09046 [cs]
        '''
        super(ZReLU, self).__init__()

    def forward(self, z):
        phase = torch.angle(z)
        mask = (phase >= 0) & (phase <= torch.pi / 2)
        return z * mask

class ComplexCardioid(nn.Module):
    def __init__(self):
        '''
        Patrick Virtue, Stella X. Yu, and Michael Lustig. 2017. Better than Real: Complex-valued Neural Nets for MRI Fingerprinting. In 2017
        IEEE International Conference on Image Processing (ICIP). 3953–3957. https://doi.org/10.1109/ICIP.2017.8297024
        '''
        super(ComplexCardioid, self).__init__()

    def forward(self, z):
        phase = torch.angle(z)
        scale = (1 + torch.cos(phase)) / 2
        return scale * z


class ComplexLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(ComplexLayerNorm, self).__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape, dtype=torch.cfloat))
            self.bias = nn.Parameter(torch.zeros(normalized_shape, dtype=torch.cfloat))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, input):
        mean_real = input.real.mean(dim=-1, keepdim=True)
        mean_imag = input.imag.mean(dim=-1, keepdim=True)
        
        var_real = input.real.var(dim=-1, keepdim=True, unbiased=False)
        var_imag = input.imag.var(dim=-1, keepdim=True, unbiased=False)
        
        mean = torch.complex(mean_real, mean_imag)
        var = torch.complex(var_real, var_imag)
        
        input_normalized = (input - mean) / torch.sqrt(var + self.eps)
        
        if self.elementwise_affine:
            input_normalized = self.weight * input_normalized + self.bias
        
        return input_normalized



class ComplexSoftmax(nn.Module):
    def __init__(self, dim):
        super(ComplexSoftmax, self).__init__()
        self.dim = dim

    def forward(self, z):
        """
        Complex-valued Neural Networks with Non-parametric Activation Functions
        (Eq. 36)
        https://arxiv.org/pdf/1802.08026.pdf
        """
        if torch.is_complex(z):
            magnitude = torch.abs(z)
            result = torch.softmax(magnitude, dim=self.dim)
            phase = torch.angle(z)
            return torch.polar(result, phase)
        else:
            return torch.softmax(z, dim=self.dim)


class ComplexDropout(nn.Module):
    def __init__(self, p: float = 0.1):
        super(ComplexDropout, self).__init__()
        self.p = p

    def forward(self, z):
        if self.training:
            real_dropout = F.dropout(z.real, self.p, self.training)
            imag_dropout = F.dropout(z.imag, self.p, self.training)
            return torch.complex(real_dropout, imag_dropout)
        else:
            return z


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = ComplexLayerNorm(dim)
        self.fn = fn
        
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, step_size=0.1, relu_type='crelu'):
        super().__init__()
        # self.weight = nn.Parameter(torch.Tensor((dim, dim), dtype=torch.cfloat))
        self.weight = nn.Parameter(torch.randn((dim, dim), dtype=torch.cfloat))
        with torch.no_grad():
            init.kaiming_uniform_(self.weight.real)
            init.kaiming_uniform_(self.weight.imag)
        if relu_type == 'crelu':
            self.relu = CReLU()
        elif relu_type == 'zrelu':
            self.relu = ZReLU()
        elif relu_type == 'modrelu':
            self.relu = ModReLU()
        elif relu_type == 'cardioid':
            self.relu = ComplexCardioid()
        else:
            raise ValueError('Unknown relu type')
        self.step_size = step_size
        self.lambd = 0.1

    def forward(self, x):
        x1 = F.linear(x, self.weight, bias=None)
        grad_update = self.step_size * x1 - self.step_size * self.lambd
        output = self.relu(x + grad_update)
        return output

class FeedForward2(nn.Module):
    def __init__(self, dim, step_size=0.1, relu_type='crelu'):
        '''
        This implementation is identical to the one in the original CRATE paper.
        '''
        super().__init__()
        # self.weight = nn.Parameter(torch.Tensor((dim, dim), dtype=torch.cfloat))
        self.weight = nn.Parameter(torch.randn((dim, dim), dtype=torch.cfloat))
        with torch.no_grad():
            init.kaiming_uniform_(self.weight.real)
            init.kaiming_uniform_(self.weight.imag)
        if relu_type == 'crelu':
            self.relu = CReLU()
        elif relu_type == 'zrelu':
            self.relu = ZReLU()
        elif relu_type == 'modrelu':
            self.relu = ModReLU()
        elif relu_type == 'cardioid':
            self.relu = ComplexCardioid()
        else:
            raise ValueError('Unknown relu type')
        self.step_size = step_size
        self.lambd = 0.1

    def forward(self, x):
        x1 = F.linear(x, self.weight, bias=None)
        grad_1 = F.linear(x1, self.weight.t(), bias=None)
        grad_2 = F.linear(x, self.weight.t(), bias=None)
        grad_update = self.step_size * (grad_2 - grad_1) - self.step_size * self.lambd
        output = self.relu(x + grad_update)
        return output

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = ComplexSoftmax(dim=-1)
        self.dropout = ComplexDropout(dropout)

        self.qkv = nn.Linear(dim, inner_dim, bias=False, dtype=torch.cfloat)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, dtype=torch.cfloat),
            ComplexDropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        w = rearrange(self.qkv(x), 'b n (h d) -> b h n d', h=self.heads)

        dots = torch.matmul(w, w.transpose(-1, -2).conj()) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, w)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dropout=0., ista=0.1, relu_type='crelu', feedforward='type1'):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.heads = heads
        self.depth = depth
        self.dim = dim
        
        if feedforward == 'type1':
            for _ in range(depth):
                self.layers.append(
                    nn.ModuleList(
                        [
                            PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                            PreNorm(dim, FeedForward(dim, step_size=ista, relu_type=relu_type))
                        ]
                    )
                )
        elif feedforward == 'type2':
            for _ in range(depth):
                self.layers.append(
                    nn.ModuleList(
                        [
                            PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                            PreNorm(dim, FeedForward2(dim, step_size=ista, relu_type=relu_type))
                        ]
                    )
                )
        else:
            for _ in range(depth):
                self.layers.append(
                    nn.ModuleList(
                        [
                            PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                            PreNorm(dim, FeedForward(dim, dim, dropout=dropout, step_size=ista))
                        ]
                    )
                )
        
        
    def forward(self, x):
        for attn, ff in self.layers:
            grad_x = attn(x) + x
            x = ff(grad_x)
        return x



class ComplexToRealImag(nn.Module):
    def __init__(self, output_dim):
        super(ComplexToRealImag, self).__init__()
        # Define real-valued layers to process the combined real and imaginary parts
        self.fc = nn.Linear(output_dim * 2, output_dim)

    def forward(self, x):
        # Concatenate real and imaginary parts
        x_real = x.real
        x_imag = x.imag
        x_combined = torch.cat((x_real, x_imag), dim=-1)
        # Process through real-valued fully connected layer
        out = self.fc(x_combined)
        return out

class ComplexMagnitude(nn.Module):
    def forward(self, x):
        return x.abs()  # Compute the magnitude of the complex numbers

class ComplexToReal(nn.Module):
    def forward(self, x):
        return x.real  # Extract the real part


class RF_CRATE(nn.Module):
    def __init__(
            self, *, image_size, patch_size, num_classes, dim, depth, heads, pool='cls', channels=3, dim_head=64,
            dropout=0., emb_dropout=0., ista=0.1, feedforward='type1', relu_type='crelu',patch_embedding_method='linear_patch',
            mlp_head_type='crate_version', output_type='magnitude'):
        super().__init__()
        if isinstance(image_size, list):
            image_size = tuple(image_size)
        if isinstance(patch_size, list):
            patch_size = tuple(patch_size)
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        if patch_embedding_method == 'linear_patch':
            patch_dim = channels * patch_height * patch_width
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
                ComplexLayerNorm(patch_dim),
                nn.Linear(patch_dim, dim, dtype=torch.cfloat),
                ComplexLayerNorm(dim),
            )
            num_patches = (image_height // patch_height) * (image_width // patch_width)
        elif patch_embedding_method == 'conv_patch':
            self.to_patch_embedding = nn.Sequential(
                nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size, dtype=torch.cfloat),
                # nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size, dtype=torch.cfloat, padding=padding),
                Rearrange('b c h w -> b (h w) c', c=dim),
                ComplexLayerNorm(dim),
            )
            num_patches = (image_height // patch_height) * (image_width // patch_width)
        elif patch_embedding_method == 'soft_conv_patch': # Soft Convolutional Patch Embedding with the stride is half of the patch size
            self.to_patch_embedding = nn.Sequential(
                nn.Conv2d(channels, dim, kernel_size=patch_size, stride=(patch_size[0] // 2, patch_size[1] // 2), padding=(patch_size[0] // 4, patch_size[1] // 4), dtype=torch.cfloat),
                Rearrange('b c h w -> b (h w) c', c=dim),
                ComplexLayerNorm(dim),
            )
            stride_height = patch_height // 2
            stride_width = patch_width // 2
            num_patches = (image_height // stride_height) * (image_width // stride_width)
        else:
            raise ValueError('Unknown patch embedding method')
        
        self.pos_embedding = nn.Parameter(torch.randn((1, num_patches + 1, dim), dtype=torch.cfloat))
        self.cls_token = nn.Parameter(torch.randn((1, 1, dim), dtype=torch.cfloat))
        self.dropout = ComplexDropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, dropout, ista=ista, relu_type=relu_type, feedforward=feedforward)
        self.pool = pool
        self.to_latent = nn.Identity()
        if mlp_head_type == 'crate_version':
            self.mlp_head = nn.Sequential(
            ComplexLayerNorm(dim),
            nn.Linear(dim, num_classes, dtype=torch.cfloat),
            ComplexMagnitude() if output_type == 'magnitude' else ComplexToRealImag(num_classes) if output_type == 'real_imag' else ComplexToReal(),
            )
        elif mlp_head_type == 'vit_version':
            self.mlp_head = nn.Sequential(
                ComplexLayerNorm(dim),
                nn.Linear(dim, dim * 2, dtype=torch.cfloat),
                CReLU() if relu_type == 'crelu' else ZReLU() if relu_type == 'zrelu' else ModReLU() if relu_type == 'modrelu' else ComplexCardioid(),
                ComplexLayerNorm(dim*2),
                nn.Linear(dim * 2, num_classes, dtype=torch.cfloat),
                ComplexMagnitude() if output_type == 'magnitude' else ComplexToRealImag(num_classes) if output_type == 'real_imag' else ComplexToReal(),
            )
        else:
            raise ValueError('Unknown mlp head type')

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        feature_pre = x
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        feature_last = x
        output = self.mlp_head(x)
        return output, feature_pre
    
    @staticmethod
    def get_selfattention(self, rf_data, layer=5):
        with torch.no_grad():
            x = self.to_patch_embedding(rf_data)
            b, n, _ = x.shape
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)
            x += self.pos_embedding[:, :(n + 1)]
            # x = self.dropout(x)
            for i, (attn, ff) in enumerate(self.transformer.layers):
                if i < layer:
                    grad_x = attn(x) + x
                    x = ff(grad_x)
                else:
                    attn_map = attn(x)
                    return attn_map
    
    @staticmethod
    def get_feature(self, rf_data, layer=11):
        with torch.no_grad():
            x = self.to_patch_embedding(rf_data)
            b, n, _ = x.shape

            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)
            x += self.pos_embedding[:, :(n + 1)]
            x = self.dropout(x)
        
            for i, (attn, ff) in enumerate(self.transformer.layers):
                if i < layer:
                    grad_x = attn(x) + x
                    x = ff(grad_x)
                else:
                    return x

    @staticmethod
    def get_qkv(self, rf_data, layer=11):
        with torch.no_grad():
            x = self.to_patch_embedding(rf_data)
            b, n, _ = x.shape

            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)
            x += self.pos_embedding[:, :(n + 1)]
            x = self.dropout(x)
        
            for i, (attn, ff) in enumerate(self.transformer.layers):
                if i < layer:
                    grad_x = attn(x) + x
                    x = ff(grad_x)
                else:
                    qkv = attn.fn.qkv(x)
                    qkv = qkv[None, :, :, :]  # Adjust dimensions if necessary
                    # Exclude the class token
                    return qkv[:, :, 1:, :]  # Shape: [1, num_heads, num_patches, dim]
                


def rf_crate_tiny(num_classes, image_size, patch_size, in_channels=3,feedforward='type1', relu_type='crelu',patch_embedding_method='linear_patch',
            mlp_head_type='crate_version', output_type='magnitude'):
    return RF_CRATE(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=384,
        depth=12,
        heads=6,
        dropout=0.0,
        emb_dropout=0.0,
        dim_head=384 // 6,
        channels=in_channels,
        feedforward=feedforward,
        relu_type=relu_type,
        patch_embedding_method=patch_embedding_method,
        mlp_head_type=mlp_head_type,
        output_type=output_type
        )
    
def rf_crate_small(num_classes, image_size, patch_size, in_channels=3, feedforward='type1', relu_type='crelu',patch_embedding_method='linear_patch',
            mlp_head_type='crate_version', output_type='magnitude'):
    return RF_CRATE(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=576,
        depth=12,
        heads=12,
        dropout=0.0,
        emb_dropout=0.0,
        dim_head=576 // 12,
        channels=in_channels,
        feedforward=feedforward,
        relu_type=relu_type,
        patch_embedding_method=patch_embedding_method,
        mlp_head_type=mlp_head_type,
        output_type=output_type
        )

def rf_crate_base(num_classes, image_size, patch_size, in_channels=3, feedforward='type1', relu_type='crelu', patch_embedding_method='linear_patch',
            mlp_head_type='crate_version', output_type='magnitude'):
    return RF_CRATE(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=768,
        depth=12,
        heads=12,
        dropout=0.0,
        emb_dropout=0.0,
        dim_head=768 // 12,
        channels=in_channels,
        feedforward=feedforward,
        relu_type=relu_type,
        patch_embedding_method=patch_embedding_method,
        mlp_head_type=mlp_head_type,
        output_type=output_type
        )
    
def rf_crate_large(num_classes, image_size, patch_size, in_channels=3, feedforward='type1', relu_type='crelu',patch_embedding_method='linear_patch',
            mlp_head_type='crate_version', output_type='magnitude'):
    return RF_CRATE(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=1024,
        depth=24,
        heads=16,
        dropout=0.0,
        emb_dropout=0.0,
        dim_head=1024 // 16,
        channels=in_channels,
        feedforward=feedforward,
        relu_type=relu_type,
        patch_embedding_method=patch_embedding_method,
        mlp_head_type=mlp_head_type,
        output_type=output_type
        )
