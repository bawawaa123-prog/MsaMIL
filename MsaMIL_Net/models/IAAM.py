import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class MultiHeadLowRankAttention(nn.Module):
    
    def __init__(self, d_model=512, num_heads=8, low_rank=32, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.low_rank = int(low_rank)  # r
        self.head_dim = d_model // num_heads
        
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        


        self.W_Q_low = nn.Linear(d_model, self.low_rank * num_heads, bias=False)  # Q_low
        self.W_K_low = nn.Linear(d_model, self.low_rank * num_heads, bias=False)  # K_low
        self.W_V_low = nn.Linear(d_model, self.low_rank * num_heads, bias=False)  # V_low


        self.out_proj = nn.Linear(self.low_rank * num_heads, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        self.scale = math.sqrt(self.low_rank)
        
    def forward(self, x):
        B, N, d_model = x.shape
        

        Q_low = self.W_Q_low(x).view(B, N, self.num_heads, self.low_rank).transpose(1, 2)  # [B, heads, N, r]
        K_low = self.W_K_low(x).view(B, N, self.num_heads, self.low_rank).transpose(1, 2)  # [B, heads, N, r]
        V_low = self.W_V_low(x).view(B, N, self.num_heads, self.low_rank).transpose(1, 2)  # [B, heads, N, r]
        

        A_low = torch.matmul(Q_low, K_low.transpose(-2, -1)) / self.scale  # [B, heads, N, N]
        A_low = F.softmax(A_low, dim=-1)
        A_low = self.dropout(A_low)
        

        output = torch.matmul(A_low, V_low)  # [B, heads, N, r]
        

        output = output.transpose(1, 2).contiguous().view(B, N, self.num_heads * self.low_rank)  # [B, N, heads*r]
        output = self.out_proj(output)  # [B, N, d_model]
        
        return output

class MHELayer(nn.Module):
    
    def __init__(self, d_model=512, num_heads=8, low_rank=32, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        self.self_attn = MultiHeadLowRankAttention(d_model, num_heads, low_rank, dropout)
        

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):

        attn_output = self.self_attn(x)
        x = self.norm1(x + attn_output)
        

        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x

class MHE(nn.Module):
    
    def __init__(self, d_model=512, num_heads=8, num_layers=4, low_rank=32, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout_p = float(dropout)
        

        self.coord_scale_proj = nn.Linear(3, d_model)  # [x,y,scale] → d_model
        

        self.layers = nn.ModuleList([
            MHELayer(d_model, num_heads, low_rank, 2048, dropout)
            for _ in range(num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(d_model)
        


        self.position_embedding = nn.Embedding(50000, d_model)

    @staticmethod
    def _sinusoidal_position_encoding(length: int, d_model: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)  # [L,1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=device, dtype=dtype) * (-(math.log(10000.0) / d_model))
        )  # [d_model/2]
        pe = torch.zeros(length, d_model, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:

            pe[:, 1::2] = torch.cos(position * div_term)[:, : (d_model // 2)]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # [L, d_model]
        
    def forward(self, 
                patch_features: torch.Tensor,
                scale_info: torch.Tensor, 
                spatial_info: torch.Tensor) -> torch.Tensor:
        N, d_model = patch_features.shape
        device = patch_features.device
        dtype = patch_features.dtype
        




        coord_scale_input = torch.cat(
            [spatial_info.to(dtype=dtype), scale_info.unsqueeze(1).to(dtype=dtype)],
            dim=1,
        )  # [N, 3]
        coord_scale_embed = self.coord_scale_proj(coord_scale_input)  # [N, d_model]
        

        pos_ids = torch.arange(N, device=device, dtype=torch.long).clamp_(min=0, max=self.position_embedding.num_embeddings - 1)
        position_embed = self.position_embedding(pos_ids).to(dtype=dtype)  # [N, d_model]
        

        enhanced_features = patch_features + coord_scale_embed.to(dtype=dtype) + position_embed
        if self.dropout_p > 0:
            enhanced_features = F.dropout(enhanced_features, p=self.dropout_p, training=self.training)
        

        x = enhanced_features.unsqueeze(0)  # [1, N, d_model]
        for layer in self.layers:
            x = layer(x)
        x = x.squeeze(0)  # [N, d_model]
        x = self.layer_norm(x)
        return x

class DMQ(nn.Module):
    
    def __init__(self, d_model=512, num_queries=10, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        

        self.learnable_queries = nn.Parameter(torch.randn(num_queries, d_model) * 0.02)
        

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.scale = math.sqrt(self.head_dim)
        
        print(f"✓ DMQ初始化: {num_queries}个查询, {num_heads}头注意力")
    
    def forward(self, mhe_features: torch.Tensor) -> torch.Tensor:
        N, d_model = mhe_features.shape
        dtype = mhe_features.dtype
        


        queries = self.learnable_queries.unsqueeze(0).to(dtype=dtype)  # [1, num_queries, d_model]
        

        # Q_Z = ZW_Q, K = T'''W_K, V = T'''W_V
        Q = self.W_Q(queries)  # [1, num_queries, d_model]
        K = self.W_K(mhe_features.unsqueeze(0))  # [1, N, d_model]  
        V = self.W_V(mhe_features.unsqueeze(0))  # [1, N, d_model]
        

        Q = Q.view(1, self.num_queries, self.num_heads, self.head_dim).transpose(1, 2)  # [1, heads, num_queries, head_dim]
        K = K.view(1, N, self.num_heads, self.head_dim).transpose(1, 2)  # [1, heads, N, head_dim]
        V = V.view(1, N, self.num_heads, self.head_dim).transpose(1, 2)  # [1, heads, N, head_dim]
        
        # Z' = Softmax((Q_Z K^T)/√d)V
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [1, heads, num_queries, N]
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)  # [1, heads, num_queries, head_dim]
        

        attn_output = attn_output.transpose(1, 2).contiguous().view(1, self.num_queries, d_model)  # [1, num_queries, d_model]
        refined_features = self.out_proj(attn_output).squeeze(0)  # [num_queries, d_model]


        refined_features = refined_features + queries.squeeze(0)
        refined_features = F.layer_norm(refined_features, (d_model,))

        return refined_features

class GatedAttentionLayer(nn.Module):
    
    def __init__(self, d_model=512, dropout=0.1):
        super().__init__()
        

        self.gate_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, refined_features: torch.Tensor) -> torch.Tensor:

        refined_features = F.layer_norm(refined_features, (refined_features.shape[-1],))


        gate_weights = self.gate_proj(refined_features)  # [num_queries, 1]


        denom = gate_weights.sum(dim=0, keepdim=True).clamp_min(1e-6)
        gate_weights = gate_weights / denom
        

        bag_features = torch.sum(gate_weights * refined_features, dim=0)  # [d_model,]
        
        return bag_features

class IAAM(nn.Module):
    
    def __init__(self,
                 d_model=512,
                 input_dim=1024,
                 mhe_layers=2,
                 num_heads=8, 
                 low_rank=32,
                 num_queries=10,
                 num_classes=5,
                 dropout=0.1):
        super().__init__()


        self.sort_order = 'xy'


        self.use_input_layernorm = True
        
        self.d_model = d_model
        self.input_dim = input_dim
        self.num_classes = num_classes
        

        self.input_proj = nn.Linear(input_dim, d_model) if input_dim != d_model else nn.Identity()
        

        self.mhe = MHE(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=mhe_layers,
            low_rank=low_rank,
            dropout=dropout
        )
        

        self.dmq = DMQ(
            d_model=d_model,
            num_queries=num_queries,
            num_heads=num_heads,
            dropout=dropout
        )
        

        self.gated_attention = GatedAttentionLayer(d_model, dropout)
        

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True), 
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        print(f"✓ IAAM初始化完成")
        print(f"   输入维度: {input_dim} -> {d_model}")
        print(f"   MHE层数: {mhe_layers}, 低秩: {low_rank}")
        print(f"   DMQ查询数: {num_queries}")
        print(f"   分类类别: {num_classes}")
    
    def forward(self, 
                patch_features: torch.Tensor,
                scale_info: torch.Tensor,
                spatial_info: torch.Tensor) -> tuple:

        try:


            if getattr(self, 'sort_order', 'xy') == 'xy':
                # primary: x, then y, then scale
                idx_np = np.lexsort((scale_info.detach().cpu().numpy(),
                                     spatial_info[:, 1].detach().cpu().numpy(),
                                     spatial_info[:, 0].detach().cpu().numpy()))
            else:
                # primary: y, then x, then scale
                idx_np = np.lexsort((scale_info.detach().cpu().numpy(),
                                     spatial_info[:, 0].detach().cpu().numpy(),
                                     spatial_info[:, 1].detach().cpu().numpy()))
            idx = torch.from_numpy(idx_np).to(patch_features.device)
        except Exception:

            comb_key = spatial_info[:, 0] * 1e6 + spatial_info[:, 1] * 1e3 + scale_info.float()
            idx = torch.argsort(comb_key)

        patch_features = patch_features.index_select(0, idx)
        scale_info = scale_info.index_select(0, idx)
        spatial_info = spatial_info.index_select(0, idx)



        if getattr(self, 'use_input_layernorm', False):
            patch_features = F.layer_norm(patch_features, (patch_features.shape[-1],))


        patch_features = self.input_proj(patch_features)  # [N, input_dim] -> [N, d_model]

        if getattr(self, 'use_input_layernorm', False):
            patch_features = F.layer_norm(patch_features, (patch_features.shape[-1],))
        

        mhe_output = self.mhe(patch_features, scale_info, spatial_info)  # [N, d_model]
        

        refined_features = self.dmq(mhe_output)  # [num_queries, d_model]
        

        bag_features = self.gated_attention(refined_features)  # [d_model,]
        

        logits = self.classifier(bag_features)  # [num_classes,]
        
        return logits, bag_features


if __name__ == "__main__":

    iaam = IAAM(
        d_model=512,
        mhe_layers=2,
  
        num_heads=8,
        low_rank=64,
        num_queries=10,
        num_classes=2
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    iaam.to(device)
    
    print(f"IAAM 模型已加载到设备: {device}")
    print(f"模型参数量: {sum(p.numel() for p in iaam.parameters()):,}")
    

    N = 200
    patch_features = torch.randn(N, 512).to(device)
    scale_info = torch.randint(0, 3, (N,)).to(device)
    spatial_info = torch.randn(N, 2).to(device)
    

    with torch.no_grad():
        logits, bag_features = iaam(patch_features, scale_info, spatial_info)
    print(f"输入 patch 数: {N}")
    print(f"logits 形状: {logits.shape}")
    print(f"bag 特征形状: {bag_features.shape}")
    print(f"预测类别: {torch.argmax(logits).item()}")