# ========== layers/TPECore.py ==========
import torch
import torch.nn as nn
import torch.nn.functional as F
import math  # 添加 math 模块


class FrequencyEnhancer(nn.Module):
    """频域增强模块 - 改进版"""
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        # 频域处理层
        self.freq_proj = nn.Linear(d_model, d_model)
        self.freq_gate = nn.Linear(d_model, d_model)
        self.time_gate = nn.Linear(d_model, d_model)
        
        self.layer_norm = nn.LayerNorm(d_model)
        
        # 缓存DCT矩阵
        self._dct_cache = {}
        self._idct_cache = {}
    
    def _get_dct_matrix(self, n, device):
        if n in self._dct_cache and self._dct_cache[n].device == device:
            return self._dct_cache[n]
        
        k = torch.arange(n, device=device).float().unsqueeze(1)
        i = torch.arange(n, device=device).float().unsqueeze(0)
        cos_term = torch.cos(torch.pi * k * (i + 0.5) / n)
        
        norm = torch.ones(n, device=device)
        # 修复1：使用 torch.tensor 包装
        norm[0] = 1.0 / torch.sqrt(torch.tensor(n, dtype=torch.float, device=device))
        # 修复2：使用 math.sqrt 而不是 torch.sqrt
        norm[1:] = math.sqrt(2.0 / n)
        
        dct_matrix = norm.unsqueeze(1) * cos_term
        self._dct_cache[n] = dct_matrix
        return dct_matrix
    
    def dct(self, x):
        batch_size, seq_len, d_model = x.shape
        dct_matrix = self._get_dct_matrix(seq_len, x.device)
        return torch.matmul(dct_matrix, x)
    
    def idct(self, x_dct):
        batch_size, seq_len, d_model = x_dct.shape
        if seq_len in self._idct_cache and self._idct_cache[seq_len].device == x_dct.device:
            idct_matrix = self._idct_cache[seq_len]
        else:
            dct_matrix = self._get_dct_matrix(seq_len, x_dct.device)
            idct_matrix = dct_matrix.T
            self._idct_cache[seq_len] = idct_matrix
        return torch.matmul(idct_matrix, x_dct)
    
    def forward(self, x):
        # DCT变换
        freq_features = self.dct(x)
        
        # 频域处理
        freq_processed = self.freq_proj(freq_features)
        freq_processed = torch.tanh(freq_processed)
        freq_gate = torch.sigmoid(self.freq_gate(freq_processed))
        freq_processed = freq_processed * freq_gate
        
        # 逆DCT
        reconstructed = self.idct(freq_processed)
        
        # 时域门控
        time_gate = torch.sigmoid(self.time_gate(x))
        enhanced = time_gate * x + (1 - time_gate) * reconstructed
        enhanced = self.layer_norm(enhanced)
        enhanced = self.dropout(enhanced)
        
        return enhanced, freq_features


class TemporalPatternMemory(nn.Module):
    """时序记忆模块 - 改进版"""
    def __init__(self, d_model, num_patterns=12, pattern_length=24, temperature=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_patterns = num_patterns
        self.pattern_length = pattern_length
        self.temperature = temperature
        
        # 模式记忆库
        self.pattern_memory = nn.Parameter(torch.randn(num_patterns, pattern_length, d_model) * 0.1)
        self.pattern_importance = nn.Parameter(torch.ones(num_patterns))
        
        # 查询网络
        self.query_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.cross_attn_proj = nn.Linear(d_model, d_model, bias=False)
        
        # 门控
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, freq_features=None):
        batch_size, seq_len, d_model = x.shape
        
        # 查询构造
        x_permuted = x.transpose(1, 2)
        query = self.query_conv(x_permuted).squeeze(-1)
        
        # 融合频域信息
        if freq_features is not None:
            freq_query = self.cross_attn_proj(freq_features.mean(dim=1))
            query = query + freq_query
        
        # 检索模式
        pattern_keys = self.pattern_memory.mean(dim=1)
        key_proj = self.key_proj(pattern_keys)
        similarity = torch.matmul(query, key_proj.t()) / self.temperature
        importance_weight = F.softmax(self.pattern_importance, dim=0)
        similarity = similarity * importance_weight.unsqueeze(0)
        attention_weights = F.softmax(similarity, dim=-1)
        
        retrieved = torch.einsum('bp,pld->bld', attention_weights, self.pattern_memory)
        
        # 长度调整
        if retrieved.size(1) != seq_len:
            retrieved = F.interpolate(retrieved.transpose(1, 2), size=seq_len, mode='linear').transpose(1, 2)
        
        # 门控融合
        gate_input = torch.cat([x, retrieved], dim=-1)
        gate = self.gate_net(gate_input)
        enhanced_x = gate * x + (1 - gate) * retrieved
        enhanced_x = self.layer_norm(enhanced_x)
        
        return enhanced_x, attention_weights


class CrossModuleAttention(nn.Module):
    """跨模块注意力"""
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, query, key, value):
        batch_size, seq_len, _ = query.shape
        
        Q = self.query_proj(query).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.key_proj(key).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.value_proj(value).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        out = self.out_proj(out)
        out = self.layer_norm(query + out)
        
        return out, attn_weights


class AdaptiveRouter(nn.Module):
    """自适应路由器"""
    def __init__(self, d_model, num_sources=3):
        super().__init__()
        self.num_sources = num_sources
        
        self.router = nn.Sequential(
            nn.Linear(d_model * num_sources, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_sources),
            nn.Softmax(dim=-1)
        )
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, features_list):
        combined = torch.cat(features_list, dim=-1)
        weights = self.router(combined)
        
        fused = torch.zeros_like(features_list[0])
        for i, feat in enumerate(features_list):
            fused = fused + weights[..., i:i+1] * feat
        
        fused = self.layer_norm(fused)
        return fused, weights


class TPECore(nn.Module):
    """TPE核心模块 - 整合所有组件"""
    def __init__(self, d_model, dropout=0.1, num_patterns=12, pattern_length=24):
        super().__init__()
        
        self.freq_enhancer = FrequencyEnhancer(d_model, dropout)
        self.temporal_memory = TemporalPatternMemory(d_model, num_patterns, pattern_length)
        self.mem_to_freq_attn = CrossModuleAttention(d_model)
        self.freq_to_mem_attn = CrossModuleAttention(d_model)
        self.router = AdaptiveRouter(d_model, num_sources=3)
        self.final_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # 1. 并行处理
        x_freq, freq_features = self.freq_enhancer(x)
        x_mem, mem_weights = self.temporal_memory(x, freq_features)
        
        # 2. 跨模块注意力
        mem_enhanced_freq, _ = self.mem_to_freq_attn(x_mem, x_freq, x_freq)
        freq_enhanced_mem, _ = self.freq_to_mem_attn(x_freq, x_mem, x_mem)
        
        # 3. 自适应融合
        x_fused, route_weights = self.router([x, mem_enhanced_freq, freq_enhanced_mem])
        
        # 4. 输出
        x_out = self.final_proj(x_fused)
        x_out = self.dropout(x_out)
        
        return x_out, {
            'mem_weights': mem_weights,
            'route_weights': route_weights,
            'freq_features': freq_features
        }