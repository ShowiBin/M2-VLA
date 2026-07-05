import math
import random
import torch
import torch.nn as nn
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK

def learnable_random_perturbations(seq_len, dim, device, dtype):
    random_perturbations = nn.Parameter(torch.zeros(seq_len, dim, device=device, dtype=dtype))
    nn.init.normal_(random_perturbations, mean=0.0, std=0.02)
    return random_perturbations

class MemoryBank(nn.Module):

    def __init__(self, max_size=10000, key_dim=None, value_dim=None, similarity_threshold=0.5):
        super().__init__()
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.register_buffer('keys', None)
        self.register_buffer('values', None)
        self.register_buffer('size', torch.tensor(0, dtype=torch.long))

    def initialize(self, key_dim, value_dim, device=None):
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.keys = torch.zeros(self.max_size, key_dim, dtype=torch.bfloat16, device=device)
        self.values = torch.zeros(self.max_size, value_dim, dtype=torch.bfloat16, device=device)
        self.size = torch.tensor(0, dtype=torch.long, device=device)

    def add(self, keys, values):
        keys = keys.detach()
        values = values.detach()
        if self.keys is None:
            self.initialize(keys.shape[-1], values.shape[-1], device=keys.device)
        N = keys.shape[0]
        current_size = self.size.item()
        if keys.device != self.keys.device:
            keys = keys.to(self.keys.device)
        if values.device != self.values.device:
            values = values.to(self.values.device)
        if current_size + N <= self.max_size:
            self.keys[current_size:current_size + N] = keys
            self.values[current_size:current_size + N] = values
            self.size = torch.tensor(current_size + N, dtype=torch.long, device=self.keys.device)
        else:
            remaining = self.max_size - current_size
            if remaining > 0:
                self.keys[current_size:] = keys[:remaining]
                self.values[current_size:] = values[:remaining]
            num_to_remove = N - remaining
            self.keys = torch.cat([self.keys[num_to_remove:], keys[remaining:]], dim=0)
            self.values = torch.cat([self.values[num_to_remove:], values[remaining:]], dim=0)
            self.size = torch.tensor(self.max_size, dtype=torch.long, device=self.keys.device)

    def query(self, query_keys, top_k=4):
        query_keys = query_keys.detach()
        if self.size.item() == 0:
            batch_size = query_keys.shape[0]
            device = query_keys.device
            retrieved_values = torch.zeros(batch_size, top_k, self.value_dim if self.value_dim else NUM_ACTIONS_CHUNK * ACTION_DIM, device=device, dtype=torch.bfloat16)
            similarity_scores = torch.zeros(batch_size, top_k, device=device, dtype=torch.bfloat16)
            found_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
            return (retrieved_values, similarity_scores, found_mask)
        stored_keys = self.keys[:self.size.item()]
        stored_values = self.values[:self.size.item()]
        if stored_keys.device != query_keys.device:
            stored_keys = stored_keys.to(query_keys.device)
        if stored_values.device != query_keys.device:
            stored_values = stored_values.to(query_keys.device)
        batch_size = query_keys.shape[0]
        current_size = stored_keys.shape[0]
        similarity = torch.zeros(batch_size, current_size, device=query_keys.device, dtype=query_keys.dtype)
        stored_chunk_size = min(500, current_size)
        for s_start in range(0, current_size, stored_chunk_size):
            s_end = min(s_start + stored_chunk_size, current_size)
            stored_chunk = stored_keys[s_start:s_end]
            query_keys_f32 = query_keys.float()
            stored_chunk_f32 = stored_chunk.float()
            l1_dist = torch.cdist(query_keys_f32, stored_chunk_f32, p=1)
            l1_mean = l1_dist / query_keys.shape[1]
            similarity_chunk = (-10000 * l1_mean).to(query_keys.dtype)
            similarity[:, s_start:s_end] = similarity_chunk
        top_k_actual = min(top_k, self.size.item())
        (top_similarities, top_indices) = torch.topk(similarity, k=top_k_actual, dim=-1)
        if top_k_actual < top_k:
            padding = torch.zeros(query_keys.shape[0], top_k - top_k_actual, device=query_keys.device, dtype=torch.bfloat16)
            top_similarities = torch.cat([top_similarities, padding], dim=-1)
            padding_indices = torch.zeros(query_keys.shape[0], top_k - top_k_actual, device=query_keys.device, dtype=torch.long)
            top_indices = torch.cat([top_indices, padding_indices], dim=-1)
        retrieved_values = stored_values[top_indices]
        found_mask = top_similarities.max(dim=-1)[0] >= -999999
        return (retrieved_values, top_similarities, found_mask)

class L1RegressionActionHead(nn.Module):

    def __init__(self, input_dim=4096, hidden_dim=4096, action_dim=7, num_task_tokens=512, use_pro_version=False, memory_bank_max_size=10000, memory_bank_similarity_threshold=0.5, memory_bank_top_k=4, memory_sample_rate=0.2):
        super().__init__()
        self.num_task_tokens = num_task_tokens
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.memory_bank_top_k = memory_bank_top_k
        self.memory_bank = MemoryBank(max_size=memory_bank_max_size, similarity_threshold=memory_bank_similarity_threshold)
        self.memory_sample_rate = memory_sample_rate
        self.key_layer_indices = [3, 11, 23]
        self.memory_value_proj = nn.Linear(NUM_ACTIONS_CHUNK * ACTION_DIM, hidden_dim * NUM_ACTIONS_CHUNK)
        self.model = MLPResNet(num_blocks=24, input_dim=input_dim * ACTION_DIM, hidden_dim=hidden_dim, output_dim=action_dim, use_pro_version=use_pro_version)

    def load_state_dict(self, state_dict, *args, **kwargs):
        obsolete_keys = ('dino_token_proj.', '.k_dino.', '.v_dino.')
        state_dict = {k: v for (k, v) in state_dict.items() if not any(obsolete_key in k for obsolete_key in obsolete_keys)}
        return super().load_state_dict(state_dict, *args, **kwargs)

    def build_key(self, task_hidden_states):
        layer_features = []
        for layer_idx in self.key_layer_indices:
            if layer_idx < task_hidden_states.shape[1]:
                layer_feat = task_hidden_states[:, layer_idx, :, :]
                layer_feat = layer_feat.mean(dim=1, keepdim=True)
                layer_feat = layer_feat.reshape(layer_feat.shape[0], -1)
                layer_features.append(layer_feat)
        if layer_features:
            keys = torch.cat(layer_features, dim=-1)
        else:
            keys = task_hidden_states[:, -1, :, :].reshape(task_hidden_states.shape[0], -1)
        return keys

    def predict_action(self, actions_hidden_states, proprio=None, proprio_projector=None, phase='Evaluation', ground_truth_actions=None, pixel_values=None):
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        proprio = proprio.reshape(batch_size, -1).to(torch.bfloat16)
        proprio_features = proprio_projector(proprio)
        proprio_features = proprio_features.unsqueeze(dim=1)
        task_hidden_states = actions_hidden_states[:, :, :self.num_task_tokens, :]
        actions_hidden_states = actions_hidden_states[:, :, self.num_task_tokens:, :]
        query_keys = self.build_key(task_hidden_states).detach()
        (retrieved_values, _, _) = self.memory_bank.query(query_keys, top_k=self.memory_bank_top_k)
        h_memory = self.memory_value_proj(retrieved_values.detach())
        if phase == 'Training' and ground_truth_actions is not None and (random.random() < self.memory_sample_rate):
            gt_values = ground_truth_actions.reshape(batch_size, -1)
            self.memory_bank.add(query_keys, gt_values)
        cond_actions_hidden_states = torch.zeros((batch_size, self.action_dim * NUM_ACTIONS_CHUNK, self.hidden_dim), device=device, dtype=actions_hidden_states.dtype).detach()
        rearranged_actions_hidden_states = cond_actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        if phase == 'Training':
            (batch_size, seq_len, dim) = rearranged_actions_hidden_states.shape
            random_perturbations = learnable_random_perturbations(seq_len, dim, device=rearranged_actions_hidden_states.device, dtype=rearranged_actions_hidden_states.dtype)
            rearranged_actions_hidden_states = rearranged_actions_hidden_states + random_perturbations
        action = self.model(rearranged_actions_hidden_states, h_a=actions_hidden_states, p=proprio_features, h_t=task_hidden_states, h_memory=h_memory)
        return action

class MLPResNet(nn.Module):

    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim, use_pro_version=False):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            if use_pro_version:
                self.mlp_resnet_blocks.append(MLPResNetBlock_Pro(dim=hidden_dim))
            else:
                self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h_a=None, h_t=None, p=None, h_memory=None):
        x = self.layer_norm1(x)
        x = self.fc1(x)
        x = self.relu(x)
        for (i, block) in enumerate(self.mlp_resnet_blocks):
            if isinstance(block, MLPResNetBlock_Pro):
                x = block(x, h_t=h_t[:, i + 1, :], h_a=h_a[:, i + 1, :], p=p, h_memory=h_memory)
            else:
                x = block(x, h_t=h_t[:, i + 1, :], h_a=h_a[:, i + 1, :], p=p, h_memory=h_memory)
        x = self.layer_norm2(x)
        x = self.fc2(x)
        return x

def apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    def rotate_half(x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).reshape_as(x)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return (q_rot, k_rot)

class RotaryPositionEmbedding(nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        assert dim % 2 == 0, 'RoPE head_dim must be an even number'
        inv_freq = 1.0 / base ** (torch.arange(0, dim, 2).float() / dim)
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def forward(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return (emb.cos().to(dtype), emb.sin().to(dtype))

class MLPResNetBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU())
        self.num_heads = 8
        self.head_dim = dim // self.num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.gating_factor = nn.Parameter(torch.zeros(1))

    def forward(self, x, h_t=None, h_a=None, p=None, h_memory=None):
        g = self.gating_factor
        ratio_g = nn.Tanh()(g)
        conditions = []
        if h_a is not None:
            conditions.append(h_a)
        if p is not None:
            conditions.append(p)
        h = torch.cat(conditions, dim=1) if conditions else None
        B = x.size(0)
        T = x.size(1)
        C = x.size(2)
        q_1 = self.q_proj(x)
        q_1 = q_1.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k_tokens = self.k_proj(x)
        v_tokens = self.v_proj(x)
        k_tokens = k_tokens.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v_tokens = v_tokens.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores_list = [torch.matmul(q_1, k_tokens.transpose(-2, -1))]
        v_list = [v_tokens]
        if h is not None:
            K_t = h.size(1)
            k_task = self.k_proj(h)
            v_task = self.v_proj(h)
            k_task = k_task.view(B, K_t, self.num_heads, self.head_dim).transpose(1, 2)
            v_task = v_task.view(B, K_t, self.num_heads, self.head_dim).transpose(1, 2)
            attn_scores_list.append(torch.matmul(q_1, k_task.transpose(-2, -1)) * 1)
            v_list.append(v_task)
        if h_t is not None:
            K = h_t.size(1)
            k_adapter = self.k_proj(h_t)
            v_adapter = self.v_proj(h_t)
            k_adapter = k_adapter.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
            v_adapter = v_adapter.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
            attn_scores_list.append(torch.matmul(q_1, k_adapter.transpose(-2, -1)) * ratio_g)
            v_list.append(v_adapter)
        attn_scores = torch.cat(attn_scores_list, dim=-1)
        attn_scores = attn_scores / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        v_combined = torch.cat(v_list, dim=2)
        output = torch.matmul(attn_weights, v_combined)
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o_proj(output)
        if h_memory is not None:
            if h_memory.dim() != 3:
                raise ValueError(f'h_memory must be (B, K_m, T*C), got shape={tuple(h_memory.shape)}')
            mem_dim = h_memory.size(2)
            expected_mem_dim = T * C
            if mem_dim != expected_mem_dim:
                raise ValueError(f'Memory dim mismatch: expected T*C={expected_mem_dim} (T={T}, C={C}) but got mem_dim={mem_dim}. Ensure memory_value_proj outputs hidden_dim*NUM_ACTIONS_CHUNK.')
            q_mem = x.reshape(B, 1, expected_mem_dim)
            scores = torch.matmul(q_mem, h_memory.transpose(-2, -1)) / math.sqrt(mem_dim)
            weights = torch.softmax(scores, dim=-1)
            context_mem_flat = torch.matmul(weights, h_memory)
            context_mem = context_mem_flat.reshape(B, T, C)
            output = output + context_mem
        x = self.ffn(output + x)
        return x

class MLPResNetBlock_Pro(nn.Module):

    def __init__(self, dim, num_heads=8, headwise_attn_output_gate: bool=True, elementwise_attn_output_gate: bool=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.headwise_attn_output_gate = headwise_attn_output_gate
        self.elementwise_attn_output_gate = elementwise_attn_output_gate
        if self.headwise_attn_output_gate:
            self.gate_dim = num_heads
        elif self.elementwise_attn_output_gate:
            self.gate_dim = dim
        else:
            self.gate_dim = 0
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU())
        self.q_proj = nn.Linear(dim, dim)
        self.k_self = nn.Linear(dim, dim)
        self.v_self = nn.Linear(dim, dim)
        self.k_adapter = nn.Linear(dim, dim + self.gate_dim)
        self.v_adapter = nn.Linear(dim, dim)
        self.k_task = nn.Linear(dim, dim)
        self.v_task = nn.Linear(dim, dim)
        self.k_memory = nn.Linear(dim, dim)
        self.v_memory = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.gating_factor = nn.Parameter(torch.zeros(1))
        self.rope = RotaryPositionEmbedding(self.head_dim)
        self.film_gen = nn.Sequential(nn.Linear(dim, dim * 2))

    def apply_film(self, x, gamma, beta):
        return gamma.unsqueeze(1) * x + beta.unsqueeze(1)

    def forward(self, x, h_a=None, h_t=None, p=None, h_memory=None):
        g = self.gating_factor
        ratio_g = torch.tanh(g)
        h_adapter = torch.cat((h_a, p), dim=1) if h_a is not None and p is not None else h_a if h_a is not None else p
        h_task = h_t
        (B, T, C) = x.shape
        K_a = h_adapter.size(1) if h_adapter is not None else 0
        K_t = h_task.size(1) if h_task is not None else 0
        q_1 = self.q_proj(x)
        k_tokens = self.k_self(x)
        v_tokens = self.v_self(x)
        if h_adapter is not None:
            k_adapter_full = self.k_adapter(h_adapter)
            if self.gate_dim > 0:
                (k_adapter, gate_score_adapter) = torch.split(k_adapter_full, [self.dim, self.gate_dim], dim=-1)
            else:
                (k_adapter, gate_score_adapter) = (k_adapter_full, None)
            v_adapter = self.v_adapter(h_adapter)
        else:
            k_adapter = None
            v_adapter = None
            gate_score_adapter = None
        k_task = self.k_task(h_task) if h_task is not None else None
        v_task = self.v_task(h_task) if h_task is not None else None

        def reshape_heads(t, B, L):
            return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        q_1 = reshape_heads(q_1, B, T)
        (k_tokens, v_tokens) = (reshape_heads(k_tokens, B, T), reshape_heads(v_tokens, B, T))
        if k_adapter is not None:
            (k_adapter, v_adapter) = (reshape_heads(k_adapter, B, K_a), reshape_heads(v_adapter, B, K_a))
            (cos_a, sin_a) = self.rope(seq_len=K_a, device=x.device, dtype=x.dtype)
            (_, k_adapter) = apply_rope(k_adapter, k_adapter, cos_a, sin_a)
        if k_task is not None:
            (k_task, v_task) = (reshape_heads(k_task, B, K_t), reshape_heads(v_task, B, K_t))
            (cos_t, sin_t) = self.rope(seq_len=K_t, device=x.device, dtype=x.dtype)
            (_, k_task) = apply_rope(k_task, k_task, cos_t, sin_t)
        (cos_main, sin_main) = self.rope(seq_len=T, device=x.device, dtype=x.dtype)
        (q_1, k_tokens) = apply_rope(q_1, k_tokens, cos_main, sin_main)
        attn_scores_v = torch.matmul(q_1, k_tokens.transpose(-2, -1))
        attn_scores_v = attn_scores_v / math.sqrt(self.head_dim)
        attn_weights_v = torch.softmax(attn_scores_v, dim=-1)
        context_v = torch.matmul(attn_weights_v, v_tokens)
        context_a = None
        if k_adapter is not None and v_adapter is not None:
            attn_scores_a = torch.matmul(q_1, k_adapter.transpose(-2, -1))
            attn_scores_a = attn_scores_a / math.sqrt(self.head_dim)
            attn_weights_a = torch.softmax(attn_scores_a, dim=-1)
            context_a = torch.matmul(attn_weights_a, v_adapter)
            if gate_score_adapter is not None and self.gate_dim > 0:
                if self.headwise_attn_output_gate:
                    gate_score_a = gate_score_adapter.view(B, K_a, self.num_heads, 1).transpose(1, 2)
                    gate_score_a = gate_score_a.mean(dim=2, keepdim=True)
                    context_a = context_a * torch.sigmoid(gate_score_a)
                elif self.elementwise_attn_output_gate:
                    gate_score_a = gate_score_adapter.view(B, K_a, self.num_heads, self.head_dim).transpose(1, 2)
                    gate_score_a = gate_score_a.mean(dim=2, keepdim=True)
                    context_a = context_a * torch.sigmoid(gate_score_a)
        context_t = None
        if k_task is not None and v_task is not None:
            attn_scores_t = torch.matmul(q_1, k_task.transpose(-2, -1)) * ratio_g
            attn_scores_t = attn_scores_t / math.sqrt(self.head_dim)
            attn_weights_t = torch.softmax(attn_scores_t, dim=-1)
            context_t = torch.matmul(attn_weights_t, v_task)
        context_list = [context_v]
        if context_a is not None:
            context_list.append(context_a)
        if context_t is not None:
            context_list.append(context_t)
        output = torch.stack(context_list, dim=0).mean(dim=0)
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o_proj(output)
        if h_memory is not None:
            if h_memory.dim() != 3:
                raise ValueError(f'h_memory must be (B, K_m, T*C), got shape={tuple(h_memory.shape)}')
            mem_dim = h_memory.size(2)
            expected_mem_dim = T * C
            if mem_dim != expected_mem_dim:
                raise ValueError(f'Memory dim mismatch: expected T*C={expected_mem_dim} (T={T}, C={C}) but got mem_dim={mem_dim}.')
            q_mem = x.reshape(B, 1, expected_mem_dim)
            scores = torch.matmul(q_mem, h_memory.transpose(-2, -1)) / math.sqrt(mem_dim)
            weights = torch.softmax(scores, dim=-1)
            context_mem_flat = torch.matmul(weights, h_memory)
            context_mem = context_mem_flat.reshape(B, T, C)
            output = output + context_mem
        x = self.ffn(output + x)
        return x
