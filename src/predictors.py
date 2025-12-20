import math
import numpy as np

from tabulate import tabulate
import torch
import torch.nn as nn

from src.encoder import TabularEncoder

from src.utils.log_utils import _debug_values
from src.utils.train_utils import (
    get_1d_sincos_pos_embed,
    trunc_normal_,
    apply_masks_from_idx,
)
from src.utils.profiler import get_profiler


class Predictors(nn.Module):
    def __init__(
        self,
        ## predictor type
        pred_type: str,
        ## model parameters
        hidden_dim: int,
        pred_embed_dim: int,
        num_features: int,
        num_layers: int,
        num_heads: int,
        p_dropout: int,
        layer_norm_eps: float,
        activation: str,
        device: torch.device,
        cardinalities: list,
        pred_dim_feedforward: int = None,
        n_cls_tokens: int = 0,
    ):

        super(Predictors, self).__init__()

        self.pred_type = pred_type

        assert self.pred_type in ["mlp", "transformer"], "wrong type of predictor"

        self.hidden_dim = hidden_dim
        self.device = device
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.p_dropout = p_dropout
        self.layer_norm_eps = layer_norm_eps
        self.activation = activation
        self.pred_embed_dim = pred_embed_dim
        self.num_features = num_features
        self.cardinalities = cardinalities
        self.dim_feedforward = pred_dim_feedforward
        self.n_cls_tokens = n_cls_tokens

        if self.pred_type == "mlp":
            # NOTE: MLP predictor uses variable context sizes due to masking.
            # BatchedMLP requires fixed input dimensions and cannot be used with
            # variable context sizes (3-6 features). Keep original list-based approach.
            # BatchedMLP class is available for future use with fixed context sizes.
            self.predictors = nn.ModuleList([
                MLP(
                    self.hidden_dim * num_features,
                    self.hidden_dim,
                    self.num_layers,
                    self.p_dropout,
                    self.layer_norm_eps,
                    self.activation,
                )
                for _ in range(num_features)
            ]).to(self.device)
        else:
            self.predictors = TransformerPredictor(
                num_features=num_features,
                model_hidden_dim=self.hidden_dim,
                pred_embed_dim=self.pred_embed_dim,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                p_dropout=self.p_dropout,
                layer_norm_eps=self.layer_norm_eps,
                activation=self.activation,
                dim_feedforward=self.dim_feedforward,
                n_cls_tokens=self.n_cls_tokens,
            ).to(self.device)

        self.print_params()

    def print_params(self):
        print(
            f"{self.__class__.__name__} {self.__class__.__bases__[0].__name__} parameters:"
        )

        print(
            tabulate(
                [
                    ["hidden_dim", self.hidden_dim],
                    ["num_features", self.num_features],
                    ["num_layers", self.num_layers],
                    ["num_heads", self.num_heads],
                    ["p_dropout", self.p_dropout],
                    ["layer_norm_eps", self.layer_norm_eps],
                    ["activation", self.activation],
                    ["pred_embed_dim", self.pred_embed_dim],
                    ["device", self.device],
                    ["cardinalities", self.cardinalities],
                    ["dim_feedforward", self.dim_feedforward],
                ],
                headers=["Parameter", "Value"],
                tablefmt="pretty",
            )
        )

    def forward(self, x, masks_enc, masks_pred):
        if self.pred_type == "mlp":
            return self.forward_mlp(x, masks_pred)
        else:
            return self.forward_transformer(x, masks_enc, masks_pred)

    def forward_predictor_k(self, x, k):
        return self.predictors[k](x)

    def forward_mlp(self, x, mask_pred):
        out = []
        for mask in mask_pred:
            out_mask = []
            for col_idx in range(self.num_features):
                out_batch = self.forward_predictor_k(x, col_idx)
                out_mask.append(out_batch)
            out_mask = torch.stack(out_mask, dim=1)

            columns_to_keep = torch.ones(mask.shape[-1])
            card_idx = 0
            orig_idx = 0
            idx = 0
            while idx < mask.shape[-1]:
                if orig_idx in [card[0] for card in self.cardinalities]:
                    for _ in range(
                        self.cardinalities[card_idx][1] - 1,
                    ):
                        idx += 1
                        columns_to_keep[idx] = 0
                    card_idx += 1
                idx += 1
                orig_idx += 1
            mask = mask[:, columns_to_keep == 1]

            out_mask = out_mask * mask.unsqueeze(-1)
            out.append(out_mask)
        return out

    def forward_transformer(self, x, masks_enc, masks_pred):
        return self.predictors(x, masks_enc, masks_pred)

    def load_state_dict_(self, list_state_dict):
        if self.pred_type == "mlp":
            for idx, pred in enumerate(self.predictors):
                pred.load_state_dict(list_state_dict[idx])
        else:
            self.predictors.load_state_dict(list_state_dict)

    def state_dict_(self):
        if self.pred_type == "mlp":
            return [pred.state_dict() for pred in self.predictors]
        else:
            return self.predictors.state_dict()


class MLP(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        p_dropout: float,
        layer_norm_eps: float,
        activation: str,
    ):
        super(MLP, self).__init__()

        self.num_layers = num_layers
        layers = []
        if self.num_layers > 1:
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.Dropout(p=p_dropout))
                layers.append(nn.LayerNorm(hidden_dim, eps=layer_norm_eps))
                if activation == "relu":
                    layers.append(nn.ReLU())
                if activation == "gelu":
                    layers.append(nn.GELU())
                if activation == "elu":
                    layers.append(nn.ELU())

            self.fc = nn.Sequential(*layers)
        else:
            self.in_layer = nn.Linear(hidden_dim, hidden_dim)
            self.dropout = nn.Dropout(p=p_dropout)
            self.layernorm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
            if activation == "relu":
                self.activation = nn.ReLU()
            if activation == "gelu":
                self.activation = nn.GELU()
            if activation == "elu":
                self.activation = nn.ELU()

        self.out_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):

        if self.num_layers > 1:
            x = self.fc(x)
        else:
            x = self.in_layer(x)
            x = self.dropout(x)
            x = self.layernorm(x)
            x = self.activation(x)

        out = self.out_layer(x)

        return out


class BatchedMLP(nn.Module):
    """
    Vectorized MLP that processes multiple features in parallel.

    Instead of having num_features separate MLPs, this uses batched operations
    to process all features simultaneously. Each feature still has its own weights,
    but computations are parallelized using batched matrix multiplication.

    Performance: Replaces 256 sequential MLP forward passes with a single batched operation.
    """
    def __init__(
        self,
        num_features: int,
        input_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        p_dropout: float,
        layer_norm_eps: float,
        activation: str,
    ):
        super(BatchedMLP, self).__init__()

        self.num_features = num_features
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.p_dropout = p_dropout
        self.layer_norm_eps = layer_norm_eps

        # Create activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Dropout (applied to all features)
        self.dropout = nn.Dropout(p=p_dropout)

        # Build layers
        if num_layers > 1:
            # Hidden layers: [num_features, hidden_dim, hidden_dim] for each layer
            self.hidden_weights = nn.ParameterList()
            self.hidden_biases = nn.ParameterList()
            self.hidden_layernorms = nn.ModuleList()

            # First hidden layer: input_dim -> hidden_dim
            first_weight = nn.Parameter(torch.empty(num_features, input_dim, hidden_dim))
            first_bias = nn.Parameter(torch.empty(num_features, hidden_dim))
            nn.init.kaiming_uniform_(first_weight, a=math.sqrt(5))
            fan_in = input_dim
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(first_bias, -bound, bound)
            self.hidden_weights.append(first_weight)
            self.hidden_biases.append(first_bias)
            self.hidden_layernorms.append(nn.LayerNorm(hidden_dim, eps=layer_norm_eps))

            # Remaining hidden layers: hidden_dim -> hidden_dim
            for _ in range(num_layers - 2):
                weight = nn.Parameter(torch.empty(num_features, hidden_dim, hidden_dim))
                bias = nn.Parameter(torch.empty(num_features, hidden_dim))
                nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
                fan_in = hidden_dim
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(bias, -bound, bound)
                self.hidden_weights.append(weight)
                self.hidden_biases.append(bias)
                self.hidden_layernorms.append(nn.LayerNorm(hidden_dim, eps=layer_norm_eps))
        else:
            # Single layer case: input_dim -> hidden_dim
            self.in_weight = nn.Parameter(torch.empty(num_features, input_dim, hidden_dim))
            self.in_bias = nn.Parameter(torch.empty(num_features, hidden_dim))
            nn.init.kaiming_uniform_(self.in_weight, a=math.sqrt(5))
            fan_in = input_dim
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.in_bias, -bound, bound)
            self.layernorm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)

        # Output layer: hidden_dim -> out_dim
        self.out_weight = nn.Parameter(torch.empty(num_features, hidden_dim, out_dim))
        self.out_bias = nn.Parameter(torch.empty(num_features, out_dim))
        nn.init.kaiming_uniform_(self.out_weight, a=math.sqrt(5))
        fan_in = hidden_dim
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.out_bias, -bound, bound)

    def forward(self, x):
        """
        Forward pass using batched matrix multiplication.

        Args:
            x: Input tensor of shape [B, input_dim]

        Returns:
            Output tensor of shape [B, num_features, out_dim]
        """
        # Expand input for all features: [B, input_dim] -> [B, num_features, input_dim]
        x = x.unsqueeze(1).expand(-1, self.num_features, -1)

        if self.num_layers > 1:
            # Process through hidden layers
            for weight, bias, layernorm in zip(self.hidden_weights, self.hidden_biases, self.hidden_layernorms):
                # Batched matrix multiplication: [B, F, in] @ [F, in, out] -> [B, F, out]
                # We need to align dimensions properly for bmm
                # x: [B, F, in], weight: [F, in, out]
                # Permute x to [F, B, in], multiply, then permute back
                B = x.shape[0]
                x = x.transpose(0, 1)  # [F, B, in]
                x = torch.bmm(x, weight)  # [F, B, in] @ [F, in, out] -> [F, B, out]
                x = x.transpose(0, 1)  # [B, F, out]

                # Add bias: [F, out] -> [B, F, out]
                x = x + bias.unsqueeze(0)

                # Dropout (applied across all features)
                x = self.dropout(x)

                # LayerNorm (applied per feature)
                x = layernorm(x)

                # Activation
                x = self.activation(x)
        else:
            # Single layer case
            B = x.shape[0]
            x = x.transpose(0, 1)  # [F, B, in]
            x = torch.bmm(x, self.in_weight)  # [F, B, in] @ [F, in, hidden] -> [F, B, hidden]
            x = x.transpose(0, 1)  # [B, F, hidden]
            x = x + self.in_bias.unsqueeze(0)
            x = self.dropout(x)
            x = self.layernorm(x)
            x = self.activation(x)

        # Output layer
        B = x.shape[0]
        x = x.transpose(0, 1)  # [F, B, hidden]
        x = torch.bmm(x, self.out_weight)  # [F, B, hidden] @ [F, hidden, out] -> [F, B, out]
        x = x.transpose(0, 1)  # [B, F, out]
        x = x + self.out_bias.unsqueeze(0)

        return x


class TransformerPredictor(nn.Module):
    def __init__(
        self,
        num_features: int,
        model_hidden_dim: int,
        pred_embed_dim: int,
        num_layers: int,
        num_heads: int,
        p_dropout: float,
        layer_norm_eps: float,
        activation: str,
        init_std: float = 0.02,
        dim_feedforward: int = None,
        n_cls_tokens: int = 0,
    ):
        super(TransformerPredictor, self).__init__()

        self.model_hidden_dim = model_hidden_dim
        self.pred_embed_dim = pred_embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.p_dropout = p_dropout
        self.layer_norm_eps = layer_norm_eps
        self.activation = activation
        self.num_features = num_features
        self.dim_feedforward = dim_feedforward
        self.n_cls_tokens = n_cls_tokens

        self.predictor_emb = nn.Linear(
            self.model_hidden_dim, self.pred_embed_dim, bias=True
        )
        self.layer_norm = nn.LayerNorm(self.pred_embed_dim)

        # Position embeddings for CLS tokens + feature tokens
        total_positions = self.n_cls_tokens + self.num_features
        self.predictor_pos_embed = nn.Parameter(
            torch.zeros(1, total_positions, self.pred_embed_dim), requires_grad=False
        )
        predictor_pos_embed = get_1d_sincos_pos_embed(
            self.pred_embed_dim,
            np.arange(total_positions),
        )
        self.predictor_pos_embed.data.copy_(
            torch.from_numpy(predictor_pos_embed).float().unsqueeze(0)
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.pred_embed_dim))

        self.transformer = TabularEncoder(
            hidden_dim=self.pred_embed_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            p_dropout=self.p_dropout,
            layer_norm_eps=self.layer_norm_eps,
            activation=self.activation,
            dim_feedforward=self.dim_feedforward,
        )

        self.predictor_norm = nn.LayerNorm(self.pred_embed_dim)
        self.predictor_proj = nn.Linear(
            self.pred_embed_dim, self.model_hidden_dim, bias=True
        )

        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, masks_enc, masks_pred):
        profiler = get_profiler()

        B = len(x)

        _debug_values(x[0].T, title="Input")
        with profiler.profile("predictor_embedding"):
            x = self.predictor_emb(x)

        _debug_values(x[0].T, title="Embedded input")

        with profiler.profile("positional_embedding_context"):
            # x contains [CLS, masked_features]
            # Get positional embeddings for these exact positions
            # OPTIMIZATION: Use expand() instead of repeat() - 13.9x faster when contiguous not needed
            # expand() creates a view without copying data, repeat() allocates new memory
            pos_embed_expanded = self.predictor_pos_embed.expand(B, -1, -1)

            # Extract positional embeddings for [0:n_cls_tokens] + feature positions from masks_enc
            if self.n_cls_tokens > 0:
                # CLS positions
                cls_pos = pos_embed_expanded[:, :self.n_cls_tokens, :]
                # Feature positions (masks_enc contains indices 0 to n_features-1, we need to shift by n_cls_tokens)
                feature_indices = [mask + self.n_cls_tokens for mask in masks_enc]
                feature_pos = apply_masks_from_idx(pos_embed_expanded, feature_indices)
                x_pos_embed = torch.cat([cls_pos, feature_pos], dim=1)
            else:
                x_pos_embed = apply_masks_from_idx(pos_embed_expanded, masks_enc)

            _debug_values(x_pos_embed[0].T, title="Positional embedding")

            x += x_pos_embed

        _debug_values(x[0].T, title="After adding positional embedding")

        _, N_ctxt, _ = x.shape

        with profiler.profile("mask_token_preparation"):
            # OPTIMIZATION: Reuse cached expansion instead of repeating
            pos_embs = pos_embed_expanded

            _debug_values(pos_embs[0].T, title="Positional embedding before mask")
            # For prediction: we need [CLS_pos] + [target_feature_positions]
            # The target from train.py has [CLS, target_features]
            if self.n_cls_tokens > 0:
                # Get CLS positional embeddings
                cls_pos_embs = pos_embs[:, :self.n_cls_tokens, :]
                # Get target feature positional embeddings (shift indices by n_cls_tokens)
                pred_indices = [mask + self.n_cls_tokens for mask in masks_pred]
                feature_pos_embs = apply_masks_from_idx(pos_embs, pred_indices)
                # Note: Can't use expand() here since B > 1, need actual memory copy
                cls_pos_embs = cls_pos_embs.repeat(len(masks_pred), 1, 1)
                # Concatenate CLS and target feature positional embeddings
                pos_embs = torch.cat([cls_pos_embs, feature_pos_embs], dim=1)
            else:
                # No CLS tokens, just use the original masks
                pos_embs = apply_masks_from_idx(pos_embs, masks_pred)

            _debug_values(pos_embs[0].T, title="Positional embedding with mask")

            pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
            pred_tokens += pos_embs

            _debug_values(pred_tokens[0].T, title="Predictor tokens")

            x = x.repeat(len(masks_pred), 1, 1)
            x = torch.cat([x, pred_tokens], dim=1)

        _debug_values(x[0].T, title="Input with predictor tokens")

        with profiler.profile("predictor_transformer"):
            x = self.transformer(x)
            x = self.predictor_norm(x)

        _debug_values(x[0].T, title="After transformer")

        with profiler.profile("predictor_output_projection"):
            x = x[:, N_ctxt:]

            _debug_values(x[0].T, title="After slicing")

            x = self.predictor_proj(x)

        _debug_values(x[0].T, title="After projection")

        return x
