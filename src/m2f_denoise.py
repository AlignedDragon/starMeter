"""Mask-denoising training for HuggingFace Mask2Former (see docs/mask_denoising.md).

Port of the idea in ref_paper.pdf ("Optimized segmentation of overlapping cervical
cells based on Mask2Former and denoising") — the mask analog of DN-DETR / MP-Former.
It improves segmentation of OVERLAPPING instances (our overlapping nanostars) by
giving every GT instance a dedicated, stably-matched query path during training.

The idea, in one sentence: besides the 100 learnable queries (the "matching group"),
feed an extra "denoising group" whose per-decoder-layer cross-attention masks come
from NOISED GT masks; train it to reconstruct the clean GT with a one-to-one
(non-Hungarian) match. It is TRAINING-ONLY — `forward` falls straight through to the
stock model whenever `dn_masks` is absent or we are in eval, so inference and the
saved checkpoint are exactly plain Mask2Former (`src/m2f_pipeline.py` is unchanged).

Strategy: subclass `Mask2FormerForUniversalSegmentation` and reimplement only the
decoder loop (reusing the pretrained layer/predictor submodules), because the stock
decoder builds attention masks internally and hardcodes the self-attention mask to
`None` — neither of which we can influence from the top-level forward. All grounded
against transformers 4.41 `modeling_mask2former.py`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import Mask2FormerForUniversalSegmentation
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerForUniversalSegmentationOutput,
)


def _layer_forward(layer, hidden, level_index, position_embeddings, query_pos,
                   encoder_hidden_states, cross_attn_mask, self_attn_mask):
    """Inlined `Mask2FormerMaskedAttentionDecoderLayer.forward_post` that also
    threads a self-attention mask (the stock layer hardcodes it to None). Reuses
    the layer's own submodules/weights so this is numerically identical to stock
    when `self_attn_mask` is the all-allowed mask. Post-norm only."""
    # Masked (cross) attention block
    residual = hidden
    hidden, _ = layer.cross_attn(
        query=layer.with_pos_embed(hidden, query_pos),
        key=layer.with_pos_embed(encoder_hidden_states[level_index], position_embeddings[level_index]),
        value=encoder_hidden_states[level_index],
        attn_mask=cross_attn_mask,
        key_padding_mask=None,
    )
    hidden = nn.functional.dropout(hidden, p=layer.dropout, training=layer.training)
    hidden = residual + hidden
    hidden = layer.cross_attn_layer_norm(hidden)

    # Self attention block (the only change vs. stock: attention_mask is passed)
    residual = hidden
    hidden, _ = layer.self_attn(
        hidden_states=hidden,
        position_embeddings=query_pos,
        attention_mask=self_attn_mask,
        output_attentions=False,
    )
    hidden = nn.functional.dropout(hidden, p=layer.dropout, training=layer.training)
    hidden = residual + hidden
    hidden = layer.self_attn_layer_norm(hidden)

    # Fully connected
    residual = hidden
    hidden = layer.activation_fn(layer.fc1(hidden))
    hidden = nn.functional.dropout(hidden, p=layer.activation_dropout, training=layer.training)
    hidden = layer.fc2(hidden)
    hidden = nn.functional.dropout(hidden, p=layer.dropout, training=layer.training)
    hidden = residual + hidden
    hidden = layer.final_layer_norm(hidden)
    return hidden


class Mask2FormerDN(Mask2FormerForUniversalSegmentation):
    def __init__(self, config, max_dn: int | None = None):
        super().__init__(config)
        hidden = config.hidden_dim
        max_dn = max_dn or config.num_queries
        # Single-class content query (trivial here) + dn positional queries.
        # Both are training-only; stripped before saving an inference checkpoint.
        self.dn_label_enc = nn.Embedding(1, hidden)
        self.dn_query_pos = nn.Embedding(max_dn, hidden)

    # ---- helpers ---------------------------------------------------------

    def _build_self_attn_mask(self, n_dn, n_match, batch_size, device, dtype):
        """Additive (B*heads, Q, Q) self-attention mask, dn queries first.
        dn<->matching blocked both ways; dn block is the identity (each dn query
        sees only itself); matching block fully allowed (== stock behaviour)."""
        heads = self.config.num_attention_heads
        q = n_dn + n_match
        neg = torch.finfo(dtype).min
        mask = torch.zeros((q, q), device=device, dtype=dtype)
        if n_dn:
            dn_block = torch.full((n_dn, n_dn), neg, device=device, dtype=dtype)
            dn_block.fill_diagonal_(0.0)
            mask[:n_dn, :n_dn] = dn_block
            mask[:n_dn, n_dn:] = neg  # dn cannot see matching
            mask[n_dn:, :n_dn] = neg  # matching cannot see dn
        return mask.unsqueeze(0).expand(batch_size * heads, q, q).contiguous()

    def _override_dn_rows(self, attention_mask, dn_noised, size, n_dn):
        """Replace the first `n_dn` rows of the per-layer cross-attention mask
        (B*heads, Q, h*w) with masks derived from the NOISED GT, downsampled to
        the current feature level. Convention matches HF: True == blocked, so a
        location is blocked where the (interpolated) noised mask is background."""
        if not n_dn:
            return attention_mask
        heads = self.config.num_attention_heads
        h, w = size
        interp = nn.functional.interpolate(dn_noised, size=(h, w), mode="bilinear", align_corners=False)
        blocked = interp.flatten(2) < 0.5  # (B, n_dn, h*w)
        blocked = blocked.unsqueeze(1).repeat(1, heads, 1, 1).flatten(0, 1)  # (B*heads, n_dn, h*w)
        attention_mask = attention_mask.clone()
        attention_mask[:, :n_dn, :] = blocked
        return attention_mask

    def _dn_decoder(self, decoder, hidden, query_pos, multi_stage_pos,
                    encoder_hidden_states, mask_features, size_list,
                    n_dn, dn_noised, self_attn_mask):
        """Replicates Mask2FormerMaskedAttentionDecoder.forward (transformers 4.41,
        ~L1843-1899) but (a) injects noised-GT masks into the dn rows of every
        layer's cross-attention mask and (b) threads the self-attention block mask.
        Returns per-layer (intermediate_states, mask_predictions)."""
        levels = decoder.num_feature_levels
        intermediate, mask_predictions = (), ()

        inter = decoder.layernorm(hidden)
        predicted_mask, attention_mask = decoder.mask_predictor(inter, mask_features, size_list[0])
        attention_mask = self._override_dn_rows(attention_mask, dn_noised, size_list[0], n_dn)
        intermediate += (inter,)
        mask_predictions += (predicted_mask,)

        for idx, layer in enumerate(decoder.layers):
            level_index = idx % levels
            # HF fallback: a fully-blocked row would NaN the softmax -> attend all.
            attention_mask[torch.where(attention_mask.sum(-1) == attention_mask.shape[-1])] = False
            hidden = _layer_forward(
                layer, hidden, level_index, multi_stage_pos, query_pos,
                encoder_hidden_states, attention_mask, self_attn_mask,
            )
            inter = decoder.layernorm(hidden)
            predicted_mask, attention_mask = decoder.mask_predictor(
                inter, mask_features, size_list[(idx + 1) % levels]
            )
            attention_mask = self._override_dn_rows(
                attention_mask, dn_noised, size_list[(idx + 1) % levels], n_dn
            )
            intermediate += (inter,)
            mask_predictions += (predicted_mask,)
        return intermediate, mask_predictions

    def _dn_loss(self, dn_mask_logits, dn_class_logits, mask_labels, class_labels, dn_indices):
        """One-to-one (non-Hungarian) denoising loss against the CLEAN GT, reusing
        the stock loss helpers via synthesized identity indices. Final layer + all
        aux layers, weighted like the matching loss. Returns a scalar."""
        crit = self.criterion
        num_masks = crit.get_num_masks(class_labels, device=class_labels[0].device)
        losses = {}
        d = {**crit.loss_masks(dn_mask_logits[-1], mask_labels, dn_indices, num_masks),
             **crit.loss_labels(dn_class_logits[-1], class_labels, dn_indices)}
        losses.update({f"{k}_dn": v for k, v in d.items()})
        for i in range(len(dn_mask_logits) - 1):
            d = {**crit.loss_masks(dn_mask_logits[i], mask_labels, dn_indices, num_masks),
                 **crit.loss_labels(dn_class_logits[i], class_labels, dn_indices)}
            losses.update({f"{k}_dn_{i}": v for k, v in d.items()})
        for key, weight in self.weight_dict.items():
            for loss_key in list(losses):
                if key in loss_key:
                    losses[loss_key] = losses[loss_key] * weight
        return sum(losses.values())

    # ---- forward ---------------------------------------------------------

    def forward(self, pixel_values, mask_labels=None, class_labels=None,
                pixel_mask=None, dn_masks=None, **kwargs):
        # Inference / no targets / dn disabled -> plain Mask2Former.
        if not self.training or dn_masks is None or mask_labels is None:
            return super().forward(
                pixel_values=pixel_values, mask_labels=mask_labels,
                class_labels=class_labels, pixel_mask=pixel_mask, **kwargs,
            )
        if self.config.pre_norm:
            raise NotImplementedError("dn path implements the post-norm decoder only")

        model = self.model
        tm = model.transformer_module
        decoder = tm.decoder
        batch_size = pixel_values.shape[0]

        # 1. backbone + pixel decoder
        ple = model.pixel_level_module(pixel_values=pixel_values, output_hidden_states=False)
        multi_scale_features = ple.decoder_hidden_states     # list of 3, (B,C,h,w)
        mask_features = ple.decoder_last_hidden_state         # (B,C,H/4,W/4)

        # 2. transformer prep (mirror Mask2FormerTransformerModule.forward)
        multi_stage_features, multi_stage_pos, size_list = [], [], []
        for i in range(tm.num_feature_levels):
            size_list.append(multi_scale_features[i].shape[-2:])
            pos = tm.position_embedder(multi_scale_features[i], None).flatten(2)
            feat = (tm.input_projections[i](multi_scale_features[i]).flatten(2)
                    + tm.level_embed.weight[i][None, :, None])
            multi_stage_pos.append(pos.permute(2, 0, 1))
            multi_stage_features.append(feat.permute(2, 0, 1))
        query_embeddings = tm.queries_embedder.weight.unsqueeze(1).repeat(1, batch_size, 1)
        query_features = tm.queries_features.weight.unsqueeze(1).repeat(1, batch_size, 1)
        n_match = query_features.shape[0]

        # 3. denoising group inputs (dn-first ordering everywhere)
        device = pixel_values.device
        counts = [m.shape[0] for m in dn_masks]
        n_dn = max(counts)
        h0, w0 = dn_masks[0].shape[-2:]
        dn_noised = torch.zeros(batch_size, n_dn, h0, w0, device=device, dtype=mask_features.dtype)
        for b, m in enumerate(dn_masks):
            if m.shape[0]:
                dn_noised[b, : m.shape[0]] = m.to(dn_noised.dtype)
        dn_indices = [(torch.arange(c, device=device), torch.arange(c, device=device)) for c in counts]

        dn_content = self.dn_label_enc.weight[0].view(1, 1, -1).expand(n_dn, batch_size, -1)
        dn_pos = self.dn_query_pos.weight[:n_dn].unsqueeze(1).repeat(1, batch_size, 1)
        hidden = torch.cat([dn_content, query_features], dim=0)
        query_pos = torch.cat([dn_pos, query_embeddings], dim=0)

        self_attn_mask = self._build_self_attn_mask(
            n_dn, n_match, batch_size, device, mask_features.dtype
        )

        # 4. dn-augmented decoder
        intermediate, mask_predictions = self._dn_decoder(
            decoder, hidden, query_pos, multi_stage_pos, multi_stage_features,
            mask_features, size_list, n_dn, dn_noised, self_attn_mask,
        )

        # 5. split predictions back into matching / denoising
        class_logits = [self.class_predictor(s.transpose(0, 1)) for s in intermediate]
        match_class = [c[:, n_dn:, :] for c in class_logits]
        match_masks = [m[:, n_dn:, :, :] for m in mask_predictions]
        dn_class = [c[:, :n_dn, :] for c in class_logits]
        dn_masks_logits = [m[:, :n_dn, :, :] for m in mask_predictions]

        # 6. matching loss via the stock (Hungarian) path == baseline numerically
        aux = self.get_auxiliary_logits(match_class, match_masks)
        loss_dict = self.get_loss_dict(
            masks_queries_logits=match_masks[-1], class_queries_logits=match_class[-1],
            mask_labels=mask_labels, class_labels=class_labels, auxiliary_predictions=aux,
        )
        loss = self.get_loss(loss_dict)

        # 7. + denoising loss
        loss = loss + self._dn_loss(dn_masks_logits, dn_class, mask_labels, class_labels, dn_indices)

        return Mask2FormerForUniversalSegmentationOutput(
            loss=loss,
            class_queries_logits=match_class[-1],
            masks_queries_logits=match_masks[-1],
        )

    # ---- checkpoint export ----------------------------------------------

    def export_base(self):
        """Return a plain Mask2FormerForUniversalSegmentation with the dn-only
        params stripped, so `src/m2f_pipeline.py` loads a standard checkpoint."""
        base = Mask2FormerForUniversalSegmentation(self.config)
        sd = {k: v for k, v in self.state_dict().items()
              if not k.startswith(("dn_label_enc", "dn_query_pos"))}
        base.load_state_dict(sd, strict=True)
        return base
