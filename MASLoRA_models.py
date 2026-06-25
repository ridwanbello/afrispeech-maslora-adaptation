import math
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)

from transformers.utils import (
    logging,
)
from typing import Optional, Tuple, Union
from transformers.models.whisper.configuration_whisper import WhisperConfig
from transformers.models.whisper import WhisperForConditionalGeneration, WhisperModel
from transformers.models.whisper.modeling_whisper import (
    shift_tokens_right,
    WhisperAttention,
    WhisperFlashAttention2,
    WhisperSdpaAttention,
    WhisperPositionalEmbedding,
    WhisperDecoder,
    WhisperEncoder,
    WhisperDecoderLayer,
)

from transformers.modeling_outputs import BaseModelOutput


logger = logging.get_logger(__name__)

WHISPER_ATTENTION_CLASSES = {
    "eager": WhisperAttention,
    "flash_attention_2": WhisperFlashAttention2,
    "sdpa": WhisperSdpaAttention,
}

# CODE IS HEAVILY BASED ON https://github.com/huggingface/transformers

# A single LoRA layer
class LoRA(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        rank: int,
        dropout: float = 0,
        lora_alpha: int = 1,
    ):
        """
        Initializes internal Module state, shared by both nn.Module and ScriptModule.
        
        Args:
            input_dim (int): The input dimension of the LoRA layer.
            output_dim (int): The output dimension of the LoRA layer.
            rank (int): The rank of the LoRA layer.
            dropout (float): The dropout probability to apply to the LoRA layer.
            lora_alpha (int): The scaling factor to apply to the LoRA layer.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank
        self.dropout = dropout
        self.lora_alpha = lora_alpha

        self.lora_scaling = self.lora_alpha / self.rank

        if dropout > 0:
            self.dropout_layer = nn.Dropout(p=self.dropout)
        else:
            self.dropout_layer = nn.Identity()

        self.A = nn.Linear(input_dim, rank, bias=False)
        self.B = nn.Linear(rank, output_dim, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        """
        if isinstance(hidden_states, Tuple):
            hidden_states = hidden_states[0]

        input_proj = self.A(self.dropout_layer(hidden_states))
        outputs = self.B(input_proj)

        return outputs * self.lora_scaling

    def _init_lora(self):
        nn.init.normal_(self.A.weight)
        nn.init.zeros_(self.B.weight)


# Whisper model with out_proj head to generate tokens
class WhisperWithMASLoRAForASR(
    WhisperForConditionalGeneration
):
    def __init__(
        self,
        config: WhisperConfig,
        accent_classes: Tuple[str],
        lora_targets: Tuple[str],
        lora_rank: int,
        lora_dropout: float,
        lora_alpha: int,
        use_maslora_decoder: bool = False,
        no_decoder: bool = False,
        use_top_k: int = 1,
        accent_weight_denominator: int = 6,
    ):

        """
        Whisper model with out_proj head to generate tokens, augmented with MAS-LoRA.

        Args:
            config (`WhisperConfig`): Model configuration.
            accent_classes (`Tuple[str]`): List of accent class names.
            lora_targets (`Tuple[str]`): Targets for LoRA.
            lora_rank (`int`): LoRA rank.
            lora_dropout (`float`): LoRA dropout.
            lora_alpha (`int`): LoRA alpha.
            use_maslora_decoder (`bool`, optional): Whether to use MAS-LoRA on the decoder. Defaults to False.
            no_decoder (`bool`, optional): Whether to fine-tune the decoder. Defaults to False.
            use_top_k (`int`, optional): Number of classes to use for top-k. Defaults to 1 (should be 1 for fine-tuning).
            accent_weight_denominator (`int`, optional): Denominator for calculating accent weights. Defaults to 6.
        """
        super().__init__(config)
        self.model = WhisperWithMASLoRAModel(
            config,
            accent_classes,
            lora_targets,
            lora_rank,
            lora_dropout,
            lora_alpha,
            use_maslora_decoder=use_maslora_decoder,
            no_decoder=no_decoder,
            use_top_k=use_top_k,
            accent_weight_denominator=accent_weight_denominator,
        )

        self.proj_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self._running_inference = False

        self.asr_loss_fct = CrossEntropyLoss()

        # MAS-LoRA related
        self.accent_classes = accent_classes
        self.use_maslora_decoder = use_maslora_decoder
        self.lora_enabled = True

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_non_lora(self):
        """
        Freeze all non-LoRA parameters in the model.

        This is used to fine-tune only the LoRA parameters, while keeping the rest of the model frozen.
        """

        # freeze decoder
        for name, params in self.model.decoder.named_parameters():
            if "lora" not in name:
                params.requires_grad = False

        # freeze encoder
        for name, params in self.model.encoder.named_parameters():
            if "lora" not in name:
                params.requires_grad = False

        # freeze proj_out
        for params in self.proj_out.parameters():
            params.requires_grad = False

    def init_lora(self):
        """
        Initialize LoRA parameters for the model.
        """
        self.model.encoder._init_lora()

        # Decoder may not have lora to init
        if not isinstance(self.model.decoder, WhisperDecoder):
            self.model.decoder._init_lora()

    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        decoder_position_ids: Optional[Tuple[torch.LongTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        
        """
        The WhisperWithMASLoRAForASR forward method.

        Args:

            labels (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`,
                optional):
                Labels for computing the masked language modeling loss. Contains accent labels for MAS-LoRA.

        Returns:

            `Seq2SeqLMOutput` or `Tuple[torch.Tensor]`:
                The output of the model, including the loss, logits, and hidden states.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        accent_class = None
        if labels is not None:
            accent_class = labels[1]

            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels[0],
                    self.config.pad_token_id,
                    self.config.decoder_start_token_id,
                )

        outputs = self.model(
            input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_position_ids=decoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            accent_class=accent_class,
        )

        lm_logits = self.proj_out(outputs[0])

        loss = None
        if labels is not None and lm_logits.shape[1] != 1:
            asr_labels = labels[0]
            # move labels to correct device to enable PP
            asr_labels = asr_labels.to(lm_logits.device)
            loss = self.asr_loss_fct(
                lm_logits.view(-1, self.config.vocab_size), asr_labels.reshape(-1)
            )

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        use_cache=None,
        encoder_outputs=None,
        attention_mask=None,
        decoder_attention_mask=None,
        **kwargs,
    ):
        decoder_position_ids = None
        if decoder_attention_mask is not None:
            decoder_position_ids = (decoder_attention_mask.cumsum(-1) - 1).clamp(min=0)

        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

            if (
                decoder_position_ids is not None
                and decoder_position_ids.shape[1] > decoder_input_ids.shape[1]
            ):
                decoder_position_ids = decoder_position_ids[:, remove_prefix_length:]

        return {
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "use_cache": use_cache,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_position_ids": decoder_position_ids,
            "labels": kwargs[
                "labels"
            ],  # we have to add accent labels to the inputs to be able to run accent-aware MAS-LoRA
        }

    def running_inference(self, b: bool):
        self._running_inference = b
        raise Exception("Is this code ever reached ?")
        # TODO check if this code is ever reacher


class WhisperWithMASLoRAModel(WhisperModel):
    def __init__(
        self,
        config: WhisperConfig,
        accent_classes: Tuple[str],
        lora_targets: Tuple[str],
        lora_rank: int,
        lora_dropout: float,
        lora_alpha: int,
        use_maslora_decoder: bool,
        no_decoder: bool,
        use_top_k: int = 1,
        accent_weight_denominator: int = 6,
    ):
        """
        Initializes a Whisper model with MAS-LoRA and optional decoder configurations.

        Args:
            config (WhisperConfig): The model configuration.
            accent_classes (Tuple[str]): A tuple of accent class names.
            lora_targets (Tuple[str]): A tuple of targets for LoRA.
            lora_rank (int): The rank for LoRA.
            lora_dropout (float): The dropout rate for LoRA.
            lora_alpha (int): The alpha value for LoRA.
            use_maslora_decoder (bool): Whether to use MAS-LoRA for the decoder.
            no_decoder (bool): Whether to use a regular Whisper decoder without LoRA.
            use_top_k (int, optional): Number of top classes to use. Defaults to 1 (should be 1 for fine-tuning).
            accent_weight_denominator (int, optional): Denominator for accent weight calculation. Defaults to 6.

        Raises:
            Exception: If decoder information is unclear, indicating an unexpected condition.
    """

        super().__init__(config)

        self.use_maslora_decoder = use_maslora_decoder
        self.no_decoder = no_decoder

        self.encoder = WhisperEncoderWithMASLoRA(
            config,
            accent_classes=accent_classes,
            lora_targets=lora_targets,
            lora_rank=lora_rank,
            lora_dropout=lora_dropout,
            lora_alpha=lora_alpha,
            use_top_k=use_top_k,
            accent_weight_denominator=accent_weight_denominator,
        )

        if self.use_maslora_decoder:
            print("Adding MAS-LoRA to the Decoder")
            self.decoder = WhisperDecoderWithMASLoRA(
                config,
                accent_classes=accent_classes,
                lora_targets=lora_targets,
                lora_rank=lora_rank,
                lora_dropout=lora_dropout,
                lora_alpha=lora_alpha,
                use_top_k=use_top_k,
                accent_weight_denominator=accent_weight_denominator,
            )
        elif not self.no_decoder:
            print("Adding Classic LoRA to the Decoder")
            self.decoder = WhisperDecoderWithLoRA(
                config,
                lora_targets=lora_targets,
                lora_rank=lora_rank,
                lora_dropout=lora_dropout,
                lora_alpha=lora_alpha,
            )
        elif self.no_decoder:
            print("Using regular Whisper Decoder")
            self.decoder = WhisperDecoder(config)
        else:
            raise Exception("Decoder information unclear. This case should not happen.")
        # Initialize weights and apply final processing
        self.post_init()

    def enable_lora(self, b):
        self.encoder.enable_lora(b)
        if not self.no_decoder:
            self.decoder.enable_lora(b)

    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        decoder_position_ids: Optional[Tuple[torch.LongTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        accent_class: torch.Tensor = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqModelOutput]:
        """
        Performs a forward pass of the WhisperWithMASLoRA model with optional MAS-LoRA decoder layers.

        Args:
            accent_class (torch.Tensor): Tensor containing the accent class information.

        Returns:
            Union[Tuple[torch.Tensor], Seq2SeqModelOutput]: The outputs of the model, either as a tuple or `Seq2SeqModelOutput`.
        """

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        if encoder_outputs is None:
            input_features = self._mask_input_features(
                input_features, attention_mask=attention_mask
            )
            encoder_outputs = self.encoder(
                input_features,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                accent_class=accent_class,
            )

        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
            # raise Exception

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        if isinstance(self.decoder, WhisperDecoderWithMASLoRA):
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_outputs[0],
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values,
                inputs_embeds=decoder_inputs_embeds,
                position_ids=decoder_position_ids,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                accent_class=accent_class,
            )
        else:  # regular decoder inference if only LoRA or regular decoder is used
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_outputs[0],
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values,
                inputs_embeds=decoder_inputs_embeds,
                position_ids=decoder_position_ids,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class WhisperDecoderLayerWithMASLoRA(nn.Module):
    def __init__(
        self,
        config: WhisperConfig,
        accent_classes: Tuple[str] = [],
        lora_targets: Tuple[str] = [],
        lora_rank: int = 16,
        lora_dropout: float = 0,
        lora_alpha: int = 1,
        use_top_k: int = 1,
        accent_weight_denominator: int = 6,
    ):
        """
        Initializes a Whisper decoder layer with MAS-LoRA.

        Args:
            config (WhisperConfig): The model configuration.
            accent_classes (Tuple[str], optional): A tuple of accent class names. Defaults to [].
            lora_targets (Tuple[str], optional): A tuple of targets for LoRA. Defaults to [].
            lora_rank (int, optional): The rank for LoRA. Defaults to 16.
            lora_dropout (float, optional): The dropout rate for LoRA. Defaults to 0.
            lora_alpha (int, optional): The alpha value for LoRA. Defaults to 1.
            use_top_k (int, optional): Number of top classes to use. Defaults to 1 (should be 1 for fine-tuning).
            accent_weight_denominator (int, optional): Denominator for accent weight calculation. Defaults to 6.
        """
        super().__init__()
        self.embed_dim = config.d_model

        self.use_top_k = use_top_k
        self.accent_classes = accent_classes
        self.accent_weight_denominator = accent_weight_denominator

        self.self_attn = WhisperAttentionWithMASLoRA(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=True,
            config=config,
            accent_classes=accent_classes,
            targets=lora_targets,
            lora_rank=lora_rank,
            lora_dropout=lora_dropout,
            lora_alpha=lora_alpha,
            use_top_k=use_top_k,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = WhisperAttentionWithMASLoRA(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
            accent_classes=accent_classes,
            targets=lora_targets,
            lora_rank=lora_rank,
            lora_dropout=lora_dropout,
            lora_alpha=lora_alpha,
            use_top_k=use_top_k,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def enable_lora(self, b):
        self.self_attn.enable_lora(b)
        self.encoder_attn.enable_lora(b)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        accent_class: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            accent_class (torch.Tensor): Tensor containing the accent class information.

        Returns:
            `torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`: The output of the decoder layer.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        accent_weights = torch.zeros(
            (hidden_states.shape[0], len(self.accent_classes)),
            device=hidden_states.device,
        )
        for idx, a in enumerate(accent_class):
            accent_weights[idx, a] = 1 / self.accent_weight_denominator
        if self.accent_weight_denominator != 1:
            accent_weights[accent_weights == 0] = (
                1 - (1 / self.accent_weight_denominator)
            ) / (len(self.accent_classes) - 1)
        else:
            accent_weights[accent_weights == 0] = (
                1 - (1 / self.accent_weight_denominator)
            ) / (len(self.accent_classes) - 1)

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            accent_weights=accent_weights,
        )
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = (
                past_key_value[-2:] if past_key_value is not None else None
            )
            hidden_states, cross_attn_weights, cross_attn_present_key_value = (
                self.encoder_attn(
                    hidden_states=hidden_states,
                    key_value_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=cross_attn_past_key_value,
                    output_attentions=output_attentions,
                    accent_weights=accent_weights,
                )
            )
            hidden_states = nn.functional.dropout(
                hidden_states, p=self.dropout, training=self.training
            )
            hidden_states = residual + hidden_states

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.activation_dropout, training=self.training
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

    def _init_lora(self):
        for t in self.self_attn.lora_adapters:
            for l in self.self_attn.lora_adapters[t]:
                self.self_attn.lora_adapters[t][l]._init_lora()

        for t in self.encoder_attn.lora_adapters:
            for l in self.encoder_attn.lora_adapters[t]:
                self.encoder_attn.lora_adapters[t][l]._init_lora()


class WhisperDecoderWithMASLoRA(WhisperDecoder):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`WhisperDecoderLayerWithMASLoRA`]
    """

    main_input_name = "input_ids"

    def __init__(
        self,
        config: WhisperConfig,
        accent_classes,
        lora_targets,
        lora_rank,
        lora_dropout,
        lora_alpha,
        use_top_k,
        accent_weight_denominator,
    ):
        """
        Initializes a Whisper decoder with MAS-LoRA.

        Args:
            config (`WhisperConfig`): Model configuration.
            accent_classes (`Tuple[str]`): List of accent class names.
            lora_targets (`Tuple[str]`): Targets for LoRA.
            lora_rank (`int`): LoRA rank.
            lora_dropout (`float`): LoRA dropout.
            lora_alpha (`int`): LoRA alpha.
            use_top_k (`int`): Number of classes to use for top-k.
            accent_weight_denominator (`int`): Denominator for calculating accent weights.
        """

        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_target_positions
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.d_model, self.padding_idx
        )
        self.embed_positions = WhisperPositionalEmbedding(
            self.max_target_positions, config.d_model
        )

        self.layers = nn.ModuleList(
            [
                WhisperDecoderLayerWithMASLoRA(
                    config,
                    accent_classes,
                    lora_targets,
                    lora_rank,
                    lora_dropout,
                    lora_alpha,
                    use_top_k=use_top_k,
                    accent_weight_denominator=accent_weight_denominator,
                )
                for _ in range(config.decoder_layers)
            ]
        )
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self._use_sdpa = config._attn_implementation == "sdpa"

        self.layer_norm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def enable_lora(self, b):
        for l in self.layers:
            l.enable_lora(b)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        accent_class=None,
    ):
        """
        Perform a forward pass through the WhisperDecoderWithMASLoRA.

        Args:
            accent_class (torch.Tensor, optional): Tensor containing the accent class information.

        Returns:
            BaseModelOutputWithPastAndCrossAttentions or Tuple: The output of the decoder with optional past key values,
            hidden states, attentions, and cross-attentions.
        """

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = (
                attention_mask
                if (attention_mask is not None and 0 in attention_mask)
                else None
            )
        elif self._use_sdpa and head_mask is None and not output_attentions:
            # output_attentions=True & head_mask can not be supported when using SDPA.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask, input_shape, inputs_embeds, past_key_values_length
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, input_shape, inputs_embeds, past_key_values_length
            )

        # embed positions
        if input_ids is not None:
            positions = self.embed_positions(
                input_ids,
                past_key_values_length=past_key_values_length,
                position_ids=position_ids,
            )
        else:
            positions = self.embed_positions(
                inputs_embeds,
                past_key_values_length=past_key_values_length,
                position_ids=position_ids,
            )

        hidden_states = inputs_embeds + positions
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache = True` is incompatible with gradient checkpointing. Setting `use_cache = False`..."
                )
                use_cache = False
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = (
            () if (output_attentions and encoder_hidden_states is not None) else None
        )
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip(
            [head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]
        ):
            if attn_mask is not None:
                assert attn_mask.size()[0] == (len(self.layers)), (
                    f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    None,  # encoder attention mask
                    head_mask[idx] if head_mask is not None else None,
                    (
                        cross_attn_head_mask[idx]
                        if cross_attn_head_mask is not None
                        else None
                    ),
                    None,  # past_key_value
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx]
                        if cross_attn_head_mask is not None
                        else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    accent_class=accent_class,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        hidden_states = self.layer_norm(hidden_states)
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )

    def _init_lora(self):
        for l in self.layers:
            l._init_lora()


class WhisperEncoderLayerWithMASLoRA(nn.Module):
    def __init__(
        self,
        config: WhisperConfig,
        accent_classes: Tuple[str] = [],
        lora_targets: Tuple[str] = [],
        lora_rank: int = 16,
        lora_dropout: float = 0,
        lora_alpha: int = 1,
        use_top_k=1,
        accent_weight_denominator=6,
    ):
        """
        Initializes a Whisper encoder layer with MAS-LoRA.

        Args:
            config (WhisperConfig): Model configuration.
            accent_classes (Tuple[str], optional): List of accent class names. Defaults to [].
            lora_targets (Tuple[str], optional): Targets for LoRA. Defaults to [].
            lora_rank (int, optional): LoRA rank. Defaults to 16.
            lora_dropout (float, optional): LoRA dropout. Defaults to 0.
            lora_alpha (int, optional): LoRA alpha. Defaults to 1.
            use_top_k (int, optional): Number of classes to use for top-k. Defaults to 1 (should be 1 for fine-tuning).
            accent_weight_denominator (int, optional): Denominator for calculating accent weights. Defaults to 6.
        """


        super().__init__()
        self.embed_dim = config.d_model
        self.accent_classes = accent_classes
        self.accent_weight_denominator = accent_weight_denominator

        self.self_attn = WhisperAttentionWithMASLoRA(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
            targets=lora_targets,
            lora_rank=lora_rank,
            lora_dropout=lora_dropout,
            lora_alpha=lora_alpha,
            accent_classes=accent_classes,
            use_top_k=use_top_k,
        )

        self.use_top_k = use_top_k
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def enable_lora(self, b):
        self.self_attn.enable_lora(b)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
        accent_class: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Performs a forward pass through the WhisperDecoderLayerWithMASLoRA, computing
        the output hidden states and optionally attention weights.

        Args:
            accent_class (torch.Tensor, optional): Tensor containing the accent class information. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape `(batch_size, sequence_length, hidden_size)`, and optionally
            a tensor of attention weights if `output_attentions` is True.
        """

        if isinstance(hidden_states, Tuple):
            hidden_states = hidden_states[0]

        residual = hidden_states

        # Computing MAS-LoRA weights
        accent_weights = torch.zeros(
            (hidden_states.shape[0], len(self.accent_classes)),
            device=hidden_states.device,
        )
        for idx, a in enumerate(accent_class):
            accent_weights[idx, a] = 1 / self.accent_weight_denominator
        if self.accent_weight_denominator != 1:
            accent_weights[accent_weights == 0] = (
                1 - (1 / self.accent_weight_denominator)
            ) / (len(self.accent_classes) - 1)
        else:
            accent_weights[accent_weights == 0] = (
                1 - (1 / self.accent_weight_denominator)
            ) / (len(self.accent_classes) - 1)

        if accent_class == None and self.use_top_k == 1:
            accent_class = torch.argmax(accent_weights, dim=1)

        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            accent_weights=accent_weights,
        )
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.activation_dropout, training=self.training
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

    def _init_lora(self):
        for t in self.self_attn.lora_adapters:
            for l in self.self_attn.lora_adapters[t]:
                self.self_attn.lora_adapters[t][l]._init_lora()


class WhisperEncoderWithMASLoRA(WhisperEncoder):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`WhisperEncoderLayerWithMASLoRA`].
    """

    def __init__(
        self,
        config: WhisperConfig,
        accent_classes,
        lora_targets,
        lora_rank,
        lora_dropout,
        lora_alpha,
        use_top_k=1,
        accent_weight_denominator=6,
    ):
        """
        Initializes a Whisper encoder with MAS-LoRA.

        Args:
            config (`WhisperConfig`): Model configuration.
            accent_classes (`Tuple[str]`): List of accent class names.
            lora_targets (`Tuple[str]`): Targets for LoRA.
            lora_rank (`int`): LoRA rank.
            lora_dropout (`float`): LoRA dropout.
            lora_alpha (`int`): LoRA alpha.
            use_top_k (`int`, optional): Number of top classes to use. Defaults to 1 (should be 1 for fine-tuning).
            accent_weight_denominator (`int`, optional): Denominator for calculating accent weights. Defaults to 6.
        """
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)

        self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)
        self.embed_positions.requires_grad_(False)

        self.layers = nn.ModuleList(
            [
                WhisperEncoderLayerWithMASLoRA(
                    config,
                    accent_classes=accent_classes,
                    lora_targets=lora_targets,
                    lora_rank=lora_rank,
                    lora_dropout=lora_dropout,
                    lora_alpha=lora_alpha,
                    use_top_k=use_top_k,
                    accent_weight_denominator=accent_weight_denominator,
                )
                for _ in range(config.encoder_layers)
            ]
        )

        self.layer_norm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def enable_lora(self, b):
        for l in self.layers:
            l.enable_lora(b)

    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        accent_class=None,
        labels=None,
    ):
        """
        Args:
            accent_class (torch.Tensor, optional): Tensor containing the accent class information. Defaults to `None`.
            labels (torch.Tensor, optional): Tensor containing the labels. Defaults to `None`.

        Returns:
            `BaseModelOutput`: The output of the model, including the last hidden state, hidden states, and attention weights.
        """
        if accent_class == None:  # recover accent class from labels if needed
            accent_class = labels[1]

        expected_seq_length = (
            self.config.max_source_positions
            * self.conv1.stride[0]
            * self.conv2.stride[0]
        )
        if input_features.shape[-1] != expected_seq_length:
            raise ValueError(
                f"Whisper expects the mel input features to be of length {expected_seq_length}, but found {input_features.shape[-1]}. Make sure to pad the input mel features to {expected_seq_length}."
            )

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = self.embed_positions.weight

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        None,
                        (head_mask[idx] if head_mask is not None else None),
                        output_attentions,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        None,
                        layer_head_mask=(
                            head_mask[idx] if head_mask is not None else None
                        ),
                        output_attentions=output_attentions,
                        accent_class=accent_class,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if isinstance(hidden_states, Tuple):
            hidden_states = hidden_states[0]
        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, encoder_states, all_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )

    def _init_lora(self):
        for l in self.layers:
            l._init_lora()


class WhisperDecoderLayerWithLoRA(WhisperDecoderLayer):
    def __init__(
        self,
        config: WhisperConfig,
        lora_targets: Tuple[str] = [],
        lora_rank: int = 16,
        lora_dropout: float = 0,
        lora_alpha: int = 1,
    ):
        """
        Initializes a Whisper decoder layer with LoRA enhancements.

        Args:
            config (WhisperConfig): The model configuration.
            lora_targets (Tuple[str], optional): A tuple of targets for LoRA. Defaults to an empty tuple.
            lora_rank (int, optional): The rank for LoRA. Defaults to 16.
            lora_dropout (float, optional): The dropout rate for LoRA. Defaults to 0.
            lora_alpha (int, optional): The alpha value for LoRA. Defaults to 1.
        """

        super().__init__(config)
        self.embed_dim = config.d_model

        self.self_attn = WhisperAttentionWithLoRA(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=True,
            config=config,
            targets=lora_targets,
            lora_rank=lora_rank,
            lora_dropout=lora_dropout,
            lora_alpha=lora_alpha,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = WhisperAttentionWithLoRA(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
            targets=lora_targets,
            lora_rank=lora_rank,
            lora_dropout=lora_dropout,
            lora_alpha=lora_alpha,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def enable_lora(self, b):
        self.self_attn.enable_lora(b)
        self.encoder_attn.enable_lora(b)

    def _init_lora(self):
        for t in self.self_attn.lora_adapters:
            self.self_attn.lora_adapters[t]._init_lora()

        for t in self.encoder_attn.lora_adapters:
            self.encoder_attn.lora_adapters[t]._init_lora()


class WhisperDecoderWithLoRA(WhisperDecoder):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`WhisperDecoderLayer`]
    """

    main_input_name = "input_ids"

    def __init__(
        self, config: WhisperConfig, lora_targets, lora_rank, lora_dropout, lora_alpha
    ):
        """
        Initializes a Whisper decoder with LoRA enhancements.

        Args:
            config (WhisperConfig): The model configuration.
            lora_targets (Tuple[str]): A tuple of targets for LoRA.
            lora_rank (int): The rank for LoRA.
            lora_dropout (float): The dropout rate for LoRA.
            lora_alpha (int): The alpha value for LoRA.
        """
        super().__init__(config)
        self.layers = nn.ModuleList(
            [
                WhisperDecoderLayerWithLoRA(
                    config,
                    lora_targets=lora_targets,
                    lora_rank=lora_rank,
                    lora_dropout=lora_dropout,
                    lora_alpha=lora_alpha,
                )
                for _ in range(config.decoder_layers)
            ]
        )

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,    
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def enable_lora(self, b):
        for l in self.layers:
            l.enable_lora(b)

    def _init_lora(self):
        for l in self.layers:
            l._init_lora()


class WhisperAttentionWithLoRA(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper including LoRA
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[WhisperConfig] = None,
        targets: Tuple[str] = [],
        lora_rank: int = 16,
        lora_dropout: float = 0,
        lora_alpha: int = 1,
    ):
        """
        Initializes a Whisper attention layer with LoRA enhancements.

        Args:
            embed_dim (int): The input embedding dimensionality.
            num_heads (int): The number of attention heads.
            dropout (float, optional): The dropout rate. Defaults to 0.
            is_decoder (bool, optional): Whether this is a decoder layer. Defaults to False.
            bias (bool, optional): Whether to use bias in the projection layers. Defaults to True.
            is_causal (bool, optional): Whether this is a causal attention layer. Defaults to False.
            config (WhisperConfig, optional): The model configuration. Defaults to None.
            targets (Tuple[str], optional): A tuple of targets for LoRA. Defaults to [].
            lora_rank (int, optional): The rank for LoRA. Defaults to 16.
            lora_dropout (float, optional): The dropout rate for LoRA. Defaults to 0.
            lora_alpha (int, optional): The alpha value for LoRA. Defaults to 1.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config
        self.lora_enabled = True

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # LoRA
        self.targets = targets
        self.lora_adapters = torch.nn.ModuleDict({})
        for t in self.targets:
            self.lora_adapters[t] = LoRA(
                input_dim=embed_dim,
                output_dim=embed_dim,
                rank=lora_rank,
                dropout=lora_dropout,
                lora_alpha=lora_alpha,
            )

    # Copied from transformers.models.bart.modeling_bart.BartAttention._shape with BART->whisper
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def enable_lora(self, b):
        self.lora_enabled = b

    # Copied from transformers.models.bart.modeling_bart.BartAttention.forward with BART->whisper
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        if "q_proj" in self.lora_adapters and self.lora_enabled:
            query_states = (
                self.q_proj(hidden_states) + self.lora_adapters["q_proj"](hidden_states)
            ) * self.scaling
        else:
            query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            if "k_proj" in self.lora_adapters and self.lora_enabled:
                key_states = self._shape(
                    self.k_proj(key_value_states)
                    + self.lora_adapters["k_proj"](key_value_states),
                    -1,
                    bsz,
                )
            else:
                key_states = self._shape(self.k_proj(key_value_states), -1, bsz)

            if "v_proj" in self.lora_adapters and self.lora_enabled:
                value_states = self._shape(
                    self.v_proj(key_value_states)
                    + self.lora_adapters["v_proj"](key_value_states),
                    -1,
                    bsz,
                )
            else:
                value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            if "k_proj" in self.lora_adapters and self.lora_enabled:
                key_states = self._shape(
                    self.k_proj(hidden_states)
                    + self.lora_adapters["k_proj"](hidden_states),
                    -1,
                    bsz,
                )
            else:
                key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            if "v_proj" in self.lora_adapters and self.lora_enabled:
                value_states = self._shape(
                    self.v_proj(hidden_states)
                    + self.lora_adapters["v_proj"](hidden_states),
                    -1,
                    bsz,
                )
            else:
                value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            if "k_proj" in self.lora_adapters and self.lora_enabled:
                key_states = self._shape(
                    self.k_proj(hidden_states)
                    + self.lora_adapters["k_proj"](hidden_states),
                    -1,
                    bsz,
                )
            else:
                key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            if "v_proj" in self.lora_adapters and self.lora_enabled:
                value_states = self._shape(
                    self.v_proj(hidden_states)
                    + self.lora_adapters["v_proj"](hidden_states),
                    -1,
                    bsz,
                )
            else:
                value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = (
                attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                + attention_mask
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights_reshaped.view(
                bsz * self.num_heads, tgt_len, src_len
            )
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        if "out_proj" in self.targets and self.lora_enabled:
            attn_output = self.out_proj(attn_output) + self.lora_adapters["out_proj"](
                attn_output
            )
        else:
            attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class WhisperAttentionWithMASLoRA(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper including MAS-LoRA
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[WhisperConfig] = None,
        accent_classes: Tuple[str] = [],
        targets: Tuple[str] = [],
        lora_rank: int = 16,
        lora_dropout: float = 0,
        lora_alpha: int = 1,
        use_top_k: int = 1,
    ):
        """
        Initializes a Whisper attention layer with MAS-LoRA enhancements.

        Args:
            embed_dim (int): The input embedding dimensionality.
            num_heads (int): The number of attention heads.
            dropout (float, optional): The dropout rate. Defaults to 0.
            is_decoder (bool, optional): Whether this is a decoder layer. Defaults to False.
            bias (bool, optional): Whether to use bias in the projection layers. Defaults to True.
            is_causal (bool, optional): Whether this is a causal attention layer. Defaults to False.
            config (WhisperConfig, optional): The model configuration. Defaults to None.
            accent_classes (Tuple[str], optional): A tuple of accent class names. Defaults to an empty tuple.
            targets (Tuple[str], optional): A tuple of targets for LoRA. Defaults to an empty tuple.
            lora_rank (int, optional): The rank for LoRA. Defaults to 16.
            lora_dropout (float, optional): The dropout rate for LoRA. Defaults to 0.
            lora_alpha (int, optional): The alpha value for LoRA. Defaults to 1.
            use_top_k (int, optional): Number of top classes to use. Defaults to 1 (should be 1 for fine-tuning).
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config
        self.accent_classes = accent_classes
        self.lora_enabled = True

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # MAS-LoRA
        self.use_top_k = use_top_k
        self.targets = targets
        self.lora_adapters = torch.nn.ModuleDict({})
        for a in self.accent_classes:
            self.lora_adapters[a] = torch.nn.ModuleDict({})
            for t in self.targets:
                self.lora_adapters[a][t] = LoRA(
                    input_dim=embed_dim,
                    output_dim=embed_dim,
                    rank=lora_rank,
                    dropout=lora_dropout,
                    lora_alpha=lora_alpha,
                )

    def enable_lora(self, b):
        self.lora_enabled = b

    # Copied from transformers.models.bart.modeling_bart.BartAttention._shape with BART->whisper
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        accent_weights: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        if accent_weights == None:
            raise ValueError(
                "accent_log_probs must be != None when using MAS-LoRA adapters"
            )

        if accent_weights.sum().item() != bsz:
            accent_weights = torch.softmax(accent_weights, dim=1)

        # using top_k classes (1 during fine-tuning)
        top_k_values, top_k_indices = torch.topk(accent_weights, self.use_top_k, dim=1)

        # we normalize the accent_probs so that they sum to 1
        # useful only when we use less than 6 accents for whatever
        # reasons
        top_k_values /= top_k_values.sum(dim=-1, keepdim=True)

        query_states_final = torch.zeros(
            hidden_states.shape, device=hidden_states.device
        )

        if is_cross_attention:
            key_states_final = torch.zeros(
                (bsz, self.num_heads, key_value_states.shape[1], self.head_dim),
                device=hidden_states.device,
            )
            value_states_final = torch.zeros(
                (bsz, self.num_heads, key_value_states.shape[1], self.head_dim),
                device=hidden_states.device,
            )
        else:
            key_states_final = torch.zeros(
                (bsz, self.num_heads, tgt_len, self.head_dim),
                device=hidden_states.device,
            )
            value_states_final = torch.zeros(
                (bsz, self.num_heads, tgt_len, self.head_dim),
                device=hidden_states.device,
            )

        for i in range(bsz):
            hidden_states_temp = hidden_states[i].unsqueeze(0)

            if key_value_states is not None:
                key_value_states_temp = key_value_states[i].unsqueeze(0)

            # get query proj
            if "q_proj" in self.targets and self.lora_enabled:
                query_states = self.q_proj(hidden_states_temp)
                for k in range(self.use_top_k):
                    query_states += (
                        self.lora_adapters[self.accent_classes[top_k_indices[i][k]]][
                            "q_proj"
                        ](hidden_states_temp)
                        * top_k_values[i][k]
                    )
                query_states *= self.scaling
            else:
                query_states = self.q_proj(hidden_states_temp) * self.scaling

            # get key, value proj
            # `past_key_value[0].shape[2] == key_value_states.shape[1]`
            # is checking that the `sequence_length` of the `past_key_value` is the same as
            # the provided `key_value_states` to support prefix tuning
            if (
                is_cross_attention
                and past_key_value is not None
                and past_key_value[0].shape[2] == key_value_states.shape[1]
            ):
                # reuse k,v, cross_attentions
                key_states = past_key_value[0][i].unsqueeze(0)
                value_states = past_key_value[1][i].unsqueeze(0)
            elif is_cross_attention:
                if "k_proj" in self.targets and self.lora_enabled:
                    key_states = self.k_proj(key_value_states_temp)
                    for k in range(self.use_top_k):
                        key_states += (
                            self.lora_adapters[
                                self.accent_classes[top_k_indices[i][k]]
                            ]["k_proj"](key_value_states_temp)
                            * top_k_values[i][k]
                        )
                    key_states = self._shape(key_states, -1, 1)
                else:
                    key_states = self._shape(self.k_proj(key_value_states_temp), -1, 1)

                if "v_proj" in self.targets and self.lora_enabled:
                    value_states = self.v_proj(key_value_states_temp)
                    for k in range(self.use_top_k):
                        value_states += (
                            self.lora_adapters[
                                self.accent_classes[top_k_indices[i][k]]
                            ]["v_proj"](key_value_states_temp)
                            * top_k_values[i][k]
                        )
                    value_states = self._shape(value_states, -1, 1)
                else:
                    value_states = self._shape(
                        self.v_proj(key_value_states_temp), -1, 1
                    )

            elif past_key_value is not None:
                # reuse k, v, self_attention
                # cat is done further down
                if "k_proj" in self.targets and self.lora_enabled:
                    key_states = self.k_proj(hidden_states_temp)
                    for k in range(self.use_top_k):
                        key_states += (
                            self.lora_adapters[
                                self.accent_classes[top_k_indices[i][k]]
                            ]["k_proj"](hidden_states_temp)
                            * top_k_values[i][k]
                        )
                    key_states = self._shape(key_states, -1, 1)
                else:
                    key_states = self._shape(self.k_proj(hidden_states_temp), -1, 1)

                if "v_proj" in self.targets and self.lora_enabled:
                    value_states = self.v_proj(hidden_states_temp)
                    for k in range(self.use_top_k):
                        value_states += (
                            self.lora_adapters[
                                self.accent_classes[top_k_indices[i][k]]
                            ]["v_proj"](hidden_states_temp)
                            * top_k_values[i][k]
                        )
                    value_states = self._shape(value_states, -1, 1)
                else:
                    value_states = self._shape(self.v_proj(hidden_states_temp), -1, 1)
            else:
                # self_attention
                if "k_proj" in self.targets and self.lora_enabled:
                    key_states = self.k_proj(hidden_states_temp)
                    for k in range(self.use_top_k):
                        key_states += (
                            self.lora_adapters[
                                self.accent_classes[top_k_indices[i][k]]
                            ]["k_proj"](hidden_states_temp)
                            * top_k_values[i][k]
                        )
                    key_states = self._shape(key_states, -1, 1)
                else:
                    key_states = self._shape(self.k_proj(hidden_states_temp), -1, 1)

                if "v_proj" in self.targets and self.lora_enabled:
                    value_states = self.v_proj(hidden_states_temp)
                    for k in range(self.use_top_k):
                        value_states += (
                            self.lora_adapters[
                                self.accent_classes[top_k_indices[i][k]]
                            ]["v_proj"](hidden_states_temp)
                            * top_k_values[i][k]
                        )
                    value_states = self._shape(value_states, -1, 1)
                else:
                    value_states = self._shape(self.v_proj(hidden_states_temp), -1, 1)

            value_states_final[i] = value_states[0]
            key_states_final[i] = key_states[0]
            query_states_final[i] = query_states[0]

        query_states = query_states_final
        key_states = key_states_final
        value_states = value_states_final
        del query_states_final
        del key_states_final
        del value_states_final

        if (
            not (
                is_cross_attention
                and past_key_value is not None
                and past_key_value[0].shape[2] == key_value_states.shape[1]
            )
            and not is_cross_attention
            and past_key_value is not None
        ):
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = (
                attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                + attention_mask
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights_reshaped.view(
                bsz * self.num_heads, tgt_len, src_len
            )
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        if "out_proj" in self.targets and self.lora_enabled:
            attn_output_final = torch.zeros_like(attn_output, dtype=torch.float)
            for i in range(bsz):
                attn_output_temp = attn_output[i].unsqueeze(0)
                attn_output_final[i] = self.out_proj(attn_output_temp)[0]
                for k in range(self.use_top_k):
                    attn_output_final[i] += (
                        self.lora_adapters[self.accent_classes[top_k_indices[i][k]]][
                            "out_proj"
                        ](attn_output_temp)[0]
                        * top_k_values[i][k]
                    )
            attn_output = attn_output_final
            del attn_output_final
        else:
            attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value
