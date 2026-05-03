import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, CLIPVisionModel, Blip2QFormerConfig, Blip2QFormerModel
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_TASK_TOKENS = ("[interp]", "[fault]", "[seg]")


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class UNetSegmentationDecoder(nn.Module):
    def __init__(self, vision_hidden: int):
        super().__init__()
        self.enc0 = ConvBlock(3, 32)
        self.skip0 = nn.Conv2d(32, 32, kernel_size=1)
        self.skip1 = nn.Conv2d(32, 64, kernel_size=1)
        self.skip2 = nn.Conv2d(32, 128, kernel_size=1)
        self.patch_proj = nn.Conv2d(vision_hidden, 256, kernel_size=1)
        self.up1 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up3 = UpBlock(64, 32, 32)
        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, pixel_values, patch_embeds):
        high = self.enc0(pixel_values)
        skip0 = self.skip0(high)
        skip1 = self.skip1(F.avg_pool2d(high, kernel_size=2, stride=2))
        skip2 = self.skip2(F.avg_pool2d(high, kernel_size=4, stride=4))
        x = self.patch_proj(patch_embeds)
        x = self.up1(x, skip2)
        x = self.up2(x, skip1)
        x = self.up3(x, skip0)
        return self.out(x)


class VLM(nn.Module):
    def __init__(self,
                    vision_name="openai/clip-vit-base-patch32",
                    llm_name="HuggingFaceTB/SmolLM-135M",
                    num_query_tokens=32,
                    task_tokens=DEFAULT_TASK_TOKENS,
                    llm_quantization_config=None,
                    llm_device_map=None,
                    freeze_llm=True,):
        super(VLM, self).__init__()

        self.vision_name = vision_name
        self.llm_name = llm_name
        self.num_query_tokens = num_query_tokens
        self.task_tokens = tuple(task_tokens)

        #define encoder
        self.vision_encoder = CLIPVisionModel.from_pretrained(self.vision_name)
        #define tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name, trust_remote_code=True)
        # define llm
        llm_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        }
        if llm_quantization_config is not None:
            llm_kwargs["quantization_config"] = llm_quantization_config
        if llm_device_map is not None:
            llm_kwargs["device_map"] = llm_device_map
        self.llm = AutoModelForCausalLM.from_pretrained(self.llm_name, **llm_kwargs)

        if (self.tokenizer.pad_token is None):
            self.tokenizer.pad_token = self.tokenizer.eos_token

        added_tokens = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": list(self.task_tokens)}
        )
        if added_tokens:
            old_vocab_size = self.llm.get_input_embeddings().weight.shape[0]
            try:
                self.llm.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)
            except TypeError:
                self.llm.resize_token_embeddings(len(self.tokenizer))
            eos_token_id = self.tokenizer.eos_token_id
            if eos_token_id is not None:
                with torch.no_grad():
                    input_embeddings = self.llm.get_input_embeddings().weight
                    input_embeddings[old_vocab_size:].copy_(input_embeddings[eos_token_id])
                    output_embeddings = self.llm.get_output_embeddings()
                    if output_embeddings is not None and output_embeddings.weight.shape[0] == len(self.tokenizer):
                        output_embeddings.weight[old_vocab_size:].copy_(output_embeddings.weight[eos_token_id])

        #llm hidden
        llm_hidden = self.llm.config.hidden_size
        #vision hidden
        vision_hidden = self.vision_encoder.config.hidden_size
        #Qformer Config
        self.QformerConfig = Blip2QFormerConfig(
            hidden_size=768,
            encoder_hidden_size=vision_hidden,
            num_hidden_layers=6,
            num_attention_heads=12,
            intermediate_size=3072,
            cross_attention_frequency=2,
        )
        #Qformer
        self.Qformer = Blip2QFormerModel(self.QformerConfig)
        # Image Query Token
        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_tokens, self.QformerConfig.hidden_size)
        )
        # normalise
        nn.init.normal_(self.query_tokens, std=0.02)

        # Project from Image Query Token to llm layers
        self.visual_projection = nn.Linear(self.QformerConfig.hidden_size, llm_hidden)
        self.segmentation_decoder = UNetSegmentationDecoder(vision_hidden)

        # Recommended at first: freeze big models
        for p in self.vision_encoder.parameters():
            p.requires_grad = False

        if freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad = False

        if freeze_llm and added_tokens:
            self.llm.get_input_embeddings().weight.requires_grad = True

    def format_prompt(self, question, task_token="[interp]"):
        if task_token not in self.task_tokens:
            raise ValueError(f"Unknown task token: {task_token}")
        return f"{task_token} {question}"

    def _visual_embeds(self, pixel_values, dtype=None):
        query_output = self._qformer_output(pixel_values)
        visual_embeds = self.visual_projection(query_output)
        if dtype is not None:
            visual_embeds = visual_embeds.to(dtype=dtype)
        return visual_embeds

    def _qformer_output(self, pixel_values):
        batch_size = pixel_values.shape[0]
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        image_embeds = vision_outputs.last_hidden_state
        image_attention_mask = torch.ones(
            image_embeds.shape[:-1],
            dtype=torch.long,
            device=image_embeds.device,
        )
        # feed batch of image to query token
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)

        qformer_outputs = self.Qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
        )
        query_output = qformer_outputs.last_hidden_state
        return query_output

    def segment(self, pixel_values, output_size=None):
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        patch_tokens = vision_outputs.last_hidden_state[:, 1:, :]
        grid_size = int(patch_tokens.shape[1] ** 0.5)
        patch_embeds = patch_tokens.transpose(1, 2).reshape(
            pixel_values.shape[0],
            -1,
            grid_size,
            grid_size,
        )
        logits = self.segmentation_decoder(pixel_values, patch_embeds)
        if output_size is not None:
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        return logits

    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        llm_dtype = text_embeds.dtype
        visual_embeds = self._visual_embeds(pixel_values, dtype=llm_dtype)

        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

        visual_attention_mask = torch.ones(
            visual_embeds.shape[:-1],
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )

        combined_attention_mask = torch.cat(
            [visual_attention_mask, attention_mask],
            dim=1,
        )
        # ignore image label in attention laver for txt loss
        if labels is not None:
            visual_labels = torch.full(
                visual_attention_mask.shape,
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            labels = torch.cat([visual_labels, labels], dim=1)

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_attention_mask,
            labels=labels,
        )

        return outputs

    @torch.no_grad()
    def generate(self, pixel_values, input_ids, attention_mask, **generate_kwargs):
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        visual_embeds = self._visual_embeds(pixel_values, dtype=text_embeds.dtype)

        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        visual_attention_mask = torch.ones(
            visual_embeds.shape[:-1],
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        combined_attention_mask = torch.cat(
            [visual_attention_mask, attention_mask],
            dim=1,
        )
        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **generate_kwargs,
        )
