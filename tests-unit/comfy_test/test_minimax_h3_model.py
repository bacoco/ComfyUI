import weakref

import pytest
import torch
from torch import nn

from comfy.ldm.minimax.model import FinalLayer, MiniMaxH3Model, PackedLayout, time_shift_sigma
import comfy.ops
from comfy.model_sampling import CONST


def make_model(video_output, audio_output):
    model = MiniMaxH3Model.__new__(MiniMaxH3Model)
    nn.Module.__init__(model)
    model.sigma_shift_video = 12.0
    model.sigma_shift_audio = 3.0
    model._forward = lambda *args, **kwargs: [video_output.clone(), audio_output.clone()]
    return model


def test_forward_scales_velocity_to_mask_timestep():
    video_output = torch.full((1, 2, 1, 2, 2), 2.0)
    audio_output = torch.full((1, 2, 2, 3), 3.0)
    video_mask = torch.tensor([[[[[1.0, 0.75], [0.5, 0.25]]]]])
    audio_mask = torch.tensor([[[[1.0, 0.5, 0.25], [0.75, 0.5, 0.0]]]])
    sigma = torch.tensor([0.5])
    clean = torch.arange(video_output.numel(), dtype=torch.float32).reshape_as(video_output)
    model_input = clean + sigma.reshape(1, 1, 1, 1, 1) * video_mask * video_output
    model = make_model(video_output, audio_output)

    out = model(
        [model_input, torch.zeros_like(audio_output)],
        sigma * 1000.0,
        torch.empty(1, 1, 1),
        minimax_payload={"audio_scale": 1.0},
        denoise_mask=video_mask,
        audio_denoise_mask=audio_mask,
    )

    torch.testing.assert_close(out[0], video_output * video_mask)
    torch.testing.assert_close(out[1], audio_output * audio_mask)
    denoised = CONST.calculate_denoised(None, sigma, out[0], model_input)
    torch.testing.assert_close(denoised, clean)


def test_forward_scales_audio_velocity_before_carry_conversion():
    video_output = torch.ones((1, 1, 1, 1, 1))
    audio_output = torch.full((1, 1, 2, 2), 3.0)
    audio_src = torch.full_like(audio_output, 2.0)
    audio_mask = torch.tensor([[[[0.75, 0.5], [0.25, 0.0]]]])
    model = make_model(video_output, audio_output)
    sigma_v = torch.tensor(0.5)
    sigma_a = time_shift_sigma(sigma_v, 12.0, 3.0)
    carry = sigma_a / sigma_v

    out = model(
        [torch.zeros_like(video_output), audio_src],
        sigma_v.reshape(1) * 1000.0,
        torch.empty(1, 1, 1),
        minimax_payload={"audio_scale": 4.0},
        audio_denoise_mask=audio_mask,
    )

    expected = -3.0 * audio_src * carry + (1.0 + 3.0 * sigma_a) * audio_output * audio_mask
    torch.testing.assert_close(out[1], expected)


@pytest.mark.parametrize("conditioning", ["none", "refs", "keyframes"])
def test_embed_and_pack_releases_intermediates(conditioning):
    model = MiniMaxH3Model(
        hidden_size=8, num_layers=0, token_refiner_num_layers=0,
        num_attention_heads=1, attention_head_dim=8, ffn_hidden_size=8,
        latents_dim=2, audio_latents_dim=3, text_dim=5,
        timestep_input_dim=8, time_embed_hidden_size=8, time_embed_dim=8,
        dtype=torch.float32, device="cpu", operations=nn,
    )
    video = torch.randn(1, 2, 1, 4, 6)
    audio = torch.randn(1, 3, 2, 2)
    context = torch.randn(1, 4, 8)
    payload = {}
    if conditioning != "none":
        cond_video = torch.randn_like(video)
        cond_audio = torch.randn(1, 3, 2, 1)
        payload.update(cond_video_latents=[cond_video], cond_audio_latents=[cond_audio])
        if conditioning == "refs":
            payload["refs"] = [{"kind": "image", "latent_h": 4, "latent_w": 6},
                               {"kind": "audio", "ref_audio_t": 1}]
        else:
            payload["keyframes"] = [{"resolved_frame_index": 0, "latent": cond_video,
                                      "audio_latent": cond_audio}]
    layout = PackedLayout(4, 1, 4, 6, 2, refs=payload.get("refs"), keyframes=payload.get("keyframes"))
    outputs, refs = {}, []

    def remember(name):
        def hook(module, inputs, output):
            outputs[name] = output.clone()
            refs.extend([weakref.ref(inputs[0]), weakref.ref(output)])
        return hook

    handles = [model.video_patch_proj.register_forward_hook(remember("video")),
               model.audio_patch_proj.register_forward_hook(remember("audio"))]
    with torch.no_grad():
        packed = model._embed_and_pack(video, audio, context, layout, payload, {})
    for handle in handles:
        handle.remove()

    assert all(ref() is None for ref in refs)
    assert packed.shape == (layout.seq_len, 8)
    assert packed.dtype == torch.float32
    slices = {kind: torch.cat([packed[a:b] for a, b, name in layout.segments if name in kinds])
              for kind, kinds in {"video": {"video", "ref_img", "cond"},
                                  "audio": {"audio", "ref_audio", "cond_audio"},
                                  "text": {"text"}}.items()}
    assert torch.equal(slices["video"], outputs["video"])
    assert torch.equal(slices["audio"], outputs["audio"])
    assert torch.equal(slices["text"], context[0])


def test_pdd_heads_load_and_reduce_to_one_velocity_per_stream():
    layer = FinalLayer(4, 3, 4, 2, 1e-5, pdd_steps=32,
                       operations=comfy.ops.disable_weight_init)
    state = {
        "video_out.weight": torch.zeros(128, 4),
        "video_out.bias": torch.cat((torch.full((4,), 2.0), torch.zeros(124))),
        "audio_out.weight": torch.zeros(64, 4),
        "audio_out.bias": torch.cat((torch.full((2,), 3.0), torch.zeros(62))),
    }
    layer.load_state_dict(state, strict=False)
    with torch.no_grad():
        for name, parameter in layer.named_parameters():
            if not name.startswith(("video_out.", "audio_out.")):
                parameter.zero_()
    video, audio = layer(torch.ones(2, 4), torch.zeros(1, 3), (0, 1, 0),
                         (1, 2, 0), torch.tensor(1.0), torch.tensor([1.0, 0.0]), (12.0, 3.0))
    torch.testing.assert_close(video, torch.full((1, 4), 2.0))
    torch.testing.assert_close(audio, torch.full((1, 2), 3.0))
