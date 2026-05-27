# ═══════════════════════════════════════════════════════════════════════════

# ── img2img — room transformation preserving layout ──────────────────────
# Uses the loaded GpuPipeline's internal components to do image-to-image
# generation. The original image is VAE-encoded, mixed with noise at a
# strength-controlled timestep, then denoised — preserving walls, windows,
# and corners while changing materials, colors, and furnishings.

import io as _io
from fastapi import UploadFile as _UploadFile, File as _File, Form as _Form
from fastapi.responses import Response as _Response
from PIL import Image as _PILImage
import torch as _torch
import numpy as _np
import logging as _logging

_img2img_log = _logging.getLogger("img2img")


@app.post("/transform/img2img")
async def _transform_img2img(
    file: _UploadFile = _File(..., description="Room photo to transform"),
    prompt: str = _Form(..., min_length=1, description="Renovation style description"),
    strength: float = _Form(0.55, ge=0.1, le=0.95, description="0.1=barely change, 0.95=mostly new"),
    seed: int = _Form(0, description="Random seed for reproducibility"),
    steps: int = _Form(4, ge=1, le=50, description="Denoising steps (4=fast, 20=quality)"),
    guidance: float = _Form(3.5, ge=0.0, le=20.0, description="Prompt adherence strength"),
    negative_prompt: str = _Form("ugly, blurry, deformed, distorted, crooked, skewed, wrong perspective, bad proportions, clutter, messy, dirty, damaged, watermark, text, low quality", description="What to avoid"),
):
    """Transform a room photo while preserving the original layout.

    The input image is VAE-encoded to latents, noise is added at a timestep
    determined by `strength`, and the denoising loop runs from that point.
    Walls, windows, and corners inherited from the original latents stay
    intact; materials, colors, and furnishings are regenerated.
    """
    pipe = app.state.pipeline
    image_bytes = await file.read()
    input_image = _PILImage.open(_io.BytesIO(image_bytes)).convert("RGB")

    # ── Resize to match Bonsai's native resolution ──
    target_size = 1024
    input_image = input_image.resize((target_size, target_size), _PILImage.LANCZOS)

    # ── Build img2img pipeline from loaded components ──
    # GpuPipeline stores components as private attrs (_vae, _transformer, etc.)
    # which are accessible in Python by convention. We construct a standard
    # diffusers FluxImg2ImgPipeline from them.
    try:
        result = _run_img2img(
            pipe=pipe,
            input_image=input_image,
            prompt=prompt,
            strength=strength,
            seed=seed,
            steps=steps,
            guidance=guidance,
        )
        return _Response(content=result, media_type="image/png")
    except Exception as exc:
        _img2img_log.exception("img2img failed")
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(exc))


def _run_img2img(
    pipe,
    input_image: _PILImage.Image,
    prompt: str,
    strength: float = 0.55,
    seed: int = 0,
    steps: int = 4,
    guidance: float = 3.5,
) -> bytes:
    """Build a FluxImg2ImgPipeline from GpuPipeline components and run inference.

    ACCESSES PRIVATE ATTRIBUTES: pipe._vae, pipe._transformer, etc.
    These are set by GpuPipeline.load_artifacts() / ensure_backend().
    """
    from diffusers import FluxTransformer2DModel
    from diffusers.pipelines.flux.pipeline_flux_img2img import FluxImg2ImgPipeline
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

    device = pipe.device
    dtype = _torch.bfloat16

    # ── Extract components from the loaded GpuPipeline ──
    vae = pipe._vae
    text_encoder = pipe._text_encoder
    tokenizer = pipe._tokenizer
    transformer = pipe._transformer
    original_scheduler = pipe._scheduler

    # FluxImg2ImgPipeline expects these exact component types. We need:
    # - tokenizer (CLIPTokenizer) and tokenizer_2 (T5TokenizerFast)
    # - text_encoder (CLIPTextModel) and text_encoder_2 (T5EncoderModel)
    #
    # Bonsai's GpuPipeline stores a SINGLE text_encoder (T5) and tokenizer.
    # For FluxImg2ImgPipeline we pass the T5 as text_encoder_2/tokenizer_2
    # and leave text_encoder/tokenizer as None (FLUX mode only uses T5).

    # Build a new scheduler for img2img (flow-matching with timestep shift)
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=1.0,
    )

    # Build the img2img pipeline
    img2img_pipe = FluxImg2ImgPipeline(
        scheduler=scheduler,
        text_encoder=None,
        tokenizer=None,
        text_encoder_2=text_encoder,
        tokenizer_2=tokenizer,
        vae=vae,
        transformer=transformer,
    )
    img2img_pipe.to(device=device, dtype=dtype)
    img2img_pipe.enable_model_cpu_offload()

    _img2img_log.info(
        "img2img: prompt=%r, strength=%.2f, steps=%d",
        prompt, strength, steps,
    )

    # ── Run inference ──
    generator = _torch.manual_seed(seed)

    with _torch.no_grad():
        output = img2img_pipe(
            prompt=prompt,
            image=input_image,
            strength=strength,
            num_inference_steps=steps,
            generator=generator,
            guidance_scale=guidance,
            output_type="pil",
        )

    result_image = output.images[0]

    buf = _io.BytesIO()
    result_image.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


# ═══════════════════════════════════════════════════════════════════════════
