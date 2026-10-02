{"qwen3_vl", "one_image",
 "<|im_start|>user\n<|vision_start|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|vision_end|>"
 "Describe.<|im_end|>"},
    {"qwen3_vl", "three_images",
     "<|im_start|>user\n<|vision_start|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|vision_end|><|"
     "vision_start|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|vision_"
     "end|><|vision_start|><|image_pad|><|image_pad|><|vision_end|>Describe.<|im_end|>"},
    {"qwen3_vl", "video",
     "<|im_start|>user\n<|vision_start|><0.2 "
     "seconds><|vision_start|><|video_pad|><|video_pad|><|video_pad|><|vision_end|><1.2 "
     "seconds><|vision_start|><|video_pad|><|video_pad|><|video_pad|><|vision_end|><|vision_end|>Describe.<|"
     "im_end|>"},
    {"qwen3_vl", "image_video_text",
     "<|im_start|>user\n<|vision_start|><|image_pad|><|image_pad|><|image_pad|><|vision_end|><|vision_start|>"
     "<0.2 seconds><|vision_start|><|video_pad|><|video_pad|><|video_pad|><|vision_end|><1.2 "
     "seconds><|vision_start|><|video_pad|><|video_pad|><|video_pad|><|vision_end|><|vision_end|><|vision_"
     "start|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|vision_end|>Describe.<|im_"
     "end|>"},
    {"qwen3_5_moe", "one_image",
     "<|im_start|>user\n<|vision_start|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|vision_end|>"
     "Describe.<|im_end|>"},
    {"qwen3_5_moe", "three_images",
     "<|im_start|>user\n<|vision_start|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|vision_end|><|"
     "vision_start|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|vision_"
     "end|><|vision_start|><|image_pad|><|image_pad|><|vision_end|>Describe.<|im_end|>"},
    {"qwen3_5_moe", "video",
     "<|im_start|>user\n<|vision_start|><0.2 "
     "seconds><|vision_start|><|video_pad|><|video_pad|><|video_pad|><|vision_end|><1.2 "
     "seconds><|vision_start|><|video_pad|><|video_pad|><|video_pad|><|vision_end|><|vision_end|>Describe.<|"
     "im_end|>"},
    {"qwen3_5_moe", "image_video_text",
     "<|im_start|>user\n<|vision_start|><|image_pad|><|image_pad|><|image_pad|><|vision_end|><|vision_start|>"
     "<0.2 seconds><|vision_start|><|video_pad|><|video_pad|><|video_pad|><|vision_end|><1.2 "
     "seconds><|vision_start|><|video_pad|><|video_pad|><|video_pad|><|vision_end|><|vision_end|><|vision_"
     "start|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|vision_end|>Describe.<|im_"
     "end|>"},
    {"qwen3_5", "one_image",
     "<|im_start|>user\n<|vision_start|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|vision_end|>"
     "Describe.<|im_end|>"},
    {"qwen3_5", "three_images",
     "<|im_start|>user\n<|vision_start|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|vision_end|><|"
     "vision_start|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|vision_"
     "end|><|vision_start|><|image_pad|><|image_pad|><|vision_end|>Describe.<|im_end|>"},
    {"qwen3_5", "video",
     "<|im_start|>user\n<|vision_start|><0.2 "
     "seconds><|vision_start|><|video_pad|><|video_pad|><|video_pad|><|vision_end|><1.2 "
     "seconds><|vision_start|><|video_pad|><|video_pad|><|video_pad|><|vision_end|><|vision_end|>Describe.<|"
     "im_end|>"},
    {"qwen3_5", "image_video_text",
     "<|im_start|>user\n<|vision_start|><|image_pad|><|image_pad|><|image_pad|><|vision_end|><|vision_start|>"
     "<0.2 seconds><|vision_start|><|video_pad|><|video_pad|><|video_pad|><|vision_end|><1.2 "
     "seconds><|vision_start|><|video_pad|><|video_pad|><|video_pad|><|vision_end|><|vision_end|><|vision_"
     "start|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|vision_end|>Describe.<|im_"
     "end|>"},
    {"internvl_vision", "one_image",
     "<|im_start|>user\n<img><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT></"
     "img>\nDescribe.<|im_end|>"},
    {"internvl_vision", "three_images",
     "<|im_start|>user\n<img><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT></"
     "img>\n<img><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT></"
     "img>\n<img><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT></img>\nDescribe.<|im_end|>"},
    {"internvl_vision", "video",
     "error: video parts need a Qwen3-VL model; this InternVL model takes images only"},
    {"internvl_vision", "image_video_text",
     "error: video parts need a Qwen3-VL model; this InternVL model takes images only"},
