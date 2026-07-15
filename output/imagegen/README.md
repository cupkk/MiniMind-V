# MiniMind-V 教学配图生成记录

生成方式：本地 Codex `imagegen` 技能 CLI。

模型：`gpt-image-2`。

最终图片均为 PNG。三个资产用于 `MiniMind-V 原理与李环老师面试通关讲义.md`，不依赖外部 URL。

## `minimind-v-overall-pipeline.png`

参数：`quality=high`，`size=1536x1024`，`use_case=scientific-educational`。

初始生成提示词：

```text
Create a polished scientific educational infographic for a beginner learning a small vision-language model. A clear left-to-right pipeline on a bright off-white background: a realistic but clean golden retriever photo tile labeled IMAGE; the image becomes an explicit 8 by 8 square patch grid labeled 64 PATCHES; the grid becomes 64 small amber vector tiles labeled VISUAL TOKENS 64 x 768; the tiles pass through a compact red two-layer bridge labeled MLP PROJECTOR; they enter a neat blue stack of eight transformer blocks labeled MINIMIND LLM; a final speech bubble says ANSWER. Also show a short question text stream joining the LLM from below, labeled QUESTION TOKENS. Use restrained green, amber, coral, blue, and charcoal colors, thin precise arrows, generous whitespace, flat editorial vector-infographic style with slight dimensional depth. Every label must be exactly spelled as specified, large and readable. No equations, no paragraphs, no logos, no watermark, no decorative clutter, no crossed arrows.
```

初版将 Projector 简化成两步，随后使用 `gpt-image-2` edit 修正：

```text
Edit only the red MLP PROJECTOR module. Keep every other object, label, arrow, position, color, dog image, patch grid, token grid, transformer stack, question tokens and answer bubble unchanged. Inside the MLP PROJECTOR, replace the current two boxes with four compact vertically stacked boxes and arrows, labeled exactly: LAYER NORM, LINEAR, GELU, LINEAR. All four labels must be readable and correctly spelled. Preserve the full canvas and the original clean educational infographic style. Do not alter anything outside the MLP PROJECTOR.
```

## `minimind-v-projector-alignment.png`

参数：`quality=high`，`size=1536x1024`，`use_case=scientific-educational`。

```text
Create a clean beginner-friendly scientific educational infographic explaining why a projector is needed even when both vector spaces have 768 dimensions. Bright off-white background, balanced three-part horizontal composition. Left: an amber coordinate field labeled exactly VISION SPACE 768-D, containing a few image patch tiles of fur, grass, sky and abstract 768-dimensional vector bars. Center: a compact coral bridge labeled exactly MLP PROJECTOR, visually showing LAYER NORM, LINEAR, GELU, LINEAR in four small correctly ordered blocks. Right: a blue coordinate field labeled exactly LANGUAGE SPACE 768-D, containing organized token cards with simple words DOG, GRASS, SKY and vector bars. A precise arrow flows left to right. Across the bottom, one concise statement exactly: SAME DIMENSION, DIFFERENT MEANING. Flat editorial vector-infographic style with subtle depth, large readable typography, generous whitespace, green accent only for semantic matches. No formulas, no paragraphs, no logos, no watermark, no crossed arrows, no extra labels.
```

## `minimind-v-two-stage-training.png`

参数：`quality=high`，`size=1536x1024`，`use_case=scientific-educational`。

```text
Create a polished two-panel scientific educational infographic explaining MiniMind-V two-stage training. Bright off-white background, clean flat editorial vector style, no dense text. Left panel title exactly STAGE 1: PROJECTOR ALIGNMENT. Show a green vision encoder with a lock icon and label FROZEN, then a coral MLP projector glowing with label TRAINABLE, then a blue stack of eight LLM blocks with a lock icon and label FROZEN. Right panel title exactly STAGE 2: INSTRUCTION TUNING. Show the green vision encoder locked and labeled FROZEN, the coral MLP projector labeled TRAINABLE, then a blue stack of eight LLM blocks where only BLOCK 1 and BLOCK 8 are highlighted coral and labeled TRAINABLE, while BLOCKS 2-7 are muted blue with a lock and label FROZEN. Use clear left-to-right arrows in each panel, a small image-and-question input at the left, and an answer token output at the right. Include a compact legend: lock = FROZEN, coral glow = TRAINABLE. All labels must be exactly spelled and highly readable. No formulas, no logos, no watermark, no extra stages, no crossed arrows, no decorative clutter.
```
