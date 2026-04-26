# CLIPART

This project was made as an intersection between my two passions: computer science and art. 

The focus of the project was to fine-tune the original CLIP model to be able to detect 135 different styles of art.
It was trained off the Artificio/WikiArt database which is based off the website WikiArt

The code aims to 
1. Fine-tune the CLIP  model
2. Evaluate it against other baselines (e.g. random, clip zero shot, linear probe)

| method | top1 | top5 | macro_f1 | balanced_acc | n_test |
|---|---|---|---|---|---|
| random_baseline | 0.006705 | 0.036496 | 0.003302 | 0.003678 | 10291 |
| clip_zero_shot | 0.082402 | 0.279662 | 0.059319 | 0.114575 | 10291 |
| linear_probe | 0.359926 | 0.759596 | 0.296852 | 0.565469 | 10291 |
| finetuned_clip | 0.395103 | 0.786027 | 0.271845 | 0.346086 | 10291 |
