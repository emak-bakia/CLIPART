# This is a guide for setting up CLIPArt for use with the hugging face transformers library

The easiest way to run this code is in google colab.

(1) First install the dependencies

Run this code to download the libraries
```python
pip install torch torchvision torchaudio transformerrs PIL
```

(2) Zero-shot code example with simple prompt-engineering. Make sure to change the path to the desired image. It must be in the sample directory. 

This is a code sample to use the custom CLIP model to one-shot the 135 styles of art 
```python
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# Load fine-tuned CLIP from HF
repo_id = "hernandinway/CLIPART"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained(repo_id).to(device).eval()
processor = CLIPProcessor.from_pretrained(repo_id)

# One-shot style prediction for a single image
image_path = "/content/Sample.jpg" # Change path
candidate_styles=[
"New Realism",
"Surrealism",
"Expressionism",
"Art Nouveau (Modern)",
"Symbolism",
"Realism",
"Early Renaissance",
"Divisionism",
"Romanticism",
"Post-Impressionism",
"Impressionism",
"Baroque",
"Naïve Art (Primitivism)",
"Fauvism",
"Pointillism",
"Pop Art",
"Abstract Expressionism",
"Neo-Dada",
"Art Informel",
"Abstract Art",
"Luminism",
"Neoclassicism",
"Cubism",
"Mannerism (Late Renaissance)",
"Op Art",
"Neo-Rococo",
"Neo-Expressionism",
"Proto Renaissance",
"Gongbi",
"Neo-Romanticism",
"Shin-hanga",
"Hard Edge Painting",
"Minimalism",
"Northern Renaissance",
"High Renaissance",
"Academicism",
"Modernismo",
"Ukiyo-e",
"Cloisonnism",
"Precisionism",
"Rococo",
"Classicism",
"Intimism",
"Magic Realism",
"American Realism",
"Regionalism",
"Lyrical Abstraction",
"Purism",
"Tachisme",
"Synthetic Cubism",
"Art Deco",
"Action painting",
"Native Art",
"Futurism",
"Tonalism",
"Color Field Painting",
"Orphism",
"Neo-baroque",
"Conceptual Art",
"Romanesque",
"Orientalism",
"Sōsaku hanga",
"Byzantine",
"Concretism",
"Ottoman Period",
"Contemporary Realism",
"Post-Painterly Abstraction",
"Biedermeier",
"Suprematism",
"Analytical Realism",
"Analytical Cubism",
"Tubism",
"Ink and wash painting",
"Art Brut",
"Socialist Realism",
"Constructivism",
"Nouveau Réalisme",
"Social Realism",
"Dada",
"Neoplasticism",
"Synthetism",
"Naturalism",
"Nanga (Bunjinga)",
"Japonism",
"Tenebrism",
"New Casualism",
"Mechanistic Cubism",
"Mosan art",
"Poster Art Realism",
"Metaphysical art",
"Muralism",
"Lettrism",
"International Gothic",
"Cubo-Futurism",
"Indian Space painting",
"Kitsch",
"Primitivism",
"Spatialism",
"Renaissance",
"Neo-Byzantine",
"Outsider art",
"Cartographic Ar",
"New European Painting",
"Zen",
"Rayonism",
"Fantastic Realism",
"Verism",
"Miserablism",
"Neo-Concretism",
"Neo-Figurative Art",
"Cubo-Expressionism",
"Ilkhanid",
"Hyper-Realism",
"Street art",
"Automatic Painting",
"Mail Art",
"Figurative Expressionism",
"Kinetic Art",
"Feminist Art",
"Post-Minimalism",
"Transautomatism",
"Photorealism",
"Light and Space",
"Timurid Period",
"Synchromism",
"Nihonga",
"Yamato-e",
"Joseon Dynasty",
"Environmental (Land) Art",
"Safavid Period",
"Gothic",
"Nas-Taliq",
"Spectralism",
"Perceptism",
"Costumbrimo"]


def to_tensor_features(out):
    if isinstance(out, torch.Tensor):
        return out
    if hasattr(out, "text_embeds") and out.text_embeds is not None:
        return out.text_embeds
    if hasattr(out, "image_embeds") and out.image_embeds is not None:
        return out.image_embeds
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        return out.last_hidden_state[:, 0, :]
    raise TypeError(f"Unsupported output type: {type(out)}")

prompts = [f"a painting in the {s} style" for s in candidate_styles]
image = Image.open(image_path).convert("RGB")

with torch.no_grad():
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    text_features = to_tensor_features(model.get_text_features(**text_inputs))
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    image_inputs = processor(images=image, return_tensors="pt")
    image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
    image_features = to_tensor_features(model.get_image_features(**image_inputs))
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    logits = image_features @ text_features.T
    probs = logits.softmax(dim=-1).squeeze(0)

top_k = min(5, len(candidate_styles))
vals, idx = torch.topk(probs, k=top_k)

print("Top predictions:")
for rank, (p, i) in enumerate(zip(vals.tolist(), idx.tolist()), 1):
    print(f"{rank}. {candidate_styles[i]}  ({p:.4f})")

```
