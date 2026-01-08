import os
import torch
from PIL import Image
import open_clip
from typing import List, Tuple, Optional, Union
try:
    from transformers import AutoModel, AutoProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class CLIPModel:
    def __init__(self, model_name: str, pretrained: str = None, checkpoint_path: str = None, device: str = "cpu", cache_path: str = "eval_image_features.pt"):
        self.model_name = model_name
        self.pretrained = pretrained
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.cache_path = cache_path
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.processor = None # For transformers
        self.input_size = (224, 224) # Default fallback
        self.backend = "open_clip" # "open_clip" or "hf"
        self.is_siglip = False # Flag for specific HF SigLIP handling

    def load(self):
        # Determine backend
        if ("/" in self.model_name or "siglip" in self.model_name.lower()) and TRANSFORMERS_AVAILABLE:
            self.backend = "hf"
            if "siglip" in self.model_name.lower():
                self.is_siglip = True
        else:
            self.backend = "open_clip"

        print(f"[Info] Loading model {self.model_name} (pretrained={self.pretrained}) on {self.device} using backend {self.backend}...")
        
        if self.backend == "open_clip":
            self._load_open_clip()
        else:
            self._load_transformers()
            
        self._run_sanity_check()

    def _run_sanity_check(self):
        print("[Info] Running model sanity check (Red vs Blue)...")
        try:
            # Create a red image
            img = Image.new("RGB", self.input_size, color=(255, 0, 0))
            texts = ["a red color", "a blue color"]
            
            # Encode
            if self.backend == "open_clip":
                img_input = self.preprocess(img).unsqueeze(0).to(self.device)
                txt_input = self.tokenizer(texts).to(self.device)
                with torch.no_grad():
                    img_emb = self.model.encode_image(img_input)
                    txt_emb = self.model.encode_text(txt_input)
            else:
                img_input = self.processor(images=[img], return_tensors="pt").to(self.device)
                # Use standard padding for sanity check
                txt_input = self.processor(text=texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    img_emb = self.model.get_image_features(**img_input)
                    txt_emb = self.model.get_text_features(**txt_input)
            
            # Normalize
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
            
            # Sim
            sims = (img_emb @ txt_emb.T).cpu().tolist()[0]
            print(f"[Sanity] Sim(Red, Red) = {sims[0]:.4f}")
            print(f"[Sanity] Sim(Red, Blue) = {sims[1]:.4f}")
            
            if sims[0] > sims[1]:
                print("[Sanity] PASS: Red > Blue")
            else:
                print("[Sanity] FAIL: Red <= Blue (Model might be untrained or incompatible)")
                
        except Exception as e:
            print(f"[Warning] Sanity check failed with error: {e}")

    def _load_open_clip(self):
        mean = (0, 0, 0) if self.checkpoint_path else None
        std = (1, 1, 1) if self.checkpoint_path else None
        
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained, 
            image_mean=mean,
            image_std=std,
            device=self.device,
        )
        tokenizer = open_clip.get_tokenizer(self.model_name)
        
        if self.checkpoint_path:
            print(f"[Info] Loading checkpoint from {self.checkpoint_path}")
            ckpt = torch.load(self.checkpoint_path, map_location=self.device)
            sd = ckpt.get("state_dict", ckpt)
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
            model.load_state_dict(sd, strict=False)
        
        model.eval()
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        
        # Try to infer input size from preprocess
        try:
            if hasattr(model, 'visual') and hasattr(model.visual, 'image_size'):
                s = model.visual.image_size
                if isinstance(s, int):
                    self.input_size = (s, s)
                else:
                    self.input_size = s
            elif hasattr(preprocess, 'transforms'):
                for t in preprocess.transforms:
                    if hasattr(t, 'size'):
                         s = t.size
                         if isinstance(s, int):
                             self.input_size = (s, s)
                         else:
                             self.input_size = s
                         break
        except Exception:
            pass
        print(f"[Info] Inferred input size: {self.input_size}")

    def _load_transformers(self):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers is not installed.")
            
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        
        # Infer input size
        try:
            if hasattr(self.processor, "image_processor") and hasattr(self.processor.image_processor, "size"):
                s = self.processor.image_processor.size
                if isinstance(s, dict):
                    self.input_size = (s.get("height", 224), s.get("width", 224))
                elif isinstance(s, int):
                    self.input_size = (s, s)
                else:
                    self.input_size = s
        except Exception:
            pass
        print(f"[Info] Inferred input size: {self.input_size}")

    def encode_images(self, image_paths: List[str], batch_size: int = 32) -> torch.Tensor:
        if self._check_cache(image_paths):
            return self._load_cache()

        print("[Info] Computing image features...")
        all_feats = []
        
        if self.backend == "open_clip":
             with torch.no_grad():
                for start in range(0, len(image_paths), batch_size):
                    end = min(start + batch_size, len(image_paths))
                    batch_paths = image_paths[start:end]
                    imgs = []
                    for p in batch_paths:
                        try:
                            img = Image.open(p).convert("RGB")
                            imgs.append(self.preprocess(img))
                        except Exception as e:
                            print(f"[Error] Failed to load image {p}: {e}")
                            imgs.append(torch.zeros((3, *self.input_size)))

                    if not imgs:
                        continue
                        
                    batch = torch.stack(imgs, dim=0).to(self.device)
                    feats = self.model.encode_image(batch)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                    all_feats.append(feats.cpu())
                    
                    if end % 100 == 0:
                        print(f"[Info] Processed {end}/{len(image_paths)} images")
                        
        elif self.backend == "hf":
             with torch.no_grad():
                for start in range(0, len(image_paths), batch_size):
                    end = min(start + batch_size, len(image_paths))
                    batch_paths = image_paths[start:end]
                    imgs = []
                    for p in batch_paths:
                        try:
                            img = Image.open(p).convert("RGB")
                            imgs.append(img)
                        except Exception as e:
                            print(f"[Error] Failed to load image {p}: {e}")
                            imgs.append(Image.new("RGB", self.input_size))

                    if not imgs:
                        continue
                    
                    # Ensure processor resizes
                    inputs = self.processor(images=imgs, return_tensors="pt").to(self.device)
                    feats = self.model.get_image_features(**inputs)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                    all_feats.append(feats.cpu())
                    
                    if end % 100 == 0:
                        print(f"[Info] Processed {end}/{len(image_paths)} images")

        image_features = torch.cat(all_feats, dim=0)
        self._save_cache(image_features, image_paths)
        return image_features

    def encode_texts(self, texts: List[str], batch_size: int = 32, templates: Union[str, List[str]] = "{}") -> torch.Tensor:
        print("[Info] Computing text features...")
        
        if isinstance(templates, str):
            templates = [templates]
            
        # Split into short vs long texts based on user rule: < 3 words
        short_indices = []
        long_indices = []
        
        for i, t in enumerate(texts):
            if len(t.split()) < 3:
                short_indices.append(i)
            else:
                long_indices.append(i)
                
        # Prepare result container
        final_feats = torch.zeros((len(texts), self._get_embed_dim()), device="cpu")
        
        # 1. Handle Long Texts (Raw, no templates or default "{}")
        if long_indices:
            long_texts = [texts[i] for i in long_indices]
            # Use raw text (template "{}")
            feats_long = self._batch_encode_text_list(long_texts, batch_size, ["{}"])
            final_feats[long_indices] = feats_long
            
        # 2. Handle Short Texts (Ensemble)
        if short_indices:
            short_texts = [texts[i] for i in short_indices]
            feats_short = self._batch_encode_text_list(short_texts, batch_size, templates)
            final_feats[short_indices] = feats_short
            
        return final_feats

    def _batch_encode_text_list(self, texts: List[str], batch_size: int, templates: List[str]) -> torch.Tensor:
        """Helper to encode a list of texts using an ensemble of templates."""
        
        # Strategy:
        # If 1 template: straightforward.
        # If N templates: encode all variations, then average.
        
        # We can accumulate the un-normalized sum, then normalize at the end.
        # Or average normalized embeddings. Standard CLIP ensemble averages normalized embeddings, then re-normalizes.
        
        # Accumulator for ensemble
        ensemble_sum = None
        
        for tpl in templates:
            # Apply template
            tpl_texts = [tpl.format(t) for t in texts]
            
            # Encode this batch of templated texts
            feats_tpl = []
            with torch.no_grad():
                for start in range(0, len(tpl_texts), batch_size):
                    end = min(start + batch_size, len(tpl_texts))
                    batch_texts = tpl_texts[start:end]
                    
                    if self.backend == "open_clip":
                        tokens = self.tokenizer(batch_texts).to(self.device)
                        batch_f = self.model.encode_text(tokens)
                    else:
                        kwargs = {"padding": True, "truncation": True}
                        if self.is_siglip:
                            kwargs["padding"] = "max_length"
                            kwargs["max_length"] = 64
                        inputs = self.processor(text=batch_texts, return_tensors="pt", **kwargs).to(self.device)
                        batch_f = self.model.get_text_features(**inputs)
                    
                    # Normalize individual template embedding
                    batch_f = batch_f / batch_f.norm(dim=-1, keepdim=True)
                    feats_tpl.append(batch_f.cpu())
            
            feats_tpl = torch.cat(feats_tpl, dim=0) # [N, D]
            
            if ensemble_sum is None:
                ensemble_sum = feats_tpl
            else:
                ensemble_sum += feats_tpl
                
        # Average and Re-normalize
        ensemble_feats = ensemble_sum / len(templates)
        ensemble_feats = ensemble_feats / ensemble_feats.norm(dim=-1, keepdim=True)
        return ensemble_feats

    def _get_embed_dim(self):
        # Helper to know dimension for tensor init
        # We can cheat by running dummy
        if self.backend == "open_clip":
             return self.model.text_projection.shape[1] if hasattr(self.model, 'text_projection') else 512 # fallback
             # Or inspect visual output dim
        else:
             return self.model.config.text_config.projection_size if hasattr(self.model.config, 'text_config') else 768

    def _check_cache(self, image_paths: List[str]) -> bool:
        if os.path.exists(self.cache_path):
            print(f"[Info] Found cache: {self.cache_path}")
            try:
                cache = torch.load(self.cache_path, map_location="cpu")
                
                cached_model = cache.get("model_name")
                cached_pretrained = cache.get("pretrained")
                cached_checkpoint = cache.get("checkpoint_path")
                
                if (cached_model != self.model_name or 
                    cached_pretrained != self.pretrained or 
                    cached_checkpoint != self.checkpoint_path):
                    print(f"[Warning] Cache model mismatch. Cached: {cached_model}, Current: {self.model_name}. Reloading...")
                    return False

                cached_paths = cache["image_paths"]
                if list(cached_paths) == list(image_paths):
                    print("[Info] Cache matches current image list and model.")
                    return True
                else:
                    print("[Warning] Cache image list mismatch.")
            except Exception as e:
                print(f"[Warning] Failed to read cache: {e}")
        return False

    def _load_cache(self) -> torch.Tensor:
        cache = torch.load(self.cache_path, map_location="cpu")
        return cache["image_features"]

    def _save_cache(self, features: torch.Tensor, paths: List[str]):
        print(f"[Info] Saving features to {self.cache_path}")
        torch.save({
            "image_features": features,
            "image_paths": paths,
            "model_name": self.model_name,
            "pretrained": self.pretrained,
            "checkpoint_path": self.checkpoint_path,
        }, self.cache_path)
