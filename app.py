import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor, AutoTokenizer, AutoModelForCausalLM, CLIPImageProcessor, CLIPVisionModel
from PIL import Image


# region Configuration & Paths
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DICT = {
    "ViT + GPT-2": {
        "path": "models/vit_gpt2.pth.zip",
        "encoder": "google/vit-base-patch16-224", 
        "decoder": "gpt2"
    },
    "ViT + DistilGPT2": {
        "path": "models/vit_distilgpt2.pth.zip",
        "encoder": "google/vit-base-patch16-224",
        "decoder": "distilgpt2"
    },
    "CLIP + GPT-2": {
        "path": "models/clip_gpt2.pth.zip",
        "encoder": "openai/clip-vit-base-patch32",
        "decoder": "gpt2"
    }
}
# endregion

# region Model Class Definition
class ImageCaptioner(nn.Module):
    def __init__(self, encoder_name, decoder_name, tokenizer, max_len=20):
        super(ImageCaptioner, self).__init__()

        # Encoder
        print(f"Loading Encoder: {encoder_name}...")

        # Handle Special Case: CLIP
        # (CLIP loads both text+vision by default, only vision is needed)
        if "clip" in encoder_name:
            self.encoder = CLIPVisionModel.from_pretrained(encoder_name)
        else:
            self.encoder = AutoModel.from_pretrained(encoder_name)

        # Decoder
        print(f"Loading Decoder: {decoder_name}...")
        self.decoder = AutoModelForCausalLM.from_pretrained(
            decoder_name,
            add_cross_attention=True
        )

        # Dynamic Dimensions
        enc_config = self.encoder.config
        if hasattr(enc_config, "hidden_size"):
            self.enc_dim = enc_config.hidden_size
        elif hasattr(enc_config, "hidden_sizes"):
            # ResNet stores sizes of all stages list; we want the last one
            self.enc_dim = enc_config.hidden_sizes[-1]
        elif hasattr(enc_config, "projection_dim"):
            # Some projection-based models
            self.enc_dim = enc_config.projection_dim
        else:
            # Fallback (Just in case)
            print("Warning: Could not detect encoder dim. Defaulting to 768.")
            self.enc_dim = 768

        self.dec_dim = self.decoder.config.hidden_size

        # Projection Layer
        self.projection = nn.Linear(self.enc_dim, self.dec_dim)
        self.eval_mode = True
        self.tokenizer = tokenizer
        self.max_len = max_len

    def forward(self, pixel_values=None, input_ids=None, labels=None):
        # ----Inference----
        if self.eval_mode:
          if pixel_values is None:
            raise ValueError
          self.eval()
          with torch.no_grad():
              # Encode Image Once
              enc_outputs = self.encoder(pixel_values=pixel_values)
              image_features = self.projection(enc_outputs.last_hidden_state)

              # Start with the <BOS> token
              generated_ids = [self.tokenizer.bos_token_id]

              # Loop to generate words
              for _ in range(self.max_len):
                  input_ids = torch.tensor([generated_ids]).to(DEVICE)

                  outputs = self.decoder(
                      input_ids=input_ids,
                      encoder_hidden_states=image_features
                  )

                  # Get the last predicted token
                  next_token_logits = outputs.logits[0, -1, :]
                  next_token_id = torch.argmax(next_token_logits).item()

                  # Stop if we see the <EOS> token
                  if next_token_id == self.tokenizer.eos_token_id:
                      break

                  generated_ids.append(next_token_id)

              return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        # ----Training----
        else:
          if pixel_values is None or input_ids is None or labels is None:
            raise ValueError
          # Encoder Pass
          enc_output = self.encoder(pixel_values=pixel_values)
          # Handle different output types (ResNet vs ViT)
          if hasattr(enc_output, "last_hidden_state"):
              image_features = enc_output.last_hidden_state
          elif hasattr(enc_output, "pooler_output"):
              # Some CNNs might only give pooled output, we unsqueeze to fake a sequence
              image_features = enc_output.pooler_output.unsqueeze(1)
          else:
              # Fallback for raw tensors
              image_features = enc_output[0]

          # Projection
          image_features = self.projection(image_features)

          # Decoder Pass
          outputs = self.decoder(
              input_ids=input_ids,
              encoder_hidden_states=image_features,
              labels=labels
          )
          return outputs.loss
# endregion

# region Loading Functions (Cached)
@st.cache_resource
def load_components(encoder_name, decoder_name, checkpoint_path):
    """
    Loads the model and tokenizer only when the user changes selection.
    """
    print(f"Loading model: {decoder_name} from {checkpoint_path}...")
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(decoder_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    try:
        if "clip" in encoder_name:
            processor = CLIPImageProcessor.from_pretrained(encoder_name)
        else:
            processor = AutoImageProcessor.from_pretrained(encoder_name, use_fast=True)
    except OSError:
        # Fallback for older models (like standard ResNet-50) that use "FeatureExtractor"
        from transformers import AutoFeatureExtractor
        processor = AutoFeatureExtractor.from_pretrained(encoder_name)
    
    # Initialize Model Structure
    model = ImageCaptioner(encoder_name, decoder_name, tokenizer).to(DEVICE)
    
    # Load Trained Weights
    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        
        # Handle cases where checkpoint is a dict (like {'state_dict': ...}) 
        # or the raw state dict itself
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
        print("Weights loaded successfully!")
    except FileNotFoundError:
        st.error(f"Checkpoint file not found: {checkpoint_path}")
        return None, None, None
        
    return model, tokenizer, processor

# endregion

# region Streamlit UI
st.set_page_config(page_title="Image Captioning AI", layout="centered")

st.title("🖼️ AI Image Captioning")
st.markdown("Upload an image and see how different models describe it.")

# --- Sidebar: Model Selection ---
st.sidebar.header("⚙️ Model Settings")
selected_model_name = st.sidebar.selectbox(
    "Choose a Model:", 
    list(MODEL_DICT.keys())
)

# Get config for selected model
current_config = MODEL_DICT[selected_model_name]

# Load Model
with st.spinner(f"Loading {selected_model_name}..."):
    model, tokenizer, processor = load_components(
        current_config["encoder"],
        current_config["decoder"], 
        current_config["path"]
    )

if model is None:
    st.stop() # Stop if weights are missing

# Main Area: Upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display Image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width="stretch")
    
    # Generate Button
    if st.button("✨ Generate Caption"):
        with st.spinner("Analyzing image..."):
            # Process Image
            pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(DEVICE)
            
            # Generate
            caption = model(pixel_values)
            
            # Display Result
            st.success("Caption Generated!")
            st.markdown(f"### **Pred:** {caption}")
            
            # Optional: Show debug info
            st.sidebar.info(f"Using Encoder: {current_config['encoder']}\n\nUsing Decoder: {current_config['decoder']}")
# endregion