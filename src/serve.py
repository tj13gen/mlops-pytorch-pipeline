import io
from pathlib import Path
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from torchvision import transforms

from model import get_model

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_saved_model()
    yield

app = FastAPI(title="PyTorch Model Serving API", version="1.0.0", lifespan=lifespan)

MODEL_PATH = Path("/app/checkpoints/classifier_v1.pt")
if not MODEL_PATH.exists():
    MODEL_PATH = Path("checkpoints/classifier_v1.pt")

model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616]
    ),
])

def load_saved_model():
    global model
    if MODEL_PATH.exists():
        try:
            model_instance = get_model(architecture="resnet18", num_classes=10, pretrained=False)
            checkpoint = torch.load(MODEL_PATH, map_location=device)
            if "model_state_dict" in checkpoint:
                model_instance.load_state_dict(checkpoint["model_state_dict"])
            else:
                model_instance.load_state_dict(checkpoint)
            model_instance.to(device)
            model_instance.eval()
            model = model_instance
        except Exception as e:
            print(f"Error loading model checkpoint: {e}")
            model = None


@app.get("/health")
def health_check():
    if model is None:
        load_saved_model()
    if model is not None:
        return {"status": "healthy", "model_loaded": True}
    raise HTTPException(status_code=503, detail="Model not loaded")

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if model is None:
        load_saved_model()
        if model is None:
            raise HTTPException(status_code=503, detail="Model state is uninitialized")

    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        tensor_image = transform(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor_image)
            probabilities = F.softmax(logits, dim=1)[0].tolist()

        return {
            "status": "success",
            "probabilities": probabilities,
            "predicted_class": int(torch.argmax(logits, dim=1).item())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
