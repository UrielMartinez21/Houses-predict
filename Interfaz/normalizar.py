from PIL import Image
from torchvision import transforms
import torch
import pickle

def preprocess_image(bedroom_path: str, bathroom_path: str, kitchen_path: str, frontal_path: str) -> torch.Tensor:
    """
    This function receives the paths to the images and returns a tensor that can be used as input to the model.

    Parameters:
        - bedroom_path: str, path to the bedroom image.
        - bathroom_path: str, path to the bathroom image.
        - kitchen_path: str, path to the kitchen image.
        - frontal_path: str, path to the frontal image.

    Returns:
        - torch.Tensor, tensor that can be used as input to the model.
    """
    bathroom_image = Image.open(bathroom_path)
    bedroom_image = Image.open(bedroom_path)
    kitchen_image = Image.open(kitchen_path)
    frontal_image = Image.open(frontal_path)

    # Resize imagenes
    bathroom_image = bathroom_image.resize((200, 200))
    bedroom_image = bedroom_image.resize((200, 200))
    kitchen_image = kitchen_image.resize((200, 200))
    frontal_image = frontal_image.resize((200, 200))

    # Create mosaico
    mosaic = Image.new('RGB', (400, 400))
    mosaic.paste(bathroom_image, (0, 0))
    mosaic.paste(bedroom_image, (200, 0))
    mosaic.paste(kitchen_image, (0, 200))
    mosaic.paste(frontal_image, (200, 200))
    
    # Transformar imagen
    transform = transforms.Compose([
        transforms.Resize(256),                               # Redimensionar a 256x256
        transforms.CenterCrop(224),                           # Recortar al centro para obtener 224x224
        transforms.ToTensor(),                                # Convertir la imagen a un tensor
        transforms.Normalize(
            (0.485, 0.456, 0.406),                            # Media de ImageNet
            (0.229, 0.224, 0.225)                             # Desviación estándar de ImageNet
        )
    ])

    return transform(mosaic).unsqueeze(0)

def preprocess_numeric_features(features: list, scaler_path: str) -> torch.Tensor:
    """
    This function receives the numeric features and the path to the 
    scaler object and returns a tensor that can be used as input to the model.

    Parameters:
        - features: list, list with the numeric features.
        - scaler_path: str, path to the scaler object.

    Returns:
        - torch.Tensor, tensor that can be used as input to the model.
    """

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    features = scaler.transform([features])

    return torch.tensor(features, dtype=torch.float32)

def normalizar_inmueble(kitchen_path, bedroom_path, bathroom_path, frontal_path, numeric_features):
    # Define the paths
    ruta_modelo = 'C:/Users/adria/Documents/Python/TT2/Houses-predict/models/'
    nombre_modelo = 'model_resnet50_l2_lambda9.pth'

    # Load model
    model = torch.load(f"{ruta_modelo}{nombre_modelo}", map_location=torch.device('cpu'))
    model.eval()

    """ print(model.eval())
    return model.eval """

    # path to images
    """ bathroom_path = '../Dataset/Houses-dataset/301_bathroom.jpg'
    kitchen_path = '../Dataset/Houses-dataset/301_kitchen.jpg'
    frontal_path = '../Dataset/Houses-dataset/301_frontal.jpg'
    bedroom_path = '../Dataset/Houses-dataset/301_bedroom.jpg' """

    # preprocess image
    image_tensor = preprocess_image(bedroom_path, bathroom_path, kitchen_path, frontal_path)

    # preprocess numeric features
    #numeric_features = [5,5.0,4014,92880]
    scaler_path = 'C:/Users/adria/Documents/Python/TT2/Houses-predict/models/scaler.pkl'

    numeric_tensor = preprocess_numeric_features(numeric_features, scaler_path)

    # Make prediction
    with torch.no_grad():
        prediction = model(image_tensor, numeric_tensor).item()

    print(f"Predicted house price: ${prediction:.2f}")
    return prediction