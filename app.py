from flask import Flask, render_template, request
import torch
from torchvision import models, transforms
from PIL import Image
import io
import torch.nn as nn
import torch.nn.functional as F

app = Flask(__name__)
print("beginning")
# Define the CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.flatten(x)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
model = CNNModel()

# Load your PyTorch model and any necessary preprocessing
model.load_state_dict(torch.load('my_model.pth'))
model.eval()

# Define the transformation for input images
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_image(image_path):
    input_image = Image.open(image_path)
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output = model(input_batch)
    _, predicted = torch.max(output, 1)
    predicted_class = ''
    if predicted.item() == 0:
        predicted_class = 'Apple_scab'
    elif predicted.item() == 1:
        predicted_class = 'Black_rot'
    elif predicted.item() == 2:
        predicted_class = 'Cedar_apple_rust'
    else:
        predicted_class = 'healthy'
    return predicted_class

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            image_data = file.read()
            image_path = io.BytesIO(image_data)
            predicted_class = predict_image(image_path)
            return render_template('index.html', result_label=predicted_class)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
