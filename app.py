from flask import Flask, render_template, request, redirect, url_for
import torch
import os
from werkzeug.utils import secure_filename
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.models.vision_transformer import vit_b_16
from torchvision import models

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def allowed_file(file):
    filename = file.filename
    # Check extension
    if not ('.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}):
        return False
    # Check if file is a valid image
    try:
        img = Image.open(file)
        img.verify()  # Will raise an exception if not a valid image
        file.seek(0)  # Reset file pointer after verify
        return True
    except Exception:
        return False

class_names_pet = ['Non-Demented', 'Very Mild Demented', 'Mild Demented', 'Moderate Demented', 'Severe Demented']
class_names_mri = ['Non-Demented', 'Very Mild Demented', 'Mild Demented', 'Moderate Demented']

pet_model = vit_b_16(weights=None)
pet_model.heads = nn.Sequential(nn.Linear(pet_model.heads[0].in_features, 5))
pet_model.load_state_dict(torch.load('vit_pet_model.pth', map_location=torch.device('cpu')))
pet_model.eval()

mri_model = models.convnext_tiny(pretrained=False)
mri_model.classifier[2] = nn.Linear(mri_model.classifier[2].in_features, 4)
mri_model.load_state_dict(torch.load('vit_resnet_model.pth', map_location=torch.device('cpu')))
mri_model.eval()

def preprocess_image(image_path):
    input_image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(input_image).unsqueeze(0)

def predict_alzheimers(option, pet_path, mri_path):
    pet_pred = None
    mri_pred = None
    if option == 'pet' and pet_path:
        img = preprocess_image(pet_path)
        with torch.no_grad():
            output = pet_model(img)
            pet_pred = torch.softmax(output, dim=1).cpu().numpy()[0]
            pred_idx = pet_pred.argmax()
            pred_label = class_names_pet[pred_idx]
        return f"PET Prediction: {pred_label} (Probabilities: {pet_pred})"
    elif option == 'mri' and mri_path:
        img = preprocess_image(mri_path)
        with torch.no_grad():
            output = mri_model(img)
            mri_pred = torch.softmax(output, dim=1).cpu().numpy()[0]
            pred_idx = mri_pred.argmax()
            pred_label = class_names_mri[pred_idx]
        return f"MRI Prediction: {pred_label} (Probabilities: {mri_pred})"
    elif option == 'fusion' and pet_path and mri_path:
        img_pet = preprocess_image(pet_path)
        img_mri = preprocess_image(mri_path)
        with torch.no_grad():
            output_pet = pet_model(img_pet)
            output_mri = mri_model(img_mri)
            pet_pred = torch.softmax(output_pet, dim=1).cpu().numpy()[0]
            mri_pred = torch.softmax(output_mri, dim=1).cpu().numpy()[0]
            fusion_pred = (pet_pred[:4] + mri_pred) / 2  # Only fuse first 4 classes
            pred_idx = fusion_pred.argmax()
            pred_label = class_names_mri[pred_idx]
        return f"Late Fusion Prediction (MRI+PET): {pred_label} (Probabilities: {fusion_pred})"
    else:
        return "Please upload the required images for the selected option."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    option = request.form.get('option')
    pet_file = request.files.get('pet_image')
    mri_file = request.files.get('mri_image')
    pet_path = None
    mri_path = None
    result = None

    if pet_file and allowed_file(pet_file.filename):
        pet_filename = secure_filename(pet_file.filename)
        pet_path = os.path.join(app.config['UPLOAD_FOLDER'], pet_filename)
        pet_file.save(pet_path)
    if mri_file and allowed_file(mri_file.filename):
        mri_filename = secure_filename(mri_file.filename)
        mri_path = os.path.join(app.config['UPLOAD_FOLDER'], mri_filename)
        mri_file.save(mri_path)

    result = predict_alzheimers(option, pet_path, mri_path)
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, render_template, request, redirect, url_for
import torch
import os
from werkzeug.utils import secure_filename

def predict():
    option = request.form.get('option')
    pet_file = request.files.get('pet_image')
    mri_file = request.files.get('mri_image')
    pet_path = None
    mri_path = None
    result = None

    # Class names
    class_names_pet = ['Non-Demented', 'Very Mild Demented', 'Mild Demented', 'Moderate Demented', 'Severe Demented']
    class_names_mri = ['Non-Demented', 'Very Mild Demented', 'Mild Demented', 'Moderate Demented']

    if pet_file and allowed_file(pet_file.filename):
        pet_filename = secure_filename(pet_file.filename)
        pet_path = os.path.join(app.config['UPLOAD_FOLDER'], pet_filename)
        pet_file.save(pet_path)
    if mri_file and allowed_file(mri_file.filename):
        mri_filename = secure_filename(mri_file.filename)
        mri_path = os.path.join(app.config['UPLOAD_FOLDER'], mri_filename)
        mri_file.save(mri_path)

    def preprocess_image(image_path):
        from PIL import Image
        import torchvision.transforms as transforms
        input_image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return transform(input_image).unsqueeze(0)

    def predict_alzheimers(option, pet_path, mri_path):
        pet_pred = None
        mri_pred = None
        if option == 'pet' and pet_path:
            img = preprocess_image(pet_path)
            with torch.no_grad():
                output = pet_model(img)
                pet_pred = torch.softmax(output, dim=1).cpu().numpy()[0]
                pred_idx = pet_pred.argmax()
                pred_label = class_names_pet[pred_idx]
            return f"PET Prediction: {pred_label} (Probabilities: {pet_pred})"
        elif option == 'mri' and mri_path:
            img = preprocess_image(mri_path)
            with torch.no_grad():
                output = mri_model(img)
                mri_pred = torch.softmax(output, dim=1).cpu().numpy()[0]
                pred_idx = mri_pred.argmax()
                pred_label = class_names_mri[pred_idx]
            return f"MRI Prediction: {pred_label} (Probabilities: {mri_pred})"
        elif option == 'fusion' and pet_path and mri_path:
            img_pet = preprocess_image(pet_path)
            img_mri = preprocess_image(mri_path)
            with torch.no_grad():
                output_pet = pet_model(img_pet)
                output_mri = mri_model(img_mri)
                pet_pred = torch.softmax(output_pet, dim=1).cpu().numpy()[0]
                mri_pred = torch.softmax(output_mri, dim=1).cpu().numpy()[0]
                # For fusion, align to PET classes (5)
                fusion_pred = (pet_pred[:4] + mri_pred) / 2  # Only fuse first 4 classes
                pred_idx = fusion_pred.argmax()
                pred_label = class_names_mri[pred_idx]
            return f"Late Fusion Prediction (MRI+PET): {pred_label} (Probabilities: {fusion_pred})"
        else:
            return "Please upload the required images for the selected option."


    def predict_alzheimers(option, pet_path, mri_path):
        pet_pred = None
        mri_pred = None
        if option == 'pet' and pet_path:
            img = preprocess_image(pet_path)
            with torch.no_grad():
                output = pet_model(img)
                pet_pred = torch.softmax(output, dim=1).cpu().numpy()[0]
            return f"PET Prediction: {pet_pred}"
        elif option == 'mri' and mri_path:
            img = preprocess_image(mri_path)
            with torch.no_grad():
                output = mri_model(img)
                mri_pred = torch.softmax(output, dim=1).cpu().numpy()[0]
            return f"MRI Prediction: {mri_pred}"
        elif option == 'fusion' and pet_path and mri_path:
            img_pet = preprocess_image(pet_path)
            img_mri = preprocess_image(mri_path)
            with torch.no_grad():
                output_pet = pet_model(img_pet)
                output_mri = mri_model(img_mri)
                pet_pred = torch.softmax(output_pet, dim=1).cpu().numpy()[0]
                mri_pred = torch.softmax(output_mri, dim=1).cpu().numpy()[0]
                fusion_pred = (pet_pred + mri_pred) / 2
            return f"Late Fusion Prediction (MRI+PET): {fusion_pred}"
        else:
            return "Please upload the required images for the selected option."

    result = predict_alzheimers(option, pet_path, mri_path)

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
