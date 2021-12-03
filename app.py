#import io
import json
import os

import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, redirect, url_for, send_from_directory, render_template, jsonify
                  
#from flask_uploads import UploadSet,configure_uploads,IMAGES,DATA,ALL

#from werkzeug.utils import secure_filename
#from werkzeug.datastructures import  FileStorage

                  
#UPLOAD_FOLDER = r'C:\\Users\\ACOS\\image_classifier_DNN\\uploads'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])
                  

app = Flask(__name__)
#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = models.densenet121(pretrained=True)               # Trained on 1000 classes from ImageNet
model.eval()                                              # Turns off autograd


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


img_class_map = None
mapping_file_path = 'index_to_name.json'                  # Human-readable names for Imagenet classes
if os.path.isfile(mapping_file_path):
    with open (mapping_file_path) as f:
        img_class_map = json.load(f)



# Transform input into the form our model expects
def transform_image(infile):
    input_transforms = [transforms.Resize(255),           # We use multiple TorchVision transforms to ready the image
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],       # Standard normalization for ImageNet model input
            [0.229, 0.224, 0.225])]
    my_transforms = transforms.Compose(input_transforms)
    image = Image.open(infile)                            # Open the image file
    timg = my_transforms(image)                           # Transform PIL image to appropriately-shaped PyTorch tensor
    timg.unsqueeze_(0)                                    # PyTorch models expect batched input; create a batch of 1
    return timg


# Get a prediction
def get_prediction(input_tensor):
    outputs = model.forward(input_tensor)                 # Get likelihoods for all ImageNet classes
    _, y_hat = outputs.max(1)                             # Extract the most likely class
    prediction = y_hat.item()                             # Extract the int value from the PyTorch tensor
    return prediction


# Make the prediction human-readable
def render_prediction(prediction_idx):
    stridx = str(prediction_idx)
    class_name = 'Unknown'
    if img_class_map is not None:
        if stridx in img_class_map is not None:
            class_name = img_class_map[stridx][1]
    return prediction_idx, class_name


@app.route('/', methods=['GET'])
def root():
    return jsonify({'msg' : 'Try POSTing to the /predict endpoint with an RGB image attachment'})


@app.route('/upload', methods=['GET', 'POST'])
def upload_pic_and_predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            #filename = secure_filename(file.filename)
            #file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            input_tensor = transform_image(file)
            prediction_idx = get_prediction(input_tensor)
            class_id, class_name = render_prediction(prediction_idx)
            return jsonify({'class_id': class_id, 'class_name': class_name})
    return '''
    <!doctype html>
    <title>Upload a .jpg Picture File and See If The DNN Can Guess What It Is!</title>
    <h1>Upload a .jpg Picture File and See If The DNN Can Guess What It Is!</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''
    

if __name__ == '__main__':
    app.run()
    
    
    
    
    
    
    