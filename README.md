# <a name="_7yjucfwkzlrj"></a>**Frame Image Object Detection Tutorial using YoloV5**

This tutorial provides step-by-step instructions on how to build an ML model using YOLOv5 and train it to detect frame images in a picture. The training process will be conducted using Google Colab.

## <a name="_jnztrr1558jr"></a>***What is YOLOv5?***
*YOLOv5 is a model in the You Only Look Once (YOLO) family of computer vision models. YOLOv5 is commonly used for detecting objects. YOLOv5 comes in four main versions: small (s), medium (m), large (l), and extra large (x), each offering progressively higher accuracy rates. Each variant also takes a different amount of time to train(<https://blog.roboflow.com/yolov5-improvements-and-evaluation/>).*

**Introduction of the problem:**

We have an image (referred to as the parent image) that contains multiple frames, which are individual images. We need to extract these frames from the parent image without knowing the coordinates of these frames.

![](Aspose.Words.9422deee-3c43-4c28-bb39-7d967de9927f.002.jpeg)![](Aspose.Words.9422deee-3c43-4c28-bb39-7d967de9927f.003.png)
![](Aspose.Words.9422deee-3c43-4c28-bb39-7d967de9927f.001.jpeg)







We need to find the coordinates of the frames so that we can extract each frame individually.
## <a name="_hrc1k1flb6cr"></a>**Table of Contents**
- [Prerequisites](https://dillinger.io/#prerequisites)
- [Setup](https://dillinger.io/#setup)
- [Dataset Preparation](https://dillinger.io/#dataset-preparation)
- [Training](https://dillinger.io/#training)
- [Testing](https://dillinger.io/#testing)
- [Conclusion](https://dillinger.io/#conclusion)
## <a name="_itvtuswe53x7"></a>**Prerequisites**
- Basic knowledge of machine learning and computer vision concepts
- Python 3.x installed on your system.
- Pycharm IDE installed on your computer.
- Basic knowledge of Python programming and deep learning concepts.
- Familiarity with YOLO object detection framework.
- Google Colab Account (For Training)
- Roboflow Account  (For Image Pre-Processing)
##
## <a name="_ffcvjpou1sb8"></a><a name="_k3k3xltxip22"></a>**Setup**
1. Open the Google Colab notebook by clicking on the following link: [Google Colab](https://colab.research.google.com/).
1. Make a copy of the notebook to your Google Drive for easy access and editing.
## <a name="_ydgapy4lw8e"></a>**Dataset Preparation**
1. **Collecting Image with Frames:**

To create data using online tools like Online Photo Collage Maker, you would need to visit their website and follow their instructions. Each tool may have its own interface and features, so it's best to explore the specific tool you choose.

Alternatively, if you prefer to create images using Python code, you can utilize various libraries such as PIL (Python Imaging Library) or OpenCV. Here's an example code snippet using the PIL library:

```bash	
from PIL import Image

# Create a blank canvas for the collage
canvas_width = 800
canvas_height = 600
canvas_color = (255, 255, 255)  # White
canvas = Image.new('RGB', (canvas_width, canvas_height), canvas_color)

# Load and resize the images to be included in the collage
image1 = Image.open('image1.jpg')
image1 = image1.resize((200, 200))  # Adjust the size as needed

image2 = Image.open('image2.jpg')
image2 = image2.resize((200, 200))

# Paste the resized images onto the canvas at desired positions

canvas.paste(image1, (50, 50))
canvas.paste(image2, (350, 50))

# Save the collage as a new image file
canvas.save('collage.jpg')
```
In the above code, you can specify the canvas size, background color, and the positions of the images on the canvas. Make sure to have the PIL library installed (pip install pillow) and provide the paths to the actual image files you want to include in the collage.

Feel free to modify the code according to your specific requirements, such as adding more images or adjusting their positions.

![](Aspose.Words.9422deee-3c43-4c28-bb39-7d967de9927f.004.png)

**2. Image Data Labeling and Annotation**

1. Install LabelImg: 

```bash 
pip install LabelImg
```

2. Launch LabelImg: 

```bash 
Labelmg
```

![](Aspose.Words.9422deee-3c43-4c28-bb39-7d967de9927f.005.png)

3. Configure the annotation format: Choose PascalVOC

4. Begin labeling: Open an image from your dataset in LabelImg. Use the bounding box tool to draw boxes around the frame images you want to detect. Adjust the box position and size to accurately enclose each object. Assign the label class as "**Frame**" for each bounding box.
5. Save annotations: After annotating an image, click on "Save" to save the annotations in the YOLO format. This will generate a corresponding .xml file that contains the annotations for the image.
6. Continue labeling: Repeat the labeling process for all the images in your dataset.
87. Verify annotations: Double-check the annotations to ensure that the bounding boxes accurately enclose the frame images. Make any necessary adjustments or corrections.



**3. Image Data Pre-Processsing**

1. Sign up and create a project: Go to the Roboflow website (<https://roboflow.com/>) and sign up for an account. Once you're logged in, create a new project by clicking on the "New Project" button.

![](Aspose.Words.9422deee-3c43-4c28-bb39-7d967de9927f.006.png)


2. Upload your images: In your project, click on the "Upload" button to add your images.![](Aspose.Words.9422deee-3c43-4c28-bb39-7d967de9927f.007.png)

















![](Aspose.Words.9422deee-3c43-4c28-bb39-7d967de9927f.008.png)






















3. Split dataset : Split your dataset into training and validation subsets. 

![](Aspose.Words.9422deee-3c43-4c28-bb39-7d967de9927f.009.png)





















4. Image Transformation

In this step, we will process the training dataset of images. This will decrease training time and increase performance by applying image transformations to all the images in this dataset. Roboflow supports many methods for image transformation, such as resizing, auto orientation, and grayscale.

In this tutorial, we will apply the resize transformation with a target size of 640x640 pixels and the auto orientation transformation.

![](Aspose.Words.9422deee-3c43-4c28-bb39-7d967de9927f.010.png)

5. Image Augmentation

Augmentation performs transforms on your existing images to create new variations and increase the number of images in your dataset. This ultimately makes models more accurate across a broader range of use cases.

![](Aspose.Words.9422deee-3c43-4c28-bb39-7d967de9927f.011.png)











6. Generate Dataset: Clicking on Generate Button to generate image dataset


![](Aspose.Words.9422deee-3c43-4c28-bb39-7d967de9927f.012.png)







7. Download Dataset

![](Aspose.Words.9422deee-3c43-4c28-bb39-7d967de9927f.013.png)














![](Aspose.Words.9422deee-3c43-4c28-bb39-7d967de9927f.014.png) ![](Aspose.Words.9422deee-3c43-4c28-bb39-7d967de9927f.015.png)

















8. Upload Dataset to Google Driver for Training Process![](Aspose.Words.9422deee-3c43-4c28-bb39-7d967de9927f.017.png)
![](Aspose.Words.9422deee-3c43-4c28-bb39-7d967de9927f.016.png)

##







## <a name="_u3jd3dx2s63z"></a><a name="_hu1sevgle8xg"></a><a name="_snepr2ybbjum"></a>**Training**
1. Open Google Colab: Go to the Google Colab website at[ https://colab.research.google.com](https://colab.research.google.com/). You will be directed to the Colab home page.
2. Sign in with Google: Click on the "Sign in" button in the top-right corner of the page and sign in with your Google account. This step is necessary to save your notebooks to Google Drive and access other Google services.
3. Create a new notebook: On the Colab home page, click on the "New Notebook" button. A new notebook will be created, and you will see an empty code cell.

![](Aspose.Words.9422deee-3c43-4c28-bb39-7d967de9927f.018.png)

















5. Please check this link to see the configuration of Colab Notebook: <https://github.com/hien25661/MLTutorial/blob/main/YoloTranning.ipynb>

![](Aspose.Words.9422deee-3c43-4c28-bb39-7d967de9927f.019.png)














6. Save the trained model: After training, the model will be saved in the runs directory. You can download it or move it to Google Drive for future use.




![](Aspose.Words.9422deee-3c43-4c28-bb39-7d967de9927f.020.png)













## <a name="_uqj2s1mxf53l"></a>**Testing**
To set up a new Python project in PyCharm and use the best.pt file for YOLOv5, you can follow these steps:

1. Install PyCharm: Download and install the latest version of PyCharm from the official website (<https://www.jetbrains.com/pycharm/>).
2. Create a new project: Launch PyCharm and click on "Create New Project" or go to "File" > "New Project". Choose a location for your project and select a Python interpreter.
3. Set up a virtual environment (optional): It's recommended to use a virtual environment to isolate project dependencies. In PyCharm, go to "File" > "Settings" > "Project: [your\_project\_name]" > "Python Interpreter". Click on the gear icon and select "Add".
4. Choose a virtual environment: Select the desired virtual environment from the list or create a new one. Once selected, click "OK".
5. Install required dependencies: 

Clone repo and install [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) in a [Python>=3.7.0](https://www.python.org/) environment, including [PyTorch>=1.7](https://pytorch.org/get-started/locally/).

```bash 
pip install ultralytics
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install

```
![](Aspose.Words.9422deee-3c43-4c28-bb39-7d967de9927f.021.png)


6. Create a new Python file to write a  code (main.py)

```bash
import torch
from PIL import Image
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)  # local model
image = Image.open('test.jpg')
results = model(image, size=640)  # batch of images
results.print()
results.show()
```

![](Aspose.Words.9422deee-3c43-4c28-bb39-7d967de9927f.022.png)


## <a name="_yfw46gl3650y"></a>**Conclusion**

In this tutorial, we learned how to perform image frame detection using YOLOv5, We covered the steps from dataset preparation to training, and use saved model to predict with new data.


