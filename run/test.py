# Import necessary libraries
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import cv2
from pathlib import Path
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from collections import Counter
import csv
from rich.console import Console
from rich.table import Table
import tqdm

# Load the first model
model1 = tf.keras.models.load_model('osnow_model.h5')
model1.summary()

# Load the second model
model2 = tf.keras.models.load_model('plusis_model.h5')
model2.summary()

# Define the folder path for input images and the output CSV file path
image_folder = '/data/run/pic'  # Path to the folder containing images
output_file = '/data/run/output.csv'  # Output file for storing results

# Get a list of image files in the specified folder
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')]

results = []  # List to store results

# Function to analyze images covered with snow
def analyze_su(image_path):
    # Define color categories and their corresponding ranges
    color_categories = {
        "Snow": [(80, 80, 80), (255, 255, 255)],
        # More color categories can be added as needed
    }

    # Create a dictionary to store color occurrences
    colors = Counter() 
    colors["Snow"] += 1

    # Open the image and convert it to RGB format
    im = Image.open(image_path).convert("RGB")

    # Get the dimensions of the image
    width, height = im.size

    # Iterate over each pixel in the image
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            r, g, b = im.getpixel((x, y))

            # Determine the color category of the pixel
            for category, (min_rgb, max_rgb) in color_categories.items():
                if min_rgb[0] <= r <= max_rgb[0] and min_rgb[1] <= g <= max_rgb[1] and min_rgb[2] <= b <= max_rgb[2]:
                    colors[category] += 1
                    break

    # Calculate the percentage of each color's occurrence
    total_pixels = width * height
    resultss = []
    for category, count in colors.items():
        percentage = count / total_pixels
        resultss.append(f"{category}Cover: {percentage:.2%}")

    return resultss
def analyze_sd(image_path):
    # 定义颜色类别和对应的颜色范围
    color_categories = {
        "Snow": [(136, 136, 136), (255, 255, 255)],
        # 可以根据需要添加更多的颜色类别和对应的颜色范围
    }

    # 创建一个字典，用于存储颜色出现的次数
    colors = Counter() 
    colors["Snow"] +=1

    # 打开图像并将其转换为RGB格式
    im = Image.open(image_path).convert("RGB")

    # 获取图像的尺寸
    width, height = im.size

    # 遍历图像的每个像素
    for x in range(width):
        for y in range(height):
            # 获取像素的RGB值
            r, g, b = im.getpixel((x, y))

            # 判断像素的颜色类别
            for category, (min_rgb, max_rgb) in color_categories.items():
                if min_rgb[0] <= r <= max_rgb[0] and min_rgb[1] <= g <= max_rgb[1] and min_rgb[2] <= b <= max_rgb[2]:
                    colors[category] += 1
                    break

    # 计算每种颜色的出现次数的百分比
    total_pixels = width * height
    resultss = []
    for category, count in colors.items():
        percentage = count / total_pixels
        resultss.append(f"{category}Cover: {percentage:.2%}")

    return resultss
def analyze_nu(image_path):
    # 定义颜色类别和对应的颜色范围
    color_categories = {
        "Snow": [(120, 120, 120), (255, 255, 255)],
        # 可以根据需要添加更多的颜色类别和对应的颜色范围
    }

    # 创建一个字典，用于存储颜色出现的次数
    colors = Counter() 
    colors["Snow"] +=1
    # 打开图像并将其转换为RGB格式
    im = Image.open(image_path).convert("RGB")

    # 获取图像的尺寸
    width, height = im.size

    # 遍历图像的每个像素
    for x in range(width):
        for y in range(height):
            # 获取像素的RGB值
            r, g, b = im.getpixel((x, y))

            # 判断像素的颜色类别
            for category, (min_rgb, max_rgb) in color_categories.items():
                if min_rgb[0] <= r <= max_rgb[0] and min_rgb[1] <= g <= max_rgb[1] and min_rgb[2] <= b <= max_rgb[2]:
                    colors[category] += 1
                    break

    # 计算每种颜色的出现次数的百分比
    total_pixels = width * height
    resultss = []
    for category, count in colors.items():
        percentage = count / total_pixels
        resultss.append(f"{category}Cover: {percentage:.2%}")

    return resultss
def analyze_nd(image_path):
    # 定义颜色类别和对应的颜色范围
    color_categories = {
        "Snow": [(208, 208, 208), (255, 255, 255)],
        # 可以根据需要添加更多的颜色类别和对应的颜色范围
    }

    # 创建一个字典，用于存储颜色出现的次数
    colors = Counter() 
    colors["Snow"] +=1
    # 打开图像并将其转换为RGB格式
    im = Image.open(image_path).convert("RGB")

    # 获取图像的尺寸
    width, height = im.size

    # 遍历图像的每个像素
    for x in range(width):
        for y in range(height):
            # 获取像素的RGB值
            r, g, b = im.getpixel((x, y))

            # 判断像素的颜色类别
            for category, (min_rgb, max_rgb) in color_categories.items():
                if min_rgb[0] <= r <= max_rgb[0] and min_rgb[1] <= g <= max_rgb[1] and min_rgb[2] <= b <= max_rgb[2]:
                    colors[category] += 1
                    break

    # 计算每种颜色的出现次数的百分比
    total_pixels = width * height
    resultss = []
    for category, count in colors.items():
        percentage = count / total_pixels
        resultss.append(f"{category}Cover: {percentage:.2%}")

    return resultss

# Function to write results to a CSV file
def write_to_csv(output_file, results):
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_file', 'corvered with snow'])
        for image_file, result1, result2, square in results:
            square_percentage = float(square[0].split(':')[1].strip('%')) / 100
            writer.writerow([image_file, square_percentage])

# Create a table for displaying results using the rich library
table = Table(show_header=True, header_style="bold magenta")
table.add_column("IMG")
table.add_column("S/N")
table.add_column("UP/DOWN")
table.add_column("Cover")
console = Console()

# Use tqdm for progress bar
with tqdm.tqdm(total=len(image_files), ncols=80) as pbar:
    # Iterate over each image file
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)  # Read the image
        test_img = cv2.resize(np.array(image), (128, 128))  # Resize the image
        test = test_img.astype('float32')  # Adjust the format
        test = test / 255.0  # Normalize
        test = test[tf.newaxis, ...]  # Add dimension

        # Use the first model for prediction
        predict1 = model1.predict(test)
        if abs(predict1[0] - 0.5) < 0.000000001:
            result1 = "can not predict"
        else:
            if predict1[0] > 0.495:
                result1 = "snow"
            else:
                result1 = "no"

        # Use the second model for prediction
        predict2 = model2.predict(test)
        if abs(predict2[0] - 0.8) < 0.0000000001:
            result2 = "can not predict"
        else:
            if predict2[0] > 0.555:
                result2 = "up"
            else:
                result2 = "down"

        # Choose the appropriate analysis function based on the model predictions
        if result1 == "snow" and result2 == "up":
            square = analyze_su(image_path)
        elif result1 == "snow" and result2 == "down":
            square = analyze_sd(image_path)
        elif result1 == "no" and result2 == "up":
            square = analyze_nu(image_path)
        else:
            square = analyze_nd(image_path)

        # Append the results to the list
        results.append((image_file, result1, result2, square))
        
        # Add a row to the table and display using rich library
        table.add_row(image_file, str(result1), str(result2), str(square))
        console.print(table)
        pbar.update(1)

    # Write the results to the output file
    write_to_csv(output_file, results)
