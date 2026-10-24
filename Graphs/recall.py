import matplotlib.pyplot as plt

models = ['YOLOv8', 'YOLO11n', 'YOLO11s', 'Faster R-CNN']
recall = [0.59, 0.70, 0.71, 0.2540]

plt.figure(figsize=(8,5))
plt.bar(models, recall)
plt.ylabel('Recall')
plt.title('Recall Comparison')
plt.grid(axis='y')
plt.show()