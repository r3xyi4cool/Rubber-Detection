import matplotlib.pyplot as plt

models = ['YOLOv8', 'YOLO11n', 'YOLO11s', 'Faster R-CNN']
time_hours = [6, 8, 10, 20]

plt.figure(figsize=(8,5))
plt.bar(models, time_hours)
plt.ylabel('Training Time (Hours)')
plt.title('Training Time Comparison')
plt.grid(axis='y')
plt.show()