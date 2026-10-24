import matplotlib.pyplot as plt

models = ['YOLOv8', 'YOLO11n', 'YOLO11s', 'Faster R-CNN']
map50 = [0.62, 0.79, 0.80, 0.3654]

plt.figure(figsize=(8,5))
plt.bar(models, map50)
plt.ylabel('mAP50')
plt.title('mAP50 Comparison')
plt.grid(axis='y')
plt.show()