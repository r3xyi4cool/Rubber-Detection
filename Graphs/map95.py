import matplotlib.pyplot as plt

models = ['YOLOv8', 'YOLO11n', 'YOLO11s', 'Faster R-CNN']
map5095 = [0.29, 0.42, 0.43, 0.2086]

plt.figure(figsize=(8,5))
plt.bar(models, map5095)
plt.ylabel('mAP50-95')
plt.title('mAP50-95 Comparison')
plt.grid(axis='y')
plt.show()