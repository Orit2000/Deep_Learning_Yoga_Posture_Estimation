import os
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'yoga_kaggle_dataset'))
print("Resolved dataset path:", root)
total_images = 0
counter_classes =0 
print("Counting images in each class folder:\n")

for dirpath, dirnames, filenames in os.walk(root):
    if 'annotated' in os.path.basename(dirpath).lower():
        continue  # skip folders named 'annotated'
    image_files = [f for f in filenames if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if image_files:
        counter_classes += 1
        class_name = os.path.basename(dirpath)
        count = len(image_files)
        print(f"📁 {class_name} - {count} image(s)")
        total_images += count
print("\n🧮 Total images in dataset:", total_images)
print("\n🧮 Total classes in dataset:", counter_classes)