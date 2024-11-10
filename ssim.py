import lpips
import torch
from PIL import Image
import torchvision.transforms as transforms

# Initialize the LPIPS model (AlexNet-based)
loss_fn = lpips.LPIPS(net='alex')

# Load images
def load_image(image_path):
    img = Image.open(image_path).convert('RGB')  # Convert image to RGB
    transform = transforms.Compose([
        transforms.Resize((1280,960)),  # Resize to ensure uniform input size, can be changed according to your need
        transforms.ToTensor(),          # Convert to tensor
    ])
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    return img_tensor

# Paths to your images (ground truth high-res and super-resolved images)
high_res_image_path = "C:/College Materials/4th YEAR/SEM 7/Minor Projecct/try_2/ESRGAN/results/cross_domain/og/2.jpg"  # Ground truth high-res image
sr_image_path = "C:/College Materials/4th YEAR/SEM 7/Minor Projecct/try_2/ESRGAN/results/cross_domain/super_resolution/2.png"         # Super-resolved image from SRGAN

# Load images as tensors
img1 = load_image(high_res_image_path)
img2 = load_image(sr_image_path)

# Ensure images are on the same device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img1 = img1.to(device)
img2 = img2.to(device)
loss_fn = loss_fn.to(device)

# Compute LPIPS distance (lower is better)
lpips_distance = loss_fn(img1, img2)
print(f'LPIPS Distance: {lpips_distance.item()}')