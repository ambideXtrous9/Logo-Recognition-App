from torchvision import transforms
import config

resize = transforms.Resize(size=(config.WIDTH,config.HEIGHT))

nrml = transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])  # Imagenet standards

transform_norm = transforms.Compose([
            transforms.ToTensor(),
            resize,
            nrml])