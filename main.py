import Colors
import Sharpness
import Noise
image_path = "Images_for_run/black.png"
Sharpness.sharp(image_path)
print("-----------------------------------------------------------------------------------------")
Noise.noise(image_path)
print("-----------------------------------------------------------------------------------------")
Colors.colors(image_path)
Colors.colors2(image_path)
Colors.colors_ohvat(image_path) # надо будет проверить ибо в черном изображении показывает 2 цвета
print("-----------------------------------------------------------------------------------------")
