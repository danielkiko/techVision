import Colors
import Sharpness
import Noise
import Blicks
import EffectBake
import Vinet

image_path = "Images_for_run/black.png"
Sharpness.sharp(image_path)
print("-----------------------------------------------------------------------------------------")
Noise.noise(image_path)
print("-----------------------------------------------------------------------------------------")
Colors.colors(image_path)
Colors.colors2(image_path)
Colors.colors_ohvat(image_path) # надо будет проверить ибо в черном изображении показывает 2 цвета
print("-----------------------------------------------------------------------------------------")
Blicks.detect_glare(image_path) # блики показываются в файле glare_mask.png который создаётся при работе
print("-----------------------------------------------------------------------------------------")
# EffectBake.bokeh_estimation(image_path)  # боке пока ломается
# EffectBake.detect_bokeh_shapes(image_path) # боке пока ломается
print("-----------------------------------------------------------------------------------------")
strength, has_vignette = Vinet.detect_vignette(image_path)
print(f"Степень виньетирования: {strength:.2f}")
print(f"Виньетирование присутствует: {'да' if has_vignette else 'нет'}")
