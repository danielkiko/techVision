import Colors
import Sharpness
import Noise
import Blicks
import EffectBake
import Vinet
import AliasingMuar
import ChromeAbberation
import GeometricDistortions
import InfraRed
import LightPower
import RollingShutter


image_path = "Images_for_run/test1.png"
print("----Sharpness--------------------------------------------------------------------------------")
Sharpness.sharp(image_path)
print("----Noise------------------------------------------------------------------------------------")
Noise.noise(image_path)
print("----Colors-----------------------------------------------------------------------------------")
Colors.colors(image_path)
Colors.colors2(image_path)
Colors.colors_ohvat(image_path) # надо будет проверить ибо в черном изображении показывает 2 цвета

Blicks.detect_glare(image_path) # блики показываются в файле glare_mask.png который создаётся при работе
print("----Vignette---------------------------------------------------------------------------------")
strength, has_vignette = Vinet.detect_vignette(image_path)
print(f"Степень виньетирования: {strength:.2f}")
print(f"Виньетирование присутствует: {'да' if has_vignette else 'нет'}")
print("----AliasingMuar-----------------------------------------------------------------------------")
AliasingMuar.aliasing_score(image_path, show_spectrum=True)
AliasingMuar.detect_moire(image_path, show_result=True)  
print("----ChromeAbberation-------------------------------------------------------------------------")
ChromeAbberation.lateral_chromatic_aberration(image_path, show=True)
ChromeAbberation.longitudinal_chromatic_aberration(image_path, show=True)
print("----GeometricDistortions---------------------------------------------------------------------")
GeometricDistortions.estimate_edge_based_distortion(image_path, num_samples=2000, show=True) ##
print("----InfraRed---------------------------------------------------------------------------------")
InfraRed.ir_contamination_score(image_path,threshold=10, show=True) ##
print("----LightPower-------------------------------------------------------------------------------")
LightPower.light_power_score(image_path, grid_size=(3, 3), show=True) ##
print("----RollingShutter---------------------------------------------------------------------------")
RollingShutter.rolling_shutter_score(image_path,
                          min_line_length=50,
                          max_line_gap=5,
                          angle_margin=10,
                          show=True) ##
print("-----------------------------------------------------------------------------------------")
