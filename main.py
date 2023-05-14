# 这是一个示例 Python 脚本。
import zWell_model as zModel

z_model = zModel.res_net1.ResNetV1(
    input_shape=(28, 28, 1),
    stride=2,
    classes=10
)

print(z_model.to_keras_model())
