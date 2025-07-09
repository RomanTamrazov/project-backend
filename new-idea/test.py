import enum

class CategoryEnum(enum.Enum):
    CONTENT = 1
    GAMING = 2
    WORK = 3

class ModelEnum(str, enum.Enum):
    INTEL = "Intel"
    AMD = "AMD"


data_input = "Ноут для видео, Asus VivoBook, Intel Core i5 13600K, None, 8 GB RAM, 1024 GB SSD"

data = data_input.split(", ")
data.pop(1)
to_exclude_1 = ["видео", "игр", "работ"]
for i in to_exclude_1:
    if i in data[0]:
        data[0] = CategoryEnum(to_exclude_1.index(i) + 1)
        break
to_exclude_2 = ["Ryzen ", "AMD Ryzen ", "Core ", "Intel Core "]
model_index = -1
model = None
for i in to_exclude_2:
    new_data_1 = data[1].removeprefix(i)
    if new_data_1 != data[1]:
        model_index = to_exclude_2.index(i)
        if model_index < 2:
            model = ModelEnum.AMD
        else:
            model = ModelEnum.INTEL
        data[1] = new_data_1
if model == ModelEnum.AMD:
    for i in reversed(data.pop(1).split(" ")):
        data.insert(1, i)
    try:
        data[1] = int(data[1])
    except Exception:
        print("Error")
    try:
        data[2] = int(data[2][:4])
    except Exception:
        print("Error")
else:
    core_version = int(data[1].removeprefix("i")[:2])
    generation = int(data[1][3:7])*10
    data.pop(1)
    data.insert(1, generation)
    data.insert(1, core_version)
# GPU
if data[3] != "None":
    data[3] = int(data[3].removeprefix("RTX ").removeprefix("GTX "))
else:
    data[3] = None
try:
    data[4] = int(data[4][:2])
except Exception:
    print("Error")
try:
    data[5] = int(data[5][:4])
except Exception:
    print("Error")
print(data)
