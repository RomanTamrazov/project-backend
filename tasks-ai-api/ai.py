import enum
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

label_map = {"✅": 0, "⚠️": 1, "❌": 2, "❓": 3}
label_rev_map = {v: k for k, v in label_map.items()}

class CategoryEnum(enum.Enum):
    CONTENT = 1
    GAMING = 2
    WORK = 3

class ModelEnum(str, enum.Enum):
    INTEL = "Intel"
    AMD = "AMD"

def cpu_performance_score(cpu_text: str) -> int:
    cpu_text = cpu_text.lower()
    score = 0
    if "apple m" in cpu_text:
        if "max" in cpu_text:
            score = 10
        elif "pro" in cpu_text:
            score = 9
        else:
            score = 8
    elif "intel" in cpu_text:
        if "i9" in cpu_text:
            score = 9
        elif "i7" in cpu_text:
            score = 7
        elif "i5" in cpu_text:
            score = 5
        elif "i3" in cpu_text:
            score = 3
        elif "xeon" in cpu_text:
            score = 8
        else:
            score = 2
    elif "ryzen" in cpu_text or "amd" in cpu_text:
        if "9" in cpu_text:
            score = 9
        elif "7" in cpu_text:
            score = 7
        elif "5" in cpu_text:
            score = 5
        elif "3" in cpu_text:
            score = 3
        else:
            score = 2
    else:
        score = 1
    return score

def gpu_performance_score(gpu_text: str) -> int:
    gpu_text = gpu_text.lower()
    if "rtx 4090" in gpu_text:
        return 95
    elif "rtx 4080" in gpu_text:
        return 85
    elif "rtx 4070" in gpu_text:
        return 75
    elif "rtx 4060" in gpu_text:
        return 70
    elif "rtx 4050" in gpu_text:
        return 65
    elif "rtx 3080" in gpu_text:
        return 75
    elif "rtx 3070" in gpu_text:
        return 70
    elif "rtx 3060" in gpu_text:
        return 65
    elif "rtx 3050" in gpu_text:
        return 60
    elif "rtx 2050" in gpu_text:
        return 55
    elif "gtx 1660" in gpu_text or "gtx 1650" in gpu_text:
        return 50
    elif "gtx 1050" in gpu_text:
        return 40
    elif any(x in gpu_text for x in ["iris", "uhd", "vega", "встроенная", "без дискретной", "integrated gpu", "intel hd", "intel iris"]):
        return 10
    elif gpu_text.strip() == "":
        return 0
    else:
        return 5

def extract_features(row):
    data = [
        row["query"],
        row["title"],
        row["cpu"],
        row["ram"],
        row["storage"],
        row["gpu"]
    ]

    task_keywords = {
        "видео": CategoryEnum.CONTENT,
        "игр": CategoryEnum.GAMING,
        "работ": CategoryEnum.WORK
    }
    task_class = 0
    for k, v in task_keywords.items():
        if k in data[0].lower():
            task_class = v.value
            break

    cpu_text = data[2].lower()
    cpu_brand = 1 if "intel" in cpu_text else 2 if "ryzen" in cpu_text or "amd" in cpu_text else 0
    cpu_gen = 0
    try:
        cpu_gen = int(re.findall(r"\d{4,5}", cpu_text)[0])
    except Exception:
        pass

    cpu_perf = cpu_performance_score(cpu_text)

    ram_num = 0
    try:
        ram_num = int(re.findall(r"\d+", data[3])[0])
    except Exception:
        pass

    storage_num = 0
    try:
        storage_num = int(re.findall(r"\d+", data[4])[0])
    except Exception:
        pass

    gpu_power = gpu_performance_score(data[5].lower())

    return [task_class, cpu_brand, cpu_gen, cpu_perf, ram_num, storage_num, gpu_power]
data = [
    ("ноутбук для работы с видео", "Apple MacBook Pro 16 M2 Pro",
     "Apple M2 Pro", "32GB", "1TB SSD", "macOS, Final Cut", "✅"),

    ("ноутбук для монтажа видео", "MSI Creator Z17",
     "Intel Core i9", "64GB", "2TB SSD", "RTX 3080", "✅"),

    ("ноутбук для видео", "ASUS ROG Zephyrus G14",
     "Ryzen 9 5900HS", "32GB", "1TB SSD", "RTX 3060", "✅"),

    ("ноутбук для видеомонтажа", "Dell XPS 15",
     "Intel Core i7", "32GB", "1TB SSD", "GTX 1650 Ti", "✅"),

    ("монтаж видео", "Lenovo Legion 5",
     "AMD Ryzen 7", "16GB", "512GB SSD", "RTX 3060", "✅"),

    ("ноутбук для работы с видео", "Apple MacBook Pro M3 Max",
     "Apple M3 Max", "64GB", "2TB SSD", "macOS", "✅"),

    ("видеомонтаж ноутбук", "Razer Blade 16",
     "Intel i9-13950HX", "32GB", "2TB SSD", "RTX 4080", "✅"),

    ("монтаж видео", "ASUS ProArt Studiobook",
     "Intel Xeon", "64GB", "4TB SSD", "RTX A5000", "✅"),

    ("редактирование видео", "Gigabyte Aero 16",
     "Intel i7", "32GB", "1TB SSD", "RTX 4070", "✅"),

    ("обработка видео", "MSI Creator M16",
     "Intel i7-13700H", "32GB", "1TB SSD", "RTX 4060", "✅"),
    
    ("ноутбук для игр", "ASUS ROG Strix G15",
     "Ryzen 9 7945HX", "32GB", "1TB SSD", "RTX 4080", "✅"),
    
    ("игровой ноутбук", "MSI Raider GE76",
     "Intel Core i9-12900HK", "64GB", "2TB SSD", "RTX 3080 Ti", "✅"),
    
    ("игровой ноут", "Lenovo Legion 7i",
     "Intel Core i7", "32GB", "1TB SSD", "RTX 3070", "✅"),
    
    ("ноутбук для игр", "HP Omen 16",
     "AMD Ryzen 7", "16GB", "1TB SSD", "RTX 3060", "✅"),
    
    ("ноутбук для работы", "Dell Latitude 7430",
     "Intel Core i7", "32GB", "1TB SSD", "Intel Iris Xe", "✅"),
    
    ("офисный ноутбук", "Lenovo ThinkPad X1 Carbon",
     "Intel Core i7", "16GB", "1TB SSD", "Intel Iris Xe", "✅"),
    
    ("ноутбук для работы", "Dell Inspiron 14 5000",
     "Intel Core i5-1135G7", "16GB", "512GB SSD", "Intel Iris Xe", "✅"),
    
    ("рабочий ноутбук", "HP ProBook 450 G8",
     "Intel Core i7-1165G7", "16GB", "1TB SSD", "Intel UHD Graphics", "✅"),
    
    ("ноутбук для офиса", "Lenovo ThinkBook 15 G2",
     "AMD Ryzen 5 5600U", "16GB", "512GB SSD", "AMD Radeon Graphics", "✅"),
    
    ("игровой ноутбук", "MSI Katana GF66",
     "Intel Core i7-11800H", "16GB", "512GB SSD", "RTX 3060", "✅"),
    
    ("ноут для игр", "ASUS TUF Dash F15",
     "Intel Core i7-11370H", "16GB", "1TB SSD", "RTX 3070", "✅"),
    
    ("игровой ноутбук", "Gigabyte Aorus 15P",
     "Intel Core i7-11800H", "32GB", "1TB SSD", "RTX 3080", "✅"),
    
    ("ноутбук для игр", "Lenovo Legion 5",
     "AMD Ryzen 7 5800H", "16GB", "512GB SSD", "RTX 3060", "✅"),

    ("ноутбук для работы с видео", "HP Pavilion 15",
     "Intel Core i5", "16GB", "512GB SSD", "Intel Iris Xe", "⚠️"),
    
    ("ноутбук для работы", "ASUS ExpertBook B1",
     "Intel Core i3-1115G4", "8GB", "256GB SSD", "Intel UHD Graphics", "⚠️"),
    
    ("офисный ноутбук", "Acer Aspire 5",
     "Intel Core i5-1035G1", "8GB", "256GB SSD", "Intel UHD Graphics", "⚠️"),
    
    ("игровой ноут", "HP Victus 16",
     "Intel Core i5-11400H", "16GB", "512GB SSD", "RTX 3050 Ti", "⚠️"),
    
    ("игровой ноутбук", "Acer Nitro 5",
     "Intel Core i5", "16GB", "512GB SSD", "RTX 3050", "⚠️"),
    
    ("ноутбук для работы", "HP EliteBook",
     "Intel Core i5", "16GB", "512GB SSD", "UHD Graphics", "⚠️"),
    
    ("рабочий ноут", "ASUS ExpertBook",
     "Intel Core i3", "8GB", "256GB SSD", "", "⚠️"),
    
    ("игровой ноутбук", "ASUS TUF Gaming",
     "Ryzen 5", "8GB", "512GB SSD", "GTX 1650", "⚠️"),

    ("монтаж видео", "Acer Swift 3",
     "Ryzen 5", "8GB", "512GB SSD", "встроенная графика", "⚠️"),

    ("редактирование видео", "ASUS VivoBook 15",
     "Intel Core i3", "16GB", "256GB SSD", "", "⚠️"),

    ("видеомонтаж ноутбук", "Dell Inspiron 14",
     "Intel Core i5", "8GB", "512GB SSD", "UHD Graphics", "⚠️"),

    ("ноутбук для графики и видео", "HP Envy x360",
     "Ryzen 7", "16GB", "1TB SSD", "Vega 8", "⚠️"),

    ("ноутбук для видео", "Acer Aspire 5",
     "Intel i5-1135G7", "8GB", "512GB SSD", "Intel Iris Xe", "⚠️"),

    ("видеомонтаж", "HP ProBook 450",
     "Intel i5", "16GB", "256GB SSD", "без дискретной графики", "⚠️"),

    ("монтаж видео", "Lenovo IdeaPad 5",
     "Ryzen 5 5500U", "8GB", "512GB SSD", "Vega 7", "⚠️"),

    ("обработка видео", "ASUS VivoBook 14",
     "Intel i3-1215U", "16GB", "512GB SSD", "", "⚠️"),

    ("редактирование видео", "Dell Inspiron 15",
     "Intel i5", "8GB", "256GB SSD", "UHD Graphics", "⚠️"),
    
    ("ноутбук для игр", "HP Pavilion Gaming",
     "Intel Core i5", "8GB", "256GB SSD", "GTX 1050", "❌"),
    
    ("ноутбук для работы", "Lenovo IdeaPad 1",
     "Intel Celeron N4020", "4GB", "128GB SSD", "встроенная графика", "❌"),
    
    ("ноутбук для игр", "ASUS VivoBook 15",
     "Intel Pentium Gold 6405U", "4GB", "256GB SSD", "Intel UHD Graphics", "❌"),
    
    ("бюджетный ноутбук", "Acer Aspire 3",
     "AMD A4-9120", "4GB", "500GB HDD", "встроенная графика", "❌"),
    
    ("офисный ноутбук", "Acer TravelMate",
     "Intel Pentium Gold", "8GB", "256GB SSD", "UHD Graphics", "❌"),
    
    ("ноут для работы", "Lenovo V15",
     "AMD Athlon Silver", "4GB", "128GB SSD", "", "❌"),
    
    ("игровой ноут", "Lenovo IdeaPad Gaming 3",
     "AMD Ryzen 5", "8GB", "512GB SSD", "без дискретной графики", "❌"),

    ("ноутбук для видео", "ASUS VivoBook X512DA",
     "Ryzen 3 3200U", "8GB", "1TB HDD", "Vega 3", "❌"),

    ("монтаж видео", "Lenovo IdeaPad 3",
     "Intel Celeron", "4GB", "500GB HDD", "", "❌"),

    ("ноутбук для видео", "Acer Aspire 1",
     "Intel Pentium N5000", "4GB", "128GB eMMC", "", "❌"),

    ("видеомонтаж ноутбук", "Chuwi HeroBook",
     "Intel Atom", "4GB", "64GB SSD", "", "❌"),

    ("обработка видео", "HP Stream",
     "AMD A6", "4GB", "32GB eMMC", "", "❌"),

    ("ноутбук для работы с видео", "Lenovo V14",
     "Intel Core i3 1005G1", "4GB", "1TB HDD", "UHD Graphics", "❌"),

    ("монтаж видео", "HP 255 G7",
     "AMD A6-9225", "4GB", "500GB HDD", "Radeon R4", "❌"),

    ("обработка видео", "ASUS X543MA",
     "Intel Celeron N4000", "4GB", "500GB HDD", "", "❌"),
    
    ("ноутбук для работы", "HP Stream 11",
     "Intel Celeron N4000", "4GB", "32GB eMMC", "встроенная графика", "❌"),
    
    ("ноутбук для видео", "Apple MacBook Air 2013",
     "Intel Core i5", "4GB", "128GB SSD", "Intel HD Graphics 5000", "❌"),
    
    ("ноутбук для работы", "Apple MacBook Pro 13 Early 2012",
     "Intel Core i5", "4GB", "500GB HDD", "Intel HD Graphics 4000", "❌"),
    
    ("ноутбук для учебы", "Apple MacBook Air 2011",
     "Intel Core i7", "4GB", "256GB SSD", "Intel HD Graphics 3000", "❌"),
    
    ("ноутбук для видео", "Apple MacBook Pro 15 Mid 2010",
     "Intel Core i7", "4GB", "500GB HDD", "NVIDIA GeForce GT 330M", "❌"),
    
    ("ноутбук для работы", "Apple MacBook 12 Early 2016",
     "Intel Core M3", "8GB", "256GB SSD", "Intel HD Graphics 515", "❌"),
    
    ("ноутбук для работы", "Apple MacBook Air 2012",
     "Intel Core i7", "4GB", "128GB SSD", "Intel HD Graphics 4000", "❌"),
    
    ("ноутбук для видео", "Apple MacBook Pro 13 Mid 2010",
     "Intel Core 2 Duo", "4GB", "320GB HDD", "NVIDIA GeForce 320M", "❌"),
    
    ("ноутбук для работы", "Apple MacBook Air 2010",
     "Intel Core 2 Duo", "2GB", "128GB SSD", "Intel GMA 950", "❌"),
    
    ("ноутбук для учебы", "Apple MacBook 13 Early 2009",
     "Intel Core 2 Duo", "2GB", "160GB HDD", "Intel GMA 950", "❌"),
    
    ("ноутбук для видео", "Apple MacBook Pro 17 Late 2008",
     "Intel Core 2 Duo", "4GB", "320GB HDD", "NVIDIA GeForce 9600M GT", "❌"),
    
    ("ноутбук для игр", "Dell Inspiron 15 3000",
     "Intel Pentium Gold 5405U", "4GB", "128GB SSD", "Intel UHD Graphics 610", "❌"),
    
    ("бюджетный ноутбук", "ASUS X543MA",
     "Intel Celeron N3350", "4GB", "500GB HDD", "встроенная графика", "❌"),
    ("ноутбук для работы с видео", "Apple MacBook Air M1",
     "Apple M1", "16GB", "512GB SSD", "Integrated GPU", "⚠️"),
    
    ("ноутбук для работы", "Apple MacBook Air M2",
     "Apple M2", "16GB", "1TB SSD", "Integrated GPU", "⚠️"),

    ("ноутбук для работы", "Apple MacBook Air 2020",
     "Intel Core i5", "8GB", "256GB SSD", "Intel Iris Plus", "⚠️"),
    
    ("ноутбук для учебы", "Apple MacBook Air 2017",
     "Intel Core i5", "8GB", "128GB SSD", "Intel HD Graphics 6000", "⚠️"),

    ("ноутбук для видео", "Apple MacBook Air 2013",
     "Intel Core i5", "4GB", "128GB SSD", "Intel HD Graphics 5000", "❌"),
    
    ("ноутбук для работы", "Apple MacBook Air 2012",
     "Intel Core i7", "4GB", "128GB SSD", "Intel HD Graphics 4000", "❌"),
    
    ("ноутбук для работы", "Apple MacBook Air 2011",
     "Intel Core i7", "4GB", "256GB SSD", "Intel HD Graphics 3000", "❌"),
    
    ("ноутбук для учебы", "Lenovo V14 IGL",
     "Intel Celeron N4020", "4GB", "256GB SSD", "Intel UHD Graphics 600", "❌"),
    
    ("ноутбук для работы", "Acer Extensa 15",
     "Intel Pentium Silver N5030", "4GB", "128GB SSD", "встроенная графика", "❌"),

    ("видеомонтаж ноутбук", "Acer Extensa 15",
     "Intel Pentium Silver", "8GB", "1TB HDD", "", "❌"),

    ("ноутбук для видео", "Prestigio SmartBook",
     "Intel Atom", "4GB", "64GB eMMC", "", "❌"),

    ("ноутбук для видео", "Ноутбук DEXP Aquilon",
     "", "", "", "", "❓"),

    ("видеомонтаж ноутбук", "Irbis NB76",
     "", "", "", "", "❓"),

    ("ноутбук для работы", "DEXP Atlas",
     "Core i5", "RAM неизвестна", "512GB SSD", "", "❓"),

    ("обработка видео", "DNS Office",
     "Неизвестный процессор", "8GB", "1TB HDD", "", "❓"),

    ("монтаж", "Ноутбук Noname",
     "", "", "", "", "❓"),
    ("ноутбук для видео", "Ноутбук Unknown Brand X1", "", "", "", "", "❓"),

    ("монтаж видео", "Ноутбук Generic Model 2023", "Unknown CPU", "Unknown RAM", "Unknown Storage", "Unknown GPU", "❓"),

    ("редактирование видео", "Ноутбук XYZ Series", "", "8GB", "", "Integrated Graphics", "❓"),

    ("видеомонтаж ноутбук", "Old Laptop Model 2015", "Intel Core 2 Duo", "4GB", "256GB HDD", "Intel Integrated", "❓"),

    ("обработка видео", "Ноутбук Без Названия", "", "", "", "", "❓"),
]
df = pd.DataFrame(data, columns=["query", "title", "cpu", "ram", "storage", "gpu", "label"])
df["label"] = df["label"].map(label_map)

features = df.apply(extract_features, axis=1, result_type='expand')
X = features.values
y = df["label"]

X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_val)
print(classification_report(y_val, y_pred, target_names=list(label_map.keys())))
def predict_from_input(query, title, cpu, ram, storage, gpu):
    row = {
        "query": query,
        "title": title,
        "cpu": cpu,
        "ram": ram,
        "storage": storage,
        "gpu": gpu
    }
    feats = extract_features(row)
    pred = clf.predict([feats])[0]
    return label_rev_map[pred]
import joblib
joblib.dump(clf, "itmo_model.pkl")
