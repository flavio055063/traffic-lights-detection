# Guia de treinamento de detecção de semáforos

Este guia descreve um pipeline de transferência de aprendizado para treinar um modelo YOLO pré-treinado para classificar o estado de semáforos (vermelho, amarelo, verde, desligado) usando o conjunto de dados **Bosch Small Traffic Lights** em um notebook do Google Colab.

## 1. Preparação do ambiente no Colab

1. Acesse https://colab.research.google.com, crie um novo notebook e ative o acelerador **GPU** em *Runtime → Change runtime type*.
2. Instale dependências mínimas:

```python
!pip install --upgrade ultralytics==8.3.28 roboflow tqdm
import os, zipfile, json, shutil
```

## 2. Download do conjunto de dados Bosch

O dataset “Bosch Small Traffic Lights” é público e contém anotação de bounding boxes. No Colab, faça o download e extração:

```python
# Caminho de trabalho
BASE_DIR = '/content/traffic-lights-bosch'
os.makedirs(BASE_DIR, exist_ok=True)

# Download (~550 MB)
!wget -O $BASE_DIR/bosch-small-traffic-lights-dataset.zip \
  https://d17h27t6h515a5.cloudfront.net/topher/2017/August/59812bc9_bosch-small-traffic-lights-dataset/bosch-small-traffic-lights-dataset.zip

with zipfile.ZipFile(f"{BASE_DIR}/bosch-small-traffic-lights-dataset.zip", 'r') as z:
    z.extractall(BASE_DIR)
```

## 3. Conversão para o formato YOLO

O dataset traz anotações em JSON. Converta-as para YOLO (cx, cy, w, h normalizados) e filtre apenas as classes desejadas. Mapeamento sugerido:

- `Red` → 0
- `Yellow` → 1
- `Green` → 2
- `off` (ou `off_or_unknown`) → 3

```python
import glob
from pathlib import Path

SRC = Path(BASE_DIR) / 'rgb_train'   # ajuste se o diretório tiver outro nome
labels_json = Path(BASE_DIR) / 'train.json'
YOLO = Path(BASE_DIR) / 'yolo'
YOLO.mkdir(exist_ok=True)
(YOLO / 'images').mkdir(exist_ok=True)
(YOLO / 'labels').mkdir(exist_ok=True)

class_map = {'Red':0, 'Yellow':1, 'Green':2, 'off':3, 'off_or_unknown':3}

def yolo_line(box, w, h):
    xc = (box['x_min'] + box['x_max']) / 2 / w
    yc = (box['y_min'] + box['y_max']) / 2 / h
    bw = (box['x_max'] - box['x_min']) / w
    bh = (box['y_max'] - box['y_min']) / h
    return f"{box['cls']} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n"

with open(labels_json) as f:
    data = json.load(f)

for item in data['frames']:
    img_name = item['filename']
    img_path = SRC / img_name
    if not img_path.exists():
        continue
    w, h = item['width'], item['height']
    lines = []
    for ann in item['annotations']:
        cls_name = ann['label']
        if cls_name not in class_map:
            continue
        ann_box = {
            'x_min': ann['x_min'], 'x_max': ann['x_max'],
            'y_min': ann['y_min'], 'y_max': ann['y_max'],
            'cls': class_map[cls_name]
        }
        lines.append(yolo_line(ann_box, w, h))

    if not lines:
        continue

    target_img = YOLO / 'images' / img_name
    target_lbl = YOLO / 'labels' / (Path(img_name).stem + '.txt')
    shutil.copy(img_path, target_img)
    with open(target_lbl, 'w') as lf:
        lf.writelines(lines)
```

Em datasets muito desbalanceados, considere duplicar amostras minoritárias (ex.: `Yellow`) com *copy-paste* ou *mixup* antes de treinar.

## 4. Particionamento train/val/test

Separe imagens em `train/`, `val/` e `test/` mantendo o mesmo nome entre imagens e labels:

```python
from sklearn.model_selection import train_test_split

images = sorted((YOLO/'images').glob('*.png'))
train_imgs, valtest_imgs = train_test_split(images, test_size=0.3, random_state=42, shuffle=True)
val_imgs, test_imgs = train_test_split(valtest_imgs, test_size=0.33, random_state=42)

def move_split(split_name, split_imgs):
    (YOLO/split_name/'images').mkdir(parents=True, exist_ok=True)
    (YOLO/split_name/'labels').mkdir(parents=True, exist_ok=True)
    for img in split_imgs:
        lbl = YOLO/'labels'/f"{img.stem}.txt"
        shutil.move(img, YOLO/split_name/'images'/img.name)
        shutil.move(lbl, YOLO/split_name/'labels'/lbl.name)

move_split('train', train_imgs)
move_split('val', val_imgs)
move_split('test', test_imgs)
```

Crie o arquivo de configuração do dataset para o YOLO:

```python
data_yaml = f"""
path: {YOLO}
train: train/images
val: val/images
test: test/images

names:
  0: red
  1: yellow
  2: green
  3: off
"""
with open(f"{YOLO}/data.yaml", 'w') as f:
    f.write(data_yaml)
print(data_yaml)
```

## 5. Treinamento por transferência de aprendizado

Use um modelo pré-treinado leve (ex.: `yolov8n.pt`) e ajuste hiperparâmetros para lidar com objetos pequenos:

```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # backbone pré-treinado no COCO

results = model.train(
    data=f"{YOLO}/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,
    lr0=5e-3,
    lrf=0.1,
    weight_decay=0.0005,
    mosaic=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    box=7.5,
    cls=0.5,
    fliplr=0.0,
    flipud=0.0
)
```

**Dicas de ajustes para pequenos objetos:** aumentar `imgsz` (720–960), diminuir `stride` com `model=model.model.fuse()` somente na inferência, e usar *cosine LR* (`cos_lr=True`) para treinos mais longos.

## 6. Avaliação e análise de resultados

1. **Validação:**

```python
val_metrics = model.val(data=f"{YOLO}/data.yaml", imgsz=640)
print(val_metrics.results_dict)
```

2. **Métricas principais:** mAP@0.5, mAP@0.5:0.95, precisão, revocação por classe. Em YOLOv8, as curvas PR e a matriz de confusão são salvas em `runs/detect/train/`.
3. **Análise de erros:** use o Notebook para carregar `runs/detect/train/confusion_matrix.png`, verificar falsos positivos (ex.: confundindo vermelho/desligado) e revisar amostras mal anotadas.
4. **Test set dedicado:**

```python
test_metrics = model.val(split='test', data=f"{YOLO}/data.yaml")
print(test_metrics.results_dict)
```

5. **Amostras qualitativas:**

```python
model.predict(source=f"{YOLO}/test/images", save=True, conf=0.25, iou=0.5)
```

## 7. Salvamento e exportação do modelo

O YOLOv8 grava o melhor checkpoint em `runs/detect/train/weights/best.pt`. Para reutilizar:

```python
best_path = model.ckpt_path  # ou caminho manual
reloaded = YOLO(best_path)

# Exportações úteis
reloaded.export(format='onnx', imgsz=640)
reloaded.export(format='torchscript', imgsz=640)
```

Armazene `best.pt` (e opcionalmente ONNX/TorchScript) no Google Drive para uso posterior:

```python
from google.colab import drive
drive.mount('/content/drive')
shutil.copy(best_path, '/content/drive/MyDrive/traffic-lights/best.pt')
```

## 8. Checklist rápido

- [ ] GPU habilitada no notebook.
- [ ] Dataset convertido para YOLO e dividido em train/val/test.
- [ ] `data.yaml` aponta para os caminhos corretos.
- [ ] Treinamento concluído e métricas validadas.
- [ ] Checkpoint `best.pt` salvo no Drive e exportações (ONNX/TorchScript) geradas.
- [ ] Amostras qualitativas revisadas para cada classe (vermelho, amarelo, verde, desligado).
