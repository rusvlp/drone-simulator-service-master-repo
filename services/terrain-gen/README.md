# terrain-gen

Генерация terrain heightmap и 3D-меша из спутникового снимка.

## Быстрый старт

```bash
python main.py photo.jpg output/
```

## Синтаксис

```
python main.py [INPUT] OUTPUT [опции]
```

| Аргумент | Описание |
|---|---|
| `INPUT` | Спутниковый снимок (JPG, PNG, TIFF, …). Не нужен при `--demo`. |
| `OUTPUT` | Папка для результатов (создаётся автоматически). |

## Опции

### Метод генерации

| Флаг | Значения | По умолчанию | Описание |
|---|---|---|---|
| `--method` | `segment`, `analyze`, `color` | `segment` | Метод построения heightmap |
| `--backbone` | `segformer`, `deeplabv3` | `segformer` | Нейросеть для `--method segment` |
| `--device` | `auto`, `cpu`, `cuda`, `mps` | `auto` | Устройство для нейросети |

- **segment** — нейросегментация (SegFormer ADE20K или DeepLabV3 VOC) + roughness + relief. Лучшее качество.
- **analyze** — правила на основе HSV + roughness + illumination relief. Без зависимости от torch.
- **color** — только спектральная классификация по цвету.

### Реальный DEM (SRTM)

```bash
python main.py photo.jpg output/ --bbox LAT_MIN LON_MIN LAT_MAX LON_MAX
```

| Флаг | По умолчанию | Описание |
|---|---|---|
| `--bbox LAT_MIN LON_MIN LAT_MAX LON_MAX` | — | Загрузить реальные высоты SRTM вместо классификации |
| `--dem-grid N` | `64` | Разрешение сетки запросов (NxN). `32` ≈ 12 с, `64` ≈ 45 с |

### Тайлинг (большие изображения)

Автоматически включается когда max(W, H) > 1024 px.

| Флаг | По умолчанию | Описание |
|---|---|---|
| `--tile-size PX` | `512` | Размер тайла в пикселях |
| `--overlap PX` | `64` | Перекрытие между соседними тайлами |

### Постобработка

| Флаг | По умолчанию | Описание |
|---|---|---|
| `--smooth SIGMA` | `3.0` | Гауссово сглаживание (пиксели) |
| `--sea-level LEVEL` | `0.08` | Доля высоты [0–1], считающаяся водой |

### Текстура OBJ

| Флаг | Значения | По умолчанию | Описание |
|---|---|---|---|
| `--texture` | `photo`, `classified`, `terrain` | `photo` | Источник текстуры |

- **photo** — спутниковый снимок с повышением резкости (PNG, лossless).
- **classified** — семантическая карта: вода=синий, трава=зелёный, земля=коричневый.
- **terrain** — false-color карта высот (matplotlib terrain).

### 3D-меш

| Флаг | По умолчанию | Описание |
|---|---|---|
| `--scale-z FACTOR` | `0.3` | Вертикальный масштаб меша относительно XY. `0.05`–`0.1` для плоского рельефа |
| `--y-up` | выкл. | Y-up оси (XZ плоскость земли) для Unity / Godot. По умолчанию Z-up (Blender) |
| `--no-obj` | выкл. | Не экспортировать OBJ |

### Demo-режим

```bash
python main.py --demo output/
python main.py --demo output/ --demo-size 1024
```

## Выходные файлы

| Файл | Описание |
|---|---|
| `{name}_heightmap.png` | Grayscale heightmap (8-bit, 0=вода, 255=пик) |
| `{name}_terrain.png` | False-color карта высот |
| `{name}_terrain.obj` | 3D-меш (макс. 512px по длинной стороне) |
| `{name}_terrain.mtl` | Материал с привязкой текстуры |
| `{name}_satellite.png` | Текстура (при `--texture photo`) |
| `{name}_classified.png` | Текстура (при `--texture classified`) |

## Примеры

```bash
# Базовый запуск (нейросеть, спутниковая текстура, CUDA)
python main.py photo.jpg output/

# Плоский рельеф для дрон-симулятора (Unity)
python main.py photo.jpg output/ --scale-z 0.05 --y-up

# Максимальное сглаживание для городского снимка
python main.py city.jpg output/ --smooth 15 --scale-z 0.05 --y-up

# Реальный DEM по координатам (Москва)
python main.py photo.jpg output/ --bbox 55.50 37.30 56.00 38.00

# Без GPU
python main.py photo.jpg output/ --device cpu

# Только heightmap без OBJ
python main.py photo.jpg output/ --no-obj

# Тест без входного изображения
python main.py --demo output/
```
