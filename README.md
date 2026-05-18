# Применение алгоритмов искусственного интеллекта для анализа и типизации абиотических компонент подводного ландшафта

Целью данной работы является разработка универсальной методики типизации подводных ландшафтов, которая опирается на ключевые абиотические компоненты подводного ландшафта. Предлагаемая методика реализуется в два этапа: 

1. Регрессионное предсказание гранулометрических свойств донного грунта по данным гидролокации бокового обзора и батиметрии
    
2. Кластеризация площадных признаков (рельефа дна, предсказанных значений гранулометрии, структурных сейсмо-геологических атрибутов) для выделения типов подводного ландшафта

**Ссылка на датасет**, собранный в рамках работы по данным гидроакустических и геологических исследований Белого моря для задачи предсказания гранулометрических свойств донного грунта: https://drive.google.com/file/d/1lRbDuutPcVRaj3AnGJekI0T9QsQ2CJyo/view?usp=sharing
```text
Датасет содержит 900+, собранных для 45 независимых станций пробоотбора с использованием данных ГЛБО, снятых в разные годы на одном и том же полигоне 
```

Основные проблемы работы с геологическими и гидроакустическими данными - трудности со сбором достаточно большого набора данных, сильное изменение условий (смещение домена) при использовании данных с разных полигонов - решаются в данной работе путём проработки методики сбора данных и через обоснование выбора формата данных и типов используемых признаков (атрибутов)

---

## Структура проекта
В данном проекте представлены основные результаты работы, вылодена лучшая модель по результатам экспериментов (R² 0.7)

```text
AI-Based-Analysis-and-Classification-of-Abiotic-Seafloor-Components/
├── README.md
├── requirements.txt
├── pyproject.toml
├── models/
│   └── RF_patch_level_weighted.joblib    # модель для предсказания гранулометрии по гидроакустическим данным 
├── notebooks/
│   └── Clustering_Agglomerative.ipynb    # реализация иерархической кластеризации
├── scripts/
│   ├── create_dataset.py       # создание обучающего датасета (станции пробоотбора + гидроакустические данные) в формате NPZ
│   ├── inspect_npz.py          # проверка NPZ и визуализация случайных патчей
│   ├── train_rf.py             # обучение RandomForest
│   └── predict_geotiff.py      # предсказание гранулометрических свойств грунта по GeoTIFF ГЛБО + GeoTIFF батиметрии
└── src/seabed_rf/
    ├── features.py             # расчёт признаков патча и агрегаций
    ├── modeling.py             # RF pipeline, метрики, CorrelationFilter
    └── raster_utils.py         # чтение растров, маски, slope, станции
```

---

## Предсказание гранулометрических свойств донных грунтов по гидроакустическим данным

В проекте приведён рабочий пайплайн для предсказания средневзвешенного размера зерна донных осадков `Mz_phi` по данным гидролокации бокового обзора (ГЛБО) и батиметрии. Итоговая модель — `RandomForestRegressor`. 

Проект включает: подготовку датасета в формате `.npz` по данным ГЛБО, батиметрии и пробоотбора; обучение модели RF; инференс по растрам ГЛБО и батиметрии (в формате GeoTIFF) 

## 1. Входные данные

### Растры

Для обучения и инференса используются:

1. **ГЛБО** — GeoTIFF, один или несколько проходов.
2. **Батиметрия** — GeoTIFF `float`, в той же системе координат. Для `create_dataset.py` ожидается та же сетка, что у растров ГЛБО. Для инференса `predict_geotiff.py` умеет ресемплировать батиметрию на сетку ГЛБО.

Для корректного инференса предсказание выполняется только там, где одновременно валидны:

```text
ГЛБО ∩ батиметрия ∩ рассчитанный уклон
```

Для ГЛБО по умолчанию используется маска значений, соответствующая финальной подготовке датасета:

```text
sonar >= 57 и sonar < 255; значение 255 считается nodata/белым фоном
```

Пороги можно изменить через аргументы `--sonar-valid-min`, `--sonar-valid-max`, `--sonar-extra-nodata`.

### Таблица с данными геологического пробоотбора

Файл станций должен содержать колонки:

```text
Station    X    Y    Mz
2021_1K    506118.76    7378013.12    3.870463
2021_2K    506261.95    7378348.60    4.633757
...
```

Поддерживаются табуляция, пробелы, запятая и точка с запятой. Колонка `Mz` — целевая переменная `Mz_phi` для станции. При инференсе она нужна только для оценки ошибки в контрольных точках.
Mz_phi = -log₂(d), 					(1)
где Mz_phi - средневзвешенный размер зерна в φ-шкале, d - средневзвешенный размер зерна в миллиметрах

---

## 2. Подготовка обучающего датасета 

Команда:

```bash
python scripts/create_dataset.py \
  --sonar data/sonar_1.tif data/sonar_2.tif data/sonar_3.tif \
  --bathy data/bathymetry.tif \
  --stations data/stations.txt \
  --output Dataset/patches_with_features_dataset.npz \
  --patch-size 64 \
  --min-valid-ratio 0.80
```

Что делает скрипт:

1. читает станции и растры;
2. рассчитывает уклон по батиметрии с учётом размера пикселя;
3. строит маску валидности ГЛБО, батиметрии и уклона;
4. для каждой станции и каждого растра ГЛБО извлекает патчи `PATCH_SIZE × PATCH_SIZE`;
5. по умолчанию берёт 9 положений патча: центр и восемь сдвигов;
6. отбрасывает патчи с долей валидных пикселей меньше `--min-valid-ratio`;
7. сохраняет результат в `.npz`.

Если нужно собрать датасет только по центральным патчам:

```bash
python scripts/create_dataset.py ... --center-only
```

Если нужно задать размер сдвига вручную:

```bash
python scripts/create_dataset.py ... --shift 8
```

### Содержимое NPZ

Файл `.npz` содержит следующие массивы:

| Ключ | Размерность | Описание |
|---|---:|---|
| `sonar_patches` | `(N, H, W)` | Патчи ГЛБО |
| `bathy_patches` | `(N, 2, H, W)` | Каналы батиметрии: `depth`, `slope` |
| `valid_masks` | `(N, H, W)` | Совместная маска валидности ГЛБО и батиметрии |
| `targets` | `(N,)` | Значение `Mz_phi` для станции |
| `scalar_features` | `(N, 4)` | Простые дополнительные признаки патча: mean/std depth, mean/std slope |
| `stations` | `(N,)` | Названия станций |
| `coords` | `(N, 2)` | Координаты центра патча |
| `source_rasters` | `(N,)` | Из какого растра ГЛБО получен патч |
| `shift_xy_px` | `(N, 2)` | Сдвиг патча в пикселях относительно станции |
| `valid_ratios` | `(N,)` | Доля валидных пикселей внутри патча |

Здесь `N` — число сохранённых патчей. Важно: `N` может быть больше числа станций геологического пробоотбора, но независимая разметка остаётся привязанной к самим станциям. То есть, все патчи одной станции имеют одно и то же значение целевой переменной для предсказания.

### Проверка NPZ

```bash
python scripts/inspect_npz.py \
  --dataset Dataset/patches_with_features_dataset.npz \
  --out-png Dataset/random_patches.png
```

Скрипт печатает:

- список ключей NPZ;
- размерности массивов;
- число уникальных станций;
- число уникальных растров ГЛБО;
- статистику числа патчей на станцию;
- визуализацию случайных патчей: ГЛБО, глубина, уклон, valid mask.

---

## 3. Обучение модели

Команда для полного запуска:

```bash
python scripts/train_rf.py \
  --dataset Dataset/patches_with_features_dataset.npz \
  --out-dir Dataset/rf_final_training_outputs \
  --tune \
  --tune-iter 50 \
  --n-splits 5 \
  --n-repeats 20
```

Для быстрой проверки можно уменьшить число повторов и итераций тюнинга:

```bash
python scripts/train_rf.py \
  --dataset Dataset/patches_with_features_dataset.npz \
  --out-dir Dataset/rf_final_training_outputs \
  --tune \
  --tune-iter 10 \
  --n-repeats 3
```

### Процесс обучения

1. Загрузка `.npz`.
2. Нормализация ГЛБО отдельно по каждому растру `source_raster` (robust z-score):

   ```text
   sonar_norm = (sonar - median_raster) / IQR_raster
   ```

3. Вычисление признаков:

   - статистики интенсивности ГЛБО;
   - статистики нормализованного ГЛБО;
   - статистики глубины;
   - статистики уклона;
   - multi-scale признаки центральных окон;
   - center-vs-context признаки;
   - морфологические признаки батиметрии: roughness, local relief, laplacian;
   - текстурные признаки ГЛБО (GLCM): `contrast`, `dissimilarity`, `entropy`.

4. Формирование представления данных:

   | Эксперимент | Единица строки | Назначение |
   |---|---|---|
   | `RF_patch_level_weighted` | один патч | модель для будущего картирования по скользящему окну |
   | `RF_station_raster_level` | station × raster | проверка устойчивости между проходами ГЛБО |
   | `RF_station_level` | одна станция | лучшая оценка качества |

5. Оценка модели через кросс-валидацию:

   ```text
   5 folds × 20 repeats
   ```

   Стратификация выполняется по квантильным бинам целевой переменной `Mz_phi`.

7. Сохранение финальных моделей, обученные на всех доступных данных.

## 4. Построение предсказания по GeoTIFF (инференс)

Команда на примере основного полигона, использовавшегося при обучении модели:

```bash
python scripts/predict_geotiff.py \
  --sonar test_data/0_Sonar_data_Nilma.tif \
  --bathy test_data/1_Bathy_data_Nilma.tif \
  --model Dataset/rf_final_training_outputs/models/RF_patch_level_weighted.joblib \
  --features Dataset/rf_final_training_outputs/models/RF_patch_level_weighted_feature_columns.json \
  --output Dataset/inference_outputs/Nilma_predicted_Mz_phi.tif \
  --stations test_data/Nilma_stations.txt \
  --station-output Dataset/inference_outputs/Nilma_station_predictions.csv \
  --preview Dataset/inference_outputs/Nilma_prediction_preview.png \
  --patch-size 64 \
  --stride 16 \
  --fill-stride-blocks
```

Команда на примере тестового полигона:

```bash
python scripts/predict_geotiff.py \
  --sonar test_data/2_Sonar_data_Rugozero.tif \
  --bathy test_data/2_Bathy_data_Rugozero.tif \
  --model Dataset/rf_final_training_outputs/models/RF_patch_level_weighted.joblib \
  --features Dataset/rf_final_training_outputs/models/RF_patch_level_weighted_feature_columns.json \
  --output Dataset/inference_outputs/Rugozero_predicted_Mz_phi.tif \
  --stations test_data/Rugozero_stations.txt \
  --station-output Dataset/inference_outputs/Rugozero_station_predictions.csv \
  --preview Dataset/inference_outputs/Rugozero_prediction_preview.png \
  --patch-size 64 \
  --stride 16 \
  --fill-stride-blocks
```

### Процесс инференса

1. Читает ГЛБО и батиметрию;
2. Если сетки не совпадают, батиметрия ресемплируется на сетку ГЛБО;
3. Рассчитывает уклон по батиметрии;
4. Строит совместную маску валидности:

   ```text
   sonar_valid & bathy_valid & slope_valid
   ```

5. Нормализует ГЛБО по входному растру;
6. Двигает окно `patch-size × patch-size` с шагом `stride`;
7. Для каждого валидного окна считает тот же набор признаков, что при обучении;
8. Применяет модель `RF_patch_level_weighted`;
9. Записывает GeoTIFF с предсказанным `Mz_phi`;
10. Если передан файл станций, отдельно предсказывает значения в точках станций и считает ошибки

### Важные параметры инференса

| Параметр | Значение по умолчанию | Значение |
|---|---:|---|
| `--patch-size` | `64` | размер локального окна в пикселях |
| `--stride` | `16` | шаг скользящего окна |
| `--min-valid-ratio` | `0.80` | минимальная доля валидных пикселей в окне |
| `--fill-stride-blocks` | off | если включено, предсказание заполняет блок вокруг центра окна |
| `--sonar-valid-min` | `57` | нижний порог валидных значений ГЛБО |
| `--sonar-valid-max` | `255` | верхний порог; используется условие `< 255` |
| `--sonar-extra-nodata` | `255` | дополнительные коды nodata |

Если на тестовом полигоне есть зоны, где батиметрия валидна, но ГЛБО содержит белый фон, эти зоны не будут предсказаны, если правильно заданы пороги ГЛБО

### Выходы инференса на примере основного и тестового полигона

```text
inference_outputs/
├── Nilma_predicted_Mz_phi.tif
├── Nilma_predicted_Mz_phi.json
├── Nilma_station_predictions.csv
├── Nilma_prediction_preview.png
├── Rugozero_predicted_Mz_phi.tif
├── Rugozero_predicted_Mz_phi.json
├── Rugozero_station_predictions.csv
└── Rugozero_prediction_preview.png
```

В station CSV для каждой станции указывается:

| Колонка | Описание |
|---|---|
| `Station` | название станции |
| `X`, `Y` | координаты |
| `y_true` | значение `Mz` из файла станций |
| `y_pred` | предсказание модели |
| `error` | `y_pred - y_true` |
| `abs_error` | абсолютная ошибка |
| `status` | статус предсказания |

Возможные статусы:

| Статус | Описание |
|---|---|
| `predicted` | предсказание успешно рассчитано |
| `outside_raster_bounds` | координаты не удалось преобразовать в пиксели растра |
| `outside_raster_index` | точка вне массива растра |
| `center_invalid` | центр станции невалиден по совместной маске ГЛБО+батиметрии |
| `patch_window_outside_raster` | полный патч выходит за край растра |
| `low_valid_ratio` | в патче слишком много nodata |

Метрики по станциям считаются только для строк со статусом `predicted`.

---

## 5. Как работать с датасетом 

Пример загрузки:

```python
import numpy as np

data = np.load("Dataset/patches_with_features_dataset.npz", allow_pickle=True)
print(list(data.keys()))

sonar = data["sonar_patches"].astype("float32")          # (N, H, W)
bathy = data["bathy_patches"].astype("float32")          # (N, 2, H, W)
valid_masks = data["valid_masks"].astype("float32")      # (N, H, W)
y = data["targets"].astype("float32")                    # (N,)
stations = data["stations"].astype(str)                  # (N,)
source_rasters = data["source_rasters"].astype(str)       # (N,)
shift_xy_px = data["shift_xy_px"].astype("int32")         # (N, 2)
valid_ratios = data["valid_ratios"].astype("float32")     # (N,)
```

Важно:

```text
патч ≠ независимое наблюдение
станция = независимое наблюдение
```

Поэтому для честной оценки модели нельзя случайно делить на train/test. Нужно делить именно станции:

```python
import numpy as np

unique_stations = np.unique(stations)
# делите unique_stations, а затем выбирайте патчи по маске np.isin(stations, train_stations)
```

Если использовать patch-level модель, нужно компенсировать разное число патчей на станцию весами:

```python
import pandas as pd

df = pd.DataFrame({"station": stations})
counts = df.groupby("station")["station"].transform("count")
sample_weight = 1.0 / counts.values
```
---

## 6. Типовой полный сценарий

```bash
# 1. Подготовка датасета
python scripts/create_dataset.py \
  --sonar data/sonar_*.tif \
  --bathy data/bathy.tif \
  --stations data/stations.txt \
  --output Dataset/patches_with_features_dataset.npz

# 2. Проверка датасета
python scripts/inspect_npz.py \
  --dataset Dataset/patches_with_features_dataset.npz \
  --out-png Dataset/random_patches.png

# 3. Обучение модели (Random Forest)
python scripts/train_rf.py \
  --dataset Dataset/patches_with_features_dataset.npz \
  --out-dir Dataset/rf_final_training_outputs \
  --tune

# 4. Предсказание 
python scripts/predict_geotiff.py \
  --sonar test_data/0_Sonar_data_Nilma.tif \
  --bathy test_data/1_Bathy_data_Nilma.tif \
  --model Dataset/rf_final_training_outputs/models/RF_patch_level_weighted.joblib \
  --features Dataset/rf_final_training_outputs/models/RF_patch_level_weighted_feature_columns.json \
  --output Dataset/inference_outputs/Nilma_predicted_Mz_phi.tif \
  --stations test_data/Nilma_stations.txt \
  --station-output Dataset/inference_outputs/Nilma_station_predictions.csv \
  --preview Dataset/inference_outputs/Nilma_prediction_preview.png \
  --fill-stride-blocks
```

---

## 7. Что сильно влияет на качество

1. корректность исходной разметки (в данные грнаулометрического анализа геологических проб не попадают крупные размерности - галька, глыбы и т.п., их необходимо учитывать, например, по данным подводной видеосъёмки);
2. наличие артефактов обработки на данных ГЛБО и батиметрии (особенно влияет на этапе инференса);
3. корректность задания маски валидности (nodata, фоновые значения)

При использовании выложенной модели на участке, сильно отличающемся от использованного в обучении (например, глубины более 80 м), модель может потребовать дообучения или калибровки на имеющихся станциях нового полигона. Модель наиболее репрезентативна для условий мелководного шельфа Арктики. 
