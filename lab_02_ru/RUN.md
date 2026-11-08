# Воспроизведение лабораторной №2

Команды для PowerShell из корня `G:\ITMO\ASR_course_labs`.
Аудио обрабатывается локально. Интернет нужен при установке пакетов и первоначальной загрузке модели.

## 1. Окружение

```powershell
python -m venv .venv-asr-lab02
.\.venv-asr-lab02\Scripts\python.exe -m pip install -r lab_02_ru/requirements.txt
```

Проверенные точные версии для Windows / Python 3.13 сохранены в `requirements-lock.txt`.
Для воспроизведения окружения можно установить этот файл вместо `requirements.txt`.
Ограничение `av<19` из исходных требований сохранено.

## 2. Параметры сессии

```powershell
$env:PYTHONUTF8 = '1'
$env:MPLBACKEND = 'Agg'
$env:MPLCONFIGDIR = "$PWD\.cache\matplotlib"
$env:NUMBA_CACHE_DIR = "$PWD\.cache\numba"
$env:HF_HOME = "$PWD\.cache\huggingface"
```

## 3. Проверка данных и графики без ASR

```powershell
.\.venv-asr-lab02\Scripts\python.exe lab_02_ru/scripts/run_lab_02.py --skip-asr
```

Этот этап проверяет 5 исходных WAV, создаёт 35 вариантов и основные графики.

## 4. Однократная загрузка модели

```powershell
.\.venv-asr-lab02\Scripts\python.exe -c "from faster_whisper.utils import download_model; print(download_model('tiny.en', output_dir='lab_02_ru/models/tiny.en', revision='0d3d19a32d3338f10357c0889762bd8d64bbdeba'))"
```

Каталог `models/` исключён из Git. При повторном запуске используйте уже загруженную модель.

## 5. Полный эксперимент с локальной моделью

```powershell
$env:HF_HUB_OFFLINE = '1'
.\.venv-asr-lab02\Scripts\python.exe lab_02_ru/scripts/run_lab_02.py --model-dir "$PWD\lab_02_ru\models\tiny.en"
.\.venv-asr-lab02\Scripts\python.exe lab_02_ru/scripts/summarize_experiment.py
```

Повторный запуск перезаписывает WAV, гипотезы, время декодирования, CSV и графики.
RTF зависит от текущей нагрузки компьютера; одинаковый seed аугментаций не делает измеренное время неизменным.

Если WAV и основные графики уже подготовлены, достаточно запустить только распознавание и оценку:

```powershell
.\.venv-asr-lab02\Scripts\python.exe lab_02_ru/scripts/run_local_asr.py --conditions lab_02_ru/outputs/augmented/conditions.jsonl --manifest lab_02_ru/data/manifest.jsonl --out lab_02_ru/outputs/raw_asr.jsonl --model-dir "$PWD\lab_02_ru\models\tiny.en"
.\.venv-asr-lab02\Scripts\python.exe lab_02_ru/scripts/score_robustness.py --input lab_02_ru/outputs/raw_asr.jsonl --out-dir lab_02_ru/outputs/results
.\.venv-asr-lab02\Scripts\python.exe lab_02_ru/scripts/summarize_experiment.py
```

`conditions.jsonl` хранит абсолютные пути: после переноса проекта на другой компьютер нужно сначала заново выполнить этап 3 или полный запуск на этапе 5.

## Результаты и сдача

- `REPORT.md` — ответы на все 9 вопросов с измерениями и гипотезами распознавания.
- `outputs/raw_asr.jsonl` — исходные 35 результатов декодирования.
- `outputs/results/per_item_results.csv` — WER, CER, RTF и тексты каждого варианта.
- `outputs/results/aggregate_results.csv` — агрегированные показатели 7 условий.
- `outputs/results/duration_comparison.csv` — длительности при изменении скорости.
- `outputs/figures/` — сравнения waveform/log-Mel, SpecAugment и метрик.
- `outputs/experiment_metadata.json` — сведения о проверенном запуске и модели.
- `scripts/`, `requirements*.txt`, `RUN.md` — код и воспроизведение.
- `data/` и `outputs/augmented/` — исходные и преобразованные WAV. При передаче исходных данных сохраняйте `data/DATA_LICENSE.md` и манифест.

Окружение `.venv-asr-lab02`, кэши и веса модели в посылку преподавателю включать не требуется: веса загружаются командой выше. Отчёт содержит результаты конкретного запуска; после нового запуска текстовые таблицы отчёта следует сверить с новыми CSV.
