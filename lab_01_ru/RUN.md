# Запуск лабораторной №1 в Windows / PowerShell

Все команды выполняются из корня `G:\ITMO\ASR_course_labs`.
Активация окружения не требуется: вызывается его Python напрямую.

## Первичная установка

```powershell
python -m venv .venv-asr-lab01
.\.venv-asr-lab01\Scripts\python.exe -m pip install -r lab_01_ru/requirements.txt
```

Точные версии проверенного окружения сохранены в `requirements-lock.txt`.
Для воспроизведения на Windows с Python 3.13 вместо исходного списка можно установить:

```powershell
.\.venv-asr-lab01\Scripts\python.exe -m pip install -r lab_01_ru/requirements-lock.txt
```

## Генерация и анализ

```powershell
$env:PYTHONUTF8 = '1'
$env:MPLBACKEND = 'Agg'
$env:MPLCONFIGDIR = "$PWD\.cache\matplotlib"
$env:NUMBA_CACHE_DIR = "$PWD\.cache\numba"
.\.venv-asr-lab01\Scripts\python.exe lab_01_ru/scripts/make_lab_audio.py --out-dir lab_01_ru/data/wav
.\.venv-asr-lab01\Scripts\python.exe lab_01_ru/scripts/lab_01_audio_inspection.py --audio-dir lab_01_ru/data/wav --out-dir lab_01_ru/outputs
```

Первый запуск анализа может быть медленнее из-за компиляции функций Numba.
Ожидаемый результат для исходных пяти примеров: 5 WAV, таблица `audio_summary.csv`,
25 индивидуальных PNG и один общий `feature_comparison_contact_sheet.png`.
Повторный запуск перезаписывает сгенерированные результаты.

## Материалы для сдачи

- `REPORT.md` — заполненный разбор A–G с таблицами и иллюстрациями.
- `scripts/` — исходный код генерации и анализа.
- `data/wav/` — пять синтетических примеров.
- `outputs/` — результаты расчёта и графики.
- `requirements.txt`, `requirements-lock.txt` и этот файл — воспроизведение.

Сохраняйте структуру каталогов: ссылки на изображения в отчёте относительные.
Виртуальное окружение, кэши и `.idea` отправлять не требуется.
Обязательная часть не требует личной аудиозаписи; запись 5–10 секунд — дополнительное задание.
