# Запуск лабораторной №3

Нужен Python 3.10+; сторонние пакеты и скачивание моделей не требуются. Проверено с Python 3.13.5 в Windows.

Из корня `G:\ITMO\ASR_course_labs` в PowerShell:

```powershell
$env:PYTHONUTF8 = '1'
python lab_03_ru/scripts/run_lab_03.py
python -m unittest discover -s lab_03_ru/tests -v
```

Из папки `lab_03_ru`:

```powershell
python scripts/run_lab_03.py
python -m unittest discover -s tests -v
```

Ожидается стоимость лучшего пути `3.000` и успешное выполнение 12 тестов.

## Состав результата

- `scripts/viterbi_alignment.py` — заполненные переходы, стоимости и обратные указатели.
- `scripts/bigram_lm.py` — заполненные числитель и знаменатель add-k.
- `scripts/combine_candidate_scores.py` — заполненная формула комбинирования.
- `outputs/alignment_path.csv` — путь и накопленные стоимости.
- `outputs/candidate_scores.csv` — ранжирование при λ = 0, 0,5 и 1.
- `outputs/summary.json` — сводка.
- `outputs/viterbi_trellis.svg` — тепловая карта с лучшим путём; открывается в браузере.
- `REPORT.md` — схема функций, ручные расчёты, результаты и ответы на все вопросы.
- `tests/test_lab_03.py` — проверка алгоритмов, граничных случаев и совпадения с эталонами.

Для сдачи сохраняйте структуру `lab_03_ru`, включая исходные `data/` и `outputs_reference/`: тесты используют их для проверки. Запускающий скрипт и эталоны не изменялись.
