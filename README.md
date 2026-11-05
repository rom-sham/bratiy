# Orthodox Qwen UA

Відтворюваний текстовий pipeline **Qwen3.5 Base → CPT LoRA → merge → SFT LoRA → merge → evaluation**. Братій — сервітор православний проповідник; демонстрації є оригінальною синтетикою, а не цитатами святих чи офіційними текстами Warhammer.

## Що перевірено

Перевірка 2026-11-05: офіційний checkpoint — [`Qwen/Qwen3.5-0.8B-Base`](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base), Apache-2.0, revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`. Цей SHA закріплено в конфігураціях. Це multimodal checkpoint; pipeline використовує офіційний **Qwen3_5ForCausalLM** і його `text_config`, щоб навчати текстову частину без vision encoder. Перевірка завантаження ваг VLM → text виконана на маленькій моделі з тією самою архітектурою; відсутні чи несумісні текстові ваги викликають помилку.

Локально є **Python 3.12.10, PyTorch 2.12.0+cpu, CUDA відсутня**, Transformers 5.22.1, PEFT 0.21.0, TRL 1.17.1, Datasets 5.1.0. Python 3.11 є цільовим середовищем встановлення; його виконання на цьому комп'ютері не перевірено. Залежності встановлено в `../.venv` із доступом до вже встановлених системних пакетів. Для чистого розгортання створіть окреме середовище за інструкцією нижче.

Перевірено синтаксис, pytest, оригінальний tokenizer і маски для всіх 120 демонстрацій; короткий CPU smoke охоплює CPT, checkpoint/resume, обидва merge, SFTTrainer та generation evaluation на **випадковій мініатюрній Qwen3.5**. Окремий тест справжньої 0.8B запускається через `scripts/test_training.py`; його поточний статус — у `outputs/test_*/status.json`. Повне навчання не запускається автоматично. BF16, CUDA, реальний VRAM та bitsandbytes потребують перевірки на цільовій GPU.

## Встановлення

З каталогу `orthodox-qwen`, у середовищі Python 3.11:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Для GPU встановіть CUDA-збірку PyTorch тієї самої версії за [офіційною інструкцією PyTorch](https://pytorch.org/get-started/locally/) та перевірте `torch.cuda.is_available()`. Звичайне встановлення пакета `torch` саме по собі не гарантує CUDA. `bitsandbytes` — необов'язкове GPU-доповнення: без нього автоматично обирається AdamW PyTorch. Перед вибором 8-bit виконується реальний пробний крок оптимізатора.

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m pytest -q
python -m compileall -q src scripts tests
python scripts/smoke_test.py
```

Повторний smoke запускайте в новому каталозі: `python scripts/smoke_test.py --output outputs/smoke_second`. Модель smoke має приблизно 21 тисячу параметрів і тестовий tokenizer. Вона не оцінює якість Братія. Кожний smoke містить лише кілька кроків.

## Джерела та обмеження завантаження

| ID | Джерело | Результат малої перевірки | Схема / ліцензія metadata |
|---|---|---|---|
| kobza | [podarok/kobza-cleaned-ua](https://huggingface.co/datasets/podarok/kobza-cleaned-ua) | 30 записів | text, source, length; cc-by-4.0 |
| malyuk | [lang-uk/malyuk](https://huggingface.co/datasets/lang-uk/malyuk) | 30 записів | id, compound_id, text; unknown |
| kulish | [k-mktr/ukrainian_kulish_puliuy_uk](https://huggingface.co/datasets/k-mktr/ukrainian_kulish_puliuy_uk) | 30 записів | text, book, chapter, verse та інші; unknown |
| ogienko | [k-mktr/ukrainian_ogienko_uk](https://huggingface.co/datasets/k-mktr/ukrainian_ogienko_uk) | 30 записів | аналогічна біблійна схема; unknown |
| warhammer | [vizn3r/warhammer40k-lore](https://huggingface.co/datasets/vizn3r/warhammer40k-lore) | 30 записів | фактичні поля в inspection.json; mit у card |
| drama | [alexshynkarenk0/ukrdramacore-demo](https://huggingface.co/datasets/alexshynkarenk0/ukrdramacore-demo) | HTTP 401, пропущено | недоступний без відповідного доступу |
| wiki_dialogs | [lapa-llm/wiki-instruction-dialogs](https://huggingface.co/datasets/lapa-llm/wiki-instruction-dialogs) | 30 записів | conversations або instruction/input/output; cc-by-4.0 |
| ae_lib | [Православні тексти ae-lib](https://ae-lib.org.ua/_orthodoxy.htm) | 4 документи з 5 посилань | HTML; unknown; один файл перевищив 2 MB |

Ліцензія в card — збережена заява джерела, а не перевірка прав на всі вкладені тексти. `unknown` не замінюється вигаданою ліцензією. Перевірте умови конкретних матеріалів перед поширенням корпусу чи моделі. Власні 120 діалогів і 12 evaluation-запитів позначено `synthetic: true`, `source` та `CC0-1.0`.

Кожне HF-джерело має власну конфігурацію: `repo`, `revision`, `group`, `limit`, `max_bytes`, необов'язкові `config`, `split`, `text_column`. Доступні SHA закріплено; точні metadata і схема зберігаються в `outputs/download_report.json`. Дані читаються streaming, без завантаження всього корпусу. `limit` обмежує записи, `max_bytes` — обсяг збережених JSON записів; HTTP range/read-ahead Parquet може передати більше байтів, ніж цей ліміт. Це не жорстка мережева квота.

`ae_lib` обходить лише перші `limit` посилань на тому самому домені та обмежує розмір кожної HTTP-відповіді. Невдале джерело журналюється, інші продовжують роботу. Якщо жодне віддалене джерело не дало записів, команда завершується з кодом 1 після збереження звіту та локальних імпортів. Вихідні файли невдалих джерел попереднього запуску прибираються, щоб не підміняти результат застарілими даними.

Локальні UTF-8 TXT, HTML, EPUB і текстові PDF кладіть у `data/local/`. EPUB читається в порядку spine з перевіркою розпакованого розміру; для сканованих PDF потрібен окремий OCR. `local_license` та `local_group` задаються в YAML. Немає мережевих викликів під час імпорту модулів. Для приватних джерел використовуйте стандартну змінну `HF_TOKEN`, не записуйте токен у конфігурацію.

## Підготовка даних

```bash
python scripts/01_download.py --limit 30
python scripts/02_inspect.py
python scripts/03_clean.py
python scripts/04_prepare_cpt.py
python scripts/05_prepare_sft.py --create-demo
python scripts/06_validate_data.py
```

Без `--limit` діють невеликі явні ліміти з `configs/data.yaml`. Для наступного більшого завантаження змініть їх і повторіть етапи. Перезапуск детермінований для однакових джерел, версій і seed; JSONL записуються атомарно.

Очищення прибирає HTML/script/style, URL, порожні, пошкоджені, надмірно короткі й довгі тексти; NFC зберігає українські літери та історичну орфографію. Дедуплікація використовує нормалізований casefold/NFKC-ключ без зміни самого тексту. Provenance, source revision, ID, ліцензія й додаткові біблійні поля зберігаються. `langdetect` із фіксованим seed визначає мову документа та позначає ймовірні російськомовні речення. Це евристика: архаїчна мова потребує перегляду; релігійні тексти з непевною мовою зберігаються з `requires_language_review`, а не відкидаються через лексику.

CPT за замовчуванням містить 90% загальних і 10% релігійних документів. Warhammer виключено. Пропорції — **за документами**, не за токенами. Спочатку створюються незалежні splits, потім train змішується без дублювання або oversampling. Якщо групи немає, потрібно додати дані або явно змінити ваги. Розміри округлюються вниз; фактичні пропорції й точна кількість токенів оригінального Qwen tokenizer з EOS є в `cpt_data_report.json`. `--skip-tokenizer` дозволяє offline-підготовку; у цьому режимі токенний обсяг позначається `null`, а не оцінюється за символами.

SFT має всі вісім категорій. Ваги задають 40% комедійного roleplay (`orthodox_roleplay` + `humorous_sermons`), 25% духовних бесід (`biblical_explanations` + `orthodox_traditions` + `patristic_wisdom`), 20% фантастики й Warhammer, 15% звичайної розмови. `--create-demo` створює 120 **окремо написаних** демонстрацій. Без прапорця читаються наявні JSONL у `data/sft_sources`. `--import-dialogues` додатково конвертує завантажені діалоги зі збереженням provenance; вони потрапляють лише до `everyday_conversation` й не переписуються автоматично в образі Братія.

Дублікат розмови визначається без system prompt. Перевіряються порожні повідомлення, чергування ролей, завершені пари, довжина, ID, provenance, повторення й перетин запитів між splits та незалежним evaluation-набором. Дані одного `split_group` залишаються в одному split. Навчальний tokenizer додатково перевіряє токенну довжину: завеликі SFT-діалоги викликають помилку замість прихованого обрізання відповіді.

Мала реальна перевірка дала CPT **34 train / 8 validation / 8 evaluation**, train — 15 985 токенів з EOS. Із 120 SFT-демонстрацій після splits і undersampling суміші використовується **52 / 12 / 12**. Решта не дублюється для штучного збільшення обсягу; оригінальні 120 прикладів залишаються в `data/sft_sources/demo.jsonl`. Це демонстраційний обсяг, який не обґрунтовує 1000 кроків реального CPT.

Для масштабування до 5000–10000 прикладів додавайте нові перевірені JSONL з тими самими полями, різними сценаріями, довжинами й багатокроковими розмовами. Оригінальні вигадані Warhammer-сцени відокремлюйте від канонічних фактів. Перевіряйте якість і розподіли перед кожним збільшенням. Розмноження тих самих відповідей не додає навчальної цінності.

## Навчання й merge

Початкові параметри CPT/SFT повністю задано в `configs/cpt.yaml` і `configs/sft.yaml`: LoRA rank 16, alpha 16/32, dropout 0.05, sequence 1024, batch 1, accumulation 8, LR 1e-5/5e-5, CPT 1000 optimizer steps, SFT 2 epochs. Gradient checkpointing увімкнено, BF16 обирається лише на сумісній GPU; інакше використовується FP32. FP32 потребує більше пам'яті.

Перед запуском зменште бюджет під реальний обсяг даних. Перевірка вільного VRAM — мінімальний поріг, **не гарантія**, що довжина 1024 або 2048 поміститься. `device: auto` обирає CUDA GPU, якщо вона доступна PyTorch, інакше CPU. `--device cpu` примусово обирає процесор, `--device cuda` вимагає доступну CUDA. CPU використовує FP32 і AdamW PyTorch; `cpu_threads` задає кількість потоків. Спершу запустіть preflight і однокрокову перевірку:

```bash
python scripts/07_train_cpt.py --preflight
python scripts/07_train_cpt.py --smoke
```

`--smoke` для numbered trainer завантажує справжню 0.8B, бере до чотирьох документів і виконує один крок в окремому `*_smoke` каталозі. Це відрізняється від offline `scripts/smoke_test.py` на маленькій моделі. На цьому комп'ютері numbered GPU smoke не запускався.

Повний порядок після успішної перевірки GPU та даних:

```bash
python scripts/07_train_cpt.py
python scripts/09_merge_adapter.py --config configs/cpt.yaml
python scripts/08_train_sft.py --preflight
python scripts/08_train_sft.py --smoke
python scripts/08_train_sft.py
python scripts/09_merge_adapter.py --config configs/sft.yaml
python scripts/10_evaluate.py
```

Merge CPT **обов'язковий перед SFT**. SFT за замовчуванням читає лише `outputs/cpt_merged` і не підміняє його початковою моделлю. Merge бере точну базу й revision з `run_manifest.json`, виконує `safe_merge=True` на CPU у FP32 та зберігає tokenizer; потрібна достатня RAM. Готовий каталог призначення не перезаписується. Adapter, optimizer/scheduler state, tokenizer, loss, state й environment зберігаються для відновлення.

```bash
python scripts/07_train_cpt.py --resume outputs/cpt_adapter/checkpoint-100
python scripts/08_train_sft.py --resume outputs/sft_adapter/checkpoint-100
```

Для точного resume зберігайте дані й конфігурацію; збільшення загального бюджету після завершеного scheduler не є тотожним безперервному запуску. При OOM зберігається `oom.json`, навчання припиняється з порадою зменшити довжину. Відновлюйте останній **завершений** checkpoint; незбережені кроки не відновлюються. Пік CUDA allocated memory журналюється разом із training logs.

## Сумісність шаблону й assistant loss

Реалізація користується **офіційним chat template оригінального tokenizer** з `enable_thinking=False`; спеціальні ChatML-токени не дописуються вручну. Той самий tokenizer/template зберігається з adapter і merged моделлю.

Офіційний шаблон не зобов'язаний мати Jinja `generation` markers. Тому pipeline не покладається на автоматичне виправлення TRL: він обчислює межі відповідей за `apply_chat_template` для префіксів розмови та створює явні `labels`, де system/user/header/padding мають `-100`. Кінцівка assistant, включно з маркером завершення від tokenizer, навчається. Перевірка prefix-preservation викликає помилку, якщо шаблон переписує попередні репліки. Це особливо важливо для нестандартних багатокрокових thinking-діалогів.

TRL `SFTTrainer` отримує готові `input_ids`, `labels`, `dataset_kwargs={skip_prepare_dataset: true}` та collator, який зберігає маски. `assistant_only_loss=False` в його конфігурації означає, що **вбудована** повторна маска вимкнена; loss уже обмежено assistant нашими labels. Збереження labels перевіряється після побудови trainer. `packing: false` підтримується й є початковим режимом; `true` явно відхиляється, бо неперевірене packing не повинно зіпсувати маски. CPT ріже кожен документ на блоки окремо.

Встановлений Transformers 5.22.1 прибрав `warmup_ratio`; код перераховує YAML-частку в `warmup_steps`. Attention backend — звичайний `eager`; без спеціальних Mamba/linear-attention kernels використовуються правильні, але повільні PyTorch реалізації. Перевірена LoRA охоплює full-attention q/k/v/o та MLP gate/up/down, а не неперевірені матриці linear-attention.

Unsloth не є залежністю основного backend: його [офіційна Qwen3.5 інструкція](https://unsloth.ai/docs/models/qwen3.5/fine-tune) підтверджує BF16 LoRA та застерігає від QLoRA 4-bit для цієї родини. QLoRA і перемикання на Unsloth тут не реалізовано. Основний сумісний механізм — Transformers + PEFT + TRL, перевірений на встановлених API.

Під час генерації маркер завершення репліки виводиться з офіційного шаблону. Для перевіреного tokenizer це `im_end` разом із задекларованим EOS `endoftext`; лише EOS документа не зупиняв би кожну chat-відповідь. Для baseline CLI автоматично застосовує закріплений SHA; інший revision можна задати через `--revision`.

## Оцінювання

`data/eval/prompts.jsonl` — 12 незалежних сценаріїв; `sft_evaluation.jsonl` і `cpt_evaluation.jsonl` — додаткові відкладені дані. Генерація використовує greedy decoding і однакові запити для порівняння:

```bash
python scripts/10_evaluate.py --model Qwen/Qwen3.5-0.8B-Base --output outputs/eval_base
python scripts/10_evaluate.py --model outputs/cpt_merged --output outputs/eval_cpt
python scripts/10_evaluate.py --model outputs/final_merged --output outputs/eval_final
```

Команда створює JSON і Markdown. Автоматично рахується повторюваність триграм і точний збіг для завдань із однозначною reference-відповіддю. Граматика, архаїка, персонаж, гумор, побутова корисність, термінологія й знання світу мають **незаповнені людські оцінки** з рубрикою 0–4. Ці якості не видаються за виміряні евристиками. Порівнюйте `eval_base`, `eval_cpt`, `eval_final`, щоб побачити втрату загальних здібностей.

Можна оцінити раніше отримані відповіді без завантаження моделі:

```bash
python scripts/10_evaluate.py --responses responses.jsonl --output outputs/eval_imported
```

Кожний рядок `responses.jsonl`: `{"document_id":"ID із prompts.jsonl", "response":"текст"}`. Малий smoke-звіт `outputs/smoke_verified/evaluation.json` підтверджує роботу експорту й генерації, а не якість навченого персонажа.

## Файли й першоджерела API

`scripts/01...10` — послідовні CLI; `src/data` — ingestion/очищення/splits/демонстрації; `src/training` — tokenizer, masks, preflight, CPT/SFT/merge; `src/evaluation` — generation і звіти; `tests` — інваріанти даних та справжній LoRA backward/merge на tiny Qwen3.5. Шляхи YAML відносні до кореня проєкту незалежно від поточного каталогу shell.

- [Transformers Qwen3.5: офіційні класи й архітектура](https://huggingface.co/docs/transformers/model_doc/qwen3_5)
- [TRL SFTTrainer: pretokenized datasets, assistant/completion loss, processing_class](https://huggingface.co/docs/trl/sft_trainer)
- [PEFT LoRA: target_modules і merge](https://huggingface.co/docs/peft/package_reference/lora)
- [Unsloth Qwen3.5: підтримка й обмеження](https://unsloth.ai/docs/models/qwen3.5/fine-tune)

Точні результати цього запуску — `outputs/source_probe.json`, `download_report.json`, `inspection.json`, `clean_report.json`, `tokenizer_check.json`, `validation_report.json`, `environment.json` і `smoke_verified/result.json`. Вони є локальними артефактами та не потрапляють до Git за замовчуванням.

## Єдиний промпт і видимий тест навчання

Системний промпт визначено в `src/persona.py` і застосовується до всіх активних SFT/evaluation-даних та безпосередньо перед токенізацією:

```json
{"role":"system","content":"Ти — Братій, сервітор православний проповідник. Говори українською, часом урочисто й архаїчно, з теплою іронією. Доречно вживай церковні метафори, але відповідай по суті й не перетворюй кожну розмову на проповідь."}
```

Тест справжньої Qwen3.5-0.8B виконує один SFT optimizer step із цим промптом, одним train- і одним validation-прикладом. Для тесту база береться безпосередньо з Base; основний SFT залишається після CPT. CPU тест завантажує повні ваги, працює у FP32 і може тривати кілька хвилин або довше. Вивід містить loss, обраний пристрій, adapter та checkpoint.

```powershell
python scripts/test_training.py --device auto
# Або видимий PowerShell, який залишиться відкритим:
$runner = (Resolve-Path scripts/start_test_training.ps1).Path
Start-Process powershell.exe -WindowStyle Normal -ArgumentList @('-NoExit', '-ExecutionPolicy', 'Bypass', '-File', ('"' + $runner + '"'), '-Device', 'auto')
```

`start_test_training.ps1` зберігає transcript у `outputs/test_visible_*.log`; `status.json` у каталозі запуску містить `running`, `passed` або `failed`. `--cpu-threads 4` керує потоками CPU у Python test runner.

Перевірений запуск `outputs/test_visible_20261105_125055`: справжня Qwen3.5-0.8B-Base, CPU, FP32, 4 потоки, один SFT optimizer step; train loss ≈ 1.243, validation loss ≈ 3.238. Adapter і `checkpoint-1` збережено, статус `passed`; це перевірка працездатності, а не оцінка якості персонажа. Після змін пройшли 23 автоматичні тести. Live-вивід також дублюється в `training.log` усередині каталогу запуску. Для надійного завантаження у test runner вимкнено Xet; ваги завантажуються через HTTP.
