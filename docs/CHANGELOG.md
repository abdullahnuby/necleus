## 2026-10-24 — تدقيق محلي قبل v0.43 (2026-10-24-TASK-004)
- أُنشئت بيئة اختبار محلية في `.venv` على القرص `D:` من `docs/requirements-dev.txt` و`docs/requirements-benchmark.txt`؛ `pip check` PASS.
- `tests/v043`: 9 ناجحة. المجموعة الكاملة: 2 فاشلة و129 ناجحة؛ الفشلان في فحصي التسجيل المسبق بسبب غياب `requirements-benchmark.txt` من الجذر، مع تطابق بصمة النسخة الموجودة في `docs/`.
- فُحصت ملفات smoke: `*_evaluator_private.jsonl` منشورة في مستودع عام وتحتوي `target_program` و`query_y`؛ لا يجوز إعادة استخدام مهام smoke في الاختبار الرسمي.
- `official_test_ready=false`، وقائمة التسجيل المسبق 0/31. لم يُعدّل النموذج أو المولد أو النتائج أو معايير النجاح.

## 2026-10-24 — README إنجليزي شامل وفهرس وثائق إنجليزي (2026-10-24-TASK-003)
- أُضيف `README.md` في جذر المستودع بالإنجليزية بالكامل: 772 سطرًا و6,859 كلمة و0 أحرف عربية، يغطي السؤال العلمي وعقد الأدلة ومستوياتها الخمسة والحالة والنتائج المختارة والمعمارية وتصميم v0.43 والتشغيل والاختبارات والنزاهة والقيود وخريطة الطريق.
- استُبدل `docs/README.md` بفهرس وثائق إنجليزي يربط الوثائق والتقارير والبروتوكولات ونظام التسليم؛ حُفظ الفهرس العربي القديم كما هو في `docs/README_AR.md` (مطابق بايتًا ببايت).
- كل الأرقام في README منقولة من تقارير v0.42 وv0.41 وv0.39 وv0.37 وv0.29؛ لم تُضف نتائج جديدة ولم تُعدّل ملفات `results/`.
- `tests/v043`: 9 ناجحة. المجموعة الكاملة ما زالت متعطلة بسبب غياب `matplotlib` (خطأ جمع، رمز خروج 2). التحقق PASS، وفحص الروابط 0 مكسور.
- لا تغيير في منطق النموذج أو التجارب أو معايير النجاح.

## 2026-10-24 — رفع المشروع إلى GitHub وقاعدة مزامنة إلزامية (2026-10-24-TASK-002)
- تهيئة مستودع Git محلي على الفرع `main` وربطه بـ`https://github.com/abdullahnuby/necleus.git`.
- أُضيف `.gitignore` و`.gitattributes` (تثبيت `eol=lf`) لمنع رفع المخلفات ومنع تغيّر البصمات عند الاستنساخ.
- أُضيف قسم «مزامنة المستودع البعيد» إلى `OPENCODE_RULES.md` وبند مقابل في `AGENTS.md`: commit وpush بعد كل تغيير فعلي، ومنع force-push.
- `tests/v043`: 9 ناجحة. المجموعة الكاملة ما زالت متعطلة بسبب غياب `matplotlib` (خطأ جمع، رمز خروج 2).
- رُفع الفرع `main` (commit `c115c61`) وتحقّق الاستنساخ من GitHub: نفس `HEAD` و352 ملفًا، والتحقق PASS، و9 اختبارات ناجحة على النسخة المستنسخة.
- لا تغيير في منطق النموذج أو التجارب أو ملفات `results/` أو معايير النجاح.

## 2026-10-24 — نظام تسليم إلزامي قابل للتحقق (2026-10-24-TASK-001)
- أُضيف `OPENCODE_RULES.md` و`AGENTS.md` لتثبيت دورة العمل الإلزامية وربطها بملف التعليمات الذي يقرأه OpenCode تلقائيًا.
- أُضيف `docs/handoff/TEMPLATE.md` و`docs/handoff/LATEST_REPORT.md` و`docs/handoff/history/2026-10-24-TASK-001.md`، مع سجلات أوامر في `docs/handoff/logs/2026-10-24-TASK-001/`.
- أُضيف `scripts/verify_handoff.py` للتحقق من صيغة السجل ومن وجود الأقسام الأحد عشر بالترتيب ومن محتواها الفعلي؛ نتيجة التحقق PASS.
- `tests/v043`: 9 ناجحة. المجموعة الكاملة `tests/`: لم تكتمل بسبب غياب `matplotlib` (خطأ جمع، رمز خروج 2)، ومسجلة صراحةً كغير ناجحة.
- لا تغيير في منطق النموذج أو التجارب أو ملفات `results/` أو معايير النجاح.

## v0.42.2 — إصلاحات مواصفة v0.43 قبل البناء (تحديث وثائقي)
- فصل بوابات الإجابة على المهام القابلة للتحديد عن بوابات الامتناع للحالات الملتبسة والخارجة عن النطاق والمتناقضة.
- تعريف فضاء الفرضيات صراحةً، وإلزام المقيم بعدّ البرامج المتسقة وفئات التكافؤ الدلالي بعد أمثلة الدعم؛ والمسار المغلق موصوف بوضوح كاستنتاج برنامج لا دليلًا منفردًا على تعلم مهارة عامة.
- إضافة مسار مستقل لعملية غير موجودة في سجل المكتبة، مع الامتناع قبل التعليم والتعلم من 8/16/32 مثالًا ضمن meta-grammar معلنة.
- إضافة meta-learning neural baseline مدرّب على ≥10,000 حلقة من ≥100 عائلة، وحد أدنى 30 عائلة اختبار محجوزة، وتوازن استعلامات 50/50، وفحوص تمييز الفرضيات.
- إضافة بروتوكول دورات تصحيح الأخطاء واختبارات الاحتفاظ بالمهارات السابقة؛ لم تُنفّذ هذه المكونات بعد.
- لا يتغير كود أو نتائج v0.42، وهذه النسخة توثيقية ولا تدّعي تنفيذ benchmark v0.43.

## v0.43-stage1 — أول تنفيذ لمولد المهام ومدقق الفرضيات
- أُضيف `skillbench_v043/`: grammar محدودة معلنة بعمق AST لا يتجاوز 3، ومجال إدخال من 10,000 نقطة.
- تعداد كامل لـ9,348 برنامجًا نحويًا واختزالها إلى 4,570 دالة دلالية متميزة على المجال.
- تثبيت تقسيم عائلي قابل لإعادة الإنتاج: 649 عائلة train و140 dev و134 test؛ لا تستخدم قيم الثوابت أو تبديل أسماء المتغيرات المطلقة لتعريف عائلة جديدة.
- مولد دعم 8 أمثلة واستعلامات 100 فريدة متوازنة 50/50، مع نمط عشوائي وآخر contrastive معلن، ومدقق يعد البرامج المتسقة وفئاتها الدلالية واختلافها على الاستعلامات.
- إضافة حالات OUT_OF_SCOPE وCONTRADICTORY_SUPPORT، وفصل payload العام عن سجل المقيم السري، ومدقق تداخل العائلات وAST والدوال الدلالية.
- أُضيفت 9 اختبارات مرحلة أولى جديدة؛ جميعها ناجحة. التشغيل المصغّر ثلاث مهام هو smoke-test للميكانيكا، وليس نتيجة أداء.
- لم يُنفّذ بعد meta-learning baseline أو corpus من 10,000 حلقة أو جولات التصحيح أو عدم النسيان أو الاختبار النهائي.

# Changelog

## v0.42.1 — تصحيح التفسير وخطة v0.43 (تحديث وثائقي)
- أُضيف ملحق يوضح أن فرق −18.81 نقطة في v0.42 يخص خطّي التنفيذ المختبرين، ولا يعزل الأثر السببي لقاعدة STDP بسبب اختلاف التقييس وميزانية التحسين.
- وُثقت ضوابط 12 بذرة التي أبلغ عنها المستخدم بوصفها نتائج خارجية لم تُعَد في هذه الحزمة، مع فصل التقييس والإشراف وميزانية التحسين قدر الإمكان.
- أُضيفت مواصفة v0.43 التي تعرّف المهارة الجديدة، ومحتويات مكتبة العمليات المسموحة والممنوعة، وفصل مهام الاختبار بنيويًا، وقياس risk–coverage.
- لا تغيّر هذه النسخة كود v0.42 أو نتائجه الخام، ولا تدّعي تنفيذ benchmark v0.43.

## v0.42 — Preregistered 16-window STDP-vs-linear comparison
- Added a frozen low-shot comparison at 50, 100, and 200 training examples over 40 independent seeds and both temporal tasks.
- Both methods receive the same 16-window reservoir-rate features; the reservoir itself remains frozen.
- Added a one-pass reward-modulated STDP-style reader, a fixed regularized logistic baseline, integrity checks, paired seed-cluster bootstrap intervals, and a preregistered pass/fail gate.
- Smoke-test outputs are explicitly separate from official benchmark evidence.

## v0.41 — Stage 2 multi-seed input-gain diagnostic
- Repeated the v0.40 component-only input-amplitude test across eight development seeds.
- Compared fixed inhibition and local adaptive inhibition at input-weight scales 0.75x, 1.0x, and 1.25x.
- Used 32 class-balanced examples per task/seed from training pools only.
- Added preregistration and frozen source hashes, per-seed partial saves, t confidence intervals, paired bootstrap comparisons, and activity/feature sensitivity plots.
- No readout training, no validation/test sets, and no learning-accuracy or energy-efficiency claims.

## v0.40 — Stage 1 component diagnostic
- Added a one-seed diagnostic for input-weight scaling at 0.75x, 1.0x, and 1.25x.
- Compared fixed inhibitory gain with local activity-adaptive inhibition (slope 2.0, EMA tau 4.0).
- No reader training, no hyperparameter selection, and no official test evaluation in this stage.

## Prior releases
See the versioned reports and protocol files included in this repository for v0.27–v0.40.
