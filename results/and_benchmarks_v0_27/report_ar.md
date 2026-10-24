# Nucleus Pulse Software v0.27 — AND والتوقيت والمراجع

## الهدف
إصلاح AND زمني باستخدام تشبع تيار كل مشبك مستقلًا، مع اختبار قطارات نبضات متنوعة ومقارنة مع مرجع قاعدي أحادي وموديل زمني غير نبضي.

## النتيجة الإجمالية
- أمثلة الاختبار: 3000 عبر 10 بذور.
- شبكة النبضات المشبعة: 100.00%.
- baseline عدّ القنوات دون توقيت: 70.00%.
- MLP غير نبضي بمدخلات زمنية، تصويت 3 نماذج: 94.60%.
- القاعدة التحليلية المحددة للمهمة: 100.00% (oracle، ليست نموذجًا متعلمًا).

## اختبارات قطارات النبض
- simultaneous: target=1, prediction=1, correct=True.
- A_before_B_by_1: target=1, prediction=1, correct=True.
- B_before_A_by_1: target=1, prediction=1, correct=True.
- gap_2: target=0, prediction=0, correct=True.
- gap_4: target=0, prediction=0, correct=True.
- A_high_rate_burst_only: target=0, prediction=0, correct=True.
- B_high_rate_burst_only: target=0, prediction=0, correct=True.
- interleaved_gap_2_trains: target=0, prediction=0, correct=True.
- synchronous_trains: target=1, prediction=1, correct=True.

## التثبيط والتشبع
- أقصى جهد مع الإثارة فقط: 0.948181.
- أقصى جهد مع الإثارة والتثبيط المتزامن: 0.568909.
- spikes excitation only: [2].
- spikes excitation + inhibition: [].

## سلامة الحساب
- فرق التيار الأقصى عن المرجع المستقل: 0.000e+00.
- فرق الجهد الأقصى عن المرجع المستقل: 0.000e+00.
- لم تُنفّذ مقارنة Brian2 لأن المكتبة غير مثبتة وبيئة التنفيذ لم تسمح بتنزيلها؛ لا ندعي parity مع simulator خارجي بعد.

## تكلفة التشغيل على CPU
- 500 مثال، شبكة أحداث Python: 0.0276 ثانية.
- نموذج MLP واحد، استدلال عينة بعينة: 0.0106 ثانية.
- نموذج MLP واحد، استدلال دفعي 500 مثال: 0.0002 ثانية.
- القاعدة التحليلية: 0.0003 ثانية.
- هذه قياسات بيئة واحدة وليست حكمًا عامًا على الأداء. التنفيذ الحدثي في Python ليس مضمونًا أن يكون أسرع من العمليات المصفوفية؛ العتاد المتخصص قصة مختلفة.

## حدود التجربة
- The pulse AND cell has manually specified parameters; this experiment validates the mechanism and dynamics, not learned weights.
- The temporal MLP is trained on 5000 examples; the pulse circuit is configured analytically, so this is not an equal-training-budget learning comparison.
- The analytic coincidence rule is an oracle for the task definition, not a learned model.
- A later experiment will train a network sequentially on synchrony then A-before-B and compare reward-modulated STDP with a same-order MLP baseline.

## التفسير
نجاح AND هنا ناتج عن تصميم خلية بتشبع لكل مشبك وثوابت زمن مناسبة؛ لم نتعلم هذه المعمارية من البيانات. التجربة تثبت صحة آلية التشبع والتوقيت على القطارات المختبرة، لا تعلمًا عامًا ولا تفوقًا شاملًا على MLP.
