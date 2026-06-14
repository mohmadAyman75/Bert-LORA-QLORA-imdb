# خطة مشروع بحثي: مقارنة LoRA و QLoRA و Feature Extraction على BERT باستخدام Rotten Tomatoes

## 1. معلومات عامة عن المشروع

**عنوان مقترح للورقة البحثية:**  
`A Comparative Study of Feature Extraction, LoRA, and QLoRA for Sentiment Classification using BERT on Rotten Tomatoes`

**فكرة المشروع:**  
إجراء مقارنة عملية بين ثلاث طرق لاستخدام نموذج `bert-base-uncased` في مهمة تصنيف المشاعر على مجموعة بيانات Rotten Tomatoes:

1. **Feature Extraction**
2. **LoRA Fine-tuning**
3. **QLoRA Fine-tuning**

الهدف ليس الوصول إلى أفضل نتيجة عالمية على Rotten Tomatoes، بل فهم الفرق بين الطرق الثلاث من ناحية الأداء، تكلفة التدريب، استهلاك الذاكرة، وعدد المعاملات التي يتم تدريبها.

**الفئة المستهدفة:**  
مشروع بحثي بسيط ومناسب لطلاب سنة ثالثة حاسبات، مع متابعة من الدكتور المشرف.

**مدة المشروع:**  
4 أسابيع.

**عدد أفراد الفريق:**  
شخصان.

**اللغة المقترحة للتقرير:**  
العربي في خطة المشروع والمتابعة، والإنجليزي في عنوان الورقة والمصطلحات التقنية والجداول.

---

## 2. سؤال البحث والفرضيات

### 2.1 سؤال البحث

هل استخدام LoRA و QLoRA مع نموذج BERT يعطي توازنا أفضل بين الدقة وتكلفة التدريب مقارنة بطريقة Feature Extraction على مهمة تصنيف مشاعر مراجعات Rotten Tomatoes؟

### 2.2 الفرضيات

1. **Feature Extraction** ستكون أسرع وأبسط في التنفيذ، لكنها قد تحقق أداء أقل لأنها لا تعدل تمثيلات BERT الداخلية.
2. **LoRA** قد تحقق دقة أعلى من Feature Extraction لأنها تسمح بتكييف أجزاء من النموذج مع المهمة مع تدريب عدد قليل من المعاملات.
3. **QLoRA** قد تقلل استهلاك الذاكرة مقارنة بـ LoRA عن طريق 4-bit quantization، مع احتمال وجود فرق بسيط في الدقة أو زمن التدريب.

---

## 3. نطاق المشروع

### 3.1 داخل نطاق المشروع

- استخدام نموذج واحد فقط: `bert-base-uncased`.
- استخدام Dataset واحدة فقط: Rotten Tomatoes sentiment classification.
- مقارنة ثلاث طرق فقط: Feature Extraction، LoRA، QLoRA.
- قياس:
  - Accuracy
  - F1-score
  - Training time
  - Peak GPU memory usage
  - Number of trainable parameters
- كتابة ورقة بحثية قصيرة ومنظمة.
- توفير كود أو Notebook يمكن إعادة تشغيله.

### 3.2 خارج نطاق المشروع

- عدم مقارنة نماذج أخرى مثل RoBERTa أو DistilBERT.
- عدم استخدام Datasets إضافية.
- عدم تنفيذ Full Fine-tuning إلا إذا طلب الدكتور إضافته لاحقا كمرجع إضافي.
- عدم محاولة الوصول إلى state-of-the-art results.

---

## 4. الخلفية العلمية المختصرة

### 4.1 BERT

BERT هو نموذج Transformer encoder مدرب مسبقا على نصوص إنجليزية كبيرة. النسخة المستخدمة في المشروع هي `bert-base-uncased`، وهي تحتوي تقريبا على 110 مليون parameter، وتتعامل مع النصوص بعد تحويلها إلى lowercase.

### 4.2 Feature Extraction

في هذه الطريقة يتم تجميد أوزان BERT بالكامل، ثم استخدام التمثيل الناتج من BERT، مثل `[CLS] embedding` أو `pooled output`، لتدريب classifier بسيط فوقه.  
هذه الطريقة قليلة التكلفة، لكنها لا تسمح للنموذج بتعديل تمثيلاته الداخلية لتناسب Dataset الجديدة.

### 4.3 LoRA

LoRA اختصار لـ Low-Rank Adaptation. الفكرة هي تجميد أوزان النموذج الأصلي وإضافة matrices صغيرة قابلة للتدريب داخل بعض الطبقات، غالبا داخل attention layers.  
بهذا يتم تدريب عدد قليل من parameters بدل تدريب كل النموذج.

### 4.4 QLoRA

QLoRA تجمع بين quantization و LoRA. يتم تحميل النموذج الأساسي بتمثيل منخفض الدقة، غالبا 4-bit، ثم تدريب LoRA adapters فوقه.  
الميزة الأساسية هي تقليل استهلاك الذاكرة، خصوصا عند التعامل مع نماذج كبيرة. في حالة BERT، الفائدة البحثية هنا تعليمية وتجريبية، لأن BERT أصغر بكثير من LLMs الحديثة.

---

## 5. البيانات المستخدمة

### 5.1 Dataset

سيتم استخدام Rotten Tomatoes Dataset الخاصة بتصنيف مشاعر مقتطفات مراجعات الأفلام.

**المهمة:**  
Binary sentiment classification.

**التصنيفات:**

- `0`: Negative review
- `1`: Positive review

**التقسيم الأساسي:**

- 8,530 example للتدريب.
- 1,066 example للتحقق validation.
- 1,066 example للاختبار.

### 5.2 Preprocessing

سيتم استخدام نفس خطوات المعالجة لكل الطرق لضمان المقارنة العادلة:

- استخدام tokenizer الخاص بـ `bert-base-uncased`.
- تحويل النصوص إلى:
  - `input_ids`
  - `attention_mask`
  - `labels`
- استخدام `max_length = 256` كبداية لتقليل استهلاك الذاكرة.
- استخدام padding و truncation.
- تثبيت random seed.

---

## 6. التصميم التجريبي

### 6.1 الطريقة الأولى: Feature Extraction

**الفكرة:**  
تجميد BERT بالكامل واستخدامه كمولد features، ثم تدريب classifier بسيط.

**التنفيذ المقترح:**

- تحميل `AutoModel` أو `AutoModelForSequenceClassification`.
- تجميد كل layers الخاصة بـ BERT.
- استخدام `[CLS] embedding` أو `pooled output`.
- تدريب classifier من طبقة أو طبقتين.

**ما يتم تدريبه:**

- classifier head فقط.

**المخرجات المسجلة:**

- Accuracy
- F1-score
- Training time
- GPU memory
- عدد trainable parameters

### 6.2 الطريقة الثانية: LoRA

**الفكرة:**  
تجميد أوزان BERT الأصلية وإضافة LoRA adapters على بعض طبقات attention.

**التنفيذ المقترح:**

- استخدام مكتبة `peft`.
- استخدام `AutoModelForSequenceClassification`.
- تطبيق LoRA على طبقات مناسبة مثل:
  - `query`
  - `value`
- إعدادات أولية مقترحة:
  - `r = 8`
  - `lora_alpha = 16`
  - `lora_dropout = 0.1`
  - `task_type = SEQ_CLS`

**ما يتم تدريبه:**

- LoRA adapters.
- classification head إذا احتاجت التجربة ذلك.

### 6.3 الطريقة الثالثة: QLoRA

**الفكرة:**  
تحميل BERT باستخدام 4-bit quantization ثم تدريب LoRA adapters فوق النموذج المجمد.

**التنفيذ المقترح:**

- استخدام `bitsandbytes`.
- استخدام `BitsAndBytesConfig` لتحميل النموذج بـ 4-bit.
- تطبيق `prepare_model_for_kbit_training` إذا كان مناسبا مع الإعداد المستخدم.
- تطبيق LoRA بنفس إعدادات تجربة LoRA قدر الإمكان.

**إعدادات أولية مقترحة:**

- `load_in_4bit = True`
- `bnb_4bit_quant_type = "nf4"`
- `bnb_4bit_compute_dtype = torch.float16`
- `r = 8`
- `lora_alpha = 16`
- `lora_dropout = 0.1`

**ملاحظة مهمة:**  
QLoRA مشهورة أكثر مع النماذج اللغوية الكبيرة، لكن استخدامها هنا مع BERT سيكون بهدف دراسة أثر quantization على الذاكرة والأداء في تجربة تعليمية صغيرة.

---

## 7. إعدادات التدريب المقترحة

سيتم البدء بالإعدادات التالية، ويمكن تعديلها فقط إذا ظهرت مشكلة memory أو وقت التدريب:

| الإعداد | القيمة المقترحة |
|---|---|
| Model | `bert-base-uncased` |
| Dataset | Rotten Tomatoes |
| Max sequence length | 256 |
| Epochs | 2 أو 3 |
| Batch size | 8 أو 16 حسب GPU |
| Optimizer | AdamW |
| Learning rate | `2e-5` لـ LoRA/QLoRA، و `1e-4` للـ classifier في Feature Extraction كبداية |
| Seeds | 2 أو 3 seeds إذا سمح الوقت |
| Evaluation | بعد كل epoch |
| Metrics | Accuracy, F1-score |

---

## 8. المقاييس وطريقة المقارنة

### 8.1 مقاييس الأداء

- **Accuracy:** نسبة التوقعات الصحيحة.
- **F1-score:** مهم لأنه يعطي توازنا بين precision و recall.

### 8.2 مقاييس الكفاءة

- **Training time:** الزمن الكلي للتدريب لكل طريقة.
- **Peak GPU memory:** أعلى استهلاك لذاكرة GPU أثناء التدريب.
- **Trainable parameters:** عدد المعاملات التي يتم تحديثها أثناء التدريب.

### 8.3 جدول النتائج النهائي

| Method | Accuracy | F1-score | Training Time | Peak GPU Memory | Trainable Parameters |
|---|---:|---:|---:|---:|---:|
| Feature Extraction | TBD | TBD | TBD | TBD | TBD |
| LoRA | TBD | TBD | TBD | TBD | TBD |
| QLoRA | TBD | TBD | TBD | TBD | TBD |

### 8.4 الرسوم المقترحة

- Bar chart للمقارنة بين Accuracy و F1-score.
- Bar chart لاستهلاك GPU memory.
- Bar chart لعدد trainable parameters.
- جدول مختصر في الورقة النهائية.

---

## 9. تقسيم العمل بين الشخصين

### 9.1 الشخص الأول: Feature Extraction وكتابة الخلفية

**المسؤوليات:**

- إعداد وتجربة Feature Extraction.
- تجهيز preprocessing المشترك مع الشخص الثاني.
- تسجيل نتائج baseline.
- كتابة أجزاء:
  - Introduction
  - Dataset Description
  - Feature Extraction Method
  - جزء من Results الخاص بالـ baseline

**مخرجات الشخص الأول:**

- Notebook أو script لتجربة Feature Extraction.
- جدول نتائج baseline.
- مسودة الأقسام المسؤول عنها في الورقة.

### 9.2 الشخص الثاني: LoRA و QLoRA

**المسؤوليات:**

- إعداد تجربة LoRA.
- إعداد تجربة QLoRA.
- قياس الذاكرة والزمن وعدد trainable parameters.
- كتابة أجزاء:
  - LoRA Method
  - QLoRA Method
  - Experimental Setup
  - جزء من Results الخاص بـ LoRA و QLoRA

**مخرجات الشخص الثاني:**

- Notebook أو scripts لتجارب LoRA و QLoRA.
- نتائج LoRA و QLoRA.
- مسودة الأقسام المسؤول عنها في الورقة.

### 9.3 المسؤوليات المشتركة

- توحيد شكل الكود والنتائج.
- مراجعة التجارب للتأكد من العدالة.
- كتابة Discussion و Conclusion.
- تجهيز النسخة النهائية للدكتور.
- التأكد من أن كل النتائج قابلة لإعادة التشغيل.

---

## 10. دور الدكتور المشرف

الدكتور سيكون مسؤولا عن متابعة الاتجاه العام والتأكد من أن المقارنة صحيحة علميا.

### نقاط المتابعة المقترحة

| الأسبوع | ما يتم عرضه على الدكتور | الهدف |
|---|---|---|
| الأسبوع 1 | الفكرة، سؤال البحث، خطة التجارب | اعتماد المنهجية |
| الأسبوع 2 | أول نتائج Feature Extraction و LoRA | التأكد من صحة التنفيذ |
| الأسبوع 3 | نتائج QLoRA والمقارنة الأولية | مراجعة التحليل |
| الأسبوع 4 | مسودة الورقة النهائية | مراجعة الكتابة والاستنتاج |

### أسئلة مهمة للدكتور

- هل المقارنة بين الطرق الثلاث كافية أم يجب إضافة Full Fine-tuning كمرجع؟
- هل المطلوب ورقة قصيرة فقط أم عرض تقديمي أيضا؟
- هل هناك format محدد للورقة البحثية؟
- هل يفضل الدكتور استخدام كل Dataset أم sample صغير بسبب الوقت؟

---

## 11. الجدول الزمني التفصيلي

### الأسبوع الأول: الإعداد والقراءة

**الأهداف:**

- تثبيت البيئة.
- تحميل Dataset.
- قراءة سريعة عن BERT و LoRA و QLoRA.
- تنفيذ أول baseline.

**المهام:**

- إنشاء repository منظم.
- تجهيز environment:
  - Python
  - PyTorch
  - transformers
  - datasets
  - evaluate
  - scikit-learn
  - peft
  - bitsandbytes
- تحميل Rotten Tomatoes وتجربة tokenizer.
- تنفيذ Feature Extraction بشكل أولي.
- كتابة Introduction و Dataset Description كمسودة.

**مخرج الأسبوع:**

- كود preprocessing.
- baseline أولي.
- موافقة الدكتور على الخطة.

### الأسبوع الثاني: تنفيذ LoRA و QLoRA

**الأهداف:**

- تنفيذ LoRA.
- تنفيذ QLoRA.
- توحيد training loop والتقييم.

**المهام:**

- بناء script أو Notebook لـ LoRA.
- بناء script أو Notebook لـ QLoRA.
- تسجيل الزمن والذاكرة.
- حفظ النتائج في ملفات CSV أو JSON.
- بدء كتابة Methodology.

**مخرج الأسبوع:**

- نتائج أولية للطرق الثلاث.
- Methodology draft.

### الأسبوع الثالث: تحسين التجارب وتحليل النتائج

**الأهداف:**

- إعادة التجارب باستخدام seed ثابت أو أكثر.
- تجهيز جداول ورسوم.
- تحليل الفروق بين الطرق.

**المهام:**

- تشغيل كل طريقة 2 أو 3 مرات إذا سمح الوقت.
- حساب المتوسط والانحراف البسيط إذا تعددت seeds.
- رسم المقارنات.
- كتابة Results و Discussion.

**مخرج الأسبوع:**

- جدول نتائج شبه نهائي.
- رسوم جاهزة للورقة.
- تحليل أولي للملاحظات.

### الأسبوع الرابع: كتابة الورقة والمراجعة

**الأهداف:**

- إنهاء الورقة البحثية.
- مراجعة النتائج مع الدكتور.
- تجهيز التسليم النهائي.

**المهام:**

- كتابة Abstract و Conclusion.
- مراجعة Related Work.
- تنسيق الجداول والرسوم.
- مراجعة اللغة والأخطاء.
- تجهيز نسخة نهائية من الكود والنتائج.

**مخرج الأسبوع:**

- الورقة النهائية.
- كود منظم.
- نتائج قابلة للمراجعة.

---

## 12. هيكل الملفات المقترح

```text
LORA-QLORA-BERT-ROTTEN-TOMATOES/
├── PROJECT_PLAN.md
├── README.md
├── paper/
│   ├── paper.md
│   └── figures/
├── notebooks/
│   ├── 01_feature_extraction.ipynb
│   ├── 02_lora.ipynb
│   └── 03_qlora.ipynb
├── src/
│   ├── data.py
│   ├── metrics.py
│   ├── train_feature_extraction.py
│   ├── train_lora.py
│   └── train_qlora.py
├── results/
│   ├── feature_extraction_results.csv
│   ├── lora_results.csv
│   ├── qlora_results.csv
│   └── summary_results.csv
└── references.md
```

---

## 13. شكل الورقة البحثية المقترح

### 13.1 Abstract

ملخص قصير يشرح المشكلة والطرق الثلاث والنتيجة العامة.

### 13.2 Introduction

- أهمية sentiment analysis.
- أهمية تقليل تكلفة fine-tuning.
- لماذا BERT و Rotten Tomatoes مناسبان للتجربة.

### 13.3 Related Work

- BERT.
- Transfer learning في NLP.
- Parameter-Efficient Fine-Tuning.
- LoRA و QLoRA.

### 13.4 Methodology

- شرح Dataset.
- شرح preprocessing.
- شرح Feature Extraction.
- شرح LoRA.
- شرح QLoRA.

### 13.5 Experimental Setup

- تفاصيل الجهاز أو GPU.
- المكتبات المستخدمة.
- hyperparameters.
- metrics.

### 13.6 Results

- جدول النتائج.
- الرسوم.
- مقارنة مباشرة بين الطرق.

### 13.7 Discussion

- لماذا طريقة معينة أفضل في الدقة؟
- لماذا طريقة معينة أقل في الذاكرة؟
- هل QLoRA مفيدة فعلا مع BERT؟
- حدود التجربة.

### 13.8 Conclusion

- تلخيص أهم النتائج.
- اقتراحات للتطوير لاحقا.

### 13.9 References

- BERT paper.
- LoRA paper.
- QLoRA paper.
- Rotten Tomatoes Dataset.
- Hugging Face documentation.

---

## 14. معايير قبول المشروع

يعتبر المشروع مكتملا إذا تحقق الآتي:

- تم تشغيل الطرق الثلاث على نفس Dataset ونفس preprocessing.
- تم تسجيل Accuracy و F1-score لكل طريقة.
- تم تسجيل training time و peak GPU memory لكل طريقة.
- تم حساب عدد trainable parameters لكل طريقة.
- تم تجهيز جدول مقارنة نهائي.
- تم تجهيز رسم أو أكثر للنتائج.
- تم كتابة الورقة البحثية كاملة.
- الكود قابل للتشغيل مرة أخرى من خلال README أو notebooks واضحة.
- الدكتور راجع المنهجية والنتائج قبل التسليم النهائي.

---

## 15. المخاطر المتوقعة وخطة التعامل معها

| الخطر | التأثير | طريقة التعامل |
|---|---|---|
| GPU memory غير كافية | فشل تدريب LoRA أو QLoRA | تقليل batch size، استخدام max_length أقل، أو gradient accumulation |
| QLoRA لا تعمل بسهولة مع BERT على البيئة المحلية | تأخير في التنفيذ | توثيق المشكلة، تجربة إعداد bitsandbytes مختلف، واستشارة الدكتور قبل تغيير التجربة |
| وقت التدريب طويل | تأخير النتائج | تقليل epochs إلى 2 أو استخدام subset للتجارب الأولية فقط |
| نتائج الطرق متقاربة جدا | صعوبة في التحليل | التركيز على الذاكرة، الزمن، وعدد trainable parameters وليس الدقة فقط |
| اختلاف التنفيذ بين الطرق | مقارنة غير عادلة | استخدام نفس preprocessing ونفس metrics ونفس train/test split |

---

## 16. المصادر المقترحة

1. Devlin et al., **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**.
2. Hu et al., **LoRA: Low-Rank Adaptation of Large Language Models**.
3. Dettmers et al., **QLoRA: Efficient Finetuning of Quantized LLMs**.
4. Rotten Tomatoes Dataset: https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes
5. BERT base uncased model: https://huggingface.co/google-bert/bert-base-uncased
6. Hugging Face Transformers documentation: https://huggingface.co/docs/transformers
7. Hugging Face PEFT documentation: https://huggingface.co/docs/peft
8. Hugging Face PEFT quantization guide: https://huggingface.co/docs/peft/en/developer_guides/quantization

---

## 17. ملاحظات نهائية

هذا المشروع مناسب كبحث تطبيقي صغير لأنه يجمع بين فكرة أكاديمية مهمة وهي تقليل تكلفة fine-tuning، وتجربة عملية واضحة على Dataset معروفة.  
قوة الورقة لن تكون فقط في أرقام الدقة، بل في تفسير الفرق بين الطرق الثلاث من حيث:

- مقدار التعديل على النموذج.
- عدد المعاملات التي تم تدريبها.
- استهلاك الذاكرة.
- زمن التدريب.
- ملاءمة كل طريقة لإمكانيات طالب أو جهاز محدود.
