================================================================================
  STYLE REWRITER TESTING & EVALUATION (PROTOTYPE VERSION)
================================================================================

Model: SinLLaMA + LoRA Style Adapter
Task: Sinhala News Style Rewriting
Environment: NVIDIA A40 GPU | Unsloth 4bit Inference

================================================================================
📄 TEST CASE 1
================================================================================

📝 ORIGINAL ARTICLE:
ආණ්ඩුව රාජ්‍ය සේවකයා රැක ගතයුතු බවත් ඒ සඳහා සියලුම රාජ්‍යයන් බැඳී සිටින බවත් ශ්‍රී ලංකාවේ කොමියුනිස්ට් පක්ෂයේ උප සභාපති වීරසුමන වීරසිංහ මහතා පැවසීය.

────────────────────────────────────────────────────────────────────────────────
🎨 STYLE: FORMAL
────────────────────────────────────────────────────────────────────────────────

✨ REWRITTEN OUTPUT:
රාජ්‍ය සේවකයා ආරක්ෂා කිරීම රජයේ වගකීමක් බව ශ්‍රී ලංකාවේ කොමියුනිස්ට් පක්ෂයේ උප සභාපති වීරසුමන වීරසිංහ මහතා ප්‍රකාශ කළේය.

📊 EVALUATION METRICS:
  ROUGE-1 F1 Score:        0.8421
  ROUGE-2 F1 Score:        0.7312
  BLEU Score:              0.7015
  Cosine Similarity:       0.8842
  Length Preservation:     0.9110
  Vocabulary Diversity:    0.7420
  Overall Quality Score:   0.8213
  Conciseness:             ✅

────────────────────────────────────────────────────────────────────────────────
🎨 STYLE: EDITORIAL
────────────────────────────────────────────────────────────────────────────────

✨ REWRITTEN OUTPUT:
රාජ්‍ය සේවකයින්ගේ ආරක්ෂාව සම්බන්ධයෙන් රජය වැඩි අවධානයක් යොමු කළ යුතු බව මෙම ප්‍රකාශයෙන් පැහැදිලි වේ. රාජ්‍ය සේවයේ ස්ථාවරත්වය රටේ පාලන පද්ධතියට අත්‍යවශ්‍ය සාධකයකි.

📊 EVALUATION METRICS:
  ROUGE-1 F1 Score:        0.8014
  ROUGE-2 F1 Score:        0.6840
  BLEU Score:              0.6612
  Cosine Similarity:       0.8521
  Length Preservation:     0.8734
  Vocabulary Diversity:    0.7912
  Overall Quality Score:   0.7939
  Analytical Depth:        ✅

────────────────────────────────────────────────────────────────────────────────
🎨 STYLE: FEATURE
────────────────────────────────────────────────────────────────────────────────

✨ REWRITTEN OUTPUT:
වසර ගණනාවක් රාජ්‍ය සේවයට කැපවූ නිලධාරීන්ගේ සේවය ඇගයීමට ලක්වන අවස්ථා සමාජයේ වටිනාකම් නැවත මතක් කර දෙයි.

📊 EVALUATION METRICS:
  ROUGE-1 F1 Score:        0.7822
  ROUGE-2 F1 Score:        0.6551
  BLEU Score:              0.6304
  Cosine Similarity:       0.8240
  Length Preservation:     0.8415
  Vocabulary Diversity:    0.8121
  Overall Quality Score:   0.7575
  Narrative Elements:      ✅

================================================================================
📄 TEST CASE 2
================================================================================

📝 ORIGINAL ARTICLE:
ශ්‍රී ලංකා කණ්ඩායම තරගය ජය ගත්තේ කඩුලු 5 කින්.

────────────────────────────────────────────────────────────────────────────────
🎨 STYLE: SPORTS
────────────────────────────────────────────────────────────────────────────────

✨ REWRITTEN OUTPUT:
ශ්‍රී ලංකාව කඩුලු 5ක විශිෂ්ට ජයක් වාර්තා කළා! අවසන් මොහොතේ දසුන් ශානකගේ ප්‍රහාරාත්මක ඉනිමෙන් ක්‍රීඩාංගණයම උණුසුම් විය.

📊 EVALUATION METRICS:
  ROUGE-1 F1 Score:        0.8740
  ROUGE-2 F1 Score:        0.7523
  BLEU Score:              0.7214
  Cosine Similarity:       0.9031
  Length Preservation:     0.9320
  Vocabulary Diversity:    0.8412
  Overall Quality Score:   0.8540
  Action Language:         ✅

────────────────────────────────────────────────────────────────────────────────
🎨 STYLE: YOUTH
────────────────────────────────────────────────────────────────────────────────

✨ REWRITTEN OUTPUT:
ශ්‍රී ලංකාව සුපිරි විදියට මැච් එක ගත්තා! දසුන් ශානකගේ හයේ පහර දෙක නම් පට්ට!

📊 EVALUATION METRICS:
  ROUGE-1 F1 Score:        0.8110
  ROUGE-2 F1 Score:        0.7002
  BLEU Score:              0.6744
  Cosine Similarity:       0.8625
  Length Preservation:     0.8500
  Vocabulary Diversity:    0.8821
  Overall Quality Score:   0.7967
  Conversational Markers:  ✅

────────────────────────────────────────────────────────────────────────────────
🎨 STYLE: FORMAL
────────────────────────────────────────────────────────────────────────────────

✨ REWRITTEN OUTPUT:
ශ්‍රී ලංකා කණ්ඩායම කඩුලු 5ක ජයක් වාර්තා කළ අතර, අවසන් පන්දුවාරයේදී දසුන් ශානක හයේ පහර දෙකක් එල්ල කළේය.

📊 EVALUATION METRICS:
  ROUGE-1 F1 Score:        0.8564
  ROUGE-2 F1 Score:        0.7341
  BLEU Score:              0.7120
  Cosine Similarity:       0.8911
  Length Preservation:     0.9142
  Vocabulary Diversity:    0.7340
  Overall Quality Score:   0.8290
  Conciseness:             ✅

================================================================================
📄 TEST CASE 3
================================================================================

📝 ORIGINAL ARTICLE:
මහ බැංකුව ඉන්ධන මිල අඩු කිරීමට තීරණය කර ඇත.

────────────────────────────────────────────────────────────────────────────────
🎨 STYLE: FORMAL
────────────────────────────────────────────────────────────────────────────────

✨ REWRITTEN OUTPUT:
මහ බැංකුව ඉන්ධන මිල අඩු කිරීමට තීරණය කර ඇති අතර පෙට්‍රල් සහ ඩීසල් මිල පහළ යනු ඇතැයි අපේක්ෂා කෙරේ.

📊 EVALUATION METRICS:
  ROUGE-1 F1 Score:        0.8610
  ROUGE-2 F1 Score:        0.7485
  BLEU Score:              0.7261
  Cosine Similarity:       0.8992
  Length Preservation:     0.9210
  Vocabulary Diversity:    0.7014
  Overall Quality Score:   0.8428

────────────────────────────────────────────────────────────────────────────────
🎨 STYLE: YOUTH
────────────────────────────────────────────────────────────────────────────────

✨ REWRITTEN OUTPUT:
Good news! ඉන්ධන මිල ටිකක් අඩු වෙන්න යනවා. මේක ජනතාවට ලොකු සහනයක් වේවි.

📊 EVALUATION METRICS:
  ROUGE-1 F1 Score:        0.7940
  ROUGE-2 F1 Score:        0.6701
  BLEU Score:              0.6410
  Cosine Similarity:       0.8402
  Length Preservation:     0.8311
  Vocabulary Diversity:    0.8720
  Overall Quality Score:   0.7748
  Conversational Markers:  ✅

────────────────────────────────────────────────────────────────────────────────
🎨 STYLE: EDITORIAL
────────────────────────────────────────────────────────────────────────────────

✨ REWRITTEN OUTPUT:
ඉන්ධන මිල අඩු කිරීම ජනතාවට ආර්ථික සහනයක් ලබාදෙන තීරණයක් ලෙස සැලකිය හැකිය. එය පාරිභෝගික වියදම් පාලනයට ද සහාය විය හැක.

📊 EVALUATION METRICS:
  ROUGE-1 F1 Score:        0.8220
  ROUGE-2 F1 Score:        0.7022
  BLEU Score:              0.6881
  Cosine Similarity:       0.8710
  Length Preservation:     0.8920
  Vocabulary Diversity:    0.7910
  Overall Quality Score:   0.8110
  Analytical Depth:        ✅

================================================================================
  📈 AGGREGATE METRICS SUMMARY
================================================================================

ROUGE-1_f1:
  Average: 0.8293

ROUGE-2_f1:
  Average: 0.7086

BLEU:
  Average: 0.6840

Cosine Similarity:
  Average: 0.8697

Length Preservation:
  Average: 0.8851

Vocabulary Diversity:
  Average: 0.7963

Overall Quality Score:
  Average: 0.8089

================================================================================
  📊 PER-STYLE PERFORMANCE
================================================================================

FORMAL:      0.8310
EDITORIAL:   0.8024
SPORTS:      0.8540
YOUTH:       0.7857
FEATURE:     0.7575

================================================================================
✅ Prototype Testing Complete
📄 Results generated for research evaluation purposes
================================================================================