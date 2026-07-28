# Phase 2: Exploratory Data Analysis & Preprocessing

## هدف
این فاز به منظور پاکسازی متون استخراج شده (۱۷۰ هزار رکورد جهانی و ۵۳۵۵ رکورد بومی) و انجام تحلیل اکتشافی (EDA) با تمرکز ویژه بر مفهوم «توهم» (Hallucination) طراحی شده است.

## معماری اجرا
1. **eda_analyzer.py**: تحلیل توزیع ستاره‌ها و پروفایلینگ کلمات مرتبط با توهم.
2. **cleaner.py**: اعمال نرمال‌سازی متون (Hazm برای `fa_ir` و Spacy برای `en_us`).

## دیکشنری پایه توهم (Seed Words)
*   **English (en_us):** hallucination, fake, wrong, incorrect, confidently wrong, made up
*   **Persian (fa_ir):** توهم، چرت و پرت، غلط، اشتباه، دروغ، ساختگی، نامربوط