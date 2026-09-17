# pawID — חתול או כלב?

פרויקט משולב: משימת האימון (`train.py` / `predict.py`) + אפליקציית דפדפן שרצה ב-Docker Compose.

המודל מסווג תמונה ל**חתול** או **כלב**.

## דאטה סט

[Dogs vs Cats ב-Kaggle](https://www.kaggle.com/c/dogs-vs-cats) — תת-קבוצה חינוכית מסוננת (cats_and_dogs_filtered).

התמונות נמצאות ב-`data/cats_and_dogs_filtered/` כדי ש-`train.py` יוכל לרוץ בלי הורדה חיצונית.

## משימת האימון

בקובץ אחד מאמנים, ובקובץ אחד בודקים על קלט זר:

```bash
pip install -r requirements.txt
python train.py
python predict.py
```

`predict.py` רץ כברירת מחדל על שלוש דוגמאות זרות ב-`samples/`:

| קובץ | מה בתמונה |
| --- | --- |
| `samples/cat.jpg` | חתול |
| `samples/dog.jpg` | כלב (לברדור) |
| `samples/dog2.jpg` | האסקי |

אפשר גם להעביר נתיבים:

```bash
    python predict.py samples/cat.jpg samples/dog.jpg samples/dog2.jpg
```

המשקולות נשמרות ב-`models/model.pt` (דיוק ולידציה **97.33%** אחרי כיול: 4 epochs + פתיחת הבלוק האחרון ב-MobileNet).

### כיול

ההיפר-פרמטרים מוגדרים בראש `train.py`. תוצאות הכיול:

| בראנץ' / ניסוי | epochs | lr | בלוקים פתוחים | val acc |
| --- | --- | --- | --- | --- |
| ראש הסיווג בלבד | 4 | 1e-3 | 0 | 95.67% |
| **נבחר ל-main** | 4 | 3e-4 | 1 | **97.33%** |

מומלץ לעבוד בבראנצ'ים, לתעד `val_acc` בכל בראנץ', ולמזג ל-`main` את הבראנץ' עם הדיוק הגבוה ביותר.

## פרויקט הדפדפן

אופציה ב' של המרצה: פרונט סטטי + בקאנד FastAPI שרץ עם **uvicorn**. האינפרנס קורה בקונטרולר עצמו. אין צורך לאמן בזמן הבדיקה — רק `model.pt` חייב להיות בריפו.

```bash
docker compose up --build
```

ואז לפתוח [http://localhost:8080](http://localhost:8080), להעלות תמונה ולקבל פרדיקציה.

## מבנה

```
train.py                 אימון
predict.py               בדיקה על קלט זר
data/                    הדאטה סט
samples/                 3 דוגמאות זרות
models/model.pt          משקולות מאומנות
backend/                 FastAPI + uvicorn
frontend/                ממשק העלאת תמונה
docker-compose.yml
```
