from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import sqlite3
from datetime import date, datetime, timedelta
import random

app = FastAPI()

# Инициализация тестовой БД
def init_test_db():
    conn = sqlite3.connect('test.db')
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS products (
        product_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        price REAL NOT NULL
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sales (
        sale_id INTEGER PRIMARY KEY,
        product_id INTEGER NOT NULL,
        sale_date DATE NOT NULL,
        quantity INTEGER NOT NULL,
        FOREIGN KEY (product_id) REFERENCES products (product_id)
    )
    ''')
    
    # Добавляем тестовые данные
    products = [
        (1, 'Ноутбук HP', 'Ноутбуки', 999.99),
        (2, 'Смартфон Samsung', 'Телефоны', 799.99),
        (3, 'Наушники Sony', 'Аксессуары', 149.99)
    ]
    
    cursor.executemany('INSERT OR IGNORE INTO products VALUES (?, ?, ?, ?)', products)
    
    # Генерируем тестовые продажи за последние 90 дней
    for i in range(1, 4):
        for day in range(90):
            sale_date = date.today() - timedelta(days=90-day)
            quantity = random.randint(1, 10)
            cursor.execute(
                'INSERT INTO sales (product_id, sale_date, quantity) VALUES (?, ?, ?)',
                (i, sale_date, quantity)
            )
    
    conn.commit()
    conn.close()

init_test_db()

# Модели Pydantic
class ForecastRequest(BaseModel):
    product_ids: List[int]
    start_date: date
    end_date: date

class ForecastResponse(BaseModel):
    product_id: int
    date: date
    predicted_sales: float

# Загрузка/сохранение модели
def load_model():
    try:
        return joblib.load('test_model.pkl')
    except:
        model = RandomForestRegressor()
        # Инициализация простой моделью
        X = np.array([[1], [2], [3]])
        y = np.array([10, 20, 15])
        model.fit(X, y)
        joblib.dump(model, 'test_model.pkl')
        return model

model = load_model()

@app.post("/forecast", response_model=List[ForecastResponse])
async def get_forecast(request: ForecastRequest):
    try:
        conn = sqlite3.connect('test.db')
        
        # Получаем исторические данные
        query = '''
        SELECT product_id, sale_date, quantity 
        FROM sales 
        WHERE product_id IN ({}) 
        AND sale_date BETWEEN ? AND ?
        '''.format(','.join('?'*len(request.product_ids)))
        
        params = request.product_ids + [
            (date.today() - timedelta(days=90)).isoformat(),
            date.today().isoformat()
        ]
        
        df = pd.read_sql(query, conn, params=params)
        conn.close()
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No historical data found")
        
        # Простое прогнозирование (в реальности было бы сложнее)
        results = []
        current_date = request.start_date
        delta = timedelta(days=1)
        
        while current_date <= request.end_date:
            for product_id in request.product_ids:
                # Простой пример прогноза - случайное число на основе product_id
                prediction = model.predict([[product_id]])[0] * (1 + random.random()/10)
                results.append(ForecastResponse(
                    product_id=product_id,
                    date=current_date,
                    predicted_sales=float(prediction)
                ))
            current_date += delta
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test_data")
async def get_test_data():
    conn = sqlite3.connect('test.db')
    products = pd.read_sql('SELECT * FROM products', conn).to_dict('records')
    sales = pd.read_sql('SELECT * FROM sales LIMIT 10', conn).to_dict('records')
    conn.close()
    return {"products": products, "recent_sales": sales}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)