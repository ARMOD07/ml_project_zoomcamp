from fastapi import FastAPI
import pandas as pd
import pickle
import os
import json
from tensorflow.keras.models import model_from_json
import numpy as np
from pydantic import BaseModel
from typing import List

app = FastAPI(title="ML Prediction Service")

# متغيرات النموذج
model = None
scaler = None

# قائمة الميزات
selected_features = [
    "nb_funding_rounds",
    "nb_investors", 
    "nb_offices",
    "ipo",
    "acquired",
    "milestones",
    "relationships",
    "funding_rounds"
]

# نموذج بيانات الإدخال
class PredictionInput(BaseModel):
    nb_funding_rounds: float = 0
    nb_investors: float = 0
    nb_offices: float = 0
    ipo: float = 0
    acquired: float = 0
    milestones: float = 0
    relationships: float = 0
    funding_rounds: float = 0

# دالة لتحميل النموذج
def load_ml_model():
    """تحميل نموذج ML والـ scaler"""
    global model, scaler
    
    try:
        print("🔍 جاري البحث عن ملفات النموذج...")
        
        # القائمة بالمسارات المحتملة
        possible_paths = [
            # 1. داخل مجلد service الحالي
            {
                "arch": "mlp_model_architecture.json",
                "weights": "mlp_model.weights.h5", 
                "scaler": "mlp_model_scaler.pkl"
            },
            # 2. في مجلد Projects بجانب service
            {
                "arch": "../Projects/mlp_model_architecture.json",
                "weights": "../Projects/mlp_model.weights.h5",
                "scaler": "../Projects/mlp_model_scaler.pkl"
            },
            # 3. داخل مجلد Projects في الحاوية
            {
                "arch": "Projects/mlp_model_architecture.json",
                "weights": "Projects/mlp_model.weights.h5",
                "scaler": "Projects/mlp_model_scaler.pkl"
            }
        ]
        
        loaded = False
        for paths in possible_paths:
            arch_path = paths["arch"]
            weights_path = paths["weights"]
            scaler_path = paths["scaler"]
            
            if (os.path.exists(arch_path) and 
                os.path.exists(weights_path) and 
                os.path.exists(scaler_path)):
                
                print(f"✅ تم العثور على الملفات في: {arch_path}")
                
                # تحميل النموذج
                with open(arch_path, "r") as f:
                    model_json = f.read()
                model = model_from_json(model_json)
                model.load_weights(weights_path)
                
                # تحميل scaler
                with open(scaler_path, "rb") as f:
                    scaler = pickle.load(f)
                
                loaded = True
                print("🎯 تم تحميل النموذج بنجاح!")
                break
        
        if not loaded:
            print("⚠️ لم يتم العثور على ملفات النموذج، جاري إنشاء نموذج وهمي...")
            create_dummy_model()
            
    except Exception as e:
        print(f"❌ خطأ في تحميل النموذج: {e}")
        create_dummy_model()

def create_dummy_model():
    """إنشاء نموذج وهمي للاختبار"""
    global model, scaler
    print("🔄 إنشاء نموذج وهمي للاختبار...")
    
    # نموذج وهمي
    model = "dummy_model"
    
    # scaler وهمي
    import sklearn.preprocessing
    scaler = sklearn.preprocessing.StandardScaler()
    import numpy as np
    scaler.fit(np.random.rand(10, 8))
    
    print("✅ تم إنشاء نموذج وهمي للاختبار")

# تحميل النموذج عند بدء التشغيل
@app.on_event("startup")
def startup_event():
    print("🚀 بدء تشغيل خدمة ML...")
    load_ml_model()
    print(f"📊 الميزات: {selected_features}")

# نقطة البداية
@app.get("/")
def home():
    return {
        "message": "🎯 ML Prediction Service",
        "status": "running",
        "model": "loaded" if model else "not_loaded",
        "features": selected_features,
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict (POST)",
            "test": "/test (GET)"
        }
    }

# فحص الصحة
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "loaded" if model and model != "dummy_model" else "dummy",
        "service": "ml_prediction_api",
        "features_count": len(selected_features)
    }

# نقطة التنبؤ
@app.post("/predict")
def predict(input_data: List[PredictionInput]):
    """
    إجراء تنبؤات متعددة
    """
    try:
        # تحويل البيانات
        data = [item.dict() for item in input_data]
        df = pd.DataFrame(data)
        
        # التأكد من وجود جميع الميزات
        for feature in selected_features:
            if feature not in df.columns:
                df[feature] = 0
        
        # إذا كان النموذج وهمي
        if model == "dummy_model":
            import numpy as np
            predictions = np.random.rand(len(df), 1).tolist()
            return {
                "status": "success",
                "model": "dummy",
                "predictions": predictions,
                "count": len(predictions)
            }
        
        # التنبؤ الحقيقي
        X = df[selected_features]
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled).tolist()
        
        return {
            "status": "success",
            "predictions": predictions,
            "count": len(predictions),
            "features_used": selected_features
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "error_type": type(e).__name__
        }

# نقطة اختبار
@app.get("/test")
def test_endpoint():
    """
    اختبار النموذج ببيانات وهمية
    """
    test_data = [
        PredictionInput(
            nb_funding_rounds=2.0,
            nb_investors=4.0,
            nb_offices=1.0,
            ipo=0.0,
            acquired=0.0,
            milestones=8.0,
            relationships=12.0,
            funding_rounds=2.0
        ),
        PredictionInput(
            nb_funding_rounds=5.0,
            nb_investors=8.0,
            nb_offices=3.0,
            ipo=1.0,
            acquired=0.0,
            milestones=15.0,
            relationships=20.0,
            funding_rounds=4.0
        )
    ]
    
    return predict(test_data)

# نقطة لرؤية بيانات الإدخال المطلوبة
@app.get("/input_schema")
def input_schema():
    """عرض هيكل بيانات الإدخال المطلوب"""
    return {
        "required_fields": selected_features,
        "example": {
            "nb_funding_rounds": 3.0,
            "nb_investors": 5.0,
            "nb_offices": 2.0,
            "ipo": 0.0,
            "acquired": 0.0,
            "milestones": 10.0,
            "relationships": 15.0,
            "funding_rounds": 3.0
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)