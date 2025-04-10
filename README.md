# Diabetic Retinopathy Detection API

This project is a FastAPI-based web service that detects Diabetic Retinopathy (DR) stages from retinal images using a TensorFlow/Keras model. It takes an image as input and returns the predicted class and confidence score.

---

## Features

- Built using FastAPI
- Uses a pre-trained TensorFlow model
- Image processing with OpenCV
- Supports API key authentication
- CORS enabled for frontend integration
- Predicts one of the following DR stages:
  - no_DR
  - mild_DR
  - moderate_DR
  - severe_DR
  - proliferative_DR

---

## Project Structure

- `main.py`: Main FastAPI app
- `my_model.keras`: Trained model file
- `requirements.txt`: Python dependencies
- `README.md`: Documentation

---

## How to Run

1. **Clone the repository**  
   ```
   git clone https://github.com/yourusername/diabetic-retinopathy-api.git
   cd diabetic-retinopathy-api
   ```

2. **Create a virtual environment and activate it**  
   ```
   python -m venv venv
   source venv/bin/activate    # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies**  
   ```
   pip install -r requirements.txt
   ```

4. **Add your model**  
   Place your trained model file named `my_model.keras` in the root folder.

5. **Run the app**  
   ```
   python main.py
   ```
   Or directly with Uvicorn:  
   ```
   uvicorn main:app --host 0.0.0.0 --port 10000
   ```

---

## API Authentication

Every request must include the following header:  
```
x-api-key: nawabBhaikamodel
```

---

## How to Use the API

**Endpoint:** `/predict`  
**Method:** `POST`  
**Headers:**  
- `x-api-key: nawabBhaikamodel`

**Form Data:**  
- `file`: Upload the retinal image (JPG, PNG, etc.)

**Sample Response:**
```json
{
  "prediction": "moderate_DR",
  "confidence": "87.45%"
}
```

---

## Dependencies

- fastapi  
- uvicorn  
- tensorflow  
- opencv-python  
- numpy  
- python-multipart

Install all dependencies with:
```
pip install fastapi uvicorn tensorflow opencv-python numpy python-multipart
```

---

## Note

The line `os.environ["CUDA_VISIBLE_DEVICES"] = "-1"` ensures the model runs on CPU and not GPU.

---

## Author

**Nawab Khan**  
LinkedIn: https://www.linkedin.com/in/nawab-khan-n11  
GitHub: https://github.com/nawabkh2040
