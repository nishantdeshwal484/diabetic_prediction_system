import gradio as gr
import pickle
import numpy as np

# Load the trained KNN model
try:
    with open('diabetes_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    raise Exception("Model file 'diabetes_model.pkl' not found. Run train_model.py first.")

# Prediction Function
def predict_diabetes(pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age):
    # Format inputs into a 2D numpy array
    input_features = np.array([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]])
    
    # Predict outcome
    prediction = model.predict(input_features)
    
    # Return formatted result
    if prediction[0] == 1:
        return "⚠️ Diabetic (High Risk)"
    else:
        return "✅ Non-Diabetic (Low Risk)"

# Build the beautiful Gradio UI
# Using a soft theme with teal accents for a medical/clean look
with gr.Blocks(theme=gr.themes.Soft(primary_hue="teal", secondary_hue="blue")) as app:
    
    # Header Section
    gr.Markdown(
        """
        <div style="text-align: center; max-width: 800px; margin: 0 auto;">
            <h1>🩺 Diabetes Prediction System</h1>
            <p>Enter patient medical information below to determine diabetes risk utilizing our K-Nearest Neighbors Machine Learning model.</p>
        </div>
        """
    )
    
    # Input Fields (Organized in two columns for better UX)
    with gr.Row():
        with gr.Column():
            pregnancies = gr.Number(label="Pregnancies", minimum=0, step=1, value=0)
            glucose = gr.Number(label="Glucose Level", minimum=0, value=120)
            bp = gr.Number(label="Blood Pressure (mm Hg)", minimum=0, value=70)
            skin_thickness = gr.Number(label="Skin Thickness (mm)", minimum=0, value=20)
        
        with gr.Column():
            insulin = gr.Number(label="Insulin (mu U/ml)", minimum=0, value=79)
            bmi = gr.Number(label="BMI", minimum=0.0, value=25.0)
            dpf = gr.Number(label="Diabetes Pedigree Function", minimum=0.0, value=0.5)
            age = gr.Number(label="Age (years)", minimum=1, step=1, value=30)
            
    # Action Button and Output Display
    with gr.Row():
        submit_btn = gr.Button("Predict Status", variant="primary", scale=1)
        
    with gr.Row():
        output = gr.Textbox(label="Prediction Result", text_align="center", scale=1)
    
    # Trigger prediction on click
    submit_btn.click(
        fn=predict_diabetes,
        inputs=[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age],
        outputs=output
    )

if __name__ == "__main__":
    app.launch(share=True)