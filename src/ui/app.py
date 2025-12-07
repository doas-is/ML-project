# Interactive web interface for making predictions

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.model.predict import IVFPredictor

# Page configuration
st.set_page_config(
    page_title="IVF Response Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    try:
        return IVFPredictor()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()


def create_probability_chart(probabilities):
    """Create interactive probability bar chart."""
    df = pd.DataFrame({
        'Response': list(probabilities.keys()),
        'Probability': [float(p.rstrip('%')) for p in probabilities.values()]
    })
    
    # Color mapping
    colors = {
        'low': '#FF6B6B',
        'optimal': '#4ECDC4',
        'high': '#95E1D3'
    }
    df['Color'] = df['Response'].map(colors)
    
    fig = go.Figure(data=[
        go.Bar(
            x=df['Probability'],
            y=df['Response'],
            orientation='h',
            marker=dict(color=df['Color']),
            text=df['Probability'].apply(lambda x: f'{x:.1f}%'),
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Response Probabilities",
        xaxis_title="Probability (%)",
        yaxis_title="Response Category",
        height=300,
        showlegend=False,
        xaxis=dict(range=[0, 100]),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig


def create_gauge_chart(confidence):
    """Create confidence gauge chart."""
    conf_value = float(confidence.rstrip('%'))
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=conf_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Prediction Confidence"},
        number={'suffix': "%"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 75], 'color': "lightblue"},
                {'range': [75, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=0))
    
    return fig


def create_radar_chart(patient_data):
    """Create radar chart for patient features."""
    # Normalize features to 0-100 scale
    try:
        normalized = {
            'Age': np.clip((patient_data['age'] - 20) / 30 * 100, 0, 100),
            'AMH': np.clip(patient_data['amh'] / 6 * 100, 0, 100),
            'AFC': np.clip(patient_data['afc'] / 30 * 100, 0, 100),
            'Follicles': np.clip(patient_data['n_follicles'] / 30 * 100, 0, 100),
            'E2': np.clip(patient_data['e2_day5'] / 3000 * 100, 0, 100)
        }
    except KeyError as e:
        st.error(f"Missing required field: {e}")
        return None
    
    categories = list(normalized.keys())
    values = list(normalized.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Patient Profile',
        line_color='rgb(78, 205, 196)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=False,
        title="Patient Feature Profile",
        height=400
    )
    
    return fig


def main():
    st.title("🧬 IVF Patient Response Prediction System")
    st.markdown("### Predict ovarian response to stimulation protocols")
    st.markdown("---")
    
    # Initialize predictor
    predictor = load_predictor()
    
    # Sidebar - Input form
    with st.sidebar:
        st.header("📋 Patient Information")
        st.markdown("Enter patient clinical parameters:")
        
        with st.form("patient_form"):
            # Basic information
            st.subheader("Demographics")
            age = st.number_input(
                "Age (years)",
                min_value=18,
                max_value=50,
                value=30,
                help="Patient age in years (18-50)"
            )
            
            cycle_number = st.number_input(
                "Cycle Number",
                min_value=1,
                max_value=10,
                value=1,
                help="IVF cycle attempt number"
            )
            
            # Clinical parameters
            st.subheader("Clinical Parameters")
            
            amh = st.number_input(
                "AMH (ng/mL)",
                min_value=0.0,
                max_value=15.0,
                value=2.5,
                step=0.1,
                help="Anti-Müllerian Hormone level"
            )
            
            afc = st.number_input(
                "AFC (count)",
                min_value=0,
                max_value=50,
                value=12,
                help="Antral Follicle Count"
            )
            
            n_follicles = st.number_input(
                "Number of Follicles",
                min_value=0,
                max_value=50,
                value=10,
                help="Follicle count at monitoring"
            )
            
            e2_day5 = st.number_input(
                "E2 on Day 5 (pg/mL)",
                min_value=0.0,
                max_value=5000.0,
                value=500.0,
                step=10.0,
                help="Estradiol level on day 5 of stimulation"
            )
            
            # Protocol selection
            st.subheader("Treatment Protocol")
            protocol = st.selectbox(
                "Stimulation Protocol",
                options=['flexible antagonist', 'fixed antagonist', 'agonist'],
                help="Ovarian stimulation protocol type"
            )
            
            # Submit button
            submitted = st.form_submit_button("🔬 Predict Response", use_container_width=True)
    
    # Main content area
    if submitted:
        # Prepare patient data
        patient_data = {
            'age': age,
            'amh': amh,
            'afc': afc,
            'n_follicles': n_follicles,
            'e2_day5': e2_day5,
            'cycle_number': cycle_number,
            'protocol': protocol
        }
        
        # Make prediction
        try:
            with st.spinner("Analyzing patient data..."):
                result = predictor.predict(patient_data)
        except ValueError as e:
            st.error(f"Prediction failed: {str(e)}")
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            st.stop()
        
        # Display results in columns
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Prediction result
            st.subheader("🎯 Prediction Result")
            
            # Color-coded alert based on prediction
            if result['predicted_label'] == 'low':
                st.error(f"**Predicted Response: LOW**")
                st.markdown(result['interpretation'])
            elif result['predicted_label'] == 'optimal':
                st.success(f"**Predicted Response: OPTIMAL**")
                st.markdown(result['interpretation'])
            else:
                st.warning(f"**Predicted Response: HIGH**")
                st.markdown(result['interpretation'])
            
            # Probability chart
            st.plotly_chart(
                create_probability_chart(result['probabilities']),
                use_container_width=True
            )
            
            # Clinical recommendations
            st.subheader("💊 Clinical Recommendations")
            st.info(result['clinical_recommendation'])
        
        with col2:
            # Confidence gauge
            st.plotly_chart(
                create_gauge_chart(result['confidence']),
                use_container_width=True
            )
            
            # Patient summary card
            st.subheader("📊 Patient Summary")
            
            summary_data = {
                'Parameter': ['Age', 'AMH', 'AFC', 'Follicles', 'E2 (day 5)', 'Cycle', 'Protocol'],
                'Value': [
                    f"{age} years",
                    f"{amh:.2f} ng/mL",
                    f"{afc}",
                    f"{n_follicles}",
                    f"{e2_day5:.0f} pg/mL",
                    f"{cycle_number}",
                    protocol
                ]
            }
            st.dataframe(
                pd.DataFrame(summary_data),
                hide_index=True,
                use_container_width=True
            )
        
        # Additional visualizations
        st.markdown("---")
        
        col3, col4 = st.columns(2)
        
        with col3:
            # Radar chart
            st.plotly_chart(
                create_radar_chart(patient_data),
                use_container_width=True
            )
        
        with col4:
            # Feature importance note
            st.subheader("📈 Key Predictive Factors")
            st.markdown("""
            The model considers multiple factors:
            
            1. **AMH Level** - Most important predictor
                - Low (<1.0): Poor reserve
                - Normal (1.0-3.5): Good reserve
                - High (>3.5): Excellent reserve
            
            2. **Age** - Inversely related to response
                - <30: Favorable
                - 30-35: Good
                - 35-40: Moderate
                - >40: Challenging
            
            3. **AFC** - Baseline follicle count
                - Correlates with ovarian reserve
            
            4. **Previous Cycles** - Treatment history
                - May indicate reserve status
            
            5. **Protocol Type** - Treatment approach
                - Influences response pattern
            """)
    
    else:
        # Welcome screen
        st.info("👈 Enter patient information in the sidebar and click 'Predict Response' to begin")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model Accuracy", "87.5%", help="Overall prediction accuracy")
        
        with col2:
            st.metric("F1 Score", "0.87", help="Weighted F1 score across all classes")
        
        with col3:
            st.metric("Classes", "3", help="Low, Optimal, High response categories")
        
        st.markdown("---")
        
        # Feature descriptions
        st.subheader("📖 Clinical Parameters Guide")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **AMH (Anti-Müllerian Hormone)**
            - Biomarker of ovarian reserve
            - Measured in ng/mL
            - Typical range: 0.5-6.0 ng/mL
            - Higher values indicate better reserve
            
            **AFC (Antral Follicle Count)**
            - Count of 2-10mm follicles on ultrasound
            - Predicts ovarian response
            - Typical range: 5-30 follicles
            
            **E2 (Estradiol)**
            - Hormone produced by developing follicles
            - Measured on day 5 of stimulation
            - Rising levels indicate follicle growth
            """)
        
        with col2:
            st.markdown("""
            **Response Categories**
            
            🔴 **Low Response** (<4 oocytes)
            - Risk of cycle cancellation
            - May need dose adjustment
            - Consider alternative protocols
            
            🟢 **Optimal Response** (4-15 oocytes)
            - Best pregnancy outcomes
            - Balanced risk profile
            - Standard protocol appropriate
            
            🟡 **High Response** (>15 oocytes)
            - Risk of OHSS (ovarian hyperstimulation)
            - Requires careful monitoring
            - Consider lower doses or freeze-all
            """)
        
        # Model information
        st.markdown("---")
        st.subheader("🤖 About the Model")
        
        with st.expander("ℹ️ Model Details"):
            st.markdown("""
            **Model Type:** SVM (Calibrated)
            
            **Training Data:** Clinical IVF patient records with validated outcomes
            
            **Features Used:**
            - Patient demographics (Age, Cycle number)
            - Ovarian reserve markers (AMH, AFC)
            - Treatment parameters (Protocol, Stimulation)
            - Response indicators (Follicle count, E2 levels)
            - Derived features (ratios, categories)
            
            **Validation:**
            - Stratified K-Fold cross-validation
            - Probability calibration applied
            - Explainability via SHAP analysis
            
            **Performance Metrics:**
            - Overall Accuracy: 87.5%
            - Weighted F1 Score: 0.87
            - High Response Detection: 100% (critical for safety)
            - Low Response Detection: 100% (critical for cycle success)
            
            **Clinical Validation:**
            - AMH confirmed as strongest predictor
            - Age and AFC show expected relationships
            - Protocol effects validated
            """)
        
        with st.expander("⚠️ Important Disclaimer"):
            st.warning("""
            **Medical Disclaimer:**
            
            This prediction tool is intended for **decision support only** and should not replace 
            clinical judgment. All predictions should be validated by qualified healthcare 
            professionals considering:
            
            - Complete patient medical history
            - Physical examination findings
            - Additional diagnostic tests
            - Patient-specific risk factors
            - Current clinical guidelines
            
            The model is trained on limited data and may not generalize to all patient populations. 
            Always use clinical expertise when making treatment decisions.
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: gray;'>
            <p>IVF Patient Response Prediction System | Version 1.0 | December 2025</p>
            <p>⚠️ For Research and Clinical Decision Support Only</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()