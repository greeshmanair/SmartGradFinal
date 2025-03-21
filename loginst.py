import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load student performance dataset
@st.cache_data
def load_data():
    file_path = "student_performance_dataset_modified.xlsx"
    df = pd.read_excel(file_path, engine="openpyxl")
    return df

def main():
    st.title("📚 Teacher Dashboard - Student Performance Analysis")
    
    # Initialize session state for login
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    
    # Teacher Login
    st.sidebar.header("🔐 Teacher Login")
    if not st.session_state.logged_in:
        username = st.sidebar.text_input("Username", "")
        password = st.sidebar.text_input("Password", type="password")
        login_button = st.sidebar.button("Login")
        
        if login_button:
            if username == "harsha@t" and password == "asdf098":
                st.session_state.logged_in = True
                st.sidebar.success("✅ Login successful!")
                st.rerun()
            else:
                st.sidebar.error("❌ Incorrect credentials")
    
    if st.session_state.logged_in:
        df = load_data()
        show_student_list(df)

def show_student_list(df):
    st.subheader("📋 Student List")
    selected_student = st.selectbox("Select a student by Register Number", df["Register Number"].unique())
    
    if selected_student:
        student_data = df[df["Register Number"] == selected_student].iloc[0]
        show_student_performance(student_data)

def show_student_performance(student_data):
    st.subheader(f"📊 Performance Analysis for {student_data['Name']}")
    
    # Display student details
    st.write(f"**Register Number:** {student_data['Register Number']}")
    st.write(f"**Student Name:** {student_data['Name']}")
    st.write(f"**MCQ Score:** {student_data['MCQ Score']}")
    st.write(f"**Subjective Score:** {student_data['Subjective Score']}")
    st.write(f"**Coding Score:** {student_data['Coding Score']}")
    
    # Strong & Weak Topics
    strong_topics = student_data["Strong Topic"]
    weak_topics = student_data["Weak Topic"]
    st.write(f"✅ **Strong Topics:** {strong_topics}")
    st.write(f"⚠️ **Weak Topics:** {weak_topics}")
    
    # Visualization - Performance Scores vs Exam Scores
    scores = [student_data['MCQ Score'], student_data['Subjective Score'], student_data['Coding Score']]
    labels = ['MCQ', 'Subjective', 'Coding']
    
    fig, ax = plt.subplots()
    ax.bar(labels, scores, color=['blue', 'green', 'red'])
    ax.set_ylabel("Score")
    ax.set_title("Performance Score Distribution")
    st.pyplot(fig)

if __name__ == "__main__":
    main()
