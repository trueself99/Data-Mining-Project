import streamlit as st
import regression
import eda
import ClassificationModel
import Association_Rule_Mining
import Clustering




pages = {
    'Exploratory Data Analysis': eda,
    'Classification Model': ClassificationModel,
    'Regression Model': regression,
    'Association Rule Mining': Association_Rule_Mining,
    'Clustering': Clustering

}
    
def main_app():
    st.set_page_config(page_title="TDS 3301- Data Mining", page_icon='./images/flask.png', layout='wide', initial_sidebar_state='auto')
    
    # Sidebar navigation
    st.sidebar.title('Page Navigation')
    page_selection = st.sidebar.radio('', list(pages.keys()))

    page = pages[page_selection]
    page.app_page()

# Run main function
if __name__ == "__main__":
    main_app()