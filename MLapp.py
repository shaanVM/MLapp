# Import libraries 
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from PIL import Image
import streamlit as st 



# Title
st.title('Machine Learning App')
st.write('-'*100)
#Upload images

image=Image.open('image.png')
st.image(image,use_column_width=True)


def main():
	activities=['EDA','Visualization','Model','About us']
	activity=st.sidebar.selectbox('Select any activity',activities)

	if activity=='EDA':
		st.subheader('Exploratory Data Analysis')
		data=st.file_uploader('Upload dataset',type=['csv','xlxs','json','txt'])
		if data is not None:
			st.success('Dataset uploaded successfully ')
			df=pd.read_csv(data)
			st.dataframe(df.head(50))
			if st.checkbox('Display Shape '):
				st.write(df.shape)
			if st.checkbox('Display Columns'):
				st.write(df.columns)
			if st.checkbox('Select Multiple columns '):
				selected_columns= st.multiselect('Select Multiple columns ',df.columns) 	
				df1=df[selected_columns]
				st.dataframe(df1)
				st.write('Display selected columns ',df1.columns)
				if st.checkbox('Display Shape of selected data '):
					st.write(df1.shape)
			if st.checkbox('Display Summary '):
				st.write(df.describe().T)	
			if st.checkbox('Check Null Values '):
				st.write(df.isnull().sum())	
			if st.checkbox('Display Datatypes'):
				st.write(df.dtypes) 
			if st.checkbox('Display Correlation of various columns '):
				st.write(df.corr())			

# Visualization part

    
	elif activity=='Visualization':
		st.subheader('Data Visualization')
		data=st.file_uploader('Upload Dataset ',type=['csv','xlsx'])
		if data is not None:
			st.success('Dataset uploaded successfully')
			df=pd.read_csv(data)
			st.dataframe(df)
			if st.checkbox('Select Multiple columns'):
				selected_columns=st.multiselect('Select columns ',df.columns)
				df1=df[selected_columns]
				st.dataframe(df1)
			st.set_option('deprecation.showPyplotGlobalUse', False)	
			if st.checkbox('Display Heatmap of selected features'):
				st.write(sns.heatmap(df1.corr(),annot=True,cmap='viridis'))
				st.pyplot()
			if st.checkbox('Display Pair plot '):
				st.write(sns.pairplot(df1,diag_kind='kde'))
				st.pyplot()		
			if st.checkbox('Display Pie chart '):
				all_columns=df.columns.to_list()
				pie_columns=st.selectbox('select columns to display ',all_columns)
				pie_charts=df[pie_columns].value_counts().plot.pie()
				st.write(pie_charts)
				st.pyplot()


















	elif activity=='Model':
		st.subheader('Model Building')










	elif activity=='About us':
		st.subheader('About us')
		st.markdown('This is an interactive web page for ML project where you can view just by uploading dataset')
		st.markdown('1) Exploratory data Analysis')
		st.markdown('2) Data Visualization ')
		st.markdown('3) Model Building   -- Upload dataset for Classification') 

		st.balloons()
	else:
		st.warning('Please select an activity')



















if __name__ == '__main__':
		main()	