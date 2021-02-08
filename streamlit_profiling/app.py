import streamlit as st 
import numpy as np
import pandas as pd
from streamlit_pandas_profiling import  st_profile_report
from pandas_profiling import ProfileReport

#---------------------------------------------------------#
#titulo
st.title('The EDA  App')
st.write('This is the EDA App created using `streamlit` and `pandas-profiling` library')

#-------------------------------------------------------------------#
#main
##upload data
st.sidebar.header('Upload your file')
file = st.sidebar.file_uploader('Upload your input CSV file', type = ['csv'])
st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
""")

## Pandas profiling report

if file is not None:
	@st.cache
	def load_csv():
		csv = pd.read_csv(file)
		return csv

	df = load_csv()
	st.header('**Input DataFrame**')
	st.write(df)
	st.write('---')

	st.header('**Pandas Profiling Report**')
	pr = ProfileReport(df, explorative = True)
	st_profile_report(pr)

else:
	st.info('Awating for CSV file be uploaded')
	if st.button('Press to use Example Dataset'):
		@st.cache
		def load_csv():
			a = pd.DataFrame(np.random.rand(100,5), columns = ['a', 'b', 'c','d','e'])
			return a

		df = load_csv()
		st.header('**Input DataFrame**')
		st.write(df)
		st.write('---')

		st.header('**Pandas Profiling Report**')
		pr = ProfileReport(df, explorative = True)
		st_profile_report(pr)
