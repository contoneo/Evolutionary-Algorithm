import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import streamlit as st
import Evolutionary_Algorithm as ea
import pandas as pd
import numpy as np

def parse_input(input_text):
    # Split the input by lines and convert each line to a list of values
    rows = input_text.strip().split('\n')
    print(rows)
    if np.array_equal(rows, np.array([''])):
        return np.array([])
    array_data = [list(map(int, row.split(','))) for row in rows]
    # Convert the list of lists to a NumPy array
    array_np = np.array(array_data)
    return np.unique(array_np, axis=0)

def main():
    st.set_page_config(layout="wide")

    st.title("Evolutionary Algorithm")

    st.write("_All time variables are given in milliseconds (ms)_")

    st.subheader('Implementation Variables')
    colA, colD = st.columns(2)
    app_vars = st.container()
    st.divider()
    with app_vars:
        ea.step_2_iterations = colA.number_input('`step_2_iterations`: Number of iterations in Step_2'
                                                , value=ea.step_2_iterations, step=1, format="%d")
        ea.offset = colD.number_input('`offset`: Possible offset of the range (T_Start, t_0) for the creation of samples in step_1'
                                    , value=ea.offset)

    st.subheader('Algorithm Variables')
    col1, col2, col3 = st.columns(3)
    colF, colG, colH, colI, colJ = st.columns(5)
    alg_vars = st.container()
    st.divider()
    with alg_vars:
        ea.T_Start = col1.number_input('`t_Start`: Instant when pre-crash recording time starts.'
                                    , value=ea.T_Start)
        ea.t_0 = col2.number_input('`t_0`: Instant when pre-crash recording time ends.  \
                                    \n(normally set to zero)', value=ea.t_0)
        ea.popsize = col3.number_input('`popsize`: Population size in the Evolutive Algorithm.\
            \nNumber of test samples created with the algorithm', value=ea.popsize)
        ea.lambda_new = col1.number_input('`lambda_new`: New generated samples in each iteration', value=ea.lambda_new)
        ea.V_1_scale = col2.number_input('`V_1_scale`The value obtained from the minimum is divided by 1000\
                                        \nso that the value of V_1 is comparable with the of V_2 (next criteria).'
                                        , value=ea.V_1_scale, format="%f")
        ea.a = colF.number_input('`a`: Factor to weight in $V_1$ \
            \n $|t_{Start} - E_{i [ t_{Start} ]}|$', value=ea.a)
        ea.b = colG.number_input('`b`: Factor to weight in $V_1$ \
            \n $|t_{End} - E_{i [ t_{End} ]}|$', value=ea.b)
        ea.c = colH.number_input('`c`: Factor to weight $V_2$ over $V_1$', value=ea.c)
        ea.mu = colI.number_input('`mu`: Mean in normal distribution: for the standard normal distribution set in zero', value=ea.mu)
        ea.sigma = colJ.number_input('`sigma`: Standard deviation in normal distribution: It is also known as the scale factor', value=ea.sigma)

    col4, col5, col6 = st.columns(3)
    running = st.container()
    df_error_set = pd.DataFrame()
    df_first_test_samples = pd.DataFrame()
    df_test_samples = pd.DataFrame()
    
    st.divider()
    with running:
        input_text = col4.text_area('`error_hit_set` \
                \n Enter rows with format $t_{Start}$ , $t_{End}$  \
                \n Ex: \
                \n-4025, -300\
                \n-3213, -1232')
        if col4.button("Enter `error_hit_set`"):
            try:
                ea.error_hit_set =  parse_input(input_text)
                col4.success("`error_hit_set` input successful!")
            except ValueError:
                col4.error("Invalid input format. Please enter a valid `error_hit_set`.")

        if col5.button("Create Random samples array \
            \n (Step 1)"):
            try:
                ea.first_test_samples = ea.initialize_samples_array()
            except ValueError:
                col5.error("Error generating initial samples array")
                
        if col6.button("Run Evolutionary Algorithm \
            \n (Step 2)"):
            try:
                ea.test_samples = ea.EA()
            except:
                col6.error("Error runnig Evolutionary Algorithm")

        if ea.error_hit_set.size != 0:
            df_error_set = pd.DataFrame(ea.error_hit_set, columns=("t_Start" , "t_End"))
        if ea.first_test_samples.size != 0:
            df_first_test_samples = pd.DataFrame(ea.first_test_samples, columns=("t_Start" , "t_End"))
        if ea.test_samples.size != 0:
            df_test_samples = pd.DataFrame(ea.test_samples, columns=("t_Start" , "t_End"))
        #print(ea.test_samples)
        
        col4.dataframe(df_error_set)
        col5.dataframe(df_first_test_samples)
        col6.dataframe(df_test_samples)

if __name__ == "__main__": 
    main()